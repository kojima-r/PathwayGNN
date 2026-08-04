"""Cancer drug-response (GraphCDRScan) preprocessing.

Reads the upstream GraphCDRScan bundle under ``data_cdr/processed/<folder>`` and
writes one prepared dataset in :mod:`pathwaygnn.data.format`:

* graph — the Reactome functional-interaction network as
  ``scripts/cdr/upstream/prepare_data.py`` encoded it (already undirected and
  self-loop free), with ``vertices_dic.tsv`` / ``relationships_dic.tsv`` supplying
  the node and relation names.
* node_feature ``mutation`` (sparse) — mutations per cancer-gene-census gene of the
  sample's cell line. A GDSC sample is a *(cell line, compound)* pair, so every
  compound screened against one cell line repeats the same mutation profile;
  identical profiles are stored once and ``rows/mutation.npy`` maps sample -> row.
* sample_features — the GraphCDRScan sample-feature vector verbatim: the 96/78/83
  mutational spectra of the cell line, its primary-site one-hot and the
  3 x 1024-bit compound fingerprint.
* groups — the cell line's primary site, so per-group AUC is per cancer type.
* tasks ``sensitive_drugwise`` and ``sensitive_global`` — the GDSC ``LN_IC50``
  binarised against the compound's own median and against the global median.

The upstream bundle carries ``LN_IC50`` as a real number; ``pathwaygnn`` trains
binary problems only, so the threshold is part of preprocessing and is recorded
in each ``task.json``. ``sensitive_drugwise`` is the interesting task: splitting
each compound at its own median removes the compound's overall potency from the
label, so the only usable signal is the cell line's genomics.
"""

from __future__ import annotations

from array import array
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pathwaygnn.data.format import DatasetWriter, TaskNodeFeature

SOURCE_FILES = {
    "vertices": "vertices_dic.tsv",
    "relations": "relationships_dic.tsv",
    "graph": "graph.tsv",
    "node_features": "node_features.tsv",
    "sample_features": "sample_features.tsv",
    "labels": "labels.tsv",
}

# `prepare_data.py:save_sample_features` writes one row as
#   sample_id, cancer_type, <spectra>, <primary-site one-hot>, <fingerprint bits>
SPECTRA_BLOCKS = (("spectra96", 96), ("spectra78", 78), ("spectra83", 83))
SPECTRA_DIM = sum(size for _, size in SPECTRA_BLOCKS)
FINGERPRINT_BITS = 3072

# `pd.get_dummies` sorts its categories, so the one-hot block is the primary
# sites of the used cell lines in alphabetical order. Verified against
# `data_cdr/raw/CosmicCLP_MutantExport.tsv` restricted to `used_cell_lines.csv`;
# a bundle with a different number of sites falls back to positional names.
PRIMARY_SITES = (
    "Bladder",
    "Bone",
    "Breast",
    "Central Nervous System",
    "Cervix",
    "Esophagus",
    "Haematopoietic and Lymphoid",
    "Head and Neck",
    "Kidney",
    "Large Intestine",
    "Liver",
    "Lung",
    "Ovary",
    "Pancreas",
    "Peripheral Nervous System",
    "Skin",
    "Stomach",
    "Thyroid",
    "Vulva",
)

DRUGWISE, GLOBAL = "sensitive_drugwise", "sensitive_global"


def _source(source_dir: Path, key: str) -> Path:
    path = source_dir / SOURCE_FILES[key]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run the upstream preparation first "
            f"(`bash scripts/cdr/prepare.sh`, which writes {source_dir})"
        )
    return path


def _read_dictionary(path: Path) -> list[str]:
    """``name<TAB>index`` -> a list indexed by ``index``."""
    pairs: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            name, _, index = line.rstrip("\n").rpartition("\t")
            pairs[int(index)] = name
    if sorted(pairs) != list(range(len(pairs))):
        raise ValueError(f"{path} does not number its entries 0..{len(pairs) - 1}")
    return [pairs[index] for index in range(len(pairs))]


def _read_graph(path: Path, num_nodes: int, num_relations: int):
    """``src<TAB>relation<TAB>dst`` -> sorted, de-duplicated edge tensors."""
    edges: set[tuple[int, int, int]] = set()
    rows = self_loops = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            source, relation, destination = (int(item) for item in line.split("\t"))
            rows += 1
            if not (0 <= source < num_nodes and 0 <= destination < num_nodes):
                raise ValueError(f"{path} references a node outside 0..{num_nodes - 1}")
            if not 0 <= relation < num_relations:
                raise ValueError(f"{path} references a relation outside 0..{num_relations - 1}")
            if source == destination:
                self_loops += 1
                continue
            edges.add((source, relation, destination))
    ordered = sorted(edges)
    edge_index = torch.tensor(
        [[source, destination] for source, _, destination in ordered], dtype=torch.long
    ).t().contiguous()
    edge_type = torch.tensor([relation for _, relation, _ in ordered], dtype=torch.long)
    # `graph_generator.py` already symmetrises the network, so this only records
    # whether that still holds rather than adding the reverse edges itself.
    symmetric = all((destination, relation, source) in edges for source, relation, destination in ordered)
    stats = {
        "graph_rows": rows,
        "graph_self_loops_dropped": self_loops,
        "graph_duplicate_rows_dropped": rows - self_loops - len(edges),
        "graph_symmetric": bool(symmetric),
    }
    return edge_index, edge_type, stats


def _read_labels(path: Path) -> np.ndarray:
    """``sample<TAB>LN_IC50<TAB>IC50`` -> ``LN_IC50`` indexed by sample."""
    values: dict[int, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split("\t")
            values[int(parts[0])] = float(parts[1])
    if sorted(values) != list(range(len(values))):
        raise ValueError(f"{path} does not number its samples 0..{len(values) - 1}")
    return np.asarray([values[index] for index in range(len(values))], dtype=np.float64)


def _sample_feature_names(num_sites: int) -> list[str]:
    names = [f"{block}_{position}" for block, size in SPECTRA_BLOCKS for position in range(size)]
    names += [
        f"site_{PRIMARY_SITES[position]}" if num_sites == len(PRIMARY_SITES) else f"site_{position}"
        for position in range(num_sites)
    ]
    names += [f"fingerprint_{position}" for position in range(FINGERPRINT_BITS)]
    return names


def _read_sample_features(path: Path, sample_feature_path: Path, num_samples: int):
    """Stream the sample table into a memmapped sample-level feature matrix.

    Returns the primary-site code per sample, the compound and cell-line
    identities recovered from the row (the upstream bundle drops ``DRUG_ID`` and
    ``COSMIC_ID``, but the fingerprint identifies the compound and the spectra
    identify the cell line) and the site-code -> one-hot-position mapping.
    """
    with path.open(encoding="utf-8") as handle:
        width = len(handle.readline().rstrip("\n").split("\t")) - 2
    num_sites = width - SPECTRA_DIM - FINGERPRINT_BITS
    if num_sites < 0:
        raise ValueError(
            f"{path} has {width} feature columns, fewer than the {SPECTRA_DIM} spectra plus "
            f"{FINGERPRINT_BITS} fingerprint bits this reader expects"
        )
    sample_feature_path.parent.mkdir(parents=True, exist_ok=True)
    sample_features = np.lib.format.open_memmap(
        sample_feature_path, mode="w+", dtype=np.float32, shape=(num_samples, width)
    )
    site_code = np.full(num_samples, -1, dtype=np.int64)
    drug_of = np.full(num_samples, -1, dtype=np.int64)
    cell_of = np.full(num_samples, -1, dtype=np.int64)
    drugs: dict[str, int] = {}
    cells: dict[str, int] = {}
    code_to_position: dict[int, int] = {}
    fingerprint_start = 2 + SPECTRA_DIM + num_sites
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            index = int(parts[0])
            if not 0 <= index < num_samples:
                raise ValueError(f"{path} holds sample {index}, outside 0..{num_samples - 1}")
            sample_features[index] = np.array(parts[2:], dtype=np.float32)
            code = int(parts[1])
            site_code[index] = code
            hot = [
                position
                for position, value in enumerate(parts[2 + SPECTRA_DIM : fingerprint_start])
                if value not in ("0", "False")
            ]
            if len(hot) == 1:
                previous = code_to_position.setdefault(code, hot[0])
                if previous != hot[0]:
                    raise ValueError(
                        f"{path}: cancer type {code} maps to one-hot positions "
                        f"{previous} and {hot[0]}"
                    )
            drug_of[index] = drugs.setdefault("".join(parts[fingerprint_start:]), len(drugs))
            cell_of[index] = cells.setdefault("\t".join(parts[1 : 2 + SPECTRA_DIM]), len(cells))
    sample_features.flush()
    del sample_features
    if (site_code < 0).any():
        raise ValueError(f"{path} does not cover every sample of the label table")
    return site_code, drug_of, cell_of, code_to_position, num_sites


def _read_node_features(path: Path, num_samples: int, num_nodes: int, binary: bool):
    """Stream the mutated-gene table into one CSR row per distinct profile."""
    rows = np.full(num_samples, -1, dtype=np.int64)
    profiles: dict[bytes, int] = {}
    pointer, gene, value = [0], array("q"), array("f")
    current = -1
    counts: Counter[int] = Counter()
    total_rows = 0

    def flush(sample: int) -> None:
        items = sorted(counts.items())
        key = np.asarray(items, dtype=np.int64).tobytes()
        index = profiles.get(key)
        if index is None:
            index = len(profiles)
            profiles[key] = index
            for node, count in items:
                gene.append(node)
                value.append(1.0 if binary else float(count))
            pointer.append(len(gene))
        rows[sample] = index

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            sample_text, node_text, _ = line.split("\t", 2)
            sample, node = int(sample_text), int(node_text)
            total_rows += 1
            if not 0 <= node < num_nodes:
                raise ValueError(f"{path} references node {node}, outside 0..{num_nodes - 1}")
            if sample != current:
                if current >= 0:
                    flush(current)
                if rows[sample] >= 0:
                    raise ValueError(f"{path} is not grouped by sample (saw {sample} twice)")
                current, counts = sample, Counter()
            counts[node] += 1
    if current >= 0:
        flush(current)
    missing = int((rows < 0).sum())
    if missing:
        raise ValueError(f"{path} covers {num_samples - missing} of {num_samples} samples")
    stats = {
        "node_feature_rows": total_rows,
        "distinct_mutation_profiles": len(profiles),
        "mutation_values": len(gene),
    }
    return (pointer, gene, value), rows, stats


def _binarise(ln_ic50: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """1 = sensitive, i.e. ``LN_IC50`` strictly below its reference median."""
    return (ln_ic50 < reference).astype(np.float32)


def _drugwise_median(ln_ic50: np.ndarray, drug_of: np.ndarray) -> np.ndarray:
    reference = np.empty_like(ln_ic50)
    for drug in np.unique(drug_of):
        mask = drug_of == drug
        reference[mask] = np.median(ln_ic50[mask])
    return reference


def prepare_cdr_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    binary_mutations: bool = False,
    tasks: tuple[str, ...] = (DRUGWISE, GLOBAL),
) -> dict[str, Any]:
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    unknown = [name for name in tasks if name not in (DRUGWISE, GLOBAL)]
    if unknown:
        raise ValueError(f"unknown cdr task(s) {unknown}; available: {DRUGWISE!r}, {GLOBAL!r}")

    nodes = _read_dictionary(_source(source_dir, "vertices"))
    relations = _read_dictionary(_source(source_dir, "relations"))
    edge_index, edge_type, graph_stats = _read_graph(
        _source(source_dir, "graph"), len(nodes), len(relations)
    )
    ln_ic50 = _read_labels(_source(source_dir, "labels"))
    num_samples = int(ln_ic50.size)

    writer = DatasetWriter(
        output_dir,
        "cdr",
        source={
            "source_dir": str(source_dir),
            "num_samples": num_samples,
            "binary_mutations": bool(binary_mutations),
            **graph_stats,
        },
    )
    writer.write_graph(edge_index, edge_type, nodes, relations)

    sample_feature_path = output_dir / "sample_features.npy"
    site_code, drug_of, cell_of, code_to_position, num_sites = _read_sample_features(
        _source(source_dir, "sample_features"), sample_feature_path, num_samples
    )
    sample_feature_names = _sample_feature_names(num_sites)
    positions = sorted(code_to_position.values())
    if len(set(positions)) != len(positions) or not all(0 <= p < num_sites for p in positions):
        raise ValueError("the primary-site one-hot block does not identify every cancer type once")
    # Report groups in the one-hot (alphabetical) order rather than the
    # order-of-appearance codes the upstream bundle stores.
    groups = np.asarray([code_to_position[int(code)] for code in site_code], dtype=np.int64)
    site_names = [name.removeprefix("site_") for name in sample_feature_names[SPECTRA_DIM : SPECTRA_DIM + num_sites]]

    csr, mutation_rows, node_stats = _read_node_features(
        _source(source_dir, "node_features"), num_samples, len(nodes), binary_mutations
    )
    writer.sparse_node_feature("mutation", *csr)
    writer.source.update(
        {
            "num_compounds": int(drug_of.max()) + 1,
            "num_cell_lines": int(cell_of.max()) + 1,
            "num_primary_sites": num_sites,
            "sample_feature_dim": len(sample_feature_names),
            **node_stats,
        }
    )

    sample_features = np.load(sample_feature_path, mmap_mode="r")
    thresholds = {
        DRUGWISE: _drugwise_median(ln_ic50, drug_of),
        GLOBAL: np.full_like(ln_ic50, float(np.median(ln_ic50))),
    }
    written = {}
    for seed_offset, name in enumerate(tasks):
        labels = _binarise(ln_ic50, thresholds[name])
        task_manifest = writer.write_task(
            name,
            labels,
            node_features={"mutation": TaskNodeFeature("mutation", mutation_rows)},
            sample_features=sample_features,
            sample_feature_names=sample_feature_names,
            groups=groups,
            group_names=site_names,
            seed_offset=seed_offset,
            source={
                "label": "LN_IC50 below its reference median (1 = sensitive)",
                "reference": (
                    "median LN_IC50 of the same compound"
                    if name == DRUGWISE
                    else "median LN_IC50 over every sample"
                ),
                "ln_ic50_median": float(np.median(ln_ic50)),
                "ln_ic50_min": float(ln_ic50.min()),
                "ln_ic50_max": float(ln_ic50.max()),
                "num_compounds": int(drug_of.max()) + 1,
                "num_cell_lines": int(cell_of.max()) + 1,
            },
        )
        # 3348 sample-level feature names would bury the summary the CLI prints.
        written[name] = {
            key: value for key, value in task_manifest.items() if key != "sample_feature_names"
        }
    del sample_features
    sample_feature_path.unlink()
    manifest = writer.finish()
    manifest["tasks_written"] = written
    return manifest
