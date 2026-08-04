"""Target-repositioning (SLGCN-TR) preprocessing.

Reads the bundle written by ``pathwaygnn-data tr-build-processed`` — the graph,
the disease signature, the knockdown and overexpression perturbation signatures
and the two label tables — and writes one prepared dataset:

* node_features ``disease``, ``perturbation_kd``, ``perturbation_oe`` (sparse)
* tasks ``kd_inh`` and ``oe_act``, each binding the alias ``perturbation`` to its
  own signature table and sharing the single ``disease`` table
* groups: the disease each sample targets, so that downstream per-group metrics
  and attributions are reported per disease

A perturbation profile is keyed by ``(pert_iname, cell_id)``, so one label row
(gene, disease) becomes one sample per cell line in which that gene was
perturbed — the join the reference implementation performs when it builds its
feature table. A signature table without a ``cell_id`` column (``per_cell_line:
false`` at build time, which averages the cell lines away) still reads, and then
produces exactly one sample per label row.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from pathwaygnn.data.format import DatasetWriter, TaskNodeFeature

SOURCE_FILES = {
    "graph": ("graph.tsv", "PathwayCommons12.All.hgnc.sif", "PathwayCommons12.All.hgnc.sif.tsv"),
    "disease": ("disease_specific_signature.tsv",),
    "kd_signature": ("knockdown_signature.tsv", "knockdown_signature_sample.tsv"),
    "oe_signature": ("overexpression_signature.tsv", "overexpression_signature_sample.tsv"),
    "kd_label": ("inhibitory_target_disease.tsv",),
    "oe_label": ("activatory_target_disease.tsv",),
}
DISEASE_GENE_COLUMNS = ("gene_name", "human_gene_name")
ROW_SEPARATOR = "|"
TASKS = (
    ("kd_inh", "kd_signature", "kd_label", "perturbation_kd"),
    ("oe_act", "oe_signature", "oe_label", "perturbation_oe"),
)


def _find(source_dir: Path, names: Iterable[str]) -> Path:
    for name in names:
        candidate = source_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"None of {list(names)} exists under {source_dir}")


def _read_graph(path: Path) -> tuple[list[str], list[str], torch.Tensor, torch.Tensor]:
    genes: set[str] = set()
    relations: set[str] = set()
    edges: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 3:
                continue
            src, relation, dst = row[:3]
            genes.update((src, dst))
            relations.add(relation)
            edges.add((src, relation, dst))
            edges.add((dst, relation, src))
    gene_names, relation_names = sorted(genes), sorted(relations)
    gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    relation_to_idx = {name: idx for idx, name in enumerate(relation_names)}
    ordered = sorted(edges)
    edge_index = torch.tensor(
        [[gene_to_idx[src], gene_to_idx[dst]] for src, _, dst in ordered],
        dtype=torch.long,
    ).t().contiguous()
    edge_type = torch.tensor(
        [relation_to_idx[relation] for _, relation, _ in ordered], dtype=torch.long
    )
    return gene_names, relation_names, edge_index, edge_type


def _read_disease(
    path: Path, gene_to_idx: dict[str, int], cutoff: float
) -> tuple[list[str], dict[str, list[tuple[int, float]]], int]:
    values: dict[str, list[tuple[int, float]]] = {}
    skipped = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        gene_column = next(
            (name for name in DISEASE_GENE_COLUMNS if name in (reader.fieldnames or [])), None
        )
        if gene_column is None:
            raise KeyError(f"{path.name} has none of the gene columns {DISEASE_GENE_COLUMNS}")
        for row in reader:
            gene_idx = gene_to_idx.get(row[gene_column])
            value = float(row["expression"])
            if gene_idx is None:
                skipped += 1
            elif abs(value) >= cutoff:
                values.setdefault(row["do_id"], []).append((gene_idx, value))
    return sorted(values), values, skipped


def _read_signature(
    path: Path, gene_to_idx: dict[str, int], cutoff: float
) -> tuple[list[str], dict[str, list[tuple[int, float]]], int, dict[str, list[str]]]:
    """Read one perturbation table.

    Its key columns are ``pert_iname`` plus ``cell_id`` when the bundle keeps the
    cell lines apart; a row is then named ``"<gene>|<cell line>"``. The returned
    ``by_gene`` maps a perturbed gene to every row it owns, which is what the
    label join expands over.
    """
    values: dict[str, list[tuple[int, float]]] = {}
    by_gene: dict[str, list[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        keys = 2 if len(header) > 1 and header[1] == "cell_id" else 1
        graph_columns = [
            (gene_to_idx[column], offset)
            for offset, column in enumerate(header[keys:], start=keys)
            if column in gene_to_idx
        ]
        skipped = len(header) - keys - len(graph_columns)
        for row in reader:
            if not row:
                continue
            sparse: list[tuple[int, float]] = []
            for gene_idx, offset in graph_columns:
                value = float(row[offset])
                if abs(value) >= cutoff:
                    sparse.append((gene_idx, value))
            name = ROW_SEPARATOR.join(row[:keys])
            values[name] = sparse
            by_gene.setdefault(row[0], []).append(name)
    return sorted(values), values, skipped, by_gene


def _pack(names: list[str], values: dict[str, list[tuple[int, float]]]):
    ptr, gene, value = [0], [], []
    for name in names:
        for pair in sorted(values[name]):
            gene.append(pair[0])
            value.append(pair[1])
        ptr.append(len(gene))
    return ptr, gene, value


def _read_labels(
    path: Path,
    pert_names: list[str],
    disease_names: list[str],
    by_gene: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Expand each (gene, disease) label into one sample per profile of that gene."""
    pert_to_idx = {name: idx for idx, name in enumerate(pert_names)}
    disease_to_idx = {name: idx for idx, name in enumerate(disease_names)}
    pert, disease, label = [], [], []
    skipped, label_rows = 0, 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows = by_gene.get(row["gene"], ())
            if not rows or row["doid"] not in disease_to_idx:
                skipped += 1
                continue
            label_rows += 1
            for name in rows:
                pert.append(pert_to_idx[name])
                disease.append(disease_to_idx[row["doid"]])
                label.append(float(row["label"]))
    return (
        np.asarray(pert, dtype=np.int64),
        np.asarray(disease, dtype=np.int64),
        np.asarray(label, dtype=np.float32),
        skipped,
        label_rows,
    )


def prepare_tr_dataset(
    source_dir: str | Path, output_dir: str | Path, cutoff: float = 1e-7
) -> dict[str, Any]:
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    paths = {key: _find(source_dir, names) for key, names in SOURCE_FILES.items()}
    genes, relations, edge_index, edge_type = _read_graph(paths["graph"])
    gene_to_idx = {name: idx for idx, name in enumerate(genes)}
    writer = DatasetWriter(
        output_dir, "tr", source={"source_dir": str(source_dir), "cutoff": cutoff}
    )
    writer.write_graph(edge_index, edge_type, genes, relations)

    disease_names, disease_values, disease_skipped = _read_disease(
        paths["disease"], gene_to_idx, cutoff
    )
    writer.sparse_node_feature("disease", *_pack(disease_names, disease_values))
    writer.source.update(
        {"num_diseases": len(disease_names), "disease_rows_skipped": disease_skipped}
    )

    for task_name, signature_key, label_key, node_feature in TASKS:
        pert_names, pert_values, genes_skipped, by_gene = _read_signature(
            paths[signature_key], gene_to_idx, cutoff
        )
        writer.sparse_node_feature(node_feature, *_pack(pert_names, pert_values))
        pert, disease, label, labels_skipped, label_rows = _read_labels(
            paths[label_key], pert_names, disease_names, by_gene
        )
        writer.write_task(
            task_name,
            label,
            node_features={
                "perturbation": TaskNodeFeature(node_feature, pert),
                "disease": TaskNodeFeature("disease", disease),
            },
            groups=disease,
            group_names=disease_names,
            source={
                "num_perturbations": len(pert_names),
                "num_perturbed_genes": len(by_gene),
                "signature_genes_skipped": genes_skipped,
                "label_rows_used": label_rows,
                "label_rows_skipped": labels_skipped,
                "perturbations": pert_names,
            },
        )
    return writer.finish()
