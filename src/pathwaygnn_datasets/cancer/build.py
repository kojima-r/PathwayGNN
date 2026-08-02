"""Build the upstream TCGA bundle (``data_cancer/processed``) from raw sources.

``cancer-prepare`` converts an already-built bundle into the generic format.
This module builds that bundle in the first place, so ``data_cancer/processed``
stops being an opaque input:

    rawdata_TCGA/ + PathwayCommons SIF  --cancer-build-processed-->  processed/
    processed/                          --cancer-prepare--------->   prepared/

The steps follow Inoue et al. section 2.2-2.3:

* **graph** — the PathwayCommons SIF, symmetrised, with gene symbols replaced by
  numeric HGNC ids where the HGNC export has one (CHEBI and other unmapped names
  are kept verbatim), then integer-encoded.
* **genes** — the expression rows restricted to the cancer-related gene sets
  (MSigDB plus the LM22 immune signature in the paper) and then to genes that
  exist as graph nodes.
* **samples** — every recount2 TCGA column joined to its patient through
  ``TCGA_ID.tsv`` and to the TCGA-CDR clinical table. Patients surviving beyond
  the 95th percentile of the pooled deceased-and-long-censored times are dropped,
  then per verification year the samples censored inside that year are dropped and
  the rest labelled 1 (survived) or 0.

Two deliberate differences from the shipped bundle are recorded in
``build_manifest.json`` rather than hidden:

1. **The node and relation encodings are sorted.** The shipped bundle numbered
   them in Python set-iteration order, which is not reproducible across processes.
   A rebuilt bundle is therefore self-consistent but *not* interchangeable with
   the shipped one, and an encoder pre-trained on one does not transfer.
2. **No serialized-header row.** ``<n>years_sample.tsv`` in the shipped bundle
   starts with a row of ``0..34``; that is an artifact, so a rebuilt bundle has
   one row per real sample and its per-year counts sit a handful below
   ``PAPER_SAMPLE_COUNTS`` (see ``strict_sample_counts`` in ``cancer-prepare``).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from pathwaygnn_datasets.cancer.paper import CANCER_TYPES

TRANSFORMS = ("log1p", "log1p_tpm")


def _log(**fields: Any) -> None:
    print(json.dumps({"stage": "cancer_build", **fields}), flush=True)


def _rows(path: Path, delimiter: str = "\t", encoding: str = "utf-8"):
    """Stream a delimited file as dicts; TCGA exports are not all valid UTF-8."""
    with Path(path).open(encoding=encoding, errors="replace", newline="") as handle:
        yield from csv.DictReader(handle, delimiter=delimiter)


def _number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --- HGNC identifiers -------------------------------------------------------

def read_hgnc_table(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """An HGNC complete-set export -> (symbol -> id, Ensembl gene -> id).

    Ids are the bare numbers the bundle uses, not the ``HGNC:`` form.
    """
    by_symbol: dict[str, str] = {}
    by_ensembl: dict[str, str] = {}
    for row in _rows(path):
        identifier = str(row.get("HGNC ID", "")).split("HGNC:")[-1].strip()
        if not identifier:
            continue
        symbol = str(row.get("Approved symbol", "")).strip()
        ensembl = str(row.get("Ensembl gene ID", "")).strip()
        if symbol:
            by_symbol[symbol] = identifier
        if ensembl:
            by_ensembl[ensembl] = identifier
    if not by_symbol:
        raise ValueError(f"{path} has no 'Approved symbol' column; it is not an HGNC export")
    return by_symbol, by_ensembl


# --- graph ------------------------------------------------------------------

def build_graph(sif_path: Path, by_symbol: dict[str, str], output: Path) -> dict[str, Any]:
    """Symmetrise the SIF, rename to HGNC ids and write the three graph files."""
    edges: set[tuple[str, str, str]] = set()
    with Path(sif_path).open(encoding="utf-8", newline="") as handle:
        # The `.sif` export has no header, the `.txt` one starts with
        # PARTICIPANT_A; skipping that row covers both.
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 3 or row[0] == "PARTICIPANT_A":
                continue
            source, relation, destination = (item.strip() for item in row[:3])
            if not source or not destination:
                continue
            edges.add((source, relation, destination))
            edges.add((destination, relation, source))
    names = {name for source, _, destination in edges for name in (source, destination)}
    # Unmapped names (CHEBI:*, complexes) keep their original label.
    renamed = {name: by_symbol.get(name, name) for name in names}
    # Sorted, so the encoding does not depend on set iteration order.
    nodes = sorted({renamed[name] for name in names})
    relations = sorted({relation for _, relation, _ in edges})
    node_index = {name: index for index, name in enumerate(nodes)}
    relation_index = {name: index for index, name in enumerate(relations)}
    # Renaming can merge two SIF names onto one HGNC id, which both duplicates
    # triples and creates self loops; drop both.
    triples = sorted({
        (node_index[renamed[source]], relation_index[relation], node_index[renamed[destination]])
        for source, relation, destination in edges
        if renamed[source] != renamed[destination]
    })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "graph.tsv").open("w", newline="") as handle:
        handle.write("".join(f"{a}\t{r}\t{b}\n" for a, r, b in triples))
    for name, table in (("vertices_dic.tsv", node_index), ("relationships_dic.tsv", relation_index)):
        with (output / name).open("w", newline="") as handle:
            handle.write("".join(f"{key}\t{value}\n" for key, value in table.items()))
    stats = {
        "num_nodes": len(nodes),
        "num_relations": len(relations),
        "num_edges": len(triples),
        "nodes_renamed_to_hgnc": sum(1 for name in names if renamed[name] != name),
    }
    _log(step="graph", **stats)
    return {"node_index": node_index, **stats}


# --- genes ------------------------------------------------------------------

def read_gene_sets(paths: Sequence[str | Path]) -> set[str]:
    """Gene symbols from ``.gmt`` files or one-column lists."""
    symbols: set[str] = set()
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing. `gene_sets` selects the cancer-related genes "
                "(MSigDB and LM22 in the paper); both need a manual download."
            )
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            for line in handle:
                fields = [item.strip() for item in line.rstrip("\n").split("\t")]
                # .gmt: name, description, gene, gene, ...
                chosen = fields[2:] if path.suffix.lower() == ".gmt" else fields[:1]
                symbols.update(item for item in chosen if item)
    return symbols


def read_gene_ids(path: Path) -> tuple[list[str], np.ndarray | None]:
    """The ordered identifiers of the expression rows, optionally with lengths."""
    identifiers, lengths = [], []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if not fields[0].strip():
                continue
            identifiers.append(fields[0].strip().split(".", 1)[0])
            lengths.append(_number(fields[1]) if len(fields) > 1 else None)
    if identifiers and identifiers[0].lower() in ("gene", "gene_id", "ensembl gene id"):
        identifiers, lengths = identifiers[1:], lengths[1:]
    usable = None if any(value is None for value in lengths) else np.asarray(lengths, dtype=np.float64)
    return identifiers, usable


def select_genes(
    identifiers: Sequence[str],
    by_symbol: dict[str, str],
    by_ensembl: dict[str, str],
    gene_symbols: set[str],
    node_index: dict[str, int],
    extra_map: dict[str, str] | None = None,
) -> tuple[list[int], list[int], dict[str, int]]:
    """Expression rows -> (row positions, graph node ids) for the selected genes."""
    wanted = {by_symbol[symbol] for symbol in gene_symbols if symbol in by_symbol}
    wanted |= {symbol for symbol in gene_symbols if symbol in node_index}
    rows, nodes, seen = [], [], set()
    unmapped = 0
    for position, identifier in enumerate(identifiers):
        hgnc = (extra_map or {}).get(identifier) or by_ensembl.get(identifier)
        if hgnc is None:
            hgnc = by_symbol.get(identifier, identifier if identifier in node_index else None)
        if hgnc is None:
            unmapped += 1
            continue
        if hgnc not in wanted or hgnc not in node_index or hgnc in seen:
            continue
        seen.add(hgnc)
        rows.append(position)
        nodes.append(node_index[hgnc])
    stats = {
        "expression_rows": len(identifiers),
        "gene_set_symbols": len(gene_symbols),
        "rows_without_hgnc": unmapped,
        "genes_selected": len(rows),
    }
    _log(step="genes", **stats)
    if not rows:
        raise ValueError(
            "No expression row survived gene selection. Check that `gene_ids` really "
            "lists the rows of `expression` and that `gene_sets` use gene symbols."
        )
    return rows, nodes, stats


# --- samples ----------------------------------------------------------------

def select_samples(
    columns: Sequence[str],
    metadata_path: Path,
    clinical_path: Path,
    years: Sequence[int],
    max_survival_days: float | None,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    """Apply the paper's sample selection to the expression columns."""
    barcode, project = {}, {}
    for row in _rows(metadata_path, encoding="latin-1"):
        key = str(row.get("gdc_file_id", "")).strip().lower()
        if key:
            barcode[key] = str(row.get("gdc_cases.submitter_id", "")).strip()
            project[key] = str(row.get("gdc_cases.project.project_id", "")).strip()
    clinical = {}
    with Path(clinical_path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            clinical[str(row.get("bcr_patient_barcode", "")).strip()] = row

    records = []
    missing_metadata = missing_clinical = missing_survival = unknown_type = 0
    for position, column in enumerate(columns):
        key = column.strip().lower()
        if key not in barcode:
            missing_metadata += 1
            continue
        patient = clinical.get(barcode[key])
        if patient is None:
            missing_clinical += 1
            continue
        status = str(patient.get("vital_status", "")).strip()
        if status == "Dead":
            days, dead = _number(patient.get("death_days_to", "")), True
        elif status == "Alive":
            days, dead = _number(patient.get("last_contact_days_to", "")), False
        else:
            missing_survival += 1
            continue
        if days is None:
            missing_survival += 1
            continue
        cancer = str(patient.get("type", "")).strip() or project[key].removeprefix("TCGA-")
        if cancer not in CANCER_TYPES:
            unknown_type += 1
            continue
        records.append({"column": position, "days": days, "dead": dead, "type": cancer})

    censored = np.asarray([item["days"] for item in records if not item["dead"]], dtype=np.float64)
    censored_median = float(np.median(censored)) if censored.size else 0.0
    pool = np.asarray(
        [item["days"] for item in records if item["dead"] or item["days"] > censored_median],
        dtype=np.float64,
    )
    # "the top 5% in order of survival time of censored patients with more than
    # <median> days as well as deceased patients"
    cutoff = (
        float(max_survival_days)
        if max_survival_days is not None
        else float(np.percentile(pool, 95, method="higher"))
    )
    kept = [item for item in records if item["days"] <= cutoff]
    per_year: dict[int, list[dict[str, Any]]] = {}
    year_stats = {}
    for year in years:
        limit = year * 365
        # Samples censored inside the year cannot be judged, so they are dropped;
        # everything else is labelled by whether it reached the year.
        per_year[year] = [
            {**item, "label": 1 if item["days"] >= limit else 0}
            for item in kept
            if item["dead"] or item["days"] >= limit
        ]
        labels = [item["label"] for item in per_year[year]]
        year_stats[str(year)] = {
            "samples": len(labels),
            "survival": int(sum(labels)),
            "death": int(len(labels) - sum(labels)),
        }
        _log(step="samples", year=year, **year_stats[str(year)])
    stats = {
        "expression_columns": len(columns),
        "columns_without_metadata": missing_metadata,
        "columns_without_clinical": missing_clinical,
        "columns_without_survival": missing_survival,
        "columns_with_unknown_cancer_type": unknown_type,
        "usable_columns": len(records),
        "censored_median_days": censored_median,
        "long_survival_cutoff_days": cutoff,
        "excluded_beyond_cutoff": len(records) - len(kept),
        "remaining_samples": len(kept),
        "per_year": year_stats,
    }
    return per_year, stats


# --- expression -------------------------------------------------------------

def load_expression(path: Path, rows: Sequence[int], num_columns: int) -> np.ndarray:
    """The selected rows of the counts matrix, in the order ``rows`` gives them."""
    order = np.argsort(np.asarray(rows, dtype=np.int64))
    wanted = [int(rows[position]) for position in order]
    matrix = np.empty((len(wanted), num_columns), dtype=np.float64)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        handle.readline()
        pointer = 0
        target = wanted[0]
        for index, line in enumerate(handle):
            if index != target:
                continue
            values = np.fromstring(line, sep="\t", dtype=np.float64)
            if values.size != num_columns:
                raise ValueError(
                    f"{path} row {index} has {values.size} values, expected {num_columns}"
                )
            matrix[pointer] = values
            pointer += 1
            if pointer == len(wanted):
                break
            target = wanted[pointer]
    if pointer != len(wanted):
        raise ValueError(f"{path} ended after {pointer} of {len(wanted)} selected rows")
    restored = np.empty_like(matrix)
    restored[order] = matrix
    _log(step="expression", genes=restored.shape[0], columns=restored.shape[1])
    return restored


def transform_expression(
    matrix: np.ndarray, transform: str, lengths: np.ndarray | None
) -> np.ndarray:
    """``log1p`` of the counts, or of TPM when the row lengths are known.

    The shipped bundle's values reach 20.2, above ``ln(1e6)``, so it cannot be a
    TPM transform however the paper describes it; ``log1p`` is the default.
    """
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {TRANSFORMS}, got {transform!r}")
    if transform == "log1p":
        return np.log1p(matrix)
    if lengths is None:
        raise ValueError(
            "transform 'log1p_tpm' needs a per-row length; give `gene_ids` a second "
            "column holding each row's bp_length"
        )
    rate = matrix / lengths[:, None]
    total = rate.sum(axis=0, keepdims=True)
    total[total == 0] = 1.0
    return np.log1p(rate / total * 1e6)


# --- output -----------------------------------------------------------------

def write_year(
    output: Path,
    year: int,
    samples: Sequence[dict[str, Any]],
    matrix: np.ndarray,
    node_ids: Sequence[int],
) -> None:
    genes = [str(node) for node in node_ids]
    with (output / f"{year}years_labels.tsv").open("w", newline="") as handle:
        handle.write("".join(f"{index}\t{item['label']}\n" for index, item in enumerate(samples)))
    with (output / f"{year}years_sample.tsv").open("w", newline="") as handle:
        for index, item in enumerate(samples):
            code = CANCER_TYPES.index(item["type"])
            one_hot = "\t".join("1" if position == code else "0" for position in range(len(CANCER_TYPES)))
            handle.write(f"{index}\t{code}\t{one_hot}\n")
    with (output / f"{year}years_node_input.tsv").open("w", newline="") as handle:
        for index, item in enumerate(samples):
            values = matrix[:, item["column"]].tolist()
            prefix = f"{index}\t"
            handle.write("".join(
                f"{prefix}{gene}\t{value}\n" for gene, value in zip(genes, values)
            ))
    _log(step="write", year=year, samples=len(samples), genes=len(genes))


def build_cancer_processed(cfg: dict[str, Any]) -> dict[str, Any]:
    output = Path(cfg["output_dir"])
    years = [int(year) for year in cfg.get("years", [1, 2, 3, 4, 5])]
    by_symbol, by_ensembl = read_hgnc_table(Path(cfg["hgnc_table"]))
    graph = build_graph(Path(cfg["graph_sif"]), by_symbol, output)

    expression_path = Path(cfg["expression"])
    with expression_path.open(encoding="utf-8", newline="") as handle:
        columns = handle.readline().rstrip("\n").split("\t")
    per_year, sample_stats = select_samples(
        columns,
        Path(cfg["metadata"]),
        Path(cfg["clinical"]),
        years,
        cfg.get("max_survival_days"),
    )

    identifiers, lengths = read_gene_ids(Path(cfg["gene_ids"]))
    if len(identifiers) < 2:
        raise ValueError(f"{cfg['gene_ids']} lists {len(identifiers)} rows; it cannot match "
                         f"{expression_path}")
    extra_map = None
    if cfg.get("ensembl_to_hgnc"):
        extra_map = {
            row[0].split(".", 1)[0]: row[1]
            for row in csv.reader(
                Path(cfg["ensembl_to_hgnc"]).read_text().splitlines(), delimiter="\t"
            )
            if len(row) >= 2 and row[1]
        }
    rows, node_ids, gene_stats = select_genes(
        identifiers, by_symbol, by_ensembl, read_gene_sets(cfg["gene_sets"]),
        graph["node_index"], extra_map,
    )

    matrix = load_expression(expression_path, rows, len(columns))
    matrix = transform_expression(
        matrix, str(cfg.get("transform", "log1p")),
        None if lengths is None else lengths[np.asarray(rows, dtype=np.int64)],
    )
    for year in years:
        write_year(output, year, per_year.get(year, []), matrix, node_ids)

    manifest = {
        "output_dir": str(output),
        "inputs": {key: str(cfg[key]) for key in
                   ("graph_sif", "hgnc_table", "expression", "gene_ids", "metadata", "clinical")},
        "gene_sets": [str(path) for path in cfg["gene_sets"]],
        "transform": str(cfg.get("transform", "log1p")),
        "graph": {key: value for key, value in graph.items() if key != "node_index"},
        "genes": gene_stats,
        "samples": sample_stats,
        "deviations_from_the_shipped_bundle": [
            "node and relation ids are assigned in sorted order, not set-iteration order",
            "no serialized-header row in <n>years_sample.tsv",
        ],
    }
    (output / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    from pathwaygnn.config import load_config

    print(json.dumps(build_cancer_processed(load_config(sys.argv[1])), indent=2))
