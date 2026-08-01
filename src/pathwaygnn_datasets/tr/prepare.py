"""Target-repositioning (SLGCN-TR) preprocessing.

Reads the raw PathwayCommons SIF, the disease signature, the knockdown and
overexpression perturbation signatures and the two label tables, and writes one
prepared dataset:

* channels ``disease``, ``perturbation_kd``, ``perturbation_oe`` (sparse)
* tasks ``kd_inh`` and ``oe_act``, each binding the alias ``perturbation`` to its
  own signature table and sharing the single ``disease`` table
* groups: the disease each sample targets, so that downstream per-group metrics
  and attributions are reported per disease
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from pathwaygnn.data.format import DatasetWriter, TaskChannel

RAW_FILES = {
    "graph": ("PathwayCommons12.All.hgnc.sif", "PathwayCommons12.All.hgnc.sif.tsv"),
    "disease": ("disease_specific_signature.tsv",),
    "kd_signature": ("knockdown_signature_sample.tsv",),
    "oe_signature": ("overexpression_signature_sample.tsv",),
    "kd_label": ("inhibitory_target_disease.tsv",),
    "oe_label": ("activatory_target_disease.tsv",),
}
TASKS = (
    ("kd_inh", "kd_signature", "kd_label", "perturbation_kd"),
    ("oe_act", "oe_signature", "oe_label", "perturbation_oe"),
)


def _find(raw_dir: Path, names: Iterable[str]) -> Path:
    for name in names:
        candidate = raw_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"None of {list(names)} exists under {raw_dir}")


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
        for row in csv.DictReader(handle, delimiter="\t"):
            gene_idx = gene_to_idx.get(row["human_gene_name"])
            value = float(row["expression"])
            if gene_idx is None:
                skipped += 1
            elif abs(value) >= cutoff:
                values.setdefault(row["do_id"], []).append((gene_idx, value))
    return sorted(values), values, skipped


def _read_signature(
    path: Path, gene_to_idx: dict[str, int], cutoff: float
) -> tuple[list[str], dict[str, list[tuple[int, float]]], int]:
    values: dict[str, list[tuple[int, float]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        graph_columns = [
            (gene_to_idx[column], offset)
            for offset, column in enumerate(header[1:], start=1)
            if column in gene_to_idx
        ]
        skipped = len(header) - 1 - len(graph_columns)
        for row in reader:
            if not row:
                continue
            sparse: list[tuple[int, float]] = []
            for gene_idx, offset in graph_columns:
                value = float(row[offset])
                if abs(value) >= cutoff:
                    sparse.append((gene_idx, value))
            values[row[0]] = sparse
    return sorted(values), values, skipped


def _pack(names: list[str], values: dict[str, list[tuple[int, float]]]):
    ptr, gene, value = [0], [], []
    for name in names:
        for pair in sorted(values[name]):
            gene.append(pair[0])
            value.append(pair[1])
        ptr.append(len(gene))
    return ptr, gene, value


def _read_labels(
    path: Path, pert_names: list[str], disease_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    pert_to_idx = {name: idx for idx, name in enumerate(pert_names)}
    disease_to_idx = {name: idx for idx, name in enumerate(disease_names)}
    pert, disease, label = [], [], []
    skipped = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["gene"] not in pert_to_idx or row["doid"] not in disease_to_idx:
                skipped += 1
                continue
            pert.append(pert_to_idx[row["gene"]])
            disease.append(disease_to_idx[row["doid"]])
            label.append(float(row["label"]))
    return (
        np.asarray(pert, dtype=np.int64),
        np.asarray(disease, dtype=np.int64),
        np.asarray(label, dtype=np.float32),
        skipped,
    )


def prepare_tr_dataset(
    raw_dir: str | Path, output_dir: str | Path, cutoff: float = 1e-7
) -> dict[str, Any]:
    raw_dir, output_dir = Path(raw_dir).resolve(), Path(output_dir).resolve()
    paths = {key: _find(raw_dir, names) for key, names in RAW_FILES.items()}
    genes, relations, edge_index, edge_type = _read_graph(paths["graph"])
    gene_to_idx = {name: idx for idx, name in enumerate(genes)}
    writer = DatasetWriter(
        output_dir, "tr", source={"raw_dir": str(raw_dir), "cutoff": cutoff}
    )
    writer.write_graph(edge_index, edge_type, genes, relations)

    disease_names, disease_values, disease_skipped = _read_disease(
        paths["disease"], gene_to_idx, cutoff
    )
    writer.sparse_channel("disease", *_pack(disease_names, disease_values))
    writer.source.update(
        {"num_diseases": len(disease_names), "disease_rows_skipped": disease_skipped}
    )

    for task_name, signature_key, label_key, channel in TASKS:
        pert_names, pert_values, genes_skipped = _read_signature(
            paths[signature_key], gene_to_idx, cutoff
        )
        writer.sparse_channel(channel, *_pack(pert_names, pert_values))
        pert, disease, label, labels_skipped = _read_labels(
            paths[label_key], pert_names, disease_names
        )
        writer.write_task(
            task_name,
            label,
            channels={
                "perturbation": TaskChannel(channel, pert),
                "disease": TaskChannel("disease", disease),
            },
            groups=disease,
            group_names=disease_names,
            source={
                "num_perturbations": len(pert_names),
                "signature_genes_skipped": genes_skipped,
                "label_rows_skipped": labels_skipped,
                "perturbations": pert_names,
            },
        )
    return writer.finish()
