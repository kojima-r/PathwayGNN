"""The tutorial corpus: ``data_sample/raw`` -> the prepared dataset ``sample``.

This is the smallest complete preprocessing step in the repository and doubles as
the worked example for :mod:`pathwaygnn.data.format`. Four human-readable TSV
files (60 samples, 20 genes, 27 undirected edges) become one prepared dataset
that exercises **every** concept of the format:

======================  ==================================================
raw file                what it becomes
======================  ==================================================
``graph.tsv``           the graph: 20 nodes, 3 relations, symmetrized edges
``expression.tsv``      node-level feature ``expression`` (**dense**)
``tissue_signature.tsv``node-level feature ``tissue_signature`` (**sparse**)
``samples.tsv``         tasks ``responder`` / ``relapse``, groups (tissue),
                        sample-level features (age, sex, stage, smoker)
======================  ==================================================

Two details are the whole point of the exercise:

* ``tissue_signature`` has **one row per tissue**, not per sample. Many samples
  address the same row through ``rows/tissue_signature.npy`` — exactly how
  ``data_tr`` shares one disease-signature table across tasks and how
  ``data_cdr`` collapses 107,418 samples onto 760 mutation profiles.
* ``responder`` covers all 60 samples, so it stores **no** ``rows/expression.npy``
  (an absent file means the identity mapping), while ``relapse`` only covers the
  48 samples that have follow-up and therefore stores one. Both tasks bind the
  same two aliases, so a model config written for one works for the other — the
  same property that lets ``data_cancer`` share one config across ``1year``…
  ``5year``.

Add your own corpus by copying this file: read whatever you have, call
``DatasetWriter``, and every ``pathwaygnn`` command works on it unchanged.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pathwaygnn.data.format import DatasetWriter, TaskNodeFeature

SOURCE_FILES = ("graph.tsv", "expression.tsv", "tissue_signature.tsv", "samples.tsv")
# Columns of samples.tsv that are not labels. Everything else is one binary task.
ID_COLUMN = "sample_id"
GROUP_COLUMN = "tissue"
SAMPLE_FEATURE_COLUMNS = ("age", "sex_female", "stage", "smoker")
LABEL_COLUMNS = ("responder", "relapse")
MISSING = {"", "NA", "na", "NaN", "nan", "None", "-"}


def _read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle, delimiter="\t") if row]
    return rows[0], rows[1:]


def _read_graph(path: Path) -> tuple[list[str], list[str], torch.Tensor, torch.Tensor]:
    """Read a 3-column SIF-like edge list and symmetrize it.

    Node and relation names are **sorted** before they are numbered, so that the
    same input always produces the same indices — pre-training reproducibility
    depends on that, and the real corpora do it the same way.
    """
    genes: set[str] = set()
    relations: set[str] = set()
    edges: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 3:
                continue
            source, relation, target = (field.strip() for field in row[:3])
            genes.update((source, target))
            relations.add(relation)
            edges.add((source, relation, target))
            edges.add((target, relation, source))  # undirected
    gene_names, relation_names = sorted(genes), sorted(relations)
    gene_to_index = {name: index for index, name in enumerate(gene_names)}
    relation_to_index = {name: index for index, name in enumerate(relation_names)}
    ordered = sorted(edges)
    edge_index = (
        torch.tensor(
            [[gene_to_index[source], gene_to_index[target]] for source, _, target in ordered],
            dtype=torch.long,
        )
        .t()
        .contiguous()
    )
    edge_type = torch.tensor(
        [relation_to_index[relation] for _, relation, _ in ordered], dtype=torch.long
    )
    return gene_names, relation_names, edge_index, edge_type


def _read_expression(
    path: Path, gene_to_index: dict[str, int]
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """A wide table (one row per sample, one column per gene) -> a dense matrix.

    A dense node-level feature is ``matrix.npy`` ``[rows, genes]`` plus
    ``gene_index.npy``, which says which graph node each column belongs to.
    Columns naming a gene that is not in the graph are dropped and reported.
    """
    header, rows = _read_tsv(path)
    kept = [(offset, name) for offset, name in enumerate(header[1:], start=1) if name in gene_to_index]
    dropped = [name for name in header[1:] if name not in gene_to_index]
    sample_ids = [row[0] for row in rows]
    matrix = np.array(
        [[float(row[offset]) for offset, _ in kept] for row in rows], dtype=np.float32
    )
    gene_index = np.array([gene_to_index[name] for _, name in kept], dtype=np.int64)
    return sample_ids, matrix, gene_index, dropped


def _read_signature(
    path: Path, gene_to_index: dict[str, int]
) -> tuple[list[str], list[int], list[int], list[float], int]:
    """A long table (key, gene, value) -> CSR arrays for a sparse node-level feature.

    ``ptr[i]:ptr[i + 1]`` is row ``i``'s slice of ``gene``/``value``; ``gene``
    holds graph node indices. Rows are the sorted keys of the first column.
    """
    _, rows = _read_tsv(path)
    values: dict[str, list[tuple[int, float]]] = {}
    skipped = 0
    for key, gene, value in ((row[0], row[1], row[2]) for row in rows):
        if gene not in gene_to_index:
            skipped += 1
            continue
        values.setdefault(key, []).append((gene_to_index[gene], float(value)))
    names = sorted(values)
    ptr, gene_ids, gene_values = [0], [], []
    for name in names:
        for node, value in sorted(values[name]):
            gene_ids.append(node)
            gene_values.append(value)
        ptr.append(len(gene_ids))
    return names, ptr, gene_ids, gene_values, skipped


def prepare_sample_dataset(source_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    missing = [name for name in SOURCE_FILES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{source_dir} is missing {missing}. The tutorial corpus is committed to the "
            "repository; regenerate it with `python scripts/sample/make_raw_data.py`."
        )

    # 1. The graph. Node indices from here are the only gene identifiers the rest
    #    of the pipeline uses.
    genes, relations, edge_index, edge_type = _read_graph(source_dir / "graph.tsv")
    gene_to_index = {name: index for index, name in enumerate(genes)}
    writer = DatasetWriter(output_dir, "sample", source={"source_dir": str(source_dir)})
    writer.write_graph(edge_index, edge_type, genes, relations)

    # 2. The dense node-level feature: one row per sample, every gene present.
    sample_ids, matrix, gene_index, dropped = _read_expression(
        source_dir / "expression.tsv", gene_to_index
    )
    feature_dir = writer.dense_node_feature("expression", matrix.shape[0], matrix.shape[1])
    np.save(feature_dir / "matrix.npy", matrix)
    np.save(feature_dir / "gene_index.npy", gene_index)

    # 3. The sparse node-level feature: one row per tissue, shared by its samples.
    tissues, ptr, gene_ids, values, signature_skipped = _read_signature(
        source_dir / "tissue_signature.tsv", gene_to_index
    )
    writer.sparse_node_feature("tissue_signature", ptr, gene_ids, values)

    # 4. The tasks. samples.tsv carries the group, the sample-level features and
    #    one column per binary label; `NA` means the sample is not part of that
    #    task, which is what makes `relapse` a task over a subset.
    header, rows = _read_tsv(source_dir / "samples.tsv")
    column = {name: offset for offset, name in enumerate(header)}
    order = {sample_id: position for position, sample_id in enumerate(sample_ids)}
    if [row[column[ID_COLUMN]] for row in rows] != sample_ids:
        raise ValueError(
            "samples.tsv and expression.tsv must list the same samples in the same order "
            f"({len(rows)} vs {len(sample_ids)} rows)"
        )
    tissue_to_index = {name: index for index, name in enumerate(tissues)}
    unknown = sorted({row[column[GROUP_COLUMN]] for row in rows} - set(tissue_to_index))
    if unknown:
        raise ValueError(f"samples.tsv references tissues absent from tissue_signature.tsv: {unknown}")

    writer.source.update({
        "num_samples": len(sample_ids),
        "num_genes": len(genes),
        "expression_columns_dropped": dropped,
        "signature_rows_skipped": signature_skipped,
        "tissues": tissues,
    })
    for seed_offset, label_column in enumerate(LABEL_COLUMNS):
        selected = [
            position
            for position, row in enumerate(rows)
            if row[column[label_column]] not in MISSING
        ]
        labels = np.array(
            [float(rows[position][column[label_column]]) for position in selected], dtype=np.float32
        )
        groups = np.array(
            [tissue_to_index[rows[position][column[GROUP_COLUMN]]] for position in selected],
            dtype=np.int64,
        )
        sample_features = np.array(
            [
                [float(rows[position][column[name]]) for name in SAMPLE_FEATURE_COLUMNS]
                for position in selected
            ],
            dtype=np.float32,
        )
        expression_rows = np.array([order[rows[p][column[ID_COLUMN]]] for p in selected], dtype=np.int64)
        # An absent rows/<alias>.npy means "sample i is row i". Writing it only
        # when the mapping is not the identity keeps the intent visible: here
        # `responder` omits it and `relapse` (48 of 60 samples) stores it.
        identity = expression_rows.size == matrix.shape[0] and np.array_equal(
            expression_rows, np.arange(matrix.shape[0])
        )
        writer.write_task(
            label_column,
            labels,
            node_features={
                "expression": TaskNodeFeature("expression", None if identity else expression_rows),
                # The tissue signature is addressed by tissue, so several samples
                # share one row.
                "tissue_signature": TaskNodeFeature("tissue_signature", groups),
            },
            sample_features=sample_features,
            sample_feature_names=SAMPLE_FEATURE_COLUMNS,
            groups=groups,
            group_names=tissues,
            seed_offset=seed_offset,
            source={
                "label_column": label_column,
                "sample_ids": [rows[position][column[ID_COLUMN]] for position in selected],
                "rows_identity": bool(identity),
            },
        )
    return writer.finish()
