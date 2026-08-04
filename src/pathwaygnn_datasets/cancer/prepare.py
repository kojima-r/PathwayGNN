"""TCGA cancer-survival preprocessing (Inoue et al.).

Converts the upstream bundle in ``<source>/processed`` into one prepared dataset:

* one dense node_feature per verification year (``expression_<n>year``), streamed from
  the three-column node-input TSVs into a memmappable matrix
* one task per year, binding the alias ``expression`` to that year's node_feature, with
  the 33-dimensional cancer-type one-hot as sample_features and the cancer type as the
  sample group

``seed_offset`` is the verification year, so a fold's seed does not depend on how
many years a run happens to cover.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from pathwaygnn.data.format import DatasetWriter, TaskNodeFeature
from pathwaygnn_datasets.cancer.paper import CANCER_TYPES, PAPER_SAMPLE_COUNTS

LEGACY_NOTE = (
    "sample TSV row 0 contains 0..34 (a serialized header) and is retained for "
    "public-code compatibility"
)


def _load_tsv(path: Path, dtype: type) -> np.ndarray:
    return np.loadtxt(path, delimiter="\t", dtype=dtype)


def _read_dictionary(path: Path) -> list[str]:
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            name, index = line.rstrip("\r\n").split("\t")
            pairs.append((int(index), name))
    pairs.sort()
    if [index for index, _ in pairs] != list(range(len(pairs))):
        raise ValueError(f"{path} does not contain a contiguous 0-based index")
    return [name for _, name in pairs]


def _convert_node_input(
    source: Path,
    expression_path: Path,
    gene_path: Path,
    num_samples: int,
    num_genes: int = 4448,
    chunk_bytes: int = 64 << 20,
) -> None:
    """Stream the legacy three-column TSV into a memory-mappable dense matrix."""
    if expression_path.exists() and gene_path.exists():
        existing = np.load(expression_path, mmap_mode="r")
        genes = np.load(gene_path, mmap_mode="r")
        if existing.shape == (num_samples, num_genes) and genes.shape == (num_genes,):
            print(json.dumps({"stage": "cancer_prepare", "reuse": str(expression_path)}))
            return
    expression = np.lib.format.open_memmap(
        expression_path, mode="w+", dtype=np.float32, shape=(num_samples, num_genes)
    )
    gene_ids = np.empty(num_genes, dtype=np.int64)
    position = 0
    remainder = b""
    with source.open("rb") as handle:
        first_line = handle.readline()
        if first_line.replace(b"\r", b"").rstrip(b"\n") != b"0\t1\t2":
            remainder = first_line
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                data, remainder = remainder, b""
            else:
                data = remainder + block
                split = data.rfind(b"\n")
                if split < 0:
                    remainder = data
                    continue
                data, remainder = data[: split + 1], data[split + 1 :]
            if data:
                flat = np.fromstring(data.replace(b"\r", b""), sep="\t", dtype=np.float64)
                if flat.size % 3:
                    raise ValueError(f"Malformed node feature triples in {source}")
                triples = flat.reshape(-1, 3)
                count = triples.shape[0]
                expected_sample = np.arange(position, position + count) // num_genes
                if not np.array_equal(triples[:, 0].astype(np.int64), expected_sample):
                    raise ValueError(f"Unexpected sample order in {source} near row {position}")
                gene_position = np.arange(position, position + count) % num_genes
                first_mask = expected_sample == 0
                gene_ids[gene_position[first_mask]] = triples[first_mask, 1].astype(np.int64)
                if position >= num_genes:
                    expected_gene = gene_ids[gene_position]
                    if not np.array_equal(triples[:, 1].astype(np.int64), expected_gene):
                        raise ValueError(f"Gene order changes in {source} near row {position}")
                expression.reshape(-1)[position : position + count] = triples[:, 2]
                position += count
            if not block:
                break
    expected = num_samples * num_genes
    if position != expected:
        raise ValueError(f"Expected {expected} node rows in {source}, found {position}")
    expression.flush()
    np.save(gene_path, gene_ids)


def prepare_cancer_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    years: Sequence[int] = (1, 2, 3, 4, 5),
    num_genes: int = 4448,
    strict_sample_counts: bool = True,
) -> dict[str, Any]:
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    legacy = source / "processed"
    writer = DatasetWriter(
        output,
        "cancer",
        source={"source_dir": str(source), "num_genes": num_genes, "known_legacy_issue": LEGACY_NOTE},
    )

    graph_array = _load_tsv(legacy / "graph.tsv", np.int64)
    edge_index = torch.from_numpy(graph_array[:, (0, 2)].T.copy()).long()
    edge_type = torch.from_numpy(graph_array[:, 1].copy()).long()
    writer.write_graph(
        edge_index,
        edge_type,
        _read_dictionary(legacy / "vertices_dic.tsv"),
        _read_dictionary(legacy / "relationships_dic.tsv"),
    )

    for year in years:
        print(json.dumps({"stage": "cancer_prepare", "year": int(year)}))
        labels = _load_tsv(legacy / f"{year}years_labels.tsv", np.float64)
        labels = labels[np.argsort(labels[:, 0])]
        samples = _load_tsv(legacy / f"{year}years_sample.tsv", np.float64)
        samples = samples[np.argsort(samples[:, 0])]
        num_samples = labels.shape[0]
        if samples.shape != (num_samples, 35):
            raise ValueError(
                f"Unexpected {year}-year shape: labels={labels.shape}, samples={samples.shape}"
            )
        if num_samples != PAPER_SAMPLE_COUNTS[year]:
            message = (
                f"{year}-year has {num_samples} samples, but Supplementary Table 1 reports "
                f"{PAPER_SAMPLE_COUNTS[year]}"
            )
            if strict_sample_counts:
                raise ValueError(
                    f"{message}. A bundle rebuilt by `cancer-build-processed` legitimately "
                    "differs by a few samples (it omits the serialized-header row and resolves "
                    "a handful of clinical records differently); set "
                    "`strict_sample_counts: false` to accept it."
                )
            print(json.dumps({"stage": "cancer_prepare", "warning": message}))
        node_feature = f"expression_{year}year"
        node_feature_dir = writer.dense_node_feature(node_feature, num_samples, num_genes)
        _convert_node_input(
            legacy / f"{year}years_node_input.tsv",
            node_feature_dir / "matrix.npy",
            node_feature_dir / "gene_index.npy",
            num_samples,
            num_genes,
        )
        writer.write_task(
            f"{year}year",
            labels[:, 1].astype(np.float32),
            node_features={"expression": TaskNodeFeature(node_feature)},
            sample_features=samples[:, 2:].astype(np.float32),
            sample_feature_names=CANCER_TYPES,
            groups=samples[:, 1].astype(np.int64),
            group_names=CANCER_TYPES,
            seed_offset=int(year),
            source={
                "year": int(year),
                "death": int((labels[:, 1] == 0).sum()),
                "survival": int((labels[:, 1] == 1).sum()),
                "legacy_header_row_retained": bool(np.array_equal(samples[0], np.arange(35))),
            },
        )
    return writer.finish()
