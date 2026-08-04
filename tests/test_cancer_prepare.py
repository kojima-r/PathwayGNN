import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pathwaygnn.data.format import GraphDataset
from pathwaygnn_datasets.cancer.gene_mapping import map_ensembl_ids
from pathwaygnn_datasets.cancer.paper import CANCER_TYPES
from pathwaygnn_datasets.cancer.prepare import _convert_node_input, prepare_cancer_dataset

NUM_GENES = 3


def _node_file(path: Path, header: bool, samples: int = 2) -> None:
    lines = ["0\t1\t2"] if header else []
    for sample in range(samples):
        for gene, value in zip((2, 0, 1), (1.0, 2.0, 3.0)):
            lines.append(f"{sample}\t{gene}\t{value + sample}")
    path.write_text("\r\n".join(lines) + "\r\n")


def test_cancer_stream_conversion_with_and_without_header(tmp_path: Path) -> None:
    for header in (False, True):
        source = tmp_path / f"node_{header}.tsv"
        expression = tmp_path / f"expression_{header}.npy"
        genes = tmp_path / f"genes_{header}.npy"
        _node_file(source, header)
        _convert_node_input(source, expression, genes, num_samples=2, num_genes=3, chunk_bytes=17)
        assert np.array_equal(np.load(genes), [2, 0, 1])
        assert np.array_equal(np.load(expression), [[1, 2, 3], [2, 3, 4]])


def test_cancer_stream_conversion_detects_truncation(tmp_path: Path) -> None:
    source = tmp_path / "node.tsv"
    _node_file(source, header=False, samples=2)
    with pytest.raises(ValueError, match="Expected 9 node rows"):
        _convert_node_input(
            source, tmp_path / "e.npy", tmp_path / "g.npy", num_samples=3, num_genes=3
        )


def _legacy_bundle(root: Path, samples: int) -> None:
    legacy = root / "processed"
    legacy.mkdir(parents=True)
    (legacy / "graph.tsv").write_text("0\t0\t1\n1\t1\t2\n2\t0\t0\n")
    (legacy / "vertices_dic.tsv").write_text("11998\t0\n1636\t1\n25225\t2\n")
    (legacy / "relationships_dic.tsv").write_text("interacts-with\t0\nin-complex-with\t1\n")
    labels = "\n".join(f"{index}\t{index % 2}" for index in range(samples))
    (legacy / "1years_labels.tsv").write_text(labels + "\n")
    rows = []
    for index in range(samples):
        onehot = [1 if position == index % len(CANCER_TYPES) else 0 for position in range(33)]
        rows.append("\t".join([str(index), str(index % len(CANCER_TYPES))] + [str(x) for x in onehot]))
    (legacy / "1years_sample.tsv").write_text("\n".join(rows) + "\n")
    _node_file(legacy / "1years_node_input.tsv", header=True, samples=samples)


def test_prepare_cancer_dataset(tmp_path: Path, monkeypatch) -> None:
    samples = 6
    monkeypatch.setattr(
        "pathwaygnn_datasets.cancer.prepare.PAPER_SAMPLE_COUNTS", {1: samples}
    )
    _legacy_bundle(tmp_path / "source", samples)
    manifest = prepare_cancer_dataset(
        tmp_path / "source", tmp_path / "prepared", years=[1], num_genes=NUM_GENES
    )
    assert manifest["name"] == "cancer"
    assert manifest["num_nodes"] == 3 and manifest["num_relations"] == 2
    assert manifest["node_features"]["expression_1year"]["kind"] == "dense"
    dataset = GraphDataset.open(tmp_path / "prepared", "cancer")
    # Edge order is taken verbatim from graph.tsv so that pre-training stays reproducible.
    edge_index, edge_type = dataset.graph()
    assert edge_index.tolist() == [[0, 1, 2], [1, 2, 0]]
    assert edge_type.tolist() == [0, 1, 0]
    assert dataset.node_names() == ["11998", "1636", "25225"]
    task = dataset.task("1year")
    assert task.node_feature_names == ("expression",)
    assert task.node_features[0].source == "expression_1year"
    assert task.seed_offset == 1  # the verification year, not the task position
    assert task.sample_feature_names == tuple(CANCER_TYPES)
    assert task.rows("expression") is None  # one feature row per sample
    assert task.num_samples == samples
    assert task.manifest["source"]["death"] == samples // 2
    assert np.array_equal(np.asarray(task.groups()), np.arange(samples))


def test_ensembl_mapping_reuses_auditable_cache(tmp_path: Path) -> None:
    source = tmp_path / "ensembl.txt"
    source.write_text("ENSG00000141510.18\nENSG_NOT_FOUND\n")
    normalized = "ENSG00000141510\nENSG_NOT_FOUND\n"
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({
        "input_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "results": [
            {"query": "ENSG00000141510", "hgnc": "HGNC:11998", "symbol": "TP53", "entrezgene": 7157},
            {"query": "ENSG_NOT_FOUND", "notfound": True},
        ],
    }))
    output = tmp_path / "mapping.tsv"
    result = map_ensembl_ids(source, output, cache)
    assert result["cache_reused"] is True
    assert result["mapped"] == 1 and result["unmapped"] == 1
    assert "ENSG00000141510.18\tENSG00000141510\t11998\tTP53" in output.read_text()
