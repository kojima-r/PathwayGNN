"""The tutorial corpus: the committed raw files, the generator, and `sample-prepare`.

Unlike the other dataset tests this one reads real files — ``data_sample/raw`` is
committed and only 12 KB — but it writes exclusively into ``tmp_path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from pathwaygnn.data.format import GraphDataset
from pathwaygnn.data.samples import TaskDataset
from pathwaygnn_datasets.sample.prepare import prepare_sample_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data_sample" / "raw"
RAW_FILES = ("graph.tsv", "expression.tsv", "tissue_signature.tsv", "samples.tsv", "manifest.json")


def _make_raw_data():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts.sample.make_raw_data import build
    finally:
        sys.path.pop(0)
    return build


def test_committed_raw_data_matches_the_generator(tmp_path: Path) -> None:
    """`scripts/sample/make_raw_data.py` reproduces the committed files exactly."""
    _make_raw_data()(tmp_path)
    for name in RAW_FILES:
        assert (tmp_path / name).read_bytes() == (RAW_DIR / name).read_bytes(), name


def test_prepare_writes_the_documented_dataset(tmp_path: Path) -> None:
    manifest = prepare_sample_dataset(RAW_DIR, tmp_path / "prepared")
    assert (manifest["num_nodes"], manifest["num_relations"], manifest["num_edges"]) == (20, 3, 54)
    assert manifest["tasks"] == ["responder", "relapse"]
    assert manifest["node_features"]["expression"] == {
        "kind": "dense", "num_rows": 60, "num_features": 20
    }
    assert manifest["node_features"]["tissue_signature"]["kind"] == "sparse"

    dataset = GraphDataset.open(tmp_path / "prepared", "sample")
    edge_index, edge_type = dataset.graph()
    # Every edge is symmetrized, and both directions carry the same relation.
    edges = {(int(s), int(t), int(r)) for s, t, r in zip(*edge_index, edge_type)}
    assert {(t, s, r) for s, t, r in edges} == edges
    assert dataset.node_names()[:2] == ["GROWTH1", "GROWTH2"]
    assert dataset.relation_names() == [
        "controls-expression-of", "in-complex-with", "interacts-with"
    ]

    responder = dataset.task("responder")
    relapse = dataset.task("relapse")
    assert (responder.num_samples, int(responder.labels().sum())) == (60, 30)
    assert (relapse.num_samples, int(relapse.labels().sum())) == (48, 24)
    # Both tasks expose the same aliases, so one model config serves both.
    assert responder.node_feature_names == relapse.node_feature_names == (
        "expression", "tissue_signature"
    )
    assert responder.sample_feature_names == ("age", "sex_female", "stage", "smoker")
    assert responder.group_names == ("TISSUE_A", "TISSUE_B", "TISSUE_C")
    # `responder` covers every row of the expression table, so it stores no row
    # map; `relapse` covers 48 of 60 samples and therefore stores one.
    assert responder.rows("expression") is None
    assert np.asarray(relapse.rows("expression")).size == 48
    # The tissue signature has one row per tissue, addressed by the group code.
    assert np.array_equal(np.asarray(responder.rows("tissue_signature")), responder.groups())


def test_samples_carry_the_expected_shapes(tmp_path: Path) -> None:
    prepare_sample_dataset(RAW_DIR, tmp_path / "prepared")
    dataset = GraphDataset.open(tmp_path / "prepared", "sample")
    task = dataset.task("relapse")
    data = TaskDataset(task)
    batch = data.collate()([data[index] for index in range(4)])
    assert batch.size == 4
    assert batch.sample_feature is not None and batch.sample_feature.shape == (4, 4)
    dense = batch.node_features["expression"]
    assert dense.dense and dense.value.shape == (4, 20)
    sparse = batch.node_features["tissue_signature"]
    assert not sparse.dense and sparse.value.shape[0] == int(sparse.gene.numel())
    # Only the eight marker genes of each sample's tissue are present.
    assert sparse.value.shape[0] == 4 * 8


def test_prepare_reports_a_missing_corpus(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make_raw_data"):
        prepare_sample_dataset(tmp_path, tmp_path / "prepared")
