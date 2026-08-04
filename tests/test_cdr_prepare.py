import csv
import json
from pathlib import Path

import numpy as np

from pathwaygnn.data.format import GraphDataset
from pathwaygnn.data.samples import TaskDataset
from pathwaygnn.models.encoder import RelationalGIN
from pathwaygnn.models.predictor import SampleLevelModel
from pathwaygnn_datasets.cdr.prepare import (
    FINGERPRINT_BITS,
    PRIMARY_SITES,
    SPECTRA_DIM,
    prepare_cdr_dataset,
)

# Two cell lines x three compounds; cell line 0 is Bladder, cell line 1 is Bone.
CELLS = {
    0: {"site": 0, "spectra": 1, "mutations": [(0, 2), (2, 1)]},
    1: {"site": 1, "spectra": 3, "mutations": [(1, 1)]},
}
# (cell line, compound, LN_IC50); compound 2 is potent everywhere, compound 0 weak.
SAMPLES = [
    (0, 0, 5.0),
    (1, 0, 6.0),
    (0, 1, 1.0),
    (1, 1, 3.0),
    (0, 2, -4.0),
    (1, 2, -2.0),
]


def _write(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_NONE).writerows(rows)


def make_source(root: Path) -> None:
    """A miniature copy of `data_cdr/processed/full_features`."""
    _write(root / "vertices_dic.tsv", [["1100", 0], ["3236", 1], ["7157", 2]])
    _write(root / "relationships_dic.tsv", [["activate", 0], ["catalyze", 1]])
    _write(
        root / "graph.tsv",
        # already undirected upstream, plus one duplicate and one self loop
        [[0, 0, 1], [1, 0, 0], [1, 1, 2], [2, 1, 1], [0, 0, 1], [2, 0, 2]],
    )
    _write(root / "labels.tsv", [[index, value, 2.718] for index, (_, _, value) in enumerate(SAMPLES)])

    sample_rows, node_rows = [], []
    for index, (cell, compound, _) in enumerate(SAMPLES):
        spectra = [CELLS[cell]["spectra"]] * SPECTRA_DIM
        one_hot = [
            1 if position == CELLS[cell]["site"] else 0 for position in range(len(PRIMARY_SITES))
        ]
        fingerprint = [(compound >> (position % 3)) & 1 for position in range(FINGERPRINT_BITS)]
        sample_rows.append([index, CELLS[cell]["site"], *spectra, *one_hot, *fingerprint])
        for node, count in CELLS[cell]["mutations"]:
            node_rows.extend([[index, node, "False", "True", 0.5]] * count)
    _write(root / "sample_features.tsv", sample_rows)
    _write(root / "node_features.tsv", node_rows)


def test_prepare_writes_a_generic_dataset(tmp_path: Path) -> None:
    source, prepared = tmp_path / "source", tmp_path / "prepared"
    source.mkdir()
    make_source(source)
    manifest = prepare_cdr_dataset(source, prepared)

    assert manifest["name"] == "cdr"
    assert (manifest["num_nodes"], manifest["num_relations"]) == (3, 2)
    assert manifest["num_edges"] == 4  # duplicate and self loop dropped
    assert manifest["source"]["graph_self_loops_dropped"] == 1
    assert manifest["source"]["graph_duplicate_rows_dropped"] == 1
    assert manifest["source"]["graph_symmetric"] is True
    # Six samples, but only two distinct mutation profiles.
    assert manifest["source"]["distinct_mutation_profiles"] == 2
    assert manifest["source"]["num_compounds"] == 3
    assert manifest["source"]["num_cell_lines"] == 2
    assert manifest["node_features"]["mutation"]["num_rows"] == 2
    assert manifest["tasks"] == ["sensitive_drugwise", "sensitive_global"]
    # The temporary sample-level feature matrix does not survive preprocessing.
    assert not (prepared / "sample_features.npy").exists()

    dataset = GraphDataset.open(prepared, "cdr")
    assert dataset.node_names() == ["1100", "3236", "7157"]
    task = dataset.task("sensitive_drugwise")
    assert task.node_feature_names == ("mutation",)
    # Every primary site of the bundle is named, whether or not it is used here.
    assert task.group_names == PRIMARY_SITES
    assert task.sample_feature_dim == SPECTRA_DIM + len(PRIMARY_SITES) + FINGERPRINT_BITS
    assert task.sample_feature_names[SPECTRA_DIM] == "site_Bladder"

    # Every compound is split at its own median, so exactly half of each
    # compound's samples are positive; the global split is potency-driven.
    drugwise = task.labels()
    assert drugwise.tolist() == [1, 0, 1, 0, 1, 0]
    assert dataset.task("sensitive_global").labels().tolist() == [0, 0, 1, 0, 1, 1]
    assert task.seed_offset == 0 and dataset.task("sensitive_global").seed_offset == 1
    assert json.loads((prepared / "tasks/sensitive_drugwise/task.json").read_text())["num_positive"] == 3


def test_node_feature_rows_are_shared_between_samples_of_one_cell_line(tmp_path: Path) -> None:
    source, prepared = tmp_path / "source", tmp_path / "prepared"
    source.mkdir()
    make_source(source)
    prepare_cdr_dataset(source, prepared)
    dataset = GraphDataset.open(prepared, "cdr")
    task = dataset.task("sensitive_drugwise")

    rows = np.asarray(task.rows("mutation"))
    assert rows.tolist() == [0, 1, 0, 1, 0, 1]
    ptr, gene, value = (np.asarray(item) for item in dataset.node_feature("mutation").csr())
    # Cell line 0 carries two mutations of node 0 and one of node 2.
    assert gene[ptr[0] : ptr[1]].tolist() == [0, 2]
    assert value[ptr[0] : ptr[1]].tolist() == [2.0, 1.0]

    data = TaskDataset(task)
    batch = data.collate()([data[0], data[2]])
    assert batch.node_features["mutation"].kind == "sparse"
    assert batch.sample_feature is not None
    assert batch.sample_feature.shape == (2, task.sample_feature_dim)
    encoder = RelationalGIN(dataset.num_nodes, dataset.num_relations, hidden_dim=8, dropout=0)
    logits = SampleLevelModel(
        task.node_feature_names, embedding_dim=8, hidden_dim=8, sample_feature_dim=task.sample_feature_dim,
        use_sample_features=True,
    )(batch, encoder(*dataset.graph()))
    assert logits.shape == (2,)
    logits.sum().backward()


def test_binary_mutations_drop_the_counts(tmp_path: Path) -> None:
    source, prepared = tmp_path / "source", tmp_path / "prepared"
    source.mkdir()
    make_source(source)
    prepare_cdr_dataset(source, prepared, binary_mutations=True)
    dataset = GraphDataset.open(prepared, "cdr")
    _, _, value = (np.asarray(item) for item in dataset.node_feature("mutation").csr())
    assert set(value.tolist()) == {1.0}
