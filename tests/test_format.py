from pathlib import Path

import numpy as np
import pytest
import torch

from pathwaygnn.data.format import DatasetWriter, GraphDataset, TaskChannel, open_dataset
from pathwaygnn.data.samples import TaskDataset


def test_manifest_round_trip(dataset: GraphDataset) -> None:
    assert (dataset.num_nodes, dataset.num_relations) == (24, 2)
    assert dataset.task_names == ["main"]
    task = dataset.task("main")
    assert task.channel_names == ("expression", "signature")
    assert task.seed_offset == 7
    assert task.num_samples == 24
    assert task.covariate_dim == 3 and task.num_groups == 3
    assert [channel.source for channel in task.channels] == ["expression", "signature"]
    assert task.rows("expression") is None
    assert task.rows("signature") is not None


def test_dataset_name_mismatch_is_refused(dataset: GraphDataset) -> None:
    with pytest.raises(ValueError, match="selects dataset 'cancer'"):
        open_dataset({"dataset": {"name": "cancer", "dir": str(dataset.root)}})
    with pytest.raises(FileNotFoundError, match="not a prepared pathwaygnn dataset"):
        open_dataset({"dataset": {"dir": str(dataset.root / "missing")}})
    with pytest.raises(KeyError, match="needs a `dataset:` block"):
        open_dataset({})


def test_channel_aliases_are_shared_between_tasks(tmp_path: Path) -> None:
    writer = DatasetWriter(tmp_path / "d", "shared")
    writer.write_graph(
        torch.tensor([[0, 1], [1, 0]]), torch.tensor([0, 0]), ["A", "B"], ["binds"]
    )
    writer.sparse_channel("common", [0, 1, 2], [0, 1], [1.0, 2.0])
    writer.sparse_channel("left", [0, 2], [0, 1], [3.0, 4.0])
    for name, source in (("first", "left"), ("second", "left")):
        writer.write_task(
            name,
            np.array([0.0, 1.0]),
            channels={
                "signature": TaskChannel(source, np.array([0, 0])),
                "context": TaskChannel("common", np.array([0, 1])),
            },
        )
    manifest = writer.finish()
    assert manifest["tasks"] == ["first", "second"]
    dataset = GraphDataset.open(tmp_path / "d")
    first, second = dataset.task("first"), dataset.task("second")
    assert first.channel_names == second.channel_names == ("signature", "context")
    assert first.seed_offset == 0 and second.seed_offset == 1
    # Both tasks read the same underlying tables.
    assert first.channels[0].root == second.channels[0].root


def test_writer_rejects_inconsistent_tasks(tmp_path: Path) -> None:
    writer = DatasetWriter(tmp_path / "d", "bad")
    writer.write_graph(torch.tensor([[0], [1]]), torch.tensor([0]), ["A", "B"], ["binds"])
    writer.sparse_channel("c", [0, 1], [0], [1.0])
    with pytest.raises(KeyError, match="unknown channel"):
        writer.write_task("t", np.array([1.0]), channels={"c": TaskChannel("missing")})
    with pytest.raises(ValueError, match="out of range"):
        writer.write_task("t", np.array([1.0]), channels={"c": TaskChannel("c", np.array([5]))})
    with pytest.raises(ValueError, match="maps every sample to its own row"):
        writer.write_task("t", np.array([1.0, 0.0]), channels={"c": TaskChannel("c")})
    with pytest.raises(ValueError, match="do not match"):
        writer.write_task(
            "t", np.array([1.0]), covariates=np.zeros((1, 2), dtype=np.float32),
            covariate_names=["only-one"],
        )
    with pytest.raises(ValueError, match="nodes that nodes.json does not name"):
        writer.write_graph(torch.tensor([[0], [9]]), torch.tensor([0]), ["A", "B"], ["binds"])


def test_batching_matches_the_tables(dataset: GraphDataset) -> None:
    task = dataset.task("main")
    data = TaskDataset(task)
    batch = data.collate()([data[0], data[7]])
    assert batch.size == 2
    dense = batch.channels["expression"]
    assert dense.kind == "dense" and dense.value.shape == (2, 6)
    assert torch.equal(dense.gene, torch.arange(6))
    matrix = task.channels[0].matrix()
    assert np.allclose(dense.value[1].numpy(), matrix[7])
    sparse = batch.channels["signature"]
    assert sparse.kind == "sparse" and sparse.value.shape == (6, 1)
    assert sparse.sample.tolist() == [0, 0, 0, 1, 1, 1]
    ptr, gene, _ = task.channels[1].csr()
    row = int(task.rows("signature")[7])
    assert sparse.gene[3:].tolist() == list(gene[ptr[row]:ptr[row + 1]])
    assert batch.label.tolist() == [0.0, 1.0]
    assert batch.index.tolist() == [0, 7]
    assert batch.covariate.shape == (2, 3)
    assert batch.group.tolist() == [0, 1]


def test_subset_keeps_dataset_level_indices(dataset: GraphDataset) -> None:
    data = TaskDataset(dataset.task("main"))
    subset = data.subset(np.array([5, 3]))
    assert len(subset) == 2
    assert subset[0]["index"] == 5
    assert subset.targets.tolist() == data.targets[[5, 3]].tolist()
    # The parent is untouched.
    assert len(data) == 24
