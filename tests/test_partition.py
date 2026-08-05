"""METIS partitioning, the Cluster-GCN collate, and partition-mode pre-training."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pathwaygnn.data.format import GraphDataset
from pathwaygnn.data.partition import (
    MANIFEST_NAME,
    PartitionLoader,
    PartitionStore,
    partition_settings,
    run_partitioning,
    write_partitions,
)
from pathwaygnn.models.encoder import load_encoder

NUM_PARTS = 4


@pytest.fixture
def partitioned(dataset: GraphDataset, tmp_path: Path) -> PartitionStore:
    write_partitions(dataset, tmp_path / "partitions", NUM_PARTS)
    return PartitionStore.open(tmp_path / "partitions", dataset)


def _edge_set(edge_index: torch.Tensor, edge_type: torch.Tensor) -> set[tuple[int, int, int]]:
    return {
        (int(source), int(target), int(relation))
        for source, target, relation in zip(edge_index[0], edge_index[1], edge_type)
    }


def test_partitions_cover_every_node_and_edge_once(
    dataset: GraphDataset, partitioned: PartitionStore
):
    manifest = partitioned.manifest
    assert manifest["num_parts"] == NUM_PARTS
    assert sum(part["num_nodes"] for part in manifest["parts"]) == dataset.num_nodes
    # Each directed edge is stored exactly once, in the partition owning its source.
    assert sum(part["num_edges"] for part in manifest["parts"]) == manifest["num_edges"]
    nodes = torch.cat([partitioned.load(index)["nodes"] for index in range(NUM_PARTS)])
    assert sorted(nodes.tolist()) == list(range(dataset.num_nodes))


def test_all_partitions_in_one_batch_reproduce_the_graph(
    dataset: GraphDataset, partitioned: PartitionStore
):
    batch = partitioned.collate(range(NUM_PARTS))
    edge_index, edge_type = dataset.graph()
    assert batch.num_nodes == dataset.num_nodes
    # Local ids differ from dataset ids, so compare after mapping back through `nodes`.
    mapped = batch.nodes[batch.edge_index]
    assert _edge_set(mapped, batch.edge_type) == _edge_set(edge_index, edge_type)


@pytest.mark.parametrize("selected", [(0,), (0, 2), (1, 3), (0, 1, 2)])
def test_a_batch_holds_exactly_the_induced_subgraph(
    dataset: GraphDataset, partitioned: PartitionStore, selected: tuple[int, ...]
):
    batch = partitioned.collate(selected)
    edge_index, edge_type = dataset.graph()
    inside = set(batch.nodes.tolist())
    expected = {
        edge for edge in _edge_set(edge_index, edge_type) if edge[0] in inside and edge[1] in inside
    }
    mapped = batch.nodes[batch.edge_index]
    assert _edge_set(mapped, batch.edge_type) == expected
    internal = sum(partitioned.manifest["parts"][part]["num_internal_edges"] for part in selected)
    if len(selected) == 1:
        assert batch.num_edges == internal
    else:
        assert batch.num_edges >= internal


def test_cross_partition_edges_survive(partitioned: PartitionStore):
    """Storing the whole ``col`` is what keeps the edges *between* partitions."""
    internal = sum(part["num_internal_edges"] for part in partitioned.manifest["parts"])
    whole = partitioned.collate(range(NUM_PARTS))
    assert internal < whole.num_edges == partitioned.manifest["num_edges"]


def test_collate_matches_a_naive_full_size_mapping(
    dataset: GraphDataset, partitioned: PartitionStore
):
    """The searchsorted localisation must equal an O(num_nodes) scatter mapping."""
    selected = (3, 0, 2)  # deliberately unsorted: local order follows `selected`
    batch = partitioned.collate(selected)
    partptr = partitioned.partptr
    local = torch.full((dataset.num_nodes,), -1, dtype=torch.long)
    cursor = 0
    for part in selected:
        start, end = int(partptr[part]), int(partptr[part + 1])
        local[torch.arange(start, end)] = torch.arange(cursor, cursor + end - start)
        cursor += end - start
    rows, columns, types = [], [], []
    for part in selected:
        stored = partitioned.load(part)
        keep = local[stored["col"]] >= 0
        rows.append(local[stored["row"]][keep])
        columns.append(local[stored["col"]][keep])
        types.append(stored["edge_type"][keep])
    assert _edge_set(
        torch.stack([torch.cat(rows), torch.cat(columns)]), torch.cat(types)
    ) == _edge_set(batch.edge_index, batch.edge_type)


def test_stale_partitions_are_refused(dataset: GraphDataset, partitioned: PartitionStore):
    with pytest.raises(ValueError, match="num_parts"):
        partitioned.check(dataset, num_parts=NUM_PARTS + 1)
    other = GraphDataset(
        root=dataset.root,
        name="other",
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        manifest=dataset.manifest,
    )
    with pytest.raises(ValueError, match="dataset="):
        partitioned.check(other)
    (partitioned.root / MANIFEST_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="pathwaygnn partition"):
        PartitionStore.open(partitioned.root)


def test_loader_shards_partitions_across_ranks_evenly(partitioned: PartitionStore):
    world_size = 2
    loaders = [
        PartitionLoader(partitioned, parts_per_batch=1, rank=rank, world_size=world_size, seed=3)
        for rank in range(world_size)
    ]
    # DDP all-reduces once per backward, so every rank must take the same number
    # of steps or the collective deadlocks.
    assert len({len(loader) for loader in loaders}) == 1
    # The sampler yields positions in `loader.parts`, not partition ids.
    assigned = [
        [loader.parts[index] for index in loader.partition_sampler]  # type: ignore[union-attr]
        for loader in loaders
    ]
    assert set(assigned[0]).isdisjoint(assigned[1])
    assert set(assigned[0]) | set(assigned[1]) == set(partitioned.trainable_parts)
    # ... and the batches a rank actually sees carry those partitions.
    seen = {int(part) for batch in loaders[0] for part in batch.part_ids}
    assert seen == set(assigned[0])


@pytest.mark.parametrize("num_workers", [0, 2])
def test_loader_yields_usable_batches(partitioned: PartitionStore, num_workers: int):
    """`num_workers > 0` prefetches partition files, so batches must survive workers."""
    loader = PartitionLoader(
        partitioned, parts_per_batch=2, shuffle=False, num_workers=num_workers, seed=1
    )
    batches = list(loader)
    assert len(batches) == len(loader)
    for batch in batches:
        assert batch.num_edges > 0
        assert int(batch.edge_index.max()) < batch.num_nodes
        assert int(batch.nodes.max()) < partitioned.num_nodes


def test_partitioned_pretraining_runs_and_checkpoints(dataset: GraphDataset, tmp_path: Path):
    from pathwaygnn.training.pretrain import run_pretraining

    output = tmp_path / "pretrain_partitioned"
    cfg = {
        "seed": 5,
        "device": "cpu",
        "dataset": {"name": dataset.name, "dir": str(dataset.root)},
        "model": {"hidden_dim": 4, "num_layers": 1, "dropout": 0.0},
        "training": {
            "epochs": 2,
            "batch_size": 8,
            "partition": {
                "dir": str(tmp_path / "parts"),
                "num_parts": NUM_PARTS,
                "parts_per_batch": 2,
                "shuffle": True,
            },
        },
        "output_dir": str(output),
    }
    run_pretraining(cfg)
    encoder, checkpoint = load_encoder(
        output / "best.pt", dataset.num_nodes, dataset.num_relations
    )
    # The checkpoint describes the whole graph, so `cv`/`ig` consume it unchanged.
    assert encoder.num_nodes == dataset.num_nodes
    assert checkpoint["dataset"] == dataset.name
    history = json.loads((output / "history.json").read_text())
    assert np.isfinite([record["loss"] for record in history]).all()


def test_partitioned_pretraining_never_reads_the_graph(dataset: GraphDataset, tmp_path: Path):
    """The minimal-memory contract: with partitions on disk, `graph.pt` is not needed."""
    from pathwaygnn.training.pretrain import run_pretraining

    partition_dir = tmp_path / "parts_only"
    write_partitions(dataset, partition_dir, NUM_PARTS)
    (dataset.root / "graph.pt").rename(tmp_path / "graph.pt.moved")
    run_pretraining({
        "seed": 5,
        "device": "cpu",
        "dataset": {"name": dataset.name, "dir": str(dataset.root)},
        "model": {"hidden_dim": 4, "num_layers": 1, "dropout": 0.0},
        "training": {
            "epochs": 1,
            "batch_size": 8,
            "partition": {
                "dir": str(partition_dir),
                "num_parts": NUM_PARTS,
                "parts_per_batch": 1,
                "create": False,
            },
        },
        "output_dir": str(tmp_path / "reload"),
    })
    assert (tmp_path / "reload" / "best.pt").is_file()


def test_create_false_refuses_to_cut_the_graph(dataset: GraphDataset, tmp_path: Path):
    from pathwaygnn.data.partition import ensure_partitions

    settings = partition_settings({
        "training": {"partition": {"dir": str(tmp_path / "missing"), "create": False}}
    })
    with pytest.raises(FileNotFoundError, match="create: false"):
        ensure_partitions(dataset, settings)


def test_balanced_relations_in_partition_mode(dataset: GraphDataset, tmp_path: Path):
    from pathwaygnn.training.pretrain import run_pretraining

    run_pretraining({
        "seed": 2,
        "device": "cpu",
        "dataset": {"name": dataset.name, "dir": str(dataset.root)},
        "model": {"hidden_dim": 4, "num_layers": 1, "dropout": 0.0},
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "balanced_relations": True,
            "partition": {
                "dir": str(tmp_path / "balanced"),
                "num_parts": NUM_PARTS,
                "parts_per_batch": 1,
            },
        },
        "output_dir": str(tmp_path / "balanced_out"),
    })
    assert (tmp_path / "balanced_out" / "best.pt").is_file()


def test_run_partitioning_reports_the_cut(dataset: GraphDataset, tmp_path: Path):
    summary = run_partitioning({
        "dataset": {"name": dataset.name, "dir": str(dataset.root)},
        "training": {"partition": {"dir": str(tmp_path / "cli"), "num_parts": NUM_PARTS}},
    })
    assert summary["num_parts"] == NUM_PARTS
    assert summary["num_nodes"] == dataset.num_nodes
    assert 0.0 < summary["internal_edge_fraction"] <= 1.0
    assert summary["nodes_per_part"]["min"] >= 1


def test_full_graph_pretraining_stays_deterministic(dataset: GraphDataset, tmp_path: Path):
    """No `training.partition` block means the original full-graph loop, bit for bit.

    Two runs of the same seed must agree exactly: the partition mode was added by
    branching around the sampling block, not by reordering it, so the RNG stream —
    and therefore every loss — is the one the published runs used.
    """
    from pathwaygnn.training.pretrain import run_pretraining

    histories = []
    for run in range(2):
        output = tmp_path / f"full_{run}"
        run_pretraining({
            "seed": 11,
            "device": "cpu",
            "dataset": {"name": dataset.name, "dir": str(dataset.root)},
            "model": {"hidden_dim": 4, "num_layers": 1, "dropout": 0.0},
            "training": {"epochs": 3, "steps_per_epoch": 3, "batch_size": 8},
            "output_dir": str(output),
        })
        histories.append(json.loads((output / "history.json").read_text()))
    assert histories[0] == histories[1]
    assert histories[0][0]["epoch"] == 1
