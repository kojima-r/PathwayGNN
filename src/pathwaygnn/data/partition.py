"""METIS graph partitioning, and the Cluster-GCN loader that trains on the parts.

Pre-training normally runs a **full-graph** forward every step
(:mod:`pathwaygnn.training.pretrain`): every rank holds the whole graph and the
per-step activations are ``O(num_nodes x hidden_dim x num_relations)``. That is
the right trade-off for the shipped corpora (30k nodes), but it puts a hard
ceiling on graph size — the whole PathwayCommons release does not fit.

This module lifts that ceiling the way Cluster-GCN does. The graph is cut once
with METIS into ``num_parts`` node partitions, each written to its own file, and
a training step then runs on the subgraph induced by a *batch of partitions*
instead of on the whole graph. Two properties make that useful:

* **Memory is bounded by the batch, not by the graph.** A step touches
  ``partptr[p+1] - partptr[p]`` nodes per selected partition, so peak activation
  memory is set by ``parts_per_batch``, which the config controls.
* **Partitions are the unit of distribution.** Ranks take disjoint partitions
  (:class:`PartitionLoader` wraps :class:`~torch.utils.data.DistributedSampler`),
  do their own forward/backward on their own subgraphs, and DDP averages the
  gradients — so adding ranks adds throughput without replicating the graph.

Partitioning is a *separate, offline step* (``pathwaygnn partition``) precisely
because it is the one part of the pipeline that needs the whole graph in memory
at once. Once the partition directory exists, pre-training reads only partition
files and ``dataset.json``; it never opens ``graph.pt``. That is what makes
training on a machine too small to hold the graph possible.

On-disk layout::

    <partition_dir>/partitions.json     manifest (see `write_partitions`)
    <partition_dir>/partition_00000.pt  {"nodes", "row", "col", "edge_type"}

Nodes are relabelled so that every partition owns a **contiguous** range of the
permuted node space (``partptr`` in the manifest holds the boundaries, exactly
like PyG's ``ClusterData``). ``nodes`` maps that range back to original dataset
node indices — the ids the encoder's embedding table is indexed by. ``row``/``col``
are permuted ids: ``row`` always lies inside the partition (each directed edge is
stored exactly once, in the partition owning its **source**), while ``col`` may
point anywhere in the graph. Keeping the full ``col`` is what lets a batch of
partitions recover the edges *between* the partitions it selected instead of only
their internal ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.utils import remove_self_loops, sort_edge_index, to_undirected

from pathwaygnn.data.format import GraphDataset, open_dataset, read_json, write_json

PARTITION_FORMAT = "pathwaygnn/partition/1"
MANIFEST_NAME = "partitions.json"


def partition_path(root: str | Path, index: int) -> Path:
    return Path(root) / f"partition_{index:05d}.pt"


def _pointer(index: Tensor, size: int) -> Tensor:
    """Exclusive-cumsum boundaries of a sorted, grouped index vector."""
    counts = torch.bincount(index, minlength=size)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)])


def metis_clusters(
    edge_index: Tensor, num_nodes: int, num_parts: int, recursive: bool = False
) -> Tensor:
    """Assign every node to one of ``num_parts`` clusters with METIS.

    Takes int64 ``[2, E]`` and returns int64 ``[num_nodes]`` with values in
    ``[0, num_parts)``.

    METIS wants a symmetric adjacency without self-loops, so the graph is
    symmetrized for the partitioning call only — the stored partitions keep the
    dataset's own (possibly directed) edges, and their relation types.
    """
    if num_parts < 1 or num_parts > num_nodes:
        raise ValueError(
            f"num_parts must be between 1 and the node count ({num_nodes}), got {num_parts}"
        )
    symmetric, _ = remove_self_loops(to_undirected(edge_index, num_nodes=num_nodes))
    row, col = sort_edge_index(symmetric, num_nodes=num_nodes)
    rowptr = _pointer(row, num_nodes)
    attempts: list[str] = []
    # Same backend order as torch_geometric.loader.ClusterData: whichever of the
    # two extensions was built with METIS answers.
    try:
        import pyg_lib

        return pyg_lib.partition.metis(rowptr.cpu(), col.cpu(), num_parts, recursive=recursive)
    except (ImportError, AttributeError, RuntimeError) as error:
        attempts.append(f"pyg-lib: {error}")
    try:
        return torch.ops.torch_sparse.partition(
            rowptr.cpu(), col.cpu(), None, num_parts, recursive
        )
    except (AttributeError, RuntimeError) as error:
        attempts.append(f"torch-sparse: {error}")
    raise ImportError(
        "graph partitioning needs METIS, which ships inside `pyg-lib` or `torch-sparse`; "
        "install one of them (`pip install pyg-lib` or `pip install torch-sparse`) built "
        "with METIS support. Tried " + "; ".join(attempts)
    )


def write_partitions(
    dataset: GraphDataset,
    root: str | Path,
    num_parts: int,
    recursive: bool = False,
) -> dict[str, Any]:
    """Cut ``dataset``'s graph into ``num_parts`` files and write the manifest.

    This is the only function here that holds the whole graph in memory; run it
    once, on a machine large enough, via ``pathwaygnn partition``.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    edge_index, edge_type = dataset.graph()
    num_nodes = dataset.num_nodes
    cluster = metis_clusters(edge_index, num_nodes, num_parts, recursive)

    # Relabel nodes so a partition is a contiguous range: `order` maps permuted
    # index -> original node, `permuted` is its inverse.
    order = torch.argsort(cluster, stable=True)
    partptr = _pointer(cluster, num_parts)
    permuted = torch.empty(num_nodes, dtype=torch.long)
    permuted[order] = torch.arange(num_nodes, dtype=torch.long)

    row, col = permuted[edge_index[0]], permuted[edge_index[1]]
    by_source = torch.argsort(row, stable=True)
    row, col, edge_type = row[by_source], col[by_source], edge_type[by_source]
    edge_ptr = _pointer(row, num_nodes)

    for stale in root.glob("partition_[0-9]*.pt"):
        stale.unlink()
    parts: list[dict[str, int]] = []
    for index in range(num_parts):
        start, end = int(partptr[index]), int(partptr[index + 1])
        first, last = int(edge_ptr[start]), int(edge_ptr[end])
        part_col = col[first:last]
        internal = int(((part_col >= start) & (part_col < end)).sum())
        torch.save(
            {
                "nodes": order[start:end].clone(),
                "row": row[first:last].clone(),
                "col": part_col.clone(),
                "edge_type": edge_type[first:last].clone(),
            },
            partition_path(root, index),
        )
        parts.append(
            {
                "num_nodes": end - start,
                "num_edges": last - first,
                "num_internal_edges": internal,
            }
        )

    manifest = {
        "format": PARTITION_FORMAT,
        "dataset": dataset.name,
        "num_nodes": num_nodes,
        "num_relations": dataset.num_relations,
        "num_edges": int(edge_index.size(1)),
        "num_parts": num_parts,
        "recursive": bool(recursive),
        "partptr": [int(value) for value in partptr],
        "parts": parts,
        "source": {"dir": str(Path(dataset.root).resolve())},
    }
    write_json(root / MANIFEST_NAME, manifest)
    return manifest


@dataclass(frozen=True)
class GraphPartitionBatch:
    """The subgraph induced by one batch of partitions, in **local** node ids.

    With ``n`` nodes and ``e`` edges in the subgraph:

    Fields:
        nodes: int64 ``[n]`` — original dataset node index of each local node, so it
            gathers the encoder's embedding rows (``embedding(nodes)`` -> ``[n, H]``).
        edge_index: int64 ``[2, e]`` — **local** indices in ``[0, n)``, ready for the
            convolutions.
        edge_type: int64 ``[e]`` — global relation ids, unchanged by partitioning.
        part_ids: int64 ``[parts_per_batch]`` — which partitions this batch selected.
    """

    nodes: Tensor
    edge_index: Tensor
    edge_type: Tensor
    part_ids: Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.numel())

    @property
    def num_edges(self) -> int:
        return int(self.edge_type.numel())

    def to(self, device: torch.device | str) -> "GraphPartitionBatch":
        return GraphPartitionBatch(
            nodes=self.nodes.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            part_ids=self.part_ids,
        )


@dataclass(frozen=True)
class PartitionStore:
    """Read-only handle on a partition directory. Never reads ``graph.pt``."""

    root: Path
    manifest: dict[str, Any]

    @classmethod
    def open(cls, root: str | Path, dataset: GraphDataset | None = None) -> "PartitionStore":
        root = Path(root)
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{root} holds no graph partitions ({manifest_path} is missing); "
                "run `pathwaygnn partition --config <config>` first"
            )
        manifest = read_json(manifest_path)
        if manifest.get("format") != PARTITION_FORMAT:
            raise ValueError(
                f"{manifest_path} declares format {manifest.get('format')!r}, "
                f"expected {PARTITION_FORMAT!r}"
            )
        store = cls(root=root, manifest=manifest)
        if dataset is not None:
            store.check(dataset)
        return store

    def check(self, dataset: GraphDataset, num_parts: int | None = None) -> None:
        """Refuse partitions that were cut from a different graph or into a different count.

        The counts come from ``dataset.json``, so this stays a metadata-only
        check — the point of the partition format is that the full graph is never
        loaded again.
        """
        expected = {
            "dataset": dataset.name,
            "num_nodes": dataset.num_nodes,
            "num_relations": dataset.num_relations,
            "num_edges": int(dataset.manifest["num_edges"]),
        }
        disagreements = [
            f"{key}={self.manifest.get(key)!r} (dataset: {value!r})"
            for key, value in expected.items()
            if self.manifest.get(key) != value
        ]
        if num_parts is not None and self.num_parts != num_parts:
            disagreements.append(f"num_parts={self.num_parts} (config: {num_parts})")
        if disagreements:
            raise ValueError(
                f"the partitions in {self.root} do not match this run: "
                + ", ".join(disagreements)
                + "; re-run `pathwaygnn partition` (or set `training.partition.force: true`)"
            )

    @property
    def num_parts(self) -> int:
        return int(self.manifest["num_parts"])

    @property
    def num_nodes(self) -> int:
        return int(self.manifest["num_nodes"])

    @property
    def partptr(self) -> Tensor:
        return torch.tensor(self.manifest["partptr"], dtype=torch.long)

    @property
    def trainable_parts(self) -> list[int]:
        """Partitions that hold at least one edge of their own.

        A partition made only of isolated (or purely cross-partition) nodes would
        make some batches edge-free, and an edge-free step has no loss to compute
        while DDP still expects every rank to take it. Such partitions are left
        out of training and the count is reported rather than silently dropped.
        """
        return [
            index
            for index, part in enumerate(self.manifest["parts"])
            if part["num_internal_edges"] > 0
        ]

    def load(self, index: int) -> dict[str, Tensor]:
        return torch.load(
            partition_path(self.root, index), map_location="cpu", weights_only=True
        )

    def collate(self, part_ids: Sequence[int]) -> GraphPartitionBatch:
        """Build the subgraph induced by ``part_ids``, keeping the edges between them.

        Every stored edge starts inside its own partition, so an edge survives
        exactly when its ``col`` also lands in one of the selected ranges. Because
        the ranges are contiguous and disjoint, that membership test — and the
        remapping to local ids — is one ``searchsorted`` over the batch's columns;
        no buffer proportional to the whole graph is ever allocated.
        """
        ids = torch.as_tensor(list(part_ids), dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            raise ValueError("a partition batch must select at least one partition")
        if int(ids.min()) < 0 or int(ids.max()) >= self.num_parts:
            raise IndexError(f"partition ids {ids.tolist()} outside 0..{self.num_parts - 1}")
        partptr = self.partptr
        starts, ends = partptr[ids], partptr[ids + 1]
        sizes = ends - starts
        offsets = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)[:-1]])

        nodes, rows, columns, types = [], [], [], []
        for slot in range(int(ids.numel())):
            part = self.load(int(ids[slot]))
            nodes.append(part["nodes"])
            rows.append(part["row"] - starts[slot] + offsets[slot])
            columns.append(part["col"])
            types.append(part["edge_type"])

        # Ranges sorted by start, so `searchsorted` can find each column's range.
        by_start = torch.argsort(starts)
        range_start, range_end = starts[by_start], ends[by_start]
        range_offset = offsets[by_start]
        column = torch.cat(columns)
        slot_of = torch.searchsorted(range_start, column, side="right") - 1
        safe = slot_of.clamp(min=0)
        inside = (slot_of >= 0) & (column < range_end[safe])
        local_column = range_offset[safe] + (column - range_start[safe])
        return GraphPartitionBatch(
            nodes=torch.cat(nodes),
            edge_index=torch.stack([torch.cat(rows)[inside], local_column[inside]]),
            edge_type=torch.cat(types)[inside],
            part_ids=ids,
        )


class PartitionLoader(DataLoader):
    """Iterate over batches of partitions, sharded across ranks.

    ``rank``/``world_size`` reproduce ``DistributedSampler``'s pad-and-stride
    split: every rank sees ``ceil(num_parts / world_size)`` partitions and
    therefore the **same number of steps**, which DDP's gradient all-reduce
    requires. Call :meth:`set_epoch` so the shuffle differs per epoch.
    """

    def __init__(
        self,
        store: PartitionStore,
        parts_per_batch: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.store = store
        self.parts = store.trainable_parts
        self.num_skipped = store.num_parts - len(self.parts)
        if not self.parts:
            raise ValueError(
                f"none of the {store.num_parts} partitions in {store.root} holds an edge; "
                "the graph or the partition count is wrong"
            )
        self.partition_sampler = (
            DistributedSampler(
                self.parts, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed
            )
            if world_size > 1
            else None
        )
        generator = torch.Generator().manual_seed(seed)
        super().__init__(
            self.parts,
            batch_size=parts_per_batch,
            shuffle=shuffle if self.partition_sampler is None else False,
            sampler=self.partition_sampler,
            collate_fn=store.collate,
            num_workers=num_workers,
            generator=generator if self.partition_sampler is None else None,
        )

    def set_epoch(self, epoch: int) -> None:
        if self.partition_sampler is not None:
            self.partition_sampler.set_epoch(epoch)


@dataclass(frozen=True)
class PartitionSettings:
    """The ``training.partition:`` block. ``dir`` is what turns the mode on."""

    dir: Path | None = None
    num_parts: int = 64
    parts_per_batch: int = 4
    recursive: bool = False
    shuffle: bool = True
    num_workers: int = 0
    create: bool = True
    force: bool = False

    @property
    def enabled(self) -> bool:
        return self.dir is not None


def partition_settings(cfg: dict[str, Any]) -> PartitionSettings:
    section = (cfg.get("training") or {}).get("partition") or {}
    directory = section.get("dir")
    return PartitionSettings(
        dir=Path(directory) if directory else None,
        num_parts=int(section.get("num_parts", 64)),
        parts_per_batch=int(section.get("parts_per_batch", 4)),
        recursive=bool(section.get("recursive", False)),
        shuffle=bool(section.get("shuffle", True)),
        num_workers=int(section.get("num_workers", 0)),
        create=bool(section.get("create", True)),
        force=bool(section.get("force", False)),
    )


def ensure_partitions(dataset: GraphDataset, settings: PartitionSettings) -> PartitionStore:
    """Open the partition directory, cutting the graph first if it is missing or stale.

    ``create: false`` is the minimal-memory contract: the run then *only* reads
    what is already on disk and fails with a message instead of falling back to
    loading the whole graph — which is the point on a machine that cannot hold it.
    """
    if settings.dir is None:
        raise KeyError(
            "graph partitioning needs `training.partition.dir` in the config "
            "(the directory the partition files live in)"
        )
    if not settings.force:
        try:
            store = PartitionStore.open(settings.dir)
            store.check(dataset, settings.num_parts)
            return store
        except (FileNotFoundError, ValueError) as error:
            if not settings.create:
                raise type(error)(
                    f"{error}\n`training.partition.create: false` forbids cutting the graph "
                    "here; run `pathwaygnn partition --config <config>` on a machine that "
                    "can hold it"
                ) from error
    elif not settings.create:
        raise ValueError(
            "`training.partition.force: true` asks for a rebuild but "
            "`create: false` forbids one"
        )
    write_partitions(dataset, settings.dir, settings.num_parts, settings.recursive)
    return PartitionStore.open(settings.dir, dataset)


def run_partitioning(cfg: dict[str, Any]) -> dict[str, Any]:
    """``pathwaygnn partition``: cut the configured dataset's graph with METIS."""
    dataset = open_dataset(cfg)
    settings = partition_settings(cfg)
    if settings.dir is None:
        raise KeyError(
            "add a `training.partition.dir` (and `num_parts`) block to the config; "
            "the same block is what makes `pathwaygnn pretrain` use the partitions"
        )
    store = ensure_partitions(dataset, settings)
    parts = store.manifest["parts"]
    sizes = [part["num_nodes"] for part in parts]
    return {
        "dir": str(settings.dir),
        "dataset": store.manifest["dataset"],
        "num_nodes": store.num_nodes,
        "num_edges": store.manifest["num_edges"],
        "num_parts": store.num_parts,
        "nodes_per_part": {
            "min": min(sizes),
            "max": max(sizes),
            "mean": sum(sizes) / len(sizes),
        },
        "edges_per_part": {
            "min": min(part["num_edges"] for part in parts),
            "max": max(part["num_edges"] for part in parts),
        },
        "internal_edge_fraction": (
            sum(part["num_internal_edges"] for part in parts) / store.manifest["num_edges"]
            if store.manifest["num_edges"]
            else 0.0
        ),
        "edgeless_parts": store.num_parts - len(store.trainable_parts),
    }
