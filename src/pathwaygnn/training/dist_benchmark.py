"""Measure what the partition count buys and what it costs.

Graph partitioning (:mod:`pathwaygnn.data.partition`) trades fidelity for memory:
more partitions mean smaller subgraphs per step, hence less activation memory,
but also more edges cut and more steps per pass over the graph. That trade-off is
the whole reason the mode exists, so it is measured rather than asserted.

This command sweeps ``num_parts`` x ``parts_per_batch`` on one or more prepared
datasets and records, per configuration:

* METIS partitioning wall time and the resulting on-disk size (once per
  ``num_parts``), plus the fraction of edges that stay inside a partition
* the subgraph a step actually sees (nodes, edges) and how many steps a pass over
  all partitions takes
* median batch-load time (partition files come off disk) and median
  forward+backward+step time, timed separately so I/O is visible
* peak CUDA memory for a step — parameters, gradients, optimizer state and
  activations together, i.e. what has to fit

A **full-graph** row is measured first on the same model and optimizer, so every
partitioned number has the un-partitioned baseline to be read against.

The step it times mirrors ``run_pretraining``'s and reuses its edge sampling
(:func:`pathwaygnn.training.pretrain.partition_edges`), so the measurement tracks
the real training path rather than a stand-in for it.

Results go to ``<output_dir>/results.json``; ``pathwaygnn-data dist-report``
renders them into ``docs/dist_report.{md,html}``.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pathwaygnn.data.format import GraphDataset
from pathwaygnn.data.partition import PartitionLoader, PartitionStore, write_partitions
from pathwaygnn.models.encoder import GraphPretrainer, RelationalGIN
from pathwaygnn.training.distributed import finalize, initialize
from pathwaygnn.training.pretrain import partition_edges


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    """Start a fresh peak measurement from the currently allocated bytes.

    Parameters and optimizer state stay allocated across configurations, so the
    peak reported below is *everything a step needs resident*, not activations
    alone — the number that decides whether a configuration runs at all.
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mib(device: torch.device) -> float | None:
    """Peak allocated MiB, or ``None`` on CPU where there is no equivalent counter."""
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**20


def _release(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.glob("*") if path.is_file())


def _median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000.0 if samples else float("nan")


def _pass_statistics(loader: PartitionLoader) -> dict[str, float]:
    """Walk a whole pass to size it, because single batches are not representative.

    METIS balances *nodes* per partition, not edges — one cdr partition holds 616
    edges and another 37,840 — so a step's edge count has to be averaged over a
    pass rather than sampled. The same walk yields how much of the graph a pass
    sees at all: an edge is only visible when both its endpoints land in the same
    batch, so this is the fidelity the configuration actually trains at.
    """
    nodes, edges = [], []
    for batch in loader:
        nodes.append(batch.num_nodes)
        edges.append(batch.num_edges)
    return {
        "nodes_per_step": sum(nodes) / len(nodes),
        "edges_per_step": sum(edges) / len(edges),
        "edges_per_step_min": min(edges),
        "edges_per_step_max": max(edges),
        "edges_per_pass": sum(edges),
    }


@dataclass(frozen=True)
class BenchmarkSettings:
    """Everything the sweep needs that is not the dataset itself."""

    device: torch.device
    num_parts: list[int]
    parts_per_batch: list[int]
    hidden_dim: int
    num_layers: int
    dropout: float
    steps: int
    warmup: int
    batch_size: int
    seed: int
    num_workers: int
    partition_root: Path


def _measure_dataset(
    dataset: GraphDataset, settings: BenchmarkSettings
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Sweep one dataset: (info, partitioning costs, measured rows, skipped configurations).

    The model and optimizer live only here, so their memory is released when this
    returns rather than overlapping the next dataset's.
    """
    device = settings.device
    torch.manual_seed(settings.seed)
    model = GraphPretrainer(
        RelationalGIN(
            dataset.num_nodes,
            dataset.num_relations,
            hidden_dim=settings.hidden_dim,
            num_layers=settings.num_layers,
            dropout=settings.dropout,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(settings.seed)
    info = {
        "name": dataset.name,
        "dir": str(dataset.root),
        "num_nodes": dataset.num_nodes,
        "num_relations": dataset.num_relations,
        "num_edges": int(dataset.manifest["num_edges"]),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    partitionings: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total_edges = int(dataset.manifest["num_edges"])

    def timed_step(
        graph_edge_index: torch.Tensor,
        graph_edge_type: torch.Tensor,
        positive: torch.Tensor,
        types: torch.Tensor,
        negative: torch.Tensor,
        nodes: torch.Tensor | None,
    ) -> float:
        """One forward+backward+step, mirroring ``run_pretraining``'s."""
        _synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        positive_score, negative_score = model(
            graph_edge_index, graph_edge_type, positive, types, negative, nodes
        )
        loss = (
            torch.nn.functional.softplus(-positive_score)
            + torch.nn.functional.softplus(negative_score)
        ).mean()
        loss.backward()
        optimizer.step()
        _synchronize(device)
        return time.perf_counter() - start

    # --- full-graph baseline ---------------------------------------------------
    edge_index, edge_type = dataset.graph()
    edge_index, edge_type = edge_index.to(device), edge_type.to(device)
    durations: list[float] = []
    for index in range(settings.warmup + settings.steps):
        if index == settings.warmup:
            _reset_peak(device)
        chosen = torch.randint(
            edge_type.numel(), (settings.batch_size,), generator=generator, device=device
        )
        positive = edge_index[:, chosen]
        types = edge_type[chosen]
        negative = positive.clone()
        negative[1] = torch.randint(
            dataset.num_nodes, (positive.size(1),), generator=generator, device=device
        )
        duration = timed_step(edge_index, edge_type, positive, types, negative, None)
        if index >= settings.warmup:
            durations.append(duration)
    step_ms = _median_ms(durations)
    rows.append({
        "dataset": dataset.name,
        "mode": "full_graph",
        "num_parts": None,
        "parts_per_batch": None,
        "nodes_per_step": dataset.num_nodes,
        "edges_per_step": total_edges,
        "edges_per_step_min": total_edges,
        "edges_per_step_max": total_edges,
        "edges_per_pass": total_edges,
        "edge_coverage": 1.0,
        "steps_per_pass": 1,
        "load_ms": 0.0,
        "step_ms": step_ms,
        "peak_mib": _peak_mib(device),
        "pass_seconds": step_ms / 1000.0,
    })
    del edge_index, edge_type
    _release(device)

    # --- partitioned sweep -----------------------------------------------------
    for num_parts in settings.num_parts:
        if num_parts > dataset.num_nodes:
            skipped.append({
                "dataset": dataset.name, "num_parts": num_parts,
                "reason": "num_parts exceeds the node count",
            })
            continue
        directory = settings.partition_root / dataset.name / f"parts_{num_parts:05d}"
        start = time.perf_counter()
        manifest = write_partitions(dataset, directory, num_parts)
        partition_seconds = time.perf_counter() - start
        internal = sum(part["num_internal_edges"] for part in manifest["parts"])
        sizes = [part["num_nodes"] for part in manifest["parts"]]
        partitionings.append({
            "dataset": dataset.name,
            "num_parts": num_parts,
            "seconds": partition_seconds,
            "disk_mib": _directory_bytes(directory) / 2**20,
            "nodes_per_part_min": min(sizes),
            "nodes_per_part_max": max(sizes),
            "nodes_per_part_mean": sum(sizes) / len(sizes),
            "internal_edge_fraction": internal / manifest["num_edges"],
            "edgeless_parts": sum(
                1 for part in manifest["parts"] if part["num_internal_edges"] == 0
            ),
        })
        store = PartitionStore.open(directory, dataset)

        for parts_per_batch in settings.parts_per_batch:
            if parts_per_batch > num_parts:
                skipped.append({
                    "dataset": dataset.name, "num_parts": num_parts,
                    "parts_per_batch": parts_per_batch,
                    "reason": "parts_per_batch exceeds num_parts",
                })
                continue
            loader = PartitionLoader(
                store,
                parts_per_batch=parts_per_batch,
                # Shuffled, as training is, so the timed batches are typical rather
                # than whichever partitions METIS happened to emit first; the fixed
                # seed keeps it reproducible.
                shuffle=True,
                num_workers=settings.num_workers,
                seed=settings.seed,
            )
            sizing = _pass_statistics(loader)
            iterator = iter(loader)
            loads: list[float] = []
            batches = []
            for _ in range(settings.warmup + settings.steps):
                start = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    # Fewer batches in a pass than steps to time: wrap around.
                    iterator = iter(loader)
                    start = time.perf_counter()
                    batch = next(iterator)
                loads.append(time.perf_counter() - start)
                batches.append(batch.to(device))
            step_durations: list[float] = []
            for index, batch in enumerate(batches):
                if index == settings.warmup:
                    _reset_peak(device)
                positive, types, negative = partition_edges(
                    batch, settings.batch_size, generator, device, False, dataset.num_relations
                )
                duration = timed_step(
                    batch.edge_index, batch.edge_type, positive, types, negative, batch.nodes,
                )
                if index >= settings.warmup:
                    step_durations.append(duration)
            step_ms = _median_ms(step_durations)
            load_ms = _median_ms(loads[settings.warmup:])
            steps_per_pass = len(loader)
            rows.append({
                "dataset": dataset.name,
                "mode": "partitioned",
                "num_parts": num_parts,
                "parts_per_batch": parts_per_batch,
                **sizing,
                "edge_coverage": sizing["edges_per_pass"] / total_edges,
                "steps_per_pass": steps_per_pass,
                "load_ms": load_ms,
                "step_ms": step_ms,
                "peak_mib": _peak_mib(device),
                # Derived, not measured end to end: one pass at this median cost.
                "pass_seconds": steps_per_pass * (step_ms + load_ms) / 1000.0,
            })
            del batches
            _release(device)
    return info, partitionings, rows, skipped


def run_dist_benchmark(cfg: dict[str, Any]) -> dict[str, Any]:
    context = initialize(cfg.get("device", "auto"))
    device = context.device
    try:
        entries = cfg.get("datasets")
        if not entries:
            # One `dataset:` block is also accepted, for symmetry with every other command.
            if "dataset" not in cfg:
                raise KeyError(
                    "dist-benchmark needs either a `datasets:` list of {name, dir} entries "
                    "or a single `dataset:` block"
                )
            entries = [cfg["dataset"]]
        model_cfg = cfg.get("model", {})
        settings = BenchmarkSettings(
            device=device,
            num_parts=[int(value) for value in cfg.get("num_parts", [16, 64, 256])],
            parts_per_batch=[int(value) for value in cfg.get("parts_per_batch", [1, 4, 16])],
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            steps=int(cfg.get("steps", 5)),
            warmup=int(cfg.get("warmup", 2)),
            batch_size=int(cfg.get("batch_size", 4096)),
            seed=int(cfg.get("seed", 42)),
            num_workers=int(cfg.get("num_workers", 0)),
            partition_root=Path(cfg.get("partition_root", "outputs/dist/partitions")),
        )
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        datasets: list[dict[str, Any]] = []
        partitionings: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for entry in entries:
            dataset = GraphDataset.open(entry["dir"], entry.get("name"))
            info, costs, measured, missed = _measure_dataset(dataset, settings)
            datasets.append(info)
            partitionings += costs
            rows += measured
            skipped += missed
            _release(device)

        results = {
            "environment": {
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda"
                    else (platform.processor() or platform.machine())
                ),
                "torch": torch.__version__,
                "python": platform.python_version(),
                "memory_metric": (
                    "peak CUDA bytes allocated per step (parameters, gradients, optimizer "
                    "state and activations)" if device.type == "cuda"
                    else "unavailable on CPU"
                ),
            },
            "settings": {
                "num_parts": settings.num_parts,
                "parts_per_batch": settings.parts_per_batch,
                "model": {
                    "hidden_dim": settings.hidden_dim,
                    "num_layers": settings.num_layers,
                    "dropout": settings.dropout,
                },
                "batch_size": settings.batch_size,
                "steps": settings.steps,
                "warmup": settings.warmup,
                "num_workers": settings.num_workers,
                "seed": settings.seed,
            },
            "datasets": datasets,
            "partitioning": partitionings,
            "rows": rows,
            "skipped": skipped,
        }
        with (output_dir / "results.json").open("w") as handle:
            json.dump(results, handle, indent=2)
        return {
            "output": str(output_dir / "results.json"),
            "datasets": [item["name"] for item in datasets],
            "configurations": len(rows),
            "skipped": len(skipped),
        }
    finally:
        finalize(context)
