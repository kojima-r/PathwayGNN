from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from pathwaygnn.data.format import GraphDataset, open_dataset
from pathwaygnn.data.partition import (
    GraphPartitionBatch,
    PartitionLoader,
    PartitionStore,
    ensure_partitions,
    partition_settings,
)
from pathwaygnn.models.encoder import GraphPretrainer, RelationalGIN, encoder_config
from pathwaygnn.training.distributed import DistributedContext, finalize, initialize


def partition_edges(
    batch: GraphPartitionBatch,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
    balanced: bool,
    num_relations: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Draw positive edges from one subgraph, plus destination-corrupted negatives.

    The same objective as the full-graph loop, with one difference forced by the
    method: a corrupted destination is drawn from the **subgraph's** nodes, because
    those are the only nodes this step computed an embedding for.
    """
    if balanced:
        # A subgraph need not carry every relation; balancing covers those it has.
        present = [torch.where(batch.edge_type == relation)[0] for relation in range(num_relations)]
        chosen = torch.cat([
            indices[
                torch.randint(indices.numel(), (batch_size,), generator=generator, device=device)
            ]
            for indices in present
            if indices.numel()
        ])
    else:
        chosen = torch.randint(
            batch.num_edges, (batch_size,), generator=generator, device=device
        )
    positive = batch.edge_index[:, chosen]
    types = batch.edge_type[chosen]
    negative = positive.clone()
    negative[1] = torch.randint(
        batch.num_nodes, (positive.size(1),), generator=generator, device=device
    )
    return positive, types, negative


def _open_partitions(
    cfg: dict[str, Any], dataset: GraphDataset, context: DistributedContext
) -> PartitionLoader:
    """Cut the graph on rank 0 only, then let every rank read the same files."""
    settings = partition_settings(cfg)
    if context.primary:
        ensure_partitions(dataset, settings)
    if context.distributed:
        dist.barrier()
    store = PartitionStore.open(settings.dir)
    store.check(dataset, settings.num_parts)
    loader = PartitionLoader(
        store,
        parts_per_batch=settings.parts_per_batch,
        shuffle=settings.shuffle,
        num_workers=settings.num_workers,
        rank=context.rank,
        world_size=context.world_size,
        # The base seed, not `seed + rank`: every rank has to shuffle the partition
        # order the same way for the split between them to stay disjoint.
        seed=int(cfg.get("seed", 42)),
    )
    if context.primary:
        print(json.dumps({
            "partitions": store.num_parts,
            "parts_per_batch": settings.parts_per_batch,
            "steps_per_epoch_per_rank": len(loader),
            "edgeless_parts_skipped": loader.num_skipped,
        }))
    return loader


def run_pretraining(cfg: dict[str, Any]) -> None:
    context = initialize(cfg.get("device", "auto"))
    try:
        seed = int(cfg.get("seed", 42)) + context.rank
        random.seed(seed)
        torch.manual_seed(seed)
        dataset = open_dataset(cfg)
        num_nodes, num_relations = dataset.num_nodes, dataset.num_relations
        partitioned = partition_settings(cfg).enabled
        loader: PartitionLoader | None = None
        edge_index = edge_type = None
        if partitioned:
            # The graph itself is never loaded in this mode; only partition files are.
            loader = _open_partitions(cfg, dataset, context)
        else:
            edge_index, edge_type = dataset.graph()
            edge_index, edge_type = edge_index.to(context.device), edge_type.to(context.device)
        model_cfg = cfg.get("model", {})
        dropout = float(model_cfg.get("dropout", 0.1))
        model = GraphPretrainer(
            RelationalGIN(
                num_nodes,
                num_relations,
                hidden_dim=int(model_cfg.get("hidden_dim", 64)),
                num_layers=int(model_cfg.get("num_layers", 2)),
                dropout=dropout,
            )
        ).to(context.device)
        wrapped: torch.nn.Module = model
        if context.distributed:
            wrapped = DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            )
        optimizer = torch.optim.AdamW(
            wrapped.parameters(),
            lr=float(cfg["training"].get("learning_rate", 1e-3)),
            weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
        )
        epochs = int(cfg["training"].get("epochs", 100))
        steps = int(cfg["training"].get("steps_per_epoch", 100))
        batch_size = int(cfg["training"].get("batch_size", 4096))
        output_dir = Path(cfg["output_dir"])
        if context.primary:
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "config.json").open("w") as handle:
                json.dump(cfg, handle, indent=2)
        if context.distributed:
            dist.barrier()
        generator = torch.Generator(device=context.device).manual_seed(seed)
        history = []
        checkpoint_epochs = {int(value) for value in cfg["training"].get("checkpoint_epochs", [])}
        best_loss = float("inf")
        balanced_relations = bool(cfg["training"].get("balanced_relations", False))
        relation_indices = (
            [torch.where(edge_type == relation)[0] for relation in range(num_relations)]
            if balanced_relations and not partitioned else None
        )

        def step(
            graph_edge_index: Tensor,
            graph_edge_type: Tensor,
            positive: Tensor,
            types: Tensor,
            negative: Tensor,
            nodes: Tensor | None = None,
        ) -> tuple[float, float]:
            optimizer.zero_grad(set_to_none=True)
            positive_score, negative_score = wrapped(
                graph_edge_index, graph_edge_type, positive, types, negative, nodes
            )
            loss = (
                torch.nn.functional.softplus(-positive_score)
                + torch.nn.functional.softplus(negative_score)
            ).mean()
            loss.backward()
            optimizer.step()
            return float(loss.detach()), float((positive_score > negative_score).float().mean())

        def snapshot(epoch: int) -> dict[str, Any]:
            return {
                "encoder": model.encoder.state_dict(),
                "relation": model.relation.state_dict(),
                "model_config": encoder_config(model.encoder, dropout),
                "dataset": dataset.name,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
            }

        if context.primary and 0 in checkpoint_epochs:
            torch.save(snapshot(0), output_dir / "epoch_0.pt")
        for epoch in range(1, epochs + 1):
            wrapped.train()
            loss_sum = accuracy_sum = 0.0
            if loader is not None:
                # One epoch is one pass over this rank's share of the partitions, so
                # `steps_per_epoch` does not apply; `batch_size` is still the number
                # of positive edges drawn per step, now from within the subgraph.
                loader.set_epoch(epoch)
                taken = 0
                for batch in loader:
                    batch = batch.to(context.device)
                    positive, types, negative = partition_edges(
                        batch,
                        batch_size,
                        generator,
                        context.device,
                        balanced_relations,
                        num_relations,
                    )
                    loss_value, accuracy_value = step(
                        batch.edge_index, batch.edge_type, positive, types, negative, batch.nodes
                    )
                    loss_sum += loss_value
                    accuracy_sum += accuracy_value
                    taken += 1
                divisor = max(taken, 1)
            else:
                for _ in range(steps):
                    if relation_indices is not None:
                        chosen = torch.cat([
                            indices[torch.randint(
                                indices.numel(), (batch_size,), generator=generator, device=context.device
                            )]
                            for indices in relation_indices
                        ])
                    else:
                        chosen = torch.randint(
                            edge_index.size(1),
                            (batch_size,),
                            generator=generator,
                            device=context.device,
                        )
                    positive = edge_index[:, chosen]
                    types = edge_type[chosen]
                    negative = positive.clone()
                    negative[1] = torch.randint(
                        num_nodes,
                        (positive.size(1),),
                        generator=generator,
                        device=context.device,
                    )
                    loss_value, accuracy_value = step(
                        edge_index, edge_type, positive, types, negative
                    )
                    loss_sum += loss_value
                    accuracy_sum += accuracy_value
                divisor = steps
            stats = torch.tensor(
                [loss_sum / divisor, accuracy_sum / divisor],
                device=context.device,
            )
            if context.distributed:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                stats /= context.world_size
            record = {"epoch": epoch, "loss": float(stats[0]), "accuracy": float(stats[1])}
            if context.primary:
                history.append(record)
                print(json.dumps(record))
                checkpoint = snapshot(epoch)
                torch.save(checkpoint, output_dir / "last.pt")
                if epoch in checkpoint_epochs:
                    torch.save(checkpoint, output_dir / f"epoch_{epoch}.pt")
                if record["loss"] <= best_loss:
                    best_loss = record["loss"]
                    torch.save(checkpoint, output_dir / "best.pt")
                    with (output_dir / "history.json").open("w") as handle:
                        json.dump(history, handle, indent=2)
    finally:
        finalize(context)
