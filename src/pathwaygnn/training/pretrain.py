from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from pathwaygnn.data.format import open_dataset
from pathwaygnn.models.encoder import GraphPretrainer, RelationalGIN, encoder_config
from pathwaygnn.training.distributed import finalize, initialize


def run_pretraining(cfg: dict[str, Any]) -> None:
    context = initialize(cfg.get("device", "auto"))
    try:
        seed = int(cfg.get("seed", 42)) + context.rank
        random.seed(seed)
        torch.manual_seed(seed)
        dataset = open_dataset(cfg)
        num_nodes, num_relations = dataset.num_nodes, dataset.num_relations
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
            if balanced_relations else None
        )

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
                optimizer.zero_grad(set_to_none=True)
                positive_score, negative_score = wrapped(
                    edge_index, edge_type, positive, types, negative
                )
                loss = (
                    torch.nn.functional.softplus(-positive_score)
                    + torch.nn.functional.softplus(negative_score)
                ).mean()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach())
                accuracy_sum += float((positive_score > negative_score).float().mean())
            stats = torch.tensor(
                [loss_sum / steps, accuracy_sum / steps],
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
