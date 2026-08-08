"""Single train/validation/test protocol with early stopping on validation AUC."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pathwaygnn.data.format import open_task
from pathwaygnn.data.samples import TaskDataset
from pathwaygnn.models.encoder import RelationalGIN, load_encoder
from pathwaygnn.models.predictor import SampleLevelModel, build_model
from pathwaygnn.training.cv import normalize_sample_features, standardizer
from pathwaygnn.training.metrics import binary_metrics


def stratified_split(
    targets: np.ndarray, ratios: tuple[float, float, float], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
    result: list[list[Tensor]] = [[], [], []]
    for label in (0, 1):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        first = int(indices.numel() * ratios[0])
        second = first + int(indices.numel() * ratios[1])
        for bucket, values in zip(
            result, (indices[:first], indices[first:second], indices[second:])
        ):
            bucket.append(values)
    splits = []
    for parts in result:
        merged = torch.cat(parts)
        splits.append(merged[torch.randperm(merged.numel(), generator=generator)].numpy())
    return tuple(splits)  # type: ignore[return-value]


def _evaluate(
    encoder: RelationalGIN | None,
    predictor: SampleLevelModel,
    loader: DataLoader,
    graph: tuple[Tensor, Tensor] | None,
    device: torch.device,
    mean: Tensor | None,
    std: Tensor | None,
) -> dict[str, float]:
    if encoder is not None:
        encoder.eval()
    predictor.eval()
    logits, targets, weighted_loss, count = [], [], 0.0, 0
    with torch.no_grad():
        embeddings = None if encoder is None else encoder(*graph)  # type: ignore[misc]
        for batch in loader:
            batch = normalize_sample_features(batch.to(device), mean, std)
            output = predictor(batch, embeddings)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, batch.label)
            logits.append(output.cpu())
            targets.append(batch.label.cpu())
            weighted_loss += float(loss) * batch.size
            count += batch.size
    return binary_metrics(torch.cat(targets), torch.cat(logits), weighted_loss / max(count, 1))


def run_finetuning(cfg: dict[str, Any]) -> dict[str, float]:
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if cfg.get("device", "auto") != "cpu" and torch.cuda.is_available() else "cpu"
    )
    dataset, task = open_task(cfg)
    data = TaskDataset(task)
    ratios = tuple(cfg.get("split", [0.7, 0.15, 0.15]))
    train_idx, valid_idx, test_idx = stratified_split(data.targets, ratios, seed)  # type: ignore[arg-type]
    training = cfg["training"]
    loader_args = {
        "batch_size": int(training.get("batch_size", 64)),
        "num_workers": int(training.get("num_workers", 0)),
        "collate_fn": data.collate(),
    }
    train_loader = DataLoader(data.subset(train_idx), shuffle=True, **loader_args)
    valid_loader = DataLoader(data.subset(valid_idx), shuffle=False, **loader_args)
    test_loader = DataLoader(data.subset(test_idx), shuffle=False, **loader_args)
    variant = dict(cfg.get("variant", {}))
    use_graph = bool(variant.get("use_graph", True))
    model_cfg = cfg.get("model", {})
    encoder = None
    graph: tuple[Tensor, Tensor] | None = None
    embedding_dim = int(model_cfg.get("embedding_dim", 64))
    if use_graph:
        encoder, _ = load_encoder(
            cfg["pretrained_checkpoint"],
            dataset.num_nodes,
            dataset.num_relations,
            device,
            node_names=dataset.node_names(),
            node_embeddings=model_cfg.get("node_embeddings"),
        )
        edge_index, edge_type = dataset.graph()
        graph = (edge_index.to(device), edge_type.to(device))
        embedding_dim = encoder.hidden_dim
    predictor = build_model(
        task.node_feature_names,
        task.sample_feature_dim,
        embedding_dim,
        model_cfg,
        use_graph,
        bool(variant.get("use_sample_features", False)),
    ).to(device)
    train_encoder = use_graph and bool(training.get("train_encoder", False))
    if encoder is not None:
        encoder.requires_grad_(train_encoder)
    parameters = list(predictor.parameters())
    if train_encoder:
        parameters += list(encoder.parameters())  # type: ignore[union-attr]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    mean, std = standardizer(task.sample_features(), train_idx, device)
    positive = float(data.targets[train_idx].sum())
    negative = train_idx.size - positive
    pos_weight = torch.tensor(negative / max(positive, 1.0), device=device)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_auc, stale = -1.0, 0
    patience = int(training.get("patience", 20))
    history = []
    for epoch in range(1, int(training.get("epochs", 100)) + 1):
        if encoder is not None:
            encoder.train(train_encoder)
        predictor.train()
        cached = (
            encoder(*graph).detach()  # type: ignore[misc]
            if encoder is not None and not train_encoder
            else None
        )
        for batch in train_loader:
            batch = normalize_sample_features(batch.to(device), mean, std)
            optimizer.zero_grad(set_to_none=True)
            embeddings = cached
            if encoder is not None and train_encoder:
                embeddings = encoder(*graph)  # type: ignore[misc]
            logits = predictor(batch, embeddings)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch.label, pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
        metrics = _evaluate(encoder, predictor, valid_loader, graph, device, mean, std)
        record = {"epoch": epoch, **{f"valid_{key}": value for key, value in metrics.items()}}
        history.append(record)
        print(json.dumps(record))
        if metrics["auc"] > best_auc:
            best_auc, stale = metrics["auc"], 0
            torch.save(
                {
                    "encoder": None if encoder is None else encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "model_config": predictor.config,
                    "variant": {"use_graph": use_graph, **variant},
                    "dataset": dataset.name,
                    "task": task.name,
                    "pretrained_checkpoint": cfg.get("pretrained_checkpoint"),
                    "sample_feature_mean": None if mean is None else mean.detach().cpu(),
                    "sample_feature_std": None if std is None else std.detach().cpu(),
                    "epoch": epoch,
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= patience:
                break
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    if encoder is not None:
        encoder.load_state_dict(checkpoint["encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    result = _evaluate(encoder, predictor, test_loader, graph, device, mean, std)
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(
            {
                "dataset": dataset.name,
                "task": task.name,
                "test": result,
                "history": history,
                "split": {
                    "train": train_idx.tolist(),
                    "valid": valid_idx.tolist(),
                    "test": test_idx.tolist(),
                },
            },
            handle,
            indent=2,
        )
    return result
