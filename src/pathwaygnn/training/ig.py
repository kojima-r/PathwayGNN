"""Integrated Gradients for any prepared task.

Attributes a trained cross-validation fold with respect to three input groups:
the graph node embedding matrix (when the variant uses the graph), the values of
every channel, and the covariate vector. Node rankings are written with the
dataset's own node names, and per-group rankings use the task's group names.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pathwaygnn.data.format import open_task
from pathwaygnn.data.samples import SampleBatch, TaskDataset
from pathwaygnn.models.encoder import load_encoder
from pathwaygnn.models.predictor import SampleLevelModel


def _write_ranking(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def _ranking_rows(
    score: Tensor, names: list[str], order: np.ndarray, absolute: bool
) -> list[list[object]]:
    rows = []
    for rank, index in enumerate(order):
        value = float(score[index])
        row: list[object] = [rank + 1, int(index), names[int(index)], value]
        if absolute:
            row.append(abs(value))
        rows.append(row)
    return rows


def _scatter(values: Tensor, index: Tensor, size: int) -> Tensor:
    output = torch.zeros(size, dtype=torch.float64)
    output.index_add_(0, index.cpu(), values.reshape(-1).double().cpu())
    return output


def _scale(
    batch: SampleBatch, alpha: Tensor, include_covariate: bool
) -> tuple[SampleBatch, list[Tensor]]:
    """A copy of ``batch`` whose values are scaled by ``alpha`` and differentiable.

    Only inputs the model actually consumes are made differentiable, so that
    ``autograd.grad`` stays strict about unused tensors.
    """
    inputs, channels = [], {}
    for name, channel in batch.channels.items():
        value = (channel.value * alpha).detach().requires_grad_(True)
        inputs.append(value)
        channels[name] = replace(channel, value=value)
    covariate = batch.covariate
    if include_covariate and covariate is not None:
        covariate = (covariate * alpha).detach().requires_grad_(True)
        inputs.append(covariate)
    return replace(batch, channels=channels, covariate=covariate), inputs


def run_ig(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset, task = open_task(cfg)
    fold = int(cfg.get("fold", 0))
    variant_name = cfg["variant"]
    fold_dir = Path(cfg["run_dir"]) / task.name / variant_name / f"fold_{fold}"
    checkpoint = torch.load(fold_dir / "model.pt", map_location="cpu", weights_only=False)
    if "model_config" not in checkpoint:
        raise KeyError(
            f"{fold_dir / 'model.pt'} predates the generic model format. Re-run "
            f"`pathwaygnn cv` for this condition (delete {fold_dir} to force the fold)."
        )
    model_config = dict(checkpoint["model_config"])
    device = torch.device(
        "cuda" if cfg.get("device", "auto") != "cpu" and torch.cuda.is_available() else "cpu"
    )
    use_graph = bool(model_config["use_graph"])
    encoder = None
    graph: tuple[Tensor, Tensor] | None = None
    base_embedding: Tensor | None = None
    if use_graph:
        encoder, _ = load_encoder(
            cfg["pretrained_checkpoint"], dataset.num_nodes, dataset.num_relations, device
        )
        if checkpoint["encoder"] is not None:
            encoder.load_state_dict(checkpoint["encoder"])
        encoder.eval()
        edge_index, edge_type = dataset.graph()
        graph = (edge_index.to(device), edge_type.to(device))
        base_embedding = encoder.embedding.weight.detach()
    predictor = SampleLevelModel.from_config(model_config).to(device)
    predictor.load_state_dict(checkpoint["predictor"])
    predictor.eval()
    mean = checkpoint.get("covariate_mean")
    std = checkpoint.get("covariate_std")
    mean = None if mean is None else mean.to(device)
    std = None if std is None else std.to(device)

    data = TaskDataset(task)
    collate = data.collate()
    sample_indices = np.load(fold_dir / "predictions.npz")["sample_index"]
    max_samples = cfg.get("max_samples")
    if max_samples is not None:
        generator = np.random.default_rng(int(cfg.get("seed", 100)))
        sample_indices = generator.choice(
            sample_indices, size=min(int(max_samples), len(sample_indices)), replace=False
        )
    steps = int(cfg.get("steps", 50))
    groups = task.groups()
    num_nodes = dataset.num_nodes
    graph_ig = torch.zeros((num_nodes, predictor.embedding_dim), dtype=torch.float64)
    group_graph_ig = torch.zeros((task.num_groups, num_nodes), dtype=torch.float64)
    group_counts = torch.zeros(max(task.num_groups, 1), dtype=torch.long)
    channel_ig = {name: torch.zeros(num_nodes, dtype=torch.float64) for name in task.channel_names}
    covariate_ig = torch.zeros(max(task.covariate_dim, 1), dtype=torch.float64)
    covered = {name: torch.zeros(num_nodes, dtype=torch.bool) for name in task.channel_names}

    for count, sample_index in enumerate(sample_indices, start=1):
        batch = collate([data[int(sample_index)]]).to(device)
        if batch.covariate is not None and mean is not None:
            batch.covariate = (batch.covariate - mean) / std
        value_gradient = {name: torch.zeros_like(ch.value) for name, ch in batch.channels.items()}
        covariate_gradient = (
            torch.zeros_like(batch.covariate)
            if predictor.use_covariates and batch.covariate is not None
            else None
        )
        embedding_gradient = None if base_embedding is None else torch.zeros_like(base_embedding)
        for alpha in torch.linspace(0.0, 1.0, steps, device=device):
            scaled, inputs = _scale(batch, alpha, predictor.use_covariates)
            node_embeddings = None
            if use_graph:
                scaled_embedding = (base_embedding * alpha).detach().requires_grad_(True)
                inputs.append(scaled_embedding)
                node_embeddings = encoder.forward_from_embedding(scaled_embedding, *graph)
            probability = predictor(scaled, node_embeddings).sigmoid().sum()
            gradients = list(torch.autograd.grad(probability, tuple(inputs), retain_graph=False))
            if use_graph:
                embedding_gradient += gradients.pop()  # type: ignore[operator]
            if covariate_gradient is not None:
                covariate_gradient += gradients.pop()
            for name, gradient in zip(batch.channels, gradients):
                value_gradient[name] += gradient
        for name, channel in batch.channels.items():
            attribution = channel.value * value_gradient[name] / steps
            index = (
                channel.gene.repeat(channel.value.size(0))
                if channel.dense
                else channel.gene
            )
            channel_ig[name] += _scatter(attribution, index, num_nodes)
            covered[name][index.cpu()] = True
        if covariate_gradient is not None:
            covariate_ig += (batch.covariate * covariate_gradient / steps).reshape(-1).double().cpu()
        if embedding_gradient is not None:
            sample_graph_ig = (base_embedding * embedding_gradient / steps).double().cpu()
            graph_ig += sample_graph_ig
            if groups is not None and task.num_groups:
                code = int(groups[int(sample_index)])
                if 0 <= code < task.num_groups:
                    group_graph_ig[code] += sample_graph_ig.norm(dim=1)
                    group_counts[code] += 1
        print(json.dumps({"stage": "ig", "sample": count, "total": len(sample_indices),
                          "sample_index": int(sample_index)}))

    denominator = max(len(sample_indices), 1)
    graph_score = graph_ig.norm(dim=1) / denominator
    node_names = dataset.node_names()
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    top_k = int(cfg.get("top_k", 1500))
    arrays: dict[str, np.ndarray] = {}
    if use_graph:
        order = torch.argsort(graph_score, descending=True)[:top_k].numpy()
        _write_ranking(
            output / "top_graph_nodes.tsv",
            ["rank", "node_index", "node", "ig_l2"],
            _ranking_rows(graph_score, node_names, order, absolute=False),
        )
        arrays["graph_score"] = graph_score.numpy()
    for name in task.channel_names:
        score = channel_ig[name] / denominator
        candidates = torch.where(covered[name])[0]
        order = candidates[torch.argsort(score[candidates].abs(), descending=True)][:top_k].numpy()
        _write_ranking(
            output / f"top_channel_{name}.tsv",
            ["rank", "node_index", "node", "signed_ig", "absolute_ig"],
            _ranking_rows(score, node_names, order, absolute=True),
        )
        arrays[f"channel_{name}"] = score.numpy()
    covariate_score = covariate_ig / denominator
    if task.covariate_dim:
        arrays["covariate_score"] = covariate_score[: task.covariate_dim].numpy()
    correlation = None
    if use_graph:
        edge_index, _ = dataset.graph()
        degree = torch.bincount(edge_index.reshape(-1), minlength=num_nodes).double()
        arrays["degree"] = degree.numpy()
        correlation = float(torch.corrcoef(torch.stack((degree, graph_score)))[0, 1])
    per_group = {}
    if use_graph and cfg.get("per_group_rankings", False) and task.num_groups:
        for code, name in enumerate(task.group_names):
            if not group_counts[code]:
                continue
            scores = group_graph_ig[code] / group_counts[code]
            order = torch.argsort(scores, descending=True)[:top_k].numpy()
            _write_ranking(
                output / f"top_graph_nodes_{name}.tsv",
                ["rank", "node_index", "node", "ig_l2"],
                _ranking_rows(scores, node_names, order, absolute=False),
            )
            per_group[name] = int(group_counts[code])
    elif task.num_groups:
        per_group = {
            name: int(group_counts[code])
            for code, name in enumerate(task.group_names)
            if group_counts[code]
        }
    np.savez_compressed(output / "attributions.npz", **arrays)
    result = {
        "dataset": dataset.name,
        "task": task.name,
        "variant": variant_name,
        "fold": fold,
        "num_samples": len(sample_indices),
        "integration_steps": steps,
        "degree_ig_pearson_r": correlation,
        "covariate_ig": {
            name: float(covariate_score[position])
            for position, name in enumerate(task.covariate_names)
        },
        "per_group_samples": per_group,
        "reference": cfg.get("reference", {}),
    }
    (output / "ig_summary.json").write_text(json.dumps(result, indent=2, allow_nan=True))
    return result
