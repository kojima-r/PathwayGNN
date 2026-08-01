from __future__ import annotations

import math

import torch
from torch import Tensor


def binary_auc(target: Tensor, score: Tensor) -> float:
    target, score = target.detach().cpu().float(), score.detach().cpu().float()
    positive = int(target.sum())
    negative = target.numel() - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = torch.argsort(score)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, target.numel() + 1, dtype=torch.float)
    # Average tied ranks.
    unique, inverse, counts = torch.unique(score, return_inverse=True, return_counts=True)
    if bool((counts > 1).any()):
        for group in range(unique.numel()):
            mask = inverse == group
            ranks[mask] = ranks[mask].mean()
    rank_sum = ranks[target.bool()].sum().item()
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def binary_metrics(target: Tensor, logits: Tensor, loss: float | None = None) -> dict[str, float]:
    target = target.detach().cpu().float()
    probability = logits.detach().cpu().float().sigmoid()
    prediction = probability >= 0.5
    truth = target.bool()
    tp = int((prediction & truth).sum())
    tn = int((~prediction & ~truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result = {
        "accuracy": (tp + tn) / max(target.numel(), 1),
        "auc": binary_auc(target, probability),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "predicted_positive_ratio": float(prediction.float().mean()),
        "actual_positive_ratio": float(target.mean()),
    }
    if loss is not None and not math.isnan(loss):
        result["loss"] = loss
    return result

