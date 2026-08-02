from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

THRESHOLD = 0.5
#: The metrics every evaluation records: one ranking metric plus the
#: confusion-matrix metrics at :data:`THRESHOLD`.
METRICS = ("auc", "accuracy", "precision", "recall", "f1")


def _vector(values: Tensor | np.ndarray | list) -> Tensor:
    tensor = values if isinstance(values, Tensor) else torch.as_tensor(np.asarray(values))
    return tensor.detach().cpu().reshape(-1)


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


def threshold_metrics(
    target: Tensor | np.ndarray, probability: Tensor | np.ndarray, threshold: float = THRESHOLD
) -> dict[str, float]:
    """Confusion-matrix metrics at a fixed decision threshold.

    ``probability`` is a probability, not a logit, so this also scores a stored
    ``predictions.npz`` without re-running the model.
    """
    truth = _vector(target).bool()
    prediction = _vector(probability).float() >= threshold
    tp = int((prediction & truth).sum())
    tn = int((~prediction & ~truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / max(truth.numel(), 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "predicted_positive_ratio": float(prediction.float().mean()),
        "actual_positive_ratio": float(truth.float().mean()),
    }


def binary_metrics(target: Tensor, logits: Tensor, loss: float | None = None) -> dict[str, float]:
    target = target.detach().cpu().float()
    probability = logits.detach().cpu().float().sigmoid()
    scores = threshold_metrics(target, probability)
    result = {
        "accuracy": scores["accuracy"],
        "auc": binary_auc(target, probability),
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
        "predicted_positive_ratio": scores["predicted_positive_ratio"],
        "actual_positive_ratio": scores["actual_positive_ratio"],
    }
    if loss is not None and not math.isnan(loss):
        result["loss"] = loss
    return result
