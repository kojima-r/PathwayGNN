"""Stratified k-fold cross-validation over any prepared task.

Runs one grid of ``variants x tasks x folds``, resumes at fold level and writes
``<output_dir>/<task>/<variant>/fold_<k>/{metrics.json,predictions.npz,model.pt}``
plus a per-condition ``summary.json``. Nothing here is dataset-specific: the
group breakdown, the covariate branch and the channel list all come from the
task manifest.

Every evaluation records ROC-AUC *and* accuracy/precision/recall/F1 at a 0.5
decision threshold, per epoch in ``history``, per fold in ``metrics.json`` and as
``mean_``/``std_``/``fold_`` entries in ``summary.json``. Model selection still
uses ROC-AUC alone.
"""

from __future__ import annotations

import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import Tensor
from torch.utils.data import DataLoader

from pathwaygnn.data.format import GraphDataset, Task, open_dataset
from pathwaygnn.data.samples import TaskDataset
from pathwaygnn.models.encoder import RelationalGIN, load_encoder
from pathwaygnn.models.predictor import SampleLevelModel, build_model
from pathwaygnn.training.metrics import METRICS, threshold_metrics


def auc_score(target: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(target, probability)) if np.unique(target).size == 2 else math.nan


def fold_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """ROC-AUC plus the 0.5-threshold metrics for one set of held-out predictions."""
    return {"auc": auc_score(target, probability), **threshold_metrics(target, probability)}


def standardizer(
    covariates: np.ndarray | None, train_index: np.ndarray, device: torch.device
) -> tuple[Tensor | None, Tensor | None]:
    if covariates is None:
        return None, None
    values = torch.from_numpy(np.asarray(covariates[train_index]).copy()).float().to(device)
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    std[std == 0] = 1
    return mean, std


def normalize_covariates(batch, mean: Tensor | None, std: Tensor | None):
    if batch.covariate is not None and mean is not None:
        batch.covariate = (batch.covariate - mean) / std
    return batch


def _evaluate(
    predictor: SampleLevelModel,
    encoder: RelationalGIN | None,
    loader: DataLoader,
    graph: tuple[Tensor, Tensor] | None,
    device: torch.device,
    mean: Tensor | None,
    std: Tensor | None,
    frozen_embeddings: Tensor | None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    predictor.eval()
    if encoder is not None:
        encoder.eval()
    probabilities, targets, indices = [], [], []
    with torch.no_grad():
        embeddings = frozen_embeddings
        if encoder is not None and embeddings is None:
            embeddings = encoder(*graph)  # type: ignore[misc]
        for batch in loader:
            batch = normalize_covariates(batch.to(device), mean, std)
            logits = predictor(batch, embeddings)
            probabilities.append(logits.sigmoid().cpu().numpy())
            targets.append(batch.label.cpu().numpy())
            indices.append(batch.index.numpy())
    probability, target, index = map(np.concatenate, (probabilities, targets, indices))
    return fold_metrics(target, probability), target, probability, index


def _one_fold(
    cfg: dict[str, Any],
    dataset: GraphDataset,
    task: Task,
    variant: dict[str, Any],
    variant_index: int,
    fold: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    fold_dir = output_dir / f"fold_{fold}"
    metrics_path = fold_dir / "metrics.json"
    predictions_path = fold_dir / "predictions.npz"
    model_path = fold_dir / "model.pt"
    if (
        bool(cfg["training"].get("resume", True))
        and metrics_path.exists()
        and predictions_path.exists()
        and model_path.exists()
    ):
        result = json.loads(metrics_path.read_text())
        status = "reused"
        if any(key not in result for key in METRICS):
            # Written before the threshold metrics were recorded. They are a
            # function of the stored held-out predictions, so recover them here
            # instead of forcing a retrain.
            history = result.pop("history", [])
            arrays = np.load(predictions_path)
            result = {
                **result,
                **threshold_metrics(arrays["target"], arrays["probability"]),
                "history": history,
            }
            metrics_path.write_text(json.dumps(result, indent=2, allow_nan=True))
            status = "reused (threshold metrics backfilled)"
        print(json.dumps({"stage": "cv", "status": status, "task": task.name,
                          "variant": variant["name"], "fold": fold, "auc": result["auc"]}))
        return result
    fold_seed = int(cfg.get("seed", 42)) + task.seed_offset * 1000 + fold
    fold_seed += int(variant.get("seed_index", variant_index)) * 100
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)
    started = time.time()
    device = torch.device(
        "cuda" if cfg.get("device", "auto") != "cpu" and torch.cuda.is_available() else "cpu"
    )
    train_data = TaskDataset(task, train_index)
    test_data = TaskDataset(task, test_index)
    training = cfg["training"]
    model_cfg = cfg.get("model", {})
    common = {
        "batch_size": int(training.get("batch_size", 1024)),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
        "collate_fn": train_data.collate(),
    }
    train_loader = DataLoader(
        train_data,
        shuffle=bool(training.get("shuffle", False)),
        drop_last=bool(model_cfg.get("batch_norm", False))
        and len(train_data) % common["batch_size"] == 1,
        **common,
    )
    test_loader = DataLoader(test_data, shuffle=False, **common)
    use_graph = bool(variant["use_graph"])
    train_encoder = use_graph and bool(
        variant.get("end_to_end", training.get("end_to_end", True))
    )
    encoder: RelationalGIN | None = None
    graph: tuple[Tensor, Tensor] | None = None
    frozen_embeddings: Tensor | None = None
    embedding_dim = int(model_cfg.get("embedding_dim", 32))
    if use_graph:
        # Building the encoder before the head keeps the initialisation draws in
        # the order the published runs used; do not reorder these two blocks.
        encoder, _ = load_encoder(
            cfg["pretrained_checkpoint"], dataset.num_nodes, dataset.num_relations, device
        )
        edge_index, edge_type = dataset.graph()
        graph = (edge_index.to(device), edge_type.to(device))
        encoder.requires_grad_(train_encoder)
        if not train_encoder:
            encoder.eval()
            with torch.no_grad():
                frozen_embeddings = encoder(*graph)
        embedding_dim = encoder.hidden_dim
    predictor = build_model(
        task.channel_names,
        task.covariate_dim,
        embedding_dim,
        model_cfg,
        use_graph,
        bool(variant.get("use_covariates", False)),
    ).to(device)
    params = list(predictor.parameters()) + (list(encoder.parameters()) if train_encoder else [])
    optimizer = torch.optim.AdamW(
        params,
        lr=float(training.get("learning_rate", 5e-5)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=int(training.get("scheduler_patience", 10))
    )
    mean, std = standardizer(task.covariates(), train_index, device)
    loss_clip = training.get("loss_clip")
    reduction = training.get("loss_reduction", "mean")
    grad_clip = training.get("grad_clip_value", 10.0)
    select_best = training.get("selection", "final_epoch") == "best_test_auc"
    best_auc, best_state, history = -math.inf, None, []
    for epoch in range(1, int(training.get("epochs", 150)) + 1):
        predictor.train()
        if encoder is not None:
            encoder.train(train_encoder)
        total_loss = total_samples = 0.0
        for batch in train_loader:
            batch = normalize_covariates(batch.to(device), mean, std)
            embeddings = frozen_embeddings
            if encoder is not None and train_encoder:
                embeddings = encoder(*graph)  # type: ignore[misc]
            optimizer.zero_grad(set_to_none=True)
            logits = predictor(batch, embeddings)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch.label, reduction="none"
            )
            if loss_clip is not None:
                losses = losses.clamp_min(float(loss_clip))
            loss = losses.sum() if reduction == "sum" else losses.mean()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_value_(predictor.parameters(), float(grad_clip))
            optimizer.step()
            total_loss += float(loss.detach())
            total_samples += batch.size
        metrics, target, probability, sample_index = _evaluate(
            predictor, encoder, test_loader, graph, device, mean, std, frozen_embeddings
        )
        test_auc = metrics["auc"]
        scheduler.step(test_auc)
        history.append({
            "epoch": epoch,
            "train_loss_per_sample": total_loss / total_samples,
            **{f"test_{key}": value for key, value in metrics.items()},
        })
        print(json.dumps({"task": task.name, "variant": variant["name"], "fold": fold,
                          **history[-1]}))
        if best_state is None or (select_best and test_auc >= best_auc) or not select_best:
            best_auc = test_auc
            best_state = {
                "predictor": deepcopy(predictor.state_dict()),
                "encoder": deepcopy(encoder.state_dict()) if encoder is not None else None,
                "epoch": epoch,
                "metrics": metrics,
                "target": target,
                "probability": probability,
                "sample_index": sample_index,
            }
    assert best_state is not None
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "predictor": best_state["predictor"],
            "encoder": best_state["encoder"],
            "epoch": best_state["epoch"],
            "variant": variant,
            "dataset": dataset.name,
            "task": task.name,
            "model_config": predictor.config,
            "pretrained_checkpoint": cfg.get("pretrained_checkpoint"),
            "covariate_mean": None if mean is None else mean.detach().cpu(),
            "covariate_std": None if std is None else std.detach().cpu(),
        },
        model_path,
    )
    np.savez_compressed(
        predictions_path,
        target=best_state["target"],
        probability=best_state["probability"],
        sample_index=best_state["sample_index"],
    )
    result = {
        "task": task.name,
        "variant": variant["name"],
        "fold": fold,
        "auc": best_auc,
        **{key: value for key, value in best_state["metrics"].items() if key != "auc"},
        "selected_epoch": best_state["epoch"],
        "duration_seconds": time.time() - started,
        "seed": fold_seed,
        "per_group_auc": _per_group_auc(task, best_state),
        "history": history,
    }
    metrics_path.write_text(json.dumps(result, indent=2, allow_nan=True))
    return result


def _per_group_auc(task: Task, state: dict[str, Any]) -> dict[str, float]:
    groups = task.groups()
    if groups is None or not task.group_names:
        return {}
    codes = np.asarray(groups)[state["sample_index"]]
    result = {}
    for code, name in enumerate(task.group_names):
        mask = codes == code
        result[name] = (
            auc_score(state["target"][mask], state["probability"][mask]) if mask.any() else math.nan
        )
    return result


def run_cv(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = open_dataset(cfg)
    task_names: Sequence[str] = cfg["dataset"].get("tasks") or [cfg["dataset"]["task"]]
    variants = cfg["variants"]
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if cfg.get("write_root_manifest", True):
        (root / "config.json").write_text(json.dumps(cfg, indent=2))
    results = {}
    for task_name in task_names:
        task = dataset.task(task_name)
        data = TaskDataset(task)
        splitter = StratifiedKFold(
            n_splits=int(cfg.get("folds", 5)), shuffle=True, random_state=seed
        )
        splits = list(splitter.split(np.zeros(len(data)), data.targets))
        for variant_index, variant in enumerate(variants):
            key = f"{task_name}/{variant['name']}"
            output_dir = root / task_name / variant["name"]
            fold_results = [
                _one_fold(cfg, dataset, task, variant, variant_index, fold, train, test, output_dir)
                for fold, (train, test) in enumerate(splits)
            ]
            summary = {
                "dataset": dataset.name,
                "task": task_name,
                "variant": variant,
            }
            for metric in METRICS:
                values = [float(item.get(metric, math.nan)) for item in fold_results]
                summary[f"mean_{metric}"] = float(np.nanmean(values))
                summary[f"std_{metric}"] = float(np.nanstd(values))
                summary[f"fold_{metric}"] = values
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
            results[key] = summary
    if cfg.get("write_root_manifest", True):
        (root / "cv_results.json").write_text(json.dumps(results, indent=2, allow_nan=True))
    return results
