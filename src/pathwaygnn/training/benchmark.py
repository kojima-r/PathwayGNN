"""Graph-free baselines on the same features, as a reference for the GNN."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pathwaygnn.data.format import NodeFeature, GraphDataset, Task, open_task
from pathwaygnn.training.metrics import binary_metrics


def _dependencies():
    try:
        import scipy.sparse as sparse
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
    except ImportError as error:
        raise RuntimeError("Install with `pip install -e '.[benchmark]'`.") from error
    return sparse, RandomForestClassifier, LogisticRegression, StratifiedKFold


def _node_feature_matrix(node_feature: NodeFeature, rows: np.ndarray | None, num_nodes: int, sparse):
    """One node-level feature as a ``[samples, num_nodes]`` sparse matrix."""
    if node_feature.dense:
        matrix = node_feature.matrix()
        gene_index = node_feature.gene_index()
        values = np.asarray(matrix if rows is None else matrix[rows], dtype=np.float32)
        table = sparse.csr_matrix(
            (
                values.reshape(-1),
                (
                    np.repeat(np.arange(values.shape[0]), values.shape[1]),
                    np.tile(gene_index, values.shape[0]),
                ),
            ),
            shape=(values.shape[0], num_nodes),
            dtype=np.float32,
        )
        return table
    ptr, gene, value = (np.asarray(item) for item in node_feature.csr())
    table = sparse.csr_matrix(
        (value, gene, ptr), shape=(ptr.size - 1, num_nodes), dtype=np.float32
    )
    return table if rows is None else table[rows]


def _features(dataset: GraphDataset, task: Task, sparse):
    blocks = [
        _node_feature_matrix(node_feature, task.rows(node_feature.name), dataset.num_nodes, sparse)
        for node_feature in task.node_features
    ]
    sample_features = task.sample_features()
    if sample_features is not None:
        blocks.append(sparse.csr_matrix(np.asarray(sample_features, dtype=np.float32)))
    x = sparse.hstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]
    return x, task.labels().astype(np.int64)


def run_benchmark(cfg: dict[str, Any]) -> dict[str, object]:
    sparse, RandomForest, LogisticRegression, StratifiedKFold = _dependencies()
    dataset, task = open_task(cfg)
    x, y = _features(dataset, task, sparse)
    seed = int(cfg.get("seed", 42))
    models = {
        "logistic_regression": LogisticRegression(
            C=float(cfg.get("logistic_regression", {}).get("C", 1.0)), max_iter=1000,
            class_weight="balanced", random_state=seed,
        ),
        "random_forest": RandomForest(
            n_estimators=int(cfg.get("random_forest", {}).get("n_estimators", 200)),
            max_depth=cfg.get("random_forest", {}).get("max_depth"),
            class_weight="balanced", n_jobs=int(cfg.get("n_jobs", -1)), random_state=seed,
        ),
    }
    if cfg.get("xgboost", {}).get("enabled", True):
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError("xgboost is enabled but not installed.") from error
        models["xgboost"] = XGBClassifier(
            n_estimators=int(cfg["xgboost"].get("n_estimators", 200)),
            max_depth=int(cfg["xgboost"].get("max_depth", 4)),
            learning_rate=float(cfg["xgboost"].get("learning_rate", 0.05)),
            subsample=float(cfg["xgboost"].get("subsample", 0.8)),
            n_jobs=int(cfg.get("n_jobs", -1)), random_state=seed,
        )
    splitter = StratifiedKFold(n_splits=int(cfg.get("folds", 5)), shuffle=True, random_state=seed)
    result: dict[str, object] = {"dataset": dataset.name, "task": task.name}
    for name, model in models.items():
        fold_metrics = []
        for train, test in splitter.split(x, y):
            model.fit(x[train], y[train])
            probability = model.predict_proba(x[test])[:, 1]
            logits = torch.from_numpy(probability).float().clamp(1e-7, 1 - 1e-7).logit()
            fold_metrics.append(binary_metrics(torch.from_numpy(y[test]), logits))
        mean = {key: float(np.nanmean([fold[key] for fold in fold_metrics])) for key in fold_metrics[0]}
        result[name] = {"mean": mean, "folds": fold_metrics}
        print(json.dumps({"model": name, **mean}))
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.json").write_text(json.dumps(result, indent=2))
    return result
