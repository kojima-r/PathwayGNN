"""Apply a trained model to a prepared dataset and write the prediction table.

``cv``, ``finetune`` and ``ig`` all work inside the corpus a model was trained on.
This command is the other direction: it takes a checkpoint and scores whatever
dataset the config points at — new samples that were prepared into
:mod:`pathwaygnn.data.format` after training finished, on the same graph.

Both ``cv``'s fold ``model.pt`` and ``finetune``'s ``best.pt`` carry everything an
inference run needs (the head's ``model_config``, the fine-tuned encoder state,
the variant flags and the training set's sample-feature statistics), so either can
be given as ``checkpoint:``.

Two things are load-bearing when the data is *not* the training corpus:

* **The head is checked against the task before anything runs.** A prepared task
  binds each node-level feature to a local alias, and the head's concat order is
  fixed by that alias order — so an alias list that disagrees produces a plausible
  number from the wrong inputs. That is refused, not warned about.
* **Sample features are standardized with the training set's mean and standard
  deviation**, taken from the checkpoint. Re-deriving them from the data being
  scored would silently change the model's input distribution, and would make a
  single sample unscoreable.

Labels are part of the prepared format, but data being predicted on need not have
real ones. Metrics are therefore reported only when the labels carry both classes,
and their absence is stated rather than reported as zero.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pathwaygnn.data.format import Task, open_task
from pathwaygnn.data.samples import TaskDataset
from pathwaygnn.hub import resolve_checkpoint
from pathwaygnn.models.encoder import load_encoder
from pathwaygnn.models.predictor import SampleLevelModel
from pathwaygnn.training.cv import normalize_sample_features
from pathwaygnn.training.metrics import METRICS, binary_metrics, threshold_metrics


def _check_compatible(model_config: dict[str, Any], task: Task) -> None:
    """Refuse a head whose inputs the task cannot supply in the expected order."""
    expected = list(model_config["node_features"])
    if expected != list(task.node_feature_names):
        raise ValueError(
            f"the checkpoint's head takes node-level features {expected} in that order, but task "
            f"{task.name!r} exposes {list(task.node_feature_names)}. The head's concat order is "
            "fixed by the alias order, so these have to match exactly; rename the aliases in the "
            "task manifest (`node_features` in task.json) to match the model."
        )
    if model_config.get("use_sample_features") and (
        int(model_config["sample_feature_dim"]) != task.sample_feature_dim
    ):
        raise ValueError(
            f"the checkpoint's head takes {model_config['sample_feature_dim']} sample-level "
            f"features but task {task.name!r} has {task.sample_feature_dim}"
        )


def _write_table(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def _preview(header: list[str], rows: list[list[Any]], limit: int) -> str:
    """The head of the table, aligned, so a run is readable in the terminal."""
    shown = rows[:limit]
    widths = [
        max(len(str(header[column])), *(len(str(row[column])) for row in shown or [header]))
        for column in range(len(header))
    ]
    lines = ["  ".join(str(item).ljust(width) for item, width in zip(header, widths)).rstrip()]
    lines.append("  ".join("-" * width for width in widths))
    for row in shown:
        lines.append("  ".join(str(item).ljust(width) for item, width in zip(row, widths)).rstrip())
    if len(rows) > len(shown):
        lines.append(f"... {len(rows) - len(shown)} more rows")
    return "\n".join(lines)


def run_prediction(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset, task = open_task(cfg)
    # A local path or an `hf://owner/name/model.pt` reference.
    checkpoint_path = resolve_checkpoint(cfg["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{checkpoint_path} is missing; point `checkpoint:` at a `finetune` best.pt or a "
            "`cv` fold's model.pt"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_config" not in checkpoint:
        raise KeyError(
            f"{checkpoint_path} carries no `model_config`; re-run the command that wrote it"
        )
    model_config = dict(checkpoint["model_config"])
    _check_compatible(model_config, task)

    device = torch.device(
        "cuda" if cfg.get("device", "auto") != "cpu" and torch.cuda.is_available() else "cpu"
    )
    use_graph = bool(model_config["use_graph"])
    encoder = None
    embeddings: Tensor | None = None
    if use_graph:
        if "pretrained_checkpoint" not in cfg:
            raise KeyError(
                "this checkpoint uses the graph, so `pretrained_checkpoint:` is needed to rebuild "
                "the encoder (the same one the model was trained against)"
            )
        # load_encoder refuses a graph whose node or relation count differs, which
        # is the guard against scoring data prepared over a different graph.
        encoder, _ = load_encoder(
            cfg["pretrained_checkpoint"], dataset.num_nodes, dataset.num_relations, device
        )
        if checkpoint["encoder"] is not None:
            encoder.load_state_dict(checkpoint["encoder"])
        encoder.eval()
        edge_index, edge_type = dataset.graph()
        with torch.no_grad():
            embeddings = encoder(edge_index.to(device), edge_type.to(device))
    predictor = SampleLevelModel.from_config(model_config).to(device)
    predictor.load_state_dict(checkpoint["predictor"])
    predictor.eval()
    mean = checkpoint.get("sample_feature_mean")
    std = checkpoint.get("sample_feature_std")
    mean = None if mean is None else mean.to(device)
    std = None if std is None else std.to(device)

    data = TaskDataset(task)
    loader = DataLoader(
        data,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=data.collate(),
    )
    logits, indices = [], []
    with torch.no_grad():
        for batch in loader:
            batch = normalize_sample_features(batch.to(device), mean, std)
            logits.append(predictor(batch, embeddings).cpu())
            indices.append(batch.index)
    logit = torch.cat(logits).reshape(-1)
    sample_index = torch.cat(indices).reshape(-1).numpy()
    probability = logit.sigmoid().numpy()
    threshold = float(cfg.get("threshold", 0.5))
    prediction = (probability >= threshold).astype(np.int64)

    labels = task.labels()[sample_index]
    scored = bool(np.unique(labels).size == 2)
    groups = task.groups()
    group_code = None if groups is None else np.asarray(groups)[sample_index]
    rows_by_alias = {
        alias: task.rows(alias) for alias in task.node_feature_names
    }

    header = ["sample_index", "probability", "prediction"]
    if scored:
        header.append("label")
    if group_code is not None:
        header.append("group")
    header += [f"row_{alias}" for alias in task.node_feature_names]
    table: list[list[Any]] = []
    for position, sample in enumerate(sample_index):
        row: list[Any] = [
            int(sample), f"{float(probability[position]):.6f}", int(prediction[position])
        ]
        if scored:
            row.append(int(labels[position]))
        if group_code is not None:
            code = int(group_code[position])
            row.append(
                task.group_names[code] if code < len(task.group_names) else code
            )
        for alias in task.node_feature_names:
            mapping = rows_by_alias[alias]
            row.append(int(sample) if mapping is None else int(mapping[int(sample)]))
        table.append(row)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_table(output_dir / "predictions.tsv", header, table)
    np.savez_compressed(
        output_dir / "predictions.npz",
        sample_index=sample_index,
        probability=probability,
        prediction=prediction,
        label=labels.astype(np.float32),
    )

    summary: dict[str, Any] = {
        "dataset": dataset.name,
        "task": task.name,
        "checkpoint": str(checkpoint_path),
        # The model may well have been trained on another corpus; that is the point
        # of this command, so it is recorded rather than refused.
        "trained_on": {
            "dataset": checkpoint.get("dataset"),
            "task": checkpoint.get("task"),
            "epoch": checkpoint.get("epoch"),
        },
        "variant": checkpoint.get("variant", {}),
        "num_samples": int(sample_index.size),
        "threshold": threshold,
        "predicted_positive": int(prediction.sum()),
        "predicted_positive_ratio": float(prediction.mean()),
        "probability": {
            "min": float(probability.min()),
            "mean": float(probability.mean()),
            "max": float(probability.max()),
        },
        "table": str(output_dir / "predictions.tsv"),
        "arrays": str(output_dir / "predictions.npz"),
    }
    if scored:
        metrics = binary_metrics(torch.from_numpy(labels.astype(np.float32)), logit)
        summary["metrics"] = {key: metrics[key] for key in METRICS}
    else:
        summary["metrics"] = None
        summary["labels"] = (
            "only one class present, so the labels are placeholders and no metric is reported"
        )
    if group_code is not None:
        group_header = ["group", "samples", "predicted_positive", "predicted_positive_ratio"]
        if scored:
            group_header += ["label_positive", "accuracy"]
        group_rows = []
        for code, name in enumerate(task.group_names):
            mask = group_code == code
            if not bool(mask.any()):
                continue
            line: list[Any] = [
                name, int(mask.sum()), int(prediction[mask].sum()),
                f"{float(prediction[mask].mean()):.4f}",
            ]
            if scored:
                scores = threshold_metrics(labels[mask], probability[mask], threshold)
                line += [int(labels[mask].sum()), f"{scores['accuracy']:.4f}"]
            group_rows.append(line)
        _write_table(output_dir / "predictions_by_group.tsv", group_header, group_rows)
        summary["group_table"] = str(output_dir / "predictions_by_group.tsv")

    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(_preview(header, table, int(cfg.get("preview_rows", 10))))
    return summary
