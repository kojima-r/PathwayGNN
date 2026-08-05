"""`pathwaygnn pred`: scoring a prepared dataset with a trained checkpoint.

The interesting case is data that is *not* the training corpus, so one test builds
a second synthetic dataset over the same graph and scores it with a model trained
on the first — which is what the command exists for.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import NUM_SAMPLES, build_dataset
from pathwaygnn.data.format import GraphDataset
from pathwaygnn.training.finetune import run_finetuning
from pathwaygnn.training.metrics import METRICS
from pathwaygnn.training.predict import run_prediction

VARIANT = {"use_graph": True, "use_sample_features": True}


def _finetune(dataset: GraphDataset, pretrained: Path, output: Path) -> dict:
    """Train a head the way `finetune` does, and return its reported metrics."""
    return run_finetuning({
        "seed": 3,
        "device": "cpu",
        "dataset": {"name": dataset.name, "dir": str(dataset.root), "task": "main"},
        "split": [0.6, 0.2, 0.2],
        "pretrained_checkpoint": str(pretrained),
        "variant": VARIANT,
        "model": {"hidden_dim": 4, "dropout": 0.0, "block": "plain"},
        "training": {
            "epochs": 3, "batch_size": 8, "learning_rate": 0.01, "patience": 5,
            "train_encoder": False,
        },
        "output_dir": str(output),
    })


def _pred_config(
    dataset: GraphDataset, checkpoint: Path, pretrained: Path, output: Path, **extra
) -> dict:
    return {
        "device": "cpu",
        "dataset": {"name": dataset.name, "dir": str(dataset.root), "task": "main"},
        "checkpoint": str(checkpoint),
        "pretrained_checkpoint": str(pretrained),
        "batch_size": 8,
        "output_dir": str(output),
        **extra,
    }


def _read(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    return rows[0], rows[1:]


@pytest.fixture
def trained(tmp_path: Path, dataset: GraphDataset, pretrained: Path) -> tuple[Path, dict]:
    metrics = _finetune(dataset, pretrained, tmp_path / "finetune")
    return tmp_path / "finetune" / "best.pt", metrics


def test_prediction_table_covers_every_sample(
    tmp_path: Path, dataset: GraphDataset, pretrained: Path, trained: tuple[Path, dict]
):
    checkpoint, _ = trained
    output = tmp_path / "pred"
    summary = run_prediction(_pred_config(dataset, checkpoint, pretrained, output))
    header, rows = _read(output / "predictions.tsv")
    assert header[:5] == ["sample_index", "probability", "prediction", "label", "group"]
    # One row per sample of the task, in order, plus the source row of each
    # node-level feature so a prediction can be traced back to its inputs.
    assert header[5:] == ["row_expression", "row_signature"]
    assert len(rows) == NUM_SAMPLES == summary["num_samples"]
    assert [int(row[0]) for row in rows] == list(range(NUM_SAMPLES))
    probability = np.array([float(row[1]) for row in rows])
    prediction = np.array([int(row[2]) for row in rows])
    assert ((probability >= 0) & (probability <= 1)).all()
    assert (prediction == (probability >= 0.5)).all()
    assert summary["predicted_positive"] == int(prediction.sum())
    assert set(summary["metrics"]) == set(METRICS)
    assert json.loads((output / "summary.json").read_text())["task"] == "main"


def test_threshold_moves_the_decision_not_the_probability(
    tmp_path: Path, dataset: GraphDataset, pretrained: Path, trained: tuple[Path, dict]
):
    checkpoint, _ = trained
    low = run_prediction(
        _pred_config(dataset, checkpoint, pretrained, tmp_path / "low", threshold=0.05)
    )
    high = run_prediction(
        _pred_config(dataset, checkpoint, pretrained, tmp_path / "high", threshold=0.95)
    )
    assert low["predicted_positive"] >= high["predicted_positive"]
    assert low["probability"] == high["probability"]


def test_pred_reproduces_the_training_command_on_the_same_rows(
    tmp_path: Path, dataset: GraphDataset, pretrained: Path, trained: tuple[Path, dict]
):
    """The decisive check: same model, same rows, same numbers as `finetune`."""
    checkpoint, reported = trained
    output = tmp_path / "pred"
    run_prediction(_pred_config(dataset, checkpoint, pretrained, output))
    split = json.loads((tmp_path / "finetune" / "metrics.json").read_text())["split"]["test"]
    stored = np.load(output / "predictions.npz")
    position = {int(sample): index for index, sample in enumerate(stored["sample_index"])}
    rows = [position[int(sample)] for sample in split]
    probability = stored["probability"][rows]
    label = stored["label"][rows]
    logit = torch.from_numpy(np.log(probability / (1 - probability)).astype(np.float32))
    from pathwaygnn.training.metrics import binary_metrics

    mine = binary_metrics(torch.from_numpy(label), logit)
    for key in METRICS:
        assert mine[key] == pytest.approx(reported[key], abs=1e-6), key


def test_scores_a_different_dataset_over_the_same_graph(
    tmp_path: Path, dataset: GraphDataset, pretrained: Path, trained: tuple[Path, dict]
):
    """The point of the command: new samples, prepared separately, same graph."""
    checkpoint, _ = trained
    external = build_dataset(tmp_path / "external", name="external")
    output = tmp_path / "pred_external"
    summary = run_prediction(_pred_config(external, checkpoint, pretrained, output))
    assert summary["dataset"] == "external"
    # The summary records that the model came from elsewhere rather than refusing.
    assert summary["trained_on"]["dataset"] == dataset.name
    _, rows = _read(output / "predictions.tsv")
    assert len(rows) == NUM_SAMPLES
    assert summary["metrics"] is not None


def test_single_class_labels_report_no_metrics(
    tmp_path: Path, dataset: GraphDataset, pretrained: Path, trained: tuple[Path, dict]
):
    """Real prediction data has placeholder labels; that must not become a score."""
    checkpoint, _ = trained
    unlabelled = build_dataset(tmp_path / "unlabelled", name="unlabelled")
    labels = np.zeros(NUM_SAMPLES, dtype=np.float32)
    np.save(unlabelled.root / "tasks" / "main" / "labels.npy", labels)
    output = tmp_path / "pred_unlabelled"
    summary = run_prediction(_pred_config(unlabelled, checkpoint, pretrained, output))
    assert summary["metrics"] is None
    assert "placeholder" in summary["labels"]
    header, rows = _read(output / "predictions.tsv")
    assert "label" not in header  # not a column of zeros pretending to be truth
    assert len(rows) == NUM_SAMPLES


def test_group_table_summarizes_each_group(
    tmp_path: Path, dataset: GraphDataset, pretrained: Path, trained: tuple[Path, dict]
):
    checkpoint, _ = trained
    output = tmp_path / "pred"
    summary = run_prediction(_pred_config(dataset, checkpoint, pretrained, output))
    header, rows = _read(Path(summary["group_table"]))
    assert header[:4] == ["group", "samples", "predicted_positive", "predicted_positive_ratio"]
    assert sum(int(row[1]) for row in rows) == NUM_SAMPLES
    assert {row[0] for row in rows} <= set(dataset.task("main").group_names)


def test_a_task_the_head_does_not_fit_is_refused(
    tmp_path: Path, dataset: GraphDataset, pretrained: Path, trained: tuple[Path, dict]
):
    """Alias order fixes the head's concat order, so a mismatch cannot be tolerated."""
    checkpoint, _ = trained
    other = build_dataset(tmp_path / "renamed", name="renamed")
    manifest_path = other.root / "tasks" / "main" / "task.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["node_features"] = {"expression": "expression"}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="in that order"):
        run_prediction(_pred_config(other, checkpoint, pretrained, tmp_path / "bad"))


def test_missing_checkpoint_says_where_to_get_one(
    tmp_path: Path, dataset: GraphDataset, pretrained: Path
):
    with pytest.raises(FileNotFoundError, match="model.pt"):
        run_prediction(
            _pred_config(dataset, tmp_path / "nope.pt", pretrained, tmp_path / "out")
        )


def test_graph_variant_needs_the_encoder(
    tmp_path: Path, dataset: GraphDataset, pretrained: Path, trained: tuple[Path, dict]
):
    checkpoint, _ = trained
    config = _pred_config(dataset, checkpoint, pretrained, tmp_path / "out")
    del config["pretrained_checkpoint"]
    with pytest.raises(KeyError, match="pretrained_checkpoint"):
        run_prediction(config)
