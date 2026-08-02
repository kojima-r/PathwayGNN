"""Cross-validation and Integrated Gradients are dataset-agnostic: run both on a
synthetic dataset that has one dense channel, one sparse channel, covariates and
groups, mixing graph-free and graph variants in a single grid."""

import json
from pathlib import Path

import numpy as np
import torch

from pathwaygnn.data.format import GraphDataset
from pathwaygnn.training.cv import run_cv
from pathwaygnn.training.ig import run_ig
from pathwaygnn.training.metrics import METRICS, threshold_metrics

VARIANTS = [
    {"name": "mlp", "use_graph": False, "use_covariates": False, "seed_index": 0},
    {"name": "gnn_cov", "use_graph": True, "use_covariates": True, "seed_index": 3},
]


def _cv_config(dataset: GraphDataset, pretrained: Path, output: Path) -> dict:
    return {
        "seed": 5,
        "device": "cpu",
        "folds": 2,
        "dataset": {"name": dataset.name, "dir": str(dataset.root), "tasks": ["main"]},
        "pretrained_checkpoint": str(pretrained),
        "model": {"embedding_dim": 4, "hidden_dim": 4, "batch_norm": False, "dropout": 0.0},
        "training": {
            "epochs": 2, "batch_size": 8, "learning_rate": 0.01, "num_workers": 0,
            "shuffle": False, "end_to_end": True, "selection": "final_epoch", "resume": True,
        },
        "variants": VARIANTS,
        "output_dir": str(output),
    }


def test_cv_grid_and_ig(tmp_path: Path, dataset: GraphDataset, pretrained: Path) -> None:
    output = tmp_path / "cv"
    results = run_cv(_cv_config(dataset, pretrained, output))
    assert sorted(results) == ["main/gnn_cov", "main/mlp"]
    for key, summary in results.items():
        assert summary["task"] == "main" and summary["dataset"] == dataset.name
        # ROC-AUC and the 0.5-threshold metrics are summarised the same way.
        for metric in METRICS:
            assert len(summary[f"fold_{metric}"]) == 2
            assert np.isfinite(summary[f"mean_{metric}"])
            assert np.isfinite(summary[f"std_{metric}"])
    assert json.loads((output / "cv_results.json").read_text()).keys() == results.keys()

    fold = output / "main" / "gnn_cov" / "fold_0"
    metrics = json.loads((fold / "metrics.json").read_text())
    # seed = seed + seed_offset * 1000 + fold + seed_index * 100
    assert metrics["seed"] == 5 + 7 * 1000 + 0 + 3 * 100
    assert len(metrics["history"]) == 2
    assert set(metrics["per_group_auc"]) == {"g0", "g1", "g2"}
    predictions = np.load(fold / "predictions.npz")
    assert predictions["target"].size == predictions["probability"].size == 12

    # The fold, its selected epoch and the condition summary agree, and every
    # threshold metric is exactly what the stored predictions imply.
    expected = threshold_metrics(predictions["target"], predictions["probability"])
    for metric in METRICS[1:]:
        assert metrics[metric] == expected[metric]
        assert metrics["history"][metrics["selected_epoch"] - 1][f"test_{metric}"] == metrics[metric]
        assert results["main/gnn_cov"][f"fold_{metric}"][0] == metrics[metric]
    assert metrics["history"][metrics["selected_epoch"] - 1]["test_auc"] == metrics["auc"]
    checkpoint = torch.load(fold / "model.pt", map_location="cpu", weights_only=False)
    assert checkpoint["model_config"]["channels"] == ["expression", "signature"]
    assert checkpoint["covariate_mean"].shape == (3,)
    # The graph-free variant stores no encoder and no graph flag.
    free = torch.load(output / "main" / "mlp" / "fold_0" / "model.pt",
                      map_location="cpu", weights_only=False)
    assert free["encoder"] is None and free["model_config"]["use_graph"] is False

    ig_output = tmp_path / "ig"
    summary = run_ig({
        "seed": 5,
        "device": "cpu",
        "dataset": {"name": dataset.name, "dir": str(dataset.root), "task": "main"},
        "run_dir": str(output),
        "variant": "gnn_cov",
        "fold": 0,
        "pretrained_checkpoint": str(pretrained),
        "output_dir": str(ig_output),
        "steps": 3,
        "max_samples": 4,
        "top_k": 5,
        "per_group_rankings": True,
        "reference": {"degree_ig_pearson_r": 0.727},
    })
    assert summary["num_samples"] == 4
    assert np.isfinite(summary["degree_ig_pearson_r"])
    assert set(summary["covariate_ig"]) == {"c0", "c1", "c2"}
    assert summary["reference"] == {"degree_ig_pearson_r": 0.727}
    arrays = np.load(ig_output / "attributions.npz")
    assert arrays["graph_score"].shape == (dataset.num_nodes,)
    assert arrays["degree"].shape == (dataset.num_nodes,)
    assert arrays["channel_expression"].shape == (dataset.num_nodes,)
    graph_ranking = (ig_output / "top_graph_nodes.tsv").read_text().splitlines()
    assert graph_ranking[0].split("\t") == ["rank", "node_index", "node", "ig_l2"]
    assert len(graph_ranking) == 6
    # Channel rankings only cover nodes the channel actually carries.
    expression = (ig_output / "top_channel_expression.tsv").read_text().splitlines()[1:]
    assert {int(line.split("\t")[1]) for line in expression} <= set(range(6))
    signature = (ig_output / "top_channel_signature.tsv").read_text().splitlines()[1:]
    assert signature and all(len(line.split("\t")) == 5 for line in signature)
    # One ranking per group that the sampled subset actually covers.
    assert sum(summary["per_group_samples"].values()) == 4
    assert sum(1 for _ in ig_output.glob("top_graph_nodes_g*.tsv")) == len(
        summary["per_group_samples"]
    )


def test_ig_without_graph_skips_node_attribution(
    tmp_path: Path, dataset: GraphDataset, pretrained: Path
) -> None:
    output = tmp_path / "cv"
    run_cv(_cv_config(dataset, pretrained, output))
    result = run_ig({
        "device": "cpu",
        "dataset": {"name": dataset.name, "dir": str(dataset.root), "task": "main"},
        "run_dir": str(output),
        "variant": "mlp",
        "fold": 1,
        "output_dir": str(tmp_path / "ig_free"),
        "steps": 2,
        "max_samples": 2,
        "top_k": 3,
    })
    assert result["degree_ig_pearson_r"] is None
    assert not (tmp_path / "ig_free" / "top_graph_nodes.tsv").exists()
    arrays = np.load(tmp_path / "ig_free" / "attributions.npz")
    assert "graph_score" not in arrays and "channel_signature" in arrays


def test_cv_resumes_completed_folds(tmp_path: Path, dataset: GraphDataset, pretrained: Path) -> None:
    output = tmp_path / "cv"
    config = _cv_config(dataset, pretrained, output)
    first = run_cv(config)
    stamps = {
        path: path.stat().st_mtime_ns for path in output.glob("main/*/fold_*/metrics.json")
    }
    assert len(stamps) == 4
    second = run_cv(config)
    assert second["main/mlp"]["fold_auc"] == first["main/mlp"]["fold_auc"]
    assert {path: path.stat().st_mtime_ns for path in stamps} == stamps
    config["training"]["resume"] = False
    run_cv(config)
    assert any(path.stat().st_mtime_ns != stamps[path] for path in stamps)


def test_cv_backfills_threshold_metrics_of_older_runs(
    tmp_path: Path, dataset: GraphDataset, pretrained: Path
) -> None:
    """A fold written before the threshold metrics existed is upgraded in place.

    They are a function of the stored held-out predictions, so resuming must
    recover them rather than silently reporting ROC-AUC alone.
    """
    output = tmp_path / "cv"
    config = _cv_config(dataset, pretrained, output)
    first = run_cv(config)
    fold_path = output / "main" / "mlp" / "fold_0" / "metrics.json"
    original = json.loads(fold_path.read_text())

    # Rewrite the fold the way the pre-threshold-metric version of `cv` did.
    fold_path.write_text(json.dumps(
        {key: value for key, value in original.items() if key not in METRICS[1:]}, indent=2
    ))
    second = run_cv(config)

    restored = json.loads(fold_path.read_text())
    for metric in METRICS[1:]:
        assert restored[metric] == original[metric]
        assert second["main/mlp"][f"mean_{metric}"] == first["main/mlp"][f"mean_{metric}"]
    # Backfilling must not disturb anything else about the cached fold.
    assert restored["auc"] == original["auc"]
    assert restored["history"] == original["history"]
    assert restored["seed"] == original["seed"]
