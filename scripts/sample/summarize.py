"""Print the tutorial's results as the tables README.md quotes.

Reads only what the engine wrote under ``outputs/sample/``; anything missing is
reported and skipped, so this works after a partial run too. Standard library
plus numpy, no report machinery — the real corpora have proper report commands
(``pathwaygnn-data tr-report`` and friends) that also render Markdown and HTML.

    python scripts/sample/summarize.py [--run-dir outputs/sample]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TASKS = ("responder", "relapse")
VARIANTS = ("mlp", "mlp_cov", "gnn_mlp", "gnn_mlp_cov")


def _load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def cv_table(run_dir: Path) -> None:
    print("== cross-validation (mean +- std over folds) ==")
    print(f"{'task':10s} {'variant':12s} {'ROC-AUC':>15s} {'accuracy':>9s} {'F1':>7s}   folds")
    for task in TASKS:
        for variant in VARIANTS:
            summary = _load(run_dir / "cv" / task / variant / "summary.json")
            if summary is None:
                continue
            folds = " ".join(f"{value:.2f}" for value in summary["fold_auc"])
            print(
                f"{task:10s} {variant:12s} "
                f"{summary['mean_auc']:.3f} +- {summary['std_auc']:.3f} "
                f"{summary['mean_accuracy']:9.3f} {summary['mean_f1']:7.3f}   {folds}"
            )
    print()


def per_group_table(run_dir: Path, task: str = "responder", variant: str = "gnn_mlp_cov") -> None:
    metrics = [
        _load(run_dir / "cv" / task / variant / f"fold_{fold}" / "metrics.json") for fold in range(3)
    ]
    metrics = [item for item in metrics if item]
    if not metrics:
        return
    print(f"== per-tissue ROC-AUC ({task} / {variant}) ==")
    tissues = list(metrics[0]["per_group_auc"])
    print(f"{'fold':6s} " + " ".join(f"{name:>10s}" for name in tissues))
    for fold, item in enumerate(metrics):
        print(f"{fold:<6d} " + " ".join(f"{item['per_group_auc'][name]:10.3f}" for name in tissues))
    print()


def baseline_table(run_dir: Path, task: str = "responder") -> None:
    benchmark = _load(run_dir / "benchmark" / task / "benchmark.json")
    finetune = _load(run_dir / "finetune" / task / "metrics.json")
    if benchmark is None and finetune is None:
        return
    print(f"== graph-free baselines and the single split ({task}) ==")
    for name, entry in (benchmark or {}).items():
        if isinstance(entry, dict) and "mean" in entry:
            mean = entry["mean"]
            print(f"{name:22s} AUC {mean['auc']:.3f}  accuracy {mean['accuracy']:.3f}  "
                  f"F1 {mean['f1']:.3f}   (3-fold, graph-free)")
    if finetune:
        test = finetune["test"]
        print(f"{'finetune (test split)':22s} AUC {test['auc']:.3f}  accuracy {test['accuracy']:.3f}  "
              f"F1 {test['f1']:.3f}   ({len(finetune['history'])} epochs)")
    print()


def ig_table(run_dir: Path, name: str = "responder_fold0", top: int = 10) -> None:
    directory = run_dir / "ig" / name
    summary = _load(directory / "ig_summary.json")
    if summary is None:
        return
    print(f"== Integrated Gradients ({name}: {summary['num_samples']} samples x "
          f"{summary['integration_steps']} steps) ==")
    path = directory / "top_node_feature_expression.tsv"
    if path.is_file():
        rows = list(csv.DictReader(path.open(), delimiter="\t"))
        print(f"top {top} genes by |IG| on the `expression` values "
              f"(sign = direction of the effect):")
        for row in rows[:top]:
            print(f"  {row['rank']:>3s}. {row['node']:10s} {float(row['signed_ig']):+.4f}")
    features = summary.get("sample_feature_ig", {})
    if features:
        print("sample-level feature attribution: " +
              ", ".join(f"{key}={value:+.4f}" for key, value in features.items()))
    print(f"degree/IG Pearson r on the graph embedding: {summary['degree_ig_pearson_r']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/sample")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"{run_dir} does not exist yet; run `bash scripts/sample/run_all.sh` first")
    history = _load(run_dir / "pretrain" / "history.json")
    if history:
        last = history[-1]
        # history.json is rewritten whenever the loss improves, so its length is
        # the epoch of the last improvement, not the number of epochs run.
        print(f"== pre-training ==\nbest epoch {last['epoch']}, loss {last['loss']:.4f}, "
              f"edge-ranking accuracy {last['accuracy']:.3f}\n")
    cv_table(run_dir)
    per_group_table(run_dir)
    baseline_table(run_dir)
    ig_table(run_dir)


if __name__ == "__main__":
    main()
