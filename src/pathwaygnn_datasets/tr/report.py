"""Reporting for the target-repositioning dataset.

Collects everything the engine produced for ``data_tr`` — graph pre-training,
cross-validation, holdout fine-tuning, graph-free baselines and Integrated
Gradients — into one set of tables, figures and a single document. Like the
cancer report this is dataset-specific presentation, so it lives next to the
target-repositioning preprocessing.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from pathwaygnn.data.format import GraphDataset, Task
from pathwaygnn_datasets.document import figures, mdtable, tsv, write_document

TITLE = "PathwayGNN target repositioning report"
VARIANT_COLORS = {"mlp": "#4c78a8", "gnn_mlp": "#54a24b"}
BASELINE_COLORS = {
    "logistic_regression": "#f58518",
    "random_forest": "#e45756",
    "xgboost": "#b279a2",
}
METRICS = ("auc", "accuracy", "precision", "recall", "f1")


def _auc(target: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(target, probability)) if np.unique(target).size == 2 else math.nan


def _color(name: str, position: int) -> str:
    palette = ("#4c78a8", "#54a24b", "#f58518", "#e45756", "#b279a2", "#9d755d")
    return VARIANT_COLORS.get(name, palette[position % len(palette)])


def _read(path: Path) -> Any | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _cv_conditions(cv_dir: Path, task: str) -> dict[str, dict[str, Any]]:
    """Variant -> summary, ordered graph-free first so tables read as an ablation."""
    found = {}
    for path in sorted((cv_dir / task).glob("*/summary.json")):
        found[path.parts[-2]] = json.loads(path.read_text())
    return dict(
        sorted(found.items(), key=lambda item: (bool(item[1]["variant"].get("use_graph")), item[0]))
    )


def _variant_order(conditions: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    """Every variant seen across tasks, graph-free first so tables read as an ablation."""
    uses_graph: dict[str, bool] = {}
    for task_conditions in conditions.values():
        for name, summary in task_conditions.items():
            uses_graph.setdefault(name, bool(summary["variant"].get("use_graph")))
    return sorted(uses_graph, key=lambda name: (uses_graph[name], name))


def _pooled_predictions(cv_dir: Path, task: str, variant: str):
    targets, probabilities, indices = [], [], []
    for path in sorted((cv_dir / task / variant).glob("fold_*/predictions.npz")):
        arrays = np.load(path)
        targets.append(arrays["target"])
        probabilities.append(arrays["probability"])
        indices.append(arrays["sample_index"])
    if not targets:
        return None
    return tuple(np.concatenate(item) for item in (targets, probabilities, indices))


def _fold_histories(cv_dir: Path, task: str, variant: str) -> list[list[dict[str, Any]]]:
    return [
        json.loads(path.read_text())["history"]
        for path in sorted((cv_dir / task / variant).glob("fold_*/metrics.json"))
    ]


def _fold_pos_weights(cv_dir: Path) -> list[float]:
    """The positive-class weights the folds actually trained with, if any."""
    values = [
        json.loads(path.read_text()).get("pos_weight")
        for path in sorted(cv_dir.glob("*/*/fold_*/metrics.json"))
    ]
    return [float(value) for value in values if value is not None]


def _node_feature_lengths(dataset: GraphDataset) -> dict[str, np.ndarray]:
    """Non-zero genes per row of every sparse node-level feature, keyed by its table."""
    lengths = {}
    for source, entry in dataset.manifest["node_features"].items():
        if entry["kind"] == "sparse":
            lengths[source] = np.diff(np.asarray(dataset.node_feature(source).csr()[0]))
    return lengths


def _plots(
    output: Path,
    assets: Path,
    tasks: Sequence[str],
    variants: Sequence[str],
    audit: dict[str, dict[str, Any]],
    conditions: dict[str, dict[str, dict[str, Any]]],
    pooled: dict[tuple[str, str], Any],
    per_disease: dict[str, dict[str, dict[str, Any]]],
    histories: dict[tuple[str, str], list[list[dict[str, Any]]]],
    finetune: dict[str, Any],
    benchmark: dict[str, Any],
    attributions: dict[str, dict[str, Any]],
    pretrain_history: list[dict[str, Any]] | None,
    lengths: dict[str, np.ndarray],
    top_k: int,
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made: list[str] = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, dpi=210, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    # Dataset composition and signature lengths.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = np.arange(len(tasks))
    positive = [audit[task]["num_positive"] for task in tasks]
    negative = [audit[task]["num_samples"] - audit[task]["num_positive"] for task in tasks]
    axes[0].bar(positions, negative, label="label 0", color="#4c78a8")
    axes[0].bar(positions, positive, bottom=negative, label="label 1", color="#e45756")
    for position, task in zip(positions, tasks):
        axes[0].annotate(f"{audit[task]['positive_ratio']:.1%} positive",
                         (position, audit[task]["num_samples"]), ha="center",
                         va="bottom", fontsize=8)
    axes[0].set(xticks=positions, xticklabels=tasks, ylabel="Samples", title="Label composition")
    axes[0].legend()
    for name, values in lengths.items():
        axes[1].hist(values, bins=40, histtype="step", lw=1.6,
                     label=f"{name} ({values.size} rows)")
    axes[1].set(xlabel="Non-zero genes per signature row", ylabel="Rows (log)", yscale="log",
                title="Signature length distribution per node_feature")
    axes[1].legend(fontsize=8)
    save(fig, "dataset_composition.png")

    if pretrain_history:
        epochs = [item["epoch"] for item in pretrain_history]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, [item["loss"] for item in pretrain_history], color="#4c78a8")
        twin = ax.twinx()
        twin.plot(epochs, [item["accuracy"] for item in pretrain_history], color="#e45756")
        ax.set(xlabel="Pre-training epoch", ylabel="DistMult loss",
               title="Graph pre-training diagnostics")
        twin.set_ylabel("Pairwise accuracy")
        save(fig, "pretraining_diagnostics.png")

    # Cross-validation: mean bars with the individual folds on top.
    width = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for position, variant in enumerate(variants):
        offsets = positions + position * width - 0.4 + width / 2
        means = [conditions.get(task, {}).get(variant, {}).get("mean_auc", math.nan) for task in tasks]
        errors = [conditions.get(task, {}).get(variant, {}).get("std_auc", math.nan) for task in tasks]
        ax.bar(offsets, means, width=width, yerr=errors, capsize=3, label=variant,
               color=_color(variant, position), alpha=.85)
        for offset, task in zip(offsets, tasks):
            folds = conditions.get(task, {}).get(variant, {}).get("fold_auc", [])
            ax.scatter([offset] * len(folds), folds, s=12, color="black", zorder=3)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set(xticks=positions, xticklabels=tasks, ylabel="Held-out ROC-AUC", ylim=(0, 1),
           title="Cross-validated ROC-AUC (bars: mean +/- std, dots: folds)")
    ax.legend(); ax.grid(axis="y", alpha=.25)
    save(fig, "cv_auc_by_task_variant.png")

    values, labels = [], []
    for task in tasks:
        for variant in conditions.get(task, {}):
            values.append(conditions[task][variant]["fold_auc"])
            labels.append(f"{task}\n{variant}")
    if values:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(values, tick_labels=labels, showmeans=True)
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set(ylabel="Fold ROC-AUC", title="Fold ROC-AUC distributions")
        ax.grid(axis="y", alpha=.25)
        save(fig, "cv_fold_auc_boxplot.png")

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5.5), squeeze=False)
    for column, task in enumerate(tasks):
        ax = axes[0][column]
        for position, variant in enumerate(conditions.get(task, {})):
            entry = pooled.get((task, variant))
            if entry is None:
                continue
            target, probability, _ = entry
            false_positive, true_positive, _ = roc_curve(target, probability)
            ax.plot(false_positive, true_positive, color=_color(variant, position),
                    label=f"{variant} (AUC {_auc(target, probability):.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{task}: pooled folds")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig, "cv_roc_curves.png")

    panels = [(task, variant) for task in tasks for variant in conditions.get(task, {})]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4), squeeze=False,
                                 sharey=True)
        for column, (task, variant) in enumerate(panels):
            ax = axes[0][column]
            for history in histories.get((task, variant), []):
                ax.plot([item["epoch"] for item in history],
                        [item["test_auc"] for item in history], lw=.9, alpha=.7)
            ax.axhline(0.5, color="gray", ls="--", lw=1)
            ax.set(title=f"{task} / {variant}", xlabel="Epoch", ylim=(0, 1))
            ax.grid(alpha=.2)
        axes[0][0].set_ylabel("Held-out fold ROC-AUC")
        fig.suptitle("Cross-validation training curves (one line per fold)")
        save(fig, "cv_training_curves.png")

    # Per-disease behaviour: graph-free against graph, and against sample count.
    pairs = [(task, list(conditions.get(task, {}))) for task in tasks]
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5.5), squeeze=False)
    for column, (task, names) in enumerate(pairs):
        ax = axes[0][column]
        if len(names) >= 2:
            first, second = names[0], names[-1]
            rows = per_disease.get(task, {})
            xs, ys, sizes, labels_ = [], [], [], []
            for disease, row in rows.items():
                x, y = row["auc"].get(first, math.nan), row["auc"].get(second, math.nan)
                if math.isfinite(x) and math.isfinite(y):
                    xs.append(x); ys.append(y); sizes.append(max(12, row["samples"]))
                    labels_.append(disease)
            ax.scatter(xs, ys, s=sizes, alpha=.7, color="#4c78a8")
            ax.plot([0, 1], [0, 1], "--", color="gray")
            for x, y, name in zip(xs, ys, labels_):
                ax.annotate(name.replace("DOID:", ""), (x, y), fontsize=6)
            ax.set(xlabel=f"{first} ROC-AUC", ylabel=f"{second} ROC-AUC", xlim=(0, 1), ylim=(0, 1),
                   title=f"{task}: per-disease ROC-AUC")
        ax.grid(alpha=.2)
    save(fig, "per_disease_auc_scatter.png")

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4.5), squeeze=False, sharey=True)
    for column, task in enumerate(tasks):
        ax = axes[0][column]
        for position, variant in enumerate(conditions.get(task, {})):
            rows = per_disease.get(task, {})
            xs = [row["samples"] for row in rows.values() if math.isfinite(row["auc"].get(variant, math.nan))]
            ys = [row["auc"][variant] for row in rows.values() if math.isfinite(row["auc"].get(variant, math.nan))]
            ax.scatter(xs, ys, s=18, alpha=.7, label=variant, color=_color(variant, position))
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set(xlabel="Held-out samples for the disease", title=task, xscale="log", ylim=(0, 1))
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    axes[0][0].set_ylabel("Per-disease ROC-AUC")
    save(fig, "per_disease_auc_vs_samples.png")

    # Graph-free baselines against the GNN on identical folds.
    fig, ax = plt.subplots(figsize=(9, 5))
    models = list(BASELINE_COLORS) + list(variants)
    width = 0.8 / max(len(models), 1)
    for position, model in enumerate(models):
        offsets = positions + position * width - 0.4 + width / 2
        heights = []
        for task in tasks:
            if model in BASELINE_COLORS:
                entry = benchmark.get(task, {}).get(model)
                heights.append(entry["mean"]["auc"] if entry else math.nan)
            else:
                heights.append(conditions.get(task, {}).get(model, {}).get("mean_auc", math.nan))
        ax.bar(offsets, heights, width=width, label=model,
               color=BASELINE_COLORS.get(model, _color(model, position)), alpha=.85)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set(xticks=positions, xticklabels=tasks, ylabel="ROC-AUC", ylim=(0, 1),
           title="Graph-free baselines vs the GNN pipeline (same folds, same seed)")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=.25)
    save(fig, "benchmark_vs_gnn.png")

    if finetune:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for position, (task, entry) in enumerate(finetune.items()):
            epochs = [item["epoch"] for item in entry["history"]]
            axes[0].plot(epochs, [item["valid_auc"] for item in entry["history"]],
                         label=task, color=_color(task, position))
            axes[1].plot(epochs, [item["valid_loss"] for item in entry["history"]],
                         label=task, color=_color(task, position))
        axes[0].axhline(0.5, color="gray", ls="--", lw=1)
        axes[0].set(xlabel="Epoch", ylabel="Validation ROC-AUC", title="Holdout fine-tuning")
        axes[1].set(xlabel="Epoch", ylabel="Validation loss", title="Validation loss")
        for ax in axes:
            ax.legend(); ax.grid(alpha=.2)
        save(fig, "finetune_curves.png")

    graph_tasks = [task for task in tasks if attributions.get(task, {}).get("graph_score") is not None]
    if graph_tasks:
        fig, axes = plt.subplots(1, len(graph_tasks), figsize=(6 * len(graph_tasks), 5),
                                 squeeze=False)
        for column, task in enumerate(graph_tasks):
            ax = axes[0][column]
            score = attributions[task]["graph_score"]
            degree = attributions[task]["degree"]
            top = np.argsort(score)[-top_k:]
            mask = np.ones(score.size, dtype=bool)
            mask[top] = False
            ax.scatter(degree[mask], score[mask], s=3, alpha=.25)
            ax.scatter(degree[top], score[top], s=14, alpha=.8, color="red")
            correlation = attributions[task]["pearson_r"]
            ax.set(xlabel="Degree centrality", ylabel="Graph-node IG L2",
                   title=f"{task}: degree vs attribution (r={correlation:.3f})")
            ax.grid(alpha=.2)
        save(fig, "ig_degree_vs_score.png")

        fig, axes = plt.subplots(1, len(graph_tasks), figsize=(6 * len(graph_tasks), 6),
                                 squeeze=False)
        for column, task in enumerate(graph_tasks):
            ax = axes[0][column]
            rows = attributions[task]["top_nodes"][:top_k][::-1]
            ax.barh([row[1] for row in rows], [row[2] for row in rows], color="#54a24b")
            ax.set(xlabel="IG L2", title=f"{task}: top {top_k} graph nodes")
            ax.tick_params(axis="y", labelsize=7)
        save(fig, "ig_top_nodes.png")

    assets.mkdir(parents=True, exist_ok=True)
    for name in made:
        shutil.copy2(output / name, assets / name)
    return made


def run_tr_report(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset = GraphDataset.open(cfg["dataset"]["dir"], cfg["dataset"].get("name"))
    tasks = list(cfg["dataset"].get("tasks") or dataset.task_names)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    document_name = cfg.get("document", "tr_report")
    assets_name = f"{document_name}_assets"
    docs = Path(cfg.get("docs_dir", "docs"))
    cv_dir = Path(cfg.get("cv_dir", "outputs/tr/cv"))
    finetune_dir = Path(cfg.get("finetune_dir", "outputs/tr/finetune"))
    benchmark_dir = Path(cfg.get("benchmark_dir", "outputs/tr/benchmark"))
    ig_dir = Path(cfg.get("ig_dir", "outputs/tr/ig"))
    top_k = int(cfg.get("top_k", 20))
    top_diseases = int(cfg.get("top_diseases", 15))

    # --- dataset audit -----------------------------------------------------
    audit: dict[str, dict[str, Any]] = {}
    lengths = _node_feature_lengths(dataset)
    audit_rows = []
    for name in tasks:
        task = dataset.task(name)
        labels = task.labels()
        groups = np.asarray(task.groups()) if task.groups() is not None else None
        source = task.manifest.get("source", {})
        node_feature_sizes = {
            node_feature.name: dataset.manifest["node_features"][node_feature.source]
            for node_feature in task.node_features
        }
        audit[name] = {
            "num_samples": int(labels.size),
            "num_positive": int(labels.sum()),
            "positive_ratio": float(labels.mean()),
            "diseases_used": int(np.unique(groups).size) if groups is not None else 0,
            "node_features": node_feature_sizes,
            "source": source,
        }
        audit_rows.append([
            name, int(labels.size), int(labels.sum()), float(labels.mean()),
            source.get("num_perturbations", ""), audit[name]["diseases_used"],
            len(task.group_names),
            *[int(np.mean(lengths[node_feature.source])) if node_feature.source in lengths else ""
              for node_feature in task.node_features],
            source.get("signature_genes_skipped", ""), source.get("label_rows_skipped", ""),
        ])
    audit_header = ["task", "samples", "positive", "positive_ratio", "perturbations",
                    "diseases_used", "diseases_total",
                    *[f"mean_genes_{node_feature.name}" for node_feature in dataset.task(tasks[0]).node_features],
                    "signature_genes_skipped", "label_rows_skipped"]
    tsv(output / "dataset_audit.tsv", audit_header, audit_rows)

    # --- cross-validation --------------------------------------------------
    conditions = {name: _cv_conditions(cv_dir, name) for name in tasks}
    all_variants = _variant_order(conditions)
    pooled: dict[tuple[str, str], Any] = {}
    histories: dict[tuple[str, str], list[list[dict[str, Any]]]] = {}
    cv_rows, fold_rows = [], []
    for name in tasks:
        for variant, summary in conditions[name].items():
            pooled[(name, variant)] = _pooled_predictions(cv_dir, name, variant)
            histories[(name, variant)] = _fold_histories(cv_dir, name, variant)
            entry = pooled[(name, variant)]
            cv_rows.append([
                name, variant, bool(summary["variant"].get("use_graph")),
                summary["mean_auc"], summary["std_auc"],
                *[summary.get(f"mean_{metric}", math.nan) for metric in METRICS[1:]],
                min(summary["fold_auc"]), max(summary["fold_auc"]),
                _auc(entry[0], entry[1]) if entry else math.nan,
                len(summary["fold_auc"]),
            ])
            fold_rows.append([name, variant, *summary["fold_auc"],
                              summary["mean_auc"], summary["std_auc"]])
    cv_header = ["task", "variant", "uses_graph", "mean_auc", "std_auc",
                 *[f"mean_{metric}" for metric in METRICS[1:]],
                 "min_fold_auc", "max_fold_auc", "pooled_auc", "folds"]
    tsv(output / "cv_summary.tsv", cv_header, cv_rows)
    tsv(output / "cv_fold_auc.tsv",
        ["task", "variant", *[f"fold_{index}" for index in range(5)], "mean", "std"], fold_rows)

    graph_rows = []
    for name in tasks:
        names = list(conditions[name])
        if len(names) >= 2:
            first, last = conditions[name][names[0]], conditions[name][names[-1]]
            graph_rows.append([name, names[0], first["mean_auc"], names[-1], last["mean_auc"],
                               last["mean_auc"] - first["mean_auc"]])
    tsv(output / "graph_effect.tsv",
        ["task", "baseline", "baseline_auc", "graph_variant", "graph_auc", "delta"], graph_rows)

    # --- per-disease breakdown ---------------------------------------------
    per_disease: dict[str, dict[str, dict[str, Any]]] = {}
    disease_rows = []
    for name in tasks:
        task = dataset.task(name)
        groups = task.groups()
        if groups is None:
            continue
        codes = np.asarray(groups)
        rows: dict[str, dict[str, Any]] = {}
        for variant in conditions[name]:
            entry = pooled.get((name, variant))
            if entry is None:
                continue
            target, probability, indices = entry
            sample_codes = codes[indices]
            for code, disease in enumerate(task.group_names):
                mask = sample_codes == code
                if not mask.any():
                    continue
                row = rows.setdefault(
                    disease,
                    {"samples": int(mask.sum()), "positives": int(target[mask].sum()), "auc": {}},
                )
                row["auc"][variant] = _auc(target[mask], probability[mask])
        per_disease[name] = dict(
            sorted(rows.items(), key=lambda item: item[1]["samples"], reverse=True)
        )
    for name in tasks:
        for disease, row in per_disease.get(name, {}).items():
            disease_rows.append([
                name, disease, row["samples"], row["positives"],
                *[row["auc"].get(variant, math.nan) for variant in all_variants],
            ])
    disease_header = ["task", "disease", "samples", "positives",
                      *[f"auc_{variant}" for variant in all_variants]]
    tsv(output / "per_disease_auc.tsv", disease_header, disease_rows)

    # --- holdout fine-tuning and baselines ---------------------------------
    finetune, finetune_rows = {}, []
    for name in tasks:
        entry = _read(finetune_dir / name / "metrics.json")
        if entry is None:
            continue
        finetune[name] = entry
        best = max(entry["history"], key=lambda item: item["valid_auc"])
        finetune_rows.append([
            name, len(entry["split"]["train"]), len(entry["split"]["valid"]),
            len(entry["split"]["test"]), len(entry["history"]), best["epoch"],
            best["valid_auc"], *[entry["test"][metric] for metric in METRICS],
            entry["test"]["predicted_positive_ratio"],
        ])
    finetune_header = ["task", "train", "valid", "test", "epochs_run", "best_epoch",
                       "best_valid_auc", *[f"test_{metric}" for metric in METRICS],
                       "test_predicted_positive_ratio"]
    tsv(output / "finetune_summary.tsv", finetune_header, finetune_rows)

    benchmark, benchmark_rows = {}, []
    for name in tasks:
        entry = _read(benchmark_dir / name / "benchmark.json")
        if entry is None:
            continue
        benchmark[name] = {
            key: value for key, value in entry.items() if isinstance(value, dict) and "mean" in value
        }
        for model, item in benchmark[name].items():
            benchmark_rows.append([name, model, *[item["mean"][metric] for metric in METRICS]])
        for variant, summary in conditions[name].items():
            benchmark_rows.append([
                name, f"{variant} (pathwaygnn cv)",
                *[summary.get(f"mean_{metric}", math.nan) for metric in METRICS],
            ])
    tsv(output / "benchmark_comparison.tsv", ["task", "model", *METRICS], benchmark_rows)

    # --- attributions -------------------------------------------------------
    node_names = dataset.node_names()
    attributions: dict[str, dict[str, Any]] = {}
    ig_rows, node_feature_rows = [], []
    for name in tasks:
        candidates = sorted(ig_dir.glob(f"{name}_fold*"))
        summary = _read(candidates[0] / "ig_summary.json") if candidates else None
        if summary is None:
            continue
        arrays = np.load(candidates[0] / "attributions.npz")
        graph_score = arrays["graph_score"] if "graph_score" in arrays else None
        entry: dict[str, Any] = {
            "summary": summary,
            "graph_score": graph_score,
            "degree": arrays["degree"] if "degree" in arrays else None,
            "pearson_r": summary.get("degree_ig_pearson_r") or math.nan,
        }
        if graph_score is not None:
            order = np.argsort(graph_score)[::-1][:top_k]
            entry["top_nodes"] = [
                (int(index), node_names[int(index)], float(graph_score[index]),
                 int(entry["degree"][index]))
                for index in order
            ]
            for rank, node in enumerate(entry["top_nodes"], start=1):
                ig_rows.append([name, rank, node[0], node[1], node[2], node[3]])
        for key in arrays.files:
            if not key.startswith("node_feature_"):
                continue
            score = arrays[key]
            order = np.argsort(np.abs(score))[::-1][:top_k]
            for rank, index in enumerate(order, start=1):
                node_feature_rows.append([name, key.removeprefix("node_feature_"), rank, int(index),
                                     node_names[int(index)], float(score[index])])
        attributions[name] = entry
    tsv(output / "ig_top_graph_nodes.tsv",
        ["task", "rank", "node_index", "node", "ig_l2", "degree"], ig_rows)
    tsv(output / "ig_top_node_feature_genes.tsv",
        ["task", "node_feature", "rank", "node_index", "node", "signed_ig"], node_feature_rows)

    pretrain_history = _read(Path(cfg.get("pretrain_history", "outputs/tr/pretrain/history.json")))
    made = _plots(output, docs / assets_name, tasks, all_variants, audit, conditions, pooled, per_disease,
                  histories, finetune, benchmark, attributions, pretrain_history, lengths, top_k)

    # --- document -----------------------------------------------------------
    covered = [f"{task}/{variant}" for task in tasks for variant in conditions.get(task, {})]
    collapsed = [
        f"{task}/{variant}"
        for task in tasks
        for variant, summary in conditions.get(task, {}).items()
        if summary.get("mean_f1") == 0
    ]
    weights = _fold_pos_weights(cv_dir)
    if weights:
        loss_note = (
            "`cv` weights the positive class by `pos_weight` "
            f"({min(weights):.2f}–{max(weights):.2f} across folds, i.e. negatives/positives of each "
            "fold's training split), which is the rule `finetune` uses, so both protocols optimise "
            "the same loss"
        )
    else:
        loss_note = (
            "`cv` trains with an unweighted BCE loss — unlike `finetune`, which applies `pos_weight`"
        )
    threshold_note = (
        f"{loss_note}. On imbalanced labels the 0.5 operating point still collapses onto one class "
        f"for {len(collapsed)} of {len(covered)} conditions ({', '.join(collapsed)}), which score F1 "
        "exactly 0 while their ROC-AUC is not at chance: the ranking carries signal that the default "
        "threshold does not expose."
        if collapsed else
        f"{loss_note}; the 0.5 operating point is therefore comparable between the two tables."
    )
    status = (
        f"{len(covered)} cross-validation conditions, {len(finetune)} holdout runs, "
        f"{len(benchmark)} baseline runs, {len(attributions)} attribution runs"
    )
    pretrain_line = (
        f"{len(pretrain_history)} epochs, final DistMult loss "
        f"{pretrain_history[-1]['loss']:.4f}, final pairwise accuracy "
        f"{pretrain_history[-1]['accuracy']:.4f}"
        if pretrain_history else "not run"
    )
    disease_table = []
    for name in tasks:
        for disease, row in list(per_disease.get(name, {}).items())[:top_diseases]:
            disease_table.append([
                name, disease, row["samples"], row["positives"],
                *[row["auc"].get(variant, math.nan) for variant in all_variants],
            ])
    ig_table = [
        [name, rank, node[1], node[2], node[3]]
        for name in attributions
        for rank, node in enumerate(attributions[name].get("top_nodes", [])[:top_diseases], start=1)
    ]
    node_feature_table = [[*row[:3], *row[4:]] for row in node_feature_rows if row[2] <= 10]
    md = f"""# {TITLE}

## What this report covers

Dataset **{dataset.name}** at `{dataset.manifest['source'].get('source_dir', dataset.manifest['source'].get('raw_dir', dataset.root))}`, prepared
into `{dataset.root}`: {dataset.num_nodes:,} graph nodes, {dataset.manifest['num_edges']:,} directed
edges, {dataset.num_relations} relation types, tasks {', '.join(tasks)}. Run status: {status}.
Graph pre-training: {pretrain_line}.

Every number below comes from artifacts under `outputs/tr/`, and every table is also written as TSV
under `{output}/`. Cross-validation and the graph-free baselines use the same stratified 5-fold
split (seed 42, `StratifiedKFold(shuffle=True)`), so those model comparisons are on identical folds;
attribution runs on fold 0 of the graph variant, and holdout fine-tuning uses its own 70/15/15 split.

## Dataset audit

{mdtable(audit_header, audit_rows)}

`diseases_used` counts the diseases that actually appear in a task's labels, out of
`diseases_total` in the shared disease table. The `mean_genes_*` columns are the mean number of
non-zero genes per row of each node-level feature after the 1e-7 cutoff.

## Cross-validation (`pathwaygnn cv`)

{mdtable(cv_header, cv_rows)}

`pooled_auc` is computed once over the concatenated held-out predictions of all folds, which is why
it can sit outside the min/max of the per-fold values. The `mean_accuracy`/`precision`/`recall`/`f1`
columns score the same folds at a fixed **0.5 decision threshold**; ROC-AUC is threshold-free, so a
condition can rank well and still sit at a poor operating point (or the reverse).

{threshold_note}

Effect of the graph encoder:

{mdtable(["task", "baseline", "baseline_auc", "graph_variant", "graph_auc", "delta"], graph_rows)}

## Graph-free baselines (`pathwaygnn benchmark`)

{mdtable(["task", "model", *METRICS], benchmark_rows)}

The baselines consume exactly the same features as the GNN — the sparse perturbation and disease
signatures — without the pathway graph. All five metrics are on the same footing: both sides are
the mean over the same five folds, and both threshold at 0.5.

## Holdout fine-tuning (`pathwaygnn finetune`)

{mdtable(finetune_header, finetune_rows)}

This protocol is a single stratified 70/15/15 split with early stopping on validation ROC-AUC and
`pos_weight` from the training class ratio, so its numbers are not directly comparable with the
5-fold results above. Compare `best_valid_auc` with `test_auc`: on oe_act the validation split holds
only a few dozen samples, so selecting on it overstates the held-out result. Note also that
`test_predicted_positive_ratio` reveals when a model simply predicts the positive class for
everything, which is what `pos_weight` encourages on these imbalanced labels.

## Per-disease breakdown

Top {top_diseases} diseases per task by held-out sample count:

{mdtable(disease_header, disease_table)}

The full table for every disease is in `{output}/per_disease_auc.tsv`.

## Integrated Gradients (`pathwaygnn ig`)

{mdtable(["task", "rank", "node", "ig_l2", "degree"], ig_table)}

Top 10 attributed genes per node-level feature:

{mdtable(["task", "node_feature", "rank", "node", "signed_ig"], node_feature_table)}

Degree/attribution Pearson correlation:
{', '.join(f"{name} r={attributions[name]['pearson_r']:.3f} "
           f"({attributions[name]['summary']['num_samples']} samples, "
           f"{attributions[name]['summary']['integration_steps']} steps)"
           for name in attributions)}.

Attribution mass concentrates on the highest-degree nodes, and the top of the ranking is dominated
by PathwayCommons chemical entities (`CHEBI:*`) rather than genes — the same degree-driven pattern
the cancer reproduction reports. Read the ranking as where the encoder puts its mass, not as
evidence of a disease-specific mechanism; the per-feature table below is the gene-level view.

## Plots

{figures(made, assets_name)}

## Exact commands

    conda activate gnn
    bash scripts/tr/prepare.sh
    pathwaygnn pretrain  --config configs/tr/pretrain.yaml
    pathwaygnn cv        --config configs/tr/cv.yaml
    pathwaygnn finetune  --config configs/tr/finetune_kd_inh.yaml
    pathwaygnn finetune  --config configs/tr/finetune_oe_act.yaml
    pathwaygnn benchmark --config configs/tr/benchmark_kd_inh.yaml
    pathwaygnn benchmark --config configs/tr/benchmark_oe_act.yaml
    pathwaygnn ig        --config configs/tr/ig_kd_inh.yaml
    pathwaygnn ig        --config configs/tr/ig_oe_act.yaml
    pathwaygnn-data tr-report --config configs/tr/report.yaml

## Interpretation scope

These are the numbers this pipeline currently produces on this data, not a claim that the
architecture works on this task. Read them with three caveats:

* **kd_inh sits at chance** for both variants while the tree baselines reach far higher ROC-AUC on
  the identical folds. On this task the graph pipeline extracts less than plain feature models do.
* **oe_act is small** ({audit.get('oe_act', {}).get('num_samples', 'n/a')} samples,
  {audit.get('oe_act', {}).get('num_positive', 'n/a')} positive), so its fold spread is wide and the
  mean over five folds is a weak estimate.
* **One pre-training run** feeds every downstream number; no seed sweep was performed, and the
  encoder is frozen by default (`end_to_end: false` in `configs/tr/cv.yaml`).

Per-disease ROC-AUC is undefined wherever a disease's held-out samples are single-class, and is
reported as NA in that case.
"""
    markdown_path, html_path = write_document(docs, document_name, md, TITLE)
    result = {
        "dataset": dataset.name,
        "tasks": tasks,
        "status": status,
        "tables": sorted(str(path) for path in output.glob("*.tsv")),
        "plots": [str(output / name) for name in made],
        "markdown": str(markdown_path),
        "html": str(html_path),
    }
    (output / "report.json").write_text(json.dumps(result, indent=2))
    return result
