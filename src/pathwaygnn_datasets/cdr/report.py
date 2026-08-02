"""Reporting for the cancer drug-response dataset.

Collects everything the engine produced for ``data_cdr`` — graph pre-training,
the four-way cross-validation ablation, holdout fine-tuning, graph-free
baselines and Integrated Gradients — into one set of tables, figures and a single
document. Like the cancer and target-repositioning reports this is
dataset-specific presentation, so it lives next to the drug-response
preprocessing.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from pathwaygnn.data.format import GraphDataset
from pathwaygnn_datasets.document import figures, mdtable, tsv, write_document

TITLE = "PathwayGNN cancer drug-response report"
VARIANT_COLORS = {
    "mlp": "#4c78a8",
    "mlp_cov": "#72b7b2",
    "gnn_mlp": "#54a24b",
    "gnn_mlp_cov": "#e45756",
}
BASELINE_COLORS = {
    "logistic_regression": "#f58518",
    "random_forest": "#b279a2",
    "xgboost": "#9d755d",
}
METRICS = ("auc", "accuracy", "precision", "recall", "f1")


def _auc(target: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(target, probability)) if np.unique(target).size == 2 else math.nan


def _color(name: str, position: int) -> str:
    palette = ("#4c78a8", "#54a24b", "#f58518", "#e45756", "#b279a2", "#9d755d")
    return VARIANT_COLORS.get(name, palette[position % len(palette)])


def _read(path: Path) -> Any | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _gene_symbols(path: Path | None) -> dict[str, str]:
    """HGNC numeric id -> approved symbol; the graph names nodes by id only."""
    if path is None or not path.is_file():
        return {}
    symbols: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            identifier = str(row.get("HGNC ID", "")).split("HGNC:")[-1].strip()
            symbol = str(row.get("Approved symbol", "")).strip()
            if identifier and symbol:
                symbols[identifier] = symbol
    return symbols


def _label(node: str, symbols: dict[str, str]) -> str:
    symbol = symbols.get(node)
    return f"{symbol} (HGNC:{node})" if symbol else f"HGNC:{node}"


def _cv_conditions(cv_dir: Path, task: str) -> dict[str, dict[str, Any]]:
    """Variant -> summary, ordered so the table reads as an ablation."""
    found = {}
    for path in sorted((cv_dir / task).glob("*/summary.json")):
        found[path.parts[-2]] = json.loads(path.read_text())
    return dict(sorted(found.items(), key=lambda item: _variant_key(item[0], item[1]["variant"])))


def _variant_key(name: str, variant: dict[str, Any]) -> tuple[bool, bool, str]:
    return bool(variant.get("use_graph")), bool(variant.get("use_covariates")), name


def _variant_order(conditions: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    keys: dict[str, tuple[bool, bool, str]] = {}
    for task_conditions in conditions.values():
        for name, summary in task_conditions.items():
            keys.setdefault(name, _variant_key(name, summary["variant"]))
    return sorted(keys, key=lambda name: keys[name])


def _flags(conditions: dict[str, dict[str, dict[str, Any]]]) -> dict[str, tuple[bool, bool]]:
    flags: dict[str, tuple[bool, bool]] = {}
    for task_conditions in conditions.values():
        for name, summary in task_conditions.items():
            variant = summary["variant"]
            flags.setdefault(
                name, (bool(variant.get("use_graph")), bool(variant.get("use_covariates")))
            )
    return flags


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


def _ablation(
    conditions: dict[str, dict[str, dict[str, Any]]],
    flags: dict[str, tuple[bool, bool]],
    tasks: Sequence[str],
) -> list[list[Any]]:
    """One row per (task, held-fixed setting) comparing the switched factor."""
    by_flags = {value: name for name, value in flags.items()}
    rows: list[list[Any]] = []
    for task in tasks:
        summaries = conditions.get(task, {})

        def auc(graph: bool, covariates: bool) -> float:
            name = by_flags.get((graph, covariates))
            return summaries.get(name, {}).get("mean_auc", math.nan) if name else math.nan

        for covariates in (False, True):
            base, switched = auc(False, covariates), auc(True, covariates)
            rows.append([
                task, "graph encoder", f"use_covariates={covariates}",
                by_flags.get((False, covariates), ""), base,
                by_flags.get((True, covariates), ""), switched, switched - base,
            ])
        for graph in (False, True):
            base, switched = auc(graph, False), auc(graph, True)
            rows.append([
                task, "covariates", f"use_graph={graph}",
                by_flags.get((graph, False), ""), base,
                by_flags.get((graph, True), ""), switched, switched - base,
            ])
    return rows


def _plots(
    output: Path,
    assets: Path,
    tasks: Sequence[str],
    variants: Sequence[str],
    audit: dict[str, dict[str, Any]],
    conditions: dict[str, dict[str, dict[str, Any]]],
    pooled: dict[tuple[str, str], Any],
    per_site: dict[str, dict[str, dict[str, Any]]],
    histories: dict[tuple[str, str], list[list[dict[str, Any]]]],
    finetune: dict[str, Any],
    benchmark: dict[str, Any],
    attributions: dict[str, dict[str, Any]],
    pretrain_history: list[dict[str, Any]] | None,
    profile_lengths: np.ndarray,
    site_counts: list[tuple[str, int]],
    ablation_rows: list[list[Any]],
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

    positions = np.arange(len(tasks))

    # Dataset composition: label balance, mutation-profile length, samples per site.
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    positive = [audit[task]["num_positive"] for task in tasks]
    negative = [audit[task]["num_samples"] - audit[task]["num_positive"] for task in tasks]
    axes[0].bar(positions, negative, label="resistant (0)", color="#4c78a8")
    axes[0].bar(positions, positive, bottom=negative, label="sensitive (1)", color="#e45756")
    for position, task in zip(positions, tasks):
        axes[0].annotate(f"{audit[task]['positive_ratio']:.1%}",
                         (position, audit[task]["num_samples"]), ha="center", va="bottom",
                         fontsize=8)
    axes[0].set(xticks=positions, xticklabels=tasks, ylabel="Samples", title="Label composition")
    axes[0].legend(fontsize=8)
    axes[1].hist(profile_lengths, bins=40, color="#54a24b")
    axes[1].set(xlabel="Mutated census genes per cell line", ylabel="Cell lines",
                title=f"Mutation profiles ({profile_lengths.size} distinct)")
    names = [name for name, _ in site_counts]
    axes[2].barh(np.arange(len(names)), [count for _, count in site_counts], color="#4c78a8")
    axes[2].set(yticks=np.arange(len(names)), yticklabels=names, xlabel="Samples",
                title="Samples per primary site")
    axes[2].tick_params(axis="y", labelsize=7)
    axes[2].invert_yaxis()
    save(fig, "dataset_composition.png")

    if pretrain_history:
        epochs = [item["epoch"] for item in pretrain_history]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, [item["loss"] for item in pretrain_history], color="#4c78a8")
        twin = ax.twinx()
        twin.plot(epochs, [item["accuracy"] for item in pretrain_history], color="#e45756")
        ax.set(xlabel="Pre-training epoch", ylabel="DistMult loss (blue)",
               title="Graph pre-training diagnostics")
        twin.set_ylabel("Pairwise accuracy (red)")
        save(fig, "pretraining_diagnostics.png")

    # Cross-validation: mean bars with the individual folds on top.
    width = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
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
    ax.set(xticks=positions, xticklabels=tasks, ylabel="Held-out ROC-AUC", ylim=(0.4, 1),
           title="Cross-validated ROC-AUC (bars: mean +/- std, dots: folds)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    save(fig, "cv_auc_by_task_variant.png")

    # The ablation itself: what each switch is worth, per task.
    if ablation_rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"{row[0]}\n{row[1]} | {row[2]}" for row in ablation_rows]
        deltas = [row[7] for row in ablation_rows]
        colors = ["#54a24b" if value >= 0 else "#e45756" for value in deltas]
        ax.bar(np.arange(len(deltas)), deltas, color=colors, alpha=.85)
        ax.axhline(0, color="black", lw=1)
        ax.set(xticks=np.arange(len(deltas)), ylabel="Delta mean ROC-AUC",
               title="Effect of switching one factor on, other factors fixed")
        ax.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
        ax.grid(axis="y", alpha=.25)
        save(fig, "ablation_deltas.png")

    values, labels = [], []
    for task in tasks:
        for variant in conditions.get(task, {}):
            values.append(conditions[task][variant]["fold_auc"])
            labels.append(f"{task}\n{variant}")
    if values:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.boxplot(values, tick_labels=labels, showmeans=True)
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set(ylabel="Fold ROC-AUC", title="Fold ROC-AUC distributions")
        ax.tick_params(axis="x", labelsize=7)
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
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title=f"{task}: pooled folds")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig, "cv_roc_curves.png")

    panels = [(task, variant) for task in tasks for variant in conditions.get(task, {})]
    if panels:
        columns = min(len(panels), 4)
        rows_needed = math.ceil(len(panels) / columns)
        fig, axes = plt.subplots(rows_needed, columns, figsize=(4.2 * columns, 3.6 * rows_needed),
                                 squeeze=False, sharey=True)
        for position, (task, variant) in enumerate(panels):
            ax = axes[position // columns][position % columns]
            for history in histories.get((task, variant), []):
                ax.plot([item["epoch"] for item in history],
                        [item["test_auc"] for item in history], lw=.9, alpha=.75)
            ax.axhline(0.5, color="gray", ls="--", lw=1)
            ax.set(title=f"{task}\n{variant}", xlabel="Epoch", ylim=(0.4, 1))
            ax.title.set_fontsize(9)
            ax.grid(alpha=.2)
        for position in range(len(panels), rows_needed * columns):
            axes[position // columns][position % columns].axis("off")
        axes[0][0].set_ylabel("Held-out fold ROC-AUC")
        fig.suptitle("Cross-validation training curves (one line per fold)")
        save(fig, "cv_training_curves.png")

    # Per-cancer-type behaviour: graph-free against graph, and against sample count.
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5.5), squeeze=False)
    for column, task in enumerate(tasks):
        ax = axes[0][column]
        names_here = list(conditions.get(task, {}))
        if len(names_here) >= 2:
            first, second = names_here[0], names_here[-1]
            xs, ys, sizes, labels_ = [], [], [], []
            for site, row in per_site.get(task, {}).items():
                x, y = row["auc"].get(first, math.nan), row["auc"].get(second, math.nan)
                if math.isfinite(x) and math.isfinite(y):
                    xs.append(x); ys.append(y); sizes.append(max(14, row["samples"] / 40))
                    labels_.append(site)
            ax.scatter(xs, ys, s=sizes, alpha=.7, color="#4c78a8")
            ax.plot([0, 1], [0, 1], "--", color="gray")
            for x, y, name in zip(xs, ys, labels_):
                ax.annotate(name, (x, y), fontsize=6)
            ax.set(xlabel=f"{first} ROC-AUC", ylabel=f"{second} ROC-AUC",
                   title=f"{task}: per primary site")
        ax.grid(alpha=.2)
    save(fig, "per_site_auc_scatter.png")

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4.5), squeeze=False,
                             sharey=True)
    for column, task in enumerate(tasks):
        ax = axes[0][column]
        for position, variant in enumerate(conditions.get(task, {})):
            rows_here = per_site.get(task, {})
            xs = [row["samples"] for row in rows_here.values()
                  if math.isfinite(row["auc"].get(variant, math.nan))]
            ys = [row["auc"][variant] for row in rows_here.values()
                  if math.isfinite(row["auc"].get(variant, math.nan))]
            ax.scatter(xs, ys, s=20, alpha=.75, label=variant, color=_color(variant, position))
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set(xlabel="Held-out samples for the site", title=task, xscale="log")
        ax.legend(fontsize=7); ax.grid(alpha=.2)
    axes[0][0].set_ylabel("Per-site ROC-AUC")
    save(fig, "per_site_auc_vs_samples.png")

    # Graph-free baselines against the GNN on identical folds.
    fig, ax = plt.subplots(figsize=(10, 5))
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
    ax.set(xticks=positions, xticklabels=tasks, ylabel="ROC-AUC", ylim=(0.4, 1),
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
            ax.legend(fontsize=8); ax.grid(alpha=.2)
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
            ax.set(xlabel="Degree centrality", ylabel="Graph-node IG L2",
                   title=f"{task}: degree vs attribution (r={attributions[task]['pearson_r']:.3f})")
            ax.grid(alpha=.2)
        save(fig, "ig_degree_vs_score.png")

        fig, axes = plt.subplots(1, len(graph_tasks), figsize=(6.5 * len(graph_tasks), 6),
                                 squeeze=False)
        for column, task in enumerate(graph_tasks):
            ax = axes[0][column]
            rows_here = attributions[task]["top_nodes"][:top_k][::-1]
            ax.barh([row[1] for row in rows_here], [row[2] for row in rows_here], color="#54a24b")
            ax.set(xlabel="IG L2", title=f"{task}: top {top_k} graph nodes")
            ax.tick_params(axis="y", labelsize=7)
        save(fig, "ig_top_nodes.png")

    covariate_tasks = [task for task in tasks if attributions.get(task, {}).get("top_covariates")]
    if covariate_tasks:
        fig, axes = plt.subplots(1, len(covariate_tasks), figsize=(6.5 * len(covariate_tasks), 6),
                                 squeeze=False)
        for column, task in enumerate(covariate_tasks):
            ax = axes[0][column]
            rows_here = attributions[task]["top_covariates"][:top_k][::-1]
            ax.barh([row[0] for row in rows_here], [row[1] for row in rows_here],
                    color=["#e45756" if row[1] < 0 else "#4c78a8" for row in rows_here])
            ax.axvline(0, color="black", lw=1)
            ax.set(xlabel="Signed IG", title=f"{task}: top {top_k} covariates")
            ax.tick_params(axis="y", labelsize=7)
        save(fig, "ig_top_covariates.png")

    assets.mkdir(parents=True, exist_ok=True)
    for name in made:
        shutil.copy2(output / name, assets / name)
    return made


def run_cdr_report(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset = GraphDataset.open(cfg["dataset"]["dir"], cfg["dataset"].get("name"))
    tasks = list(cfg["dataset"].get("tasks") or dataset.task_names)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    document_name = cfg.get("document", "cdr_report")
    assets_name = f"{document_name}_assets"
    docs = Path(cfg.get("docs_dir", "docs"))
    cv_dir = Path(cfg.get("cv_dir", "outputs/cdr/cv"))
    finetune_dir = Path(cfg.get("finetune_dir", "outputs/cdr/finetune"))
    benchmark_dir = Path(cfg.get("benchmark_dir", "outputs/cdr/benchmark"))
    ig_dir = Path(cfg.get("ig_dir", "outputs/cdr/ig"))
    top_k = int(cfg.get("top_k", 20))
    top_sites = int(cfg.get("top_sites", 19))
    symbol_path = cfg.get("gene_symbols", "data_cdr/raw/EnsemblToHGNC.tsv")
    symbols = _gene_symbols(Path(symbol_path) if symbol_path else None)
    source = dataset.manifest.get("source", {})

    # --- dataset audit -----------------------------------------------------
    channel = dataset.channel("mutation")
    profile_lengths = np.diff(np.asarray(channel.csr()[0]))
    audit: dict[str, dict[str, Any]] = {}
    audit_rows = []
    site_counts: list[tuple[str, int]] = []
    for name in tasks:
        task = dataset.task(name)
        labels = task.labels()
        groups = np.asarray(task.groups())
        task_source = task.manifest.get("source", {})
        audit[name] = {
            "num_samples": int(labels.size),
            "num_positive": int(labels.sum()),
            "positive_ratio": float(labels.mean()),
            "reference": task_source.get("reference", ""),
            "source": task_source,
        }
        audit_rows.append([
            name, int(labels.size), int(labels.sum()), float(labels.mean()),
            task_source.get("num_cell_lines", ""), task_source.get("num_compounds", ""),
            int(np.unique(groups).size), len(task.group_names), task.covariate_dim,
            int(channel.num_rows), float(profile_lengths.mean()),
            task_source.get("reference", ""),
        ])
        if not site_counts:
            counts = np.bincount(groups, minlength=len(task.group_names))
            site_counts = [
                (site, int(counts[code])) for code, site in enumerate(task.group_names)
            ]
            site_counts.sort(key=lambda item: item[1], reverse=True)
    audit_header = ["task", "samples", "positive", "positive_ratio", "cell_lines", "compounds",
                    "sites_used", "sites_total", "covariates", "mutation_rows",
                    "mean_genes_mutation", "label_reference"]
    tsv(output / "dataset_audit.tsv", audit_header, audit_rows)

    # --- cross-validation --------------------------------------------------
    conditions = {name: _cv_conditions(cv_dir, name) for name in tasks}
    all_variants = _variant_order(conditions)
    flags = _flags(conditions)
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
                bool(summary["variant"].get("use_covariates")),
                summary["mean_auc"], summary["std_auc"],
                *[summary.get(f"mean_{metric}", math.nan) for metric in METRICS[1:]],
                min(summary["fold_auc"]), max(summary["fold_auc"]),
                _auc(entry[0], entry[1]) if entry else math.nan,
                len(summary["fold_auc"]),
            ])
            fold_rows.append([name, variant, *summary["fold_auc"],
                              summary["mean_auc"], summary["std_auc"]])
    cv_header = ["task", "variant", "uses_graph", "uses_covariates", "mean_auc", "std_auc",
                 *[f"mean_{metric}" for metric in METRICS[1:]],
                 "min_fold_auc", "max_fold_auc", "pooled_auc", "folds"]
    tsv(output / "cv_summary.tsv", cv_header, cv_rows)
    tsv(output / "cv_fold_auc.tsv",
        ["task", "variant", *[f"fold_{index}" for index in range(5)], "mean", "std"], fold_rows)

    ablation_header = ["task", "factor", "held_fixed", "off_variant", "off_auc",
                       "on_variant", "on_auc", "delta"]
    ablation_rows = _ablation(conditions, flags, tasks)
    tsv(output / "ablation.tsv", ablation_header, ablation_rows)

    # --- per-primary-site breakdown ----------------------------------------
    per_site: dict[str, dict[str, dict[str, Any]]] = {}
    site_rows = []
    for name in tasks:
        task = dataset.task(name)
        codes = np.asarray(task.groups())
        rows_here: dict[str, dict[str, Any]] = {}
        for variant in conditions[name]:
            entry = pooled.get((name, variant))
            if entry is None:
                continue
            target, probability, indices = entry
            sample_codes = codes[indices]
            for code, site in enumerate(task.group_names):
                mask = sample_codes == code
                if not mask.any():
                    continue
                row = rows_here.setdefault(
                    site,
                    {"samples": int(mask.sum()), "positives": int(target[mask].sum()), "auc": {}},
                )
                row["auc"][variant] = _auc(target[mask], probability[mask])
        per_site[name] = dict(
            sorted(rows_here.items(), key=lambda item: item[1]["samples"], reverse=True)
        )
    for name in tasks:
        for site, row in per_site.get(name, {}).items():
            site_rows.append([
                name, site, row["samples"], row["positives"],
                *[row["auc"].get(variant, math.nan) for variant in all_variants],
            ])
    site_header = ["task", "primary_site", "samples", "positives",
                   *[f"auc_{variant}" for variant in all_variants]]
    tsv(output / "per_site_auc.tsv", site_header, site_rows)

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
    ig_rows, channel_rows, covariate_rows = [], [], []
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
                (int(index), _label(node_names[int(index)], symbols), float(graph_score[index]),
                 int(entry["degree"][index]))
                for index in order
            ]
            for rank, node in enumerate(entry["top_nodes"], start=1):
                ig_rows.append([name, rank, node[0], node[1], node[2], node[3]])
        for key in arrays.files:
            if not key.startswith("channel_"):
                continue
            score = arrays[key]
            order = np.argsort(np.abs(score))[::-1][:top_k]
            for rank, index in enumerate(order, start=1):
                channel_rows.append([name, key.removeprefix("channel_"), rank, int(index),
                                     _label(node_names[int(index)], symbols), float(score[index])])
        covariate_ig = summary.get("covariate_ig", {})
        entry["top_covariates"] = sorted(
            covariate_ig.items(), key=lambda item: abs(item[1]), reverse=True
        )[:top_k]
        for rank, (covariate, value) in enumerate(entry["top_covariates"], start=1):
            covariate_rows.append([name, rank, covariate, value])
        attributions[name] = entry
    tsv(output / "ig_top_graph_nodes.tsv",
        ["task", "rank", "node_index", "node", "ig_l2", "degree"], ig_rows)
    tsv(output / "ig_top_channel_genes.tsv",
        ["task", "channel", "rank", "node_index", "node", "signed_ig"], channel_rows)
    tsv(output / "ig_top_covariates.tsv", ["task", "rank", "covariate", "signed_ig"], covariate_rows)

    pretrain_history = _read(Path(cfg.get("pretrain_history", "outputs/cdr/pretrain/history.json")))
    made = _plots(output, docs / assets_name, tasks, all_variants, audit, conditions, pooled,
                  per_site, histories, finetune, benchmark, attributions, pretrain_history,
                  profile_lengths, site_counts, ablation_rows, top_k)

    # --- document -----------------------------------------------------------
    covered = [f"{task}/{variant}" for task in tasks for variant in conditions.get(task, {})]
    collapsed = [
        f"{task}/{variant}"
        for task in tasks
        for variant, summary in conditions.get(task, {}).items()
        if summary.get("mean_f1") == 0
    ]
    threshold_note = (
        "`cv` trains with an unweighted BCE loss — unlike `finetune`, which applies `pos_weight` — "
        "so on imbalanced labels the 0.5 operating point collapses onto the majority class. That "
        f"happens for {len(collapsed)} of {len(covered)} conditions ({', '.join(collapsed)})."
        if collapsed else
        "`cv` trains with an unweighted BCE loss — unlike `finetune`, which applies `pos_weight` — "
        "so on imbalanced labels the 0.5 operating point would drift towards the majority class. "
        "Both tasks here are ~50% positive by construction, so accuracy and F1 stay interpretable."
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
    site_table = []
    for name in tasks:
        for site, row in list(per_site.get(name, {}).items())[:top_sites]:
            site_table.append([
                name, site, row["samples"], row["positives"],
                *[row["auc"].get(variant, math.nan) for variant in all_variants],
            ])
    ig_table = [
        [name, rank, node[1], node[2], node[3]]
        for name in attributions
        for rank, node in enumerate(attributions[name].get("top_nodes", [])[:top_k], start=1)
    ]
    channel_table = [[*row[:3], *row[4:]] for row in channel_rows if row[2] <= 10]
    covariate_table = [row for row in covariate_rows if row[1] <= 10]
    def _versus(task: str) -> str:
        models = benchmark.get(task, {})
        if not models or not conditions.get(task):
            return ""
        model, entry = max(models.items(), key=lambda item: item[1]["mean"]["auc"])
        variant, summary = max(conditions[task].items(), key=lambda item: item[1]["mean_auc"])
        verb = "beats" if entry["mean"]["auc"] > summary["mean_auc"] else "trails"
        return (f"on {task} the best baseline ({model}, {entry['mean']['auc']:.4f}) {verb} the best "
                f"pathwaygnn condition ({variant}, {summary['mean_auc']:.4f})")
    versus_line = "; ".join(filter(None, (_versus(task) for task in tasks))) or "not run"
    best_line = ", ".join(
        f"{task}: {max(conditions[task].items(), key=lambda item: item[1]['mean_auc'])[0]} "
        f"{max(summary['mean_auc'] for summary in conditions[task].values()):.4f}"
        for task in tasks if conditions.get(task)
    )
    md = f"""# {TITLE}

## What this report covers

Dataset **{dataset.name}** — the GraphCDRScan corpus (GDSC1 dose response, Cell Model Passports
mutations, Reactome functional interactions) — prepared from
`{source.get('source_dir', dataset.root)}` into `{dataset.root}`:
{dataset.num_nodes:,} graph nodes, {dataset.manifest['num_edges']:,} directed edges,
{dataset.num_relations} relation types, {source.get('num_samples', 0):,} samples built from
{source.get('num_cell_lines', 0)} cell lines x {source.get('num_compounds', 0)} compounds, tasks
{', '.join(tasks)}. Run status: {status}. Graph pre-training: {pretrain_line}.
Best cross-validated condition per task — {best_line}.

A sample is one *(cell line, compound)* pair. Preprocessing turns the GDSC `LN_IC50` into a binary
label, because `pathwaygnn` trains binary problems only:

* **sensitive_drugwise** — 1 when `LN_IC50` is below the *same compound's* median. Every compound
  contributes ~50% positives, so the compound's overall potency carries no signal and the label can
  only be predicted from the cell line.
* **sensitive_global** — 1 when `LN_IC50` is below the median over all samples. Here the compound
  identity alone explains most of the label.

Each sample carries one sparse channel and one covariate vector:

* channel `mutation` — the number of mutations per Cancer-Gene-Census gene of the cell line, indexed
  by graph node. Because the profile depends only on the cell line, the
  {source.get('num_samples', 0):,} samples share {source.get('distinct_mutation_profiles', 0)}
  distinct rows through `rows/mutation.npy`.
* covariates — the GraphCDRScan sample-feature vector verbatim: the 96/78/83-context mutational
  spectra of the cell line, its primary-site one-hot and the 3 x 1024-bit RDKit compound
  fingerprint ({source.get('covariate_dim', 0):,} values).

Every number below comes from artifacts under `outputs/cdr/`, and every table is also written as TSV
under `{output}/`. Cross-validation and the graph-free baselines use the same stratified 5-fold
split (seed 42, `StratifiedKFold(shuffle=True)`), so those model comparisons are on identical folds;
attribution runs on fold 0 of `gnn_mlp_cov`, and holdout fine-tuning uses its own 70/15/15 split.

## Dataset audit

{mdtable(audit_header, audit_rows)}

`mutation_rows` is the number of distinct mutation profiles the channel stores, and
`mean_genes_mutation` the mean number of mutated census genes per profile. `sites_used` counts the
primary sites that actually appear, out of `sites_total` in the one-hot block.

## Cross-validation (`pathwaygnn cv`)

{mdtable(cv_header, cv_rows)}

`pooled_auc` is computed once over the concatenated held-out predictions of all folds, which is why
it can sit outside the min/max of the per-fold values. The `mean_accuracy`/`precision`/`recall`/`f1`
columns score the same folds at a fixed **0.5 decision threshold**; ROC-AUC is threshold-free, so a
condition can rank well and still sit at a poor operating point (or the reverse).

{threshold_note}

The grid is a two-factor ablation — the pathway graph on/off crossed with the covariate branch
on/off — so each switch can be read with the other held fixed:

{mdtable(ablation_header, ablation_rows)}

The `covariates` rows are large by construction: the covariate block carries the compound
fingerprint, and `sensitive_global` is mostly a question about the compound. The rows that speak to
the pathway graph are the `graph encoder` ones, and they are only informative where the mutation
channel is the model's *only* view of the sample (`use_covariates=False`) — with the covariates on,
the graph has little left to add.

## Graph-free baselines (`pathwaygnn benchmark`)

{mdtable(["task", "model", *METRICS], benchmark_rows)}

The baselines consume exactly the same features as the GNN — the mutation channel expanded to
`[samples, {dataset.num_nodes:,}]` plus the covariate block — without the pathway graph. All five
metrics are on the same footing: both sides are the mean over the same five folds, and both
threshold at 0.5. These are reference points, not tuned models: the features are unscaled
counts and raw bits, so `LogisticRegression` hits its `max_iter=1000` lbfgs limit without
converging, and the forest is capped at 60 trees of depth 12 to finish on a
107,418 x {dataset.num_nodes + (dataset.task(tasks[0]).covariate_dim if tasks else 0):,} matrix.

## Holdout fine-tuning (`pathwaygnn finetune`)

{mdtable(finetune_header, finetune_rows)}

This protocol is a single stratified 70/15/15 split of `gnn_mlp_cov` with early stopping on
validation ROC-AUC and `pos_weight` from the training class ratio, so its numbers are not directly
comparable with the 5-fold results above.

## Per primary site

{mdtable(site_header, site_table)}

The full table is in `{output}/per_site_auc.tsv`. Per-site ROC-AUC is undefined wherever a site's
held-out samples are single-class, and is reported as NA in that case.

## Integrated Gradients (`pathwaygnn ig`)

Top attributed graph nodes (HGNC ids resolved to approved symbols through
`{symbol_path}`):

{mdtable(["task", "rank", "node", "ig_l2", "degree"], ig_table)}

Top 10 attributed genes of the `mutation` channel:

{mdtable(["task", "channel", "rank", "node", "signed_ig"], channel_table)}

Top 10 attributed covariates:

{mdtable(["task", "rank", "covariate", "signed_ig"], covariate_table)}

Degree/attribution Pearson correlation:
{', '.join(f"{name} r={attributions[name]['pearson_r']:.3f} "
           f"({attributions[name]['summary']['num_samples']} samples, "
           f"{attributions[name]['summary']['integration_steps']} steps)"
           for name in attributions) or 'not run'}.

The graph ranking is degree-driven — the top of it is the ubiquitin/ribosomal hubs (`UBC`, `UBB`,
`UBA52`, `RPS27A`) that dominate Reactome's functional-interaction network. Read it as where the
encoder puts its mass, not as evidence of a drug-response mechanism; the `mutation` channel table is
the gene-level view, and the covariate table separates what comes from the compound fingerprint from
what comes from the cell line's spectra and site.

## Plots

{figures(made, assets_name)}

## Exact commands

    conda activate gnn
    # upstream GraphCDRScan stage, only needed to rebuild data_cdr/processed/
    python -m scripts.cdr.upstream.download_raw_data
    python -m scripts.cdr.upstream.prepare_data --config configs/cdr/upstream.json

    bash scripts/cdr/prepare.sh
    pathwaygnn pretrain  --config configs/cdr/pretrain.yaml
    pathwaygnn cv        --config configs/cdr/cv.yaml
    pathwaygnn finetune  --config configs/cdr/finetune_drugwise.yaml
    pathwaygnn finetune  --config configs/cdr/finetune_global.yaml
    pathwaygnn benchmark --config configs/cdr/benchmark_drugwise.yaml
    pathwaygnn benchmark --config configs/cdr/benchmark_global.yaml
    pathwaygnn ig        --config configs/cdr/ig_drugwise.yaml
    pathwaygnn ig        --config configs/cdr/ig_global.yaml
    pathwaygnn-data cdr-report --config configs/cdr/report.yaml

`bash scripts/cdr/reproduce.sh` runs the same list.

## Interpretation scope

These are the numbers this pipeline currently produces on this data, not a claim that the
architecture solves drug-response prediction. Read them with these caveats:

* **Tree baselines are the reference to beat.** On these folds, {versus_line}. The GNN pipeline is
  not the strongest model here; the graph earns its keep only in the covariate-free ablation, where
  it is the difference between chance and a weak but non-zero signal.
* **The two tasks are not equally hard by construction.** `sensitive_global` is largely a question
  about the compound, `sensitive_drugwise` largely a question about the cell line; comparing their
  AUCs against each other says more about the labels than about the model.
* **Folds are random over samples, not over cell lines or compounds.** A cell line appears in both
  the training and the held-out fold with different compounds, so these numbers describe filling in
  a partly observed response matrix, not generalisation to an unseen cell line.
* **The mutation channel is a scalar per gene.** GraphCDRScan's per-mutation node features (variant
  type, encoded genomic position) are reduced to a mutation count, because the sample-level head
  projects one value per gene. The spectra keep some of that information at the sample level.
* **These are current-release inputs, not the 2018 CDRscan experiment.** GraphCDRScan substitutes
  GDSC1 (Oct 2023), Cell Model Passports mutations on GRCh38 and RDKit fingerprints for the paper's
  GDSC 6.0, COSMIC v82 and PaDEL descriptors, so nothing here is comparable to published CDRscan
  numbers.
* **One pre-training run** feeds every downstream number; no seed sweep was performed, and the
  encoder is frozen during cross-validation (`end_to_end: false`).
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
