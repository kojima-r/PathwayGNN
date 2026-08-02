"""Reporting for the Inoue et al. reproduction.

Reads the cross-validation artifacts written by ``pathwaygnn cv`` (and the IG
artifacts written by ``pathwaygnn ig``) and renders the paper comparison tables,
figures and the reproduction document. This is dataset-specific presentation, so
it lives next to the cancer preprocessing rather than inside the generic engine.
"""

from __future__ import annotations

import json, math, shutil
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from pathwaygnn.data.format import GraphDataset
from pathwaygnn_datasets.document import figures, mdtable, tsv, write_document
from pathwaygnn_datasets.cancer.paper import (
    CANCER_TYPES,
    DISPLAY,
    PAPER_TABLE1,
    VARIANT_NAMES,
    YEARS,
)

SWEEP_EPOCHS = (0, 10, 20, 30, 40, 50)


def _summaries(root: Path) -> dict[str, dict[str, Any]]:
    """Condition -> summary, keyed by directory so old and new runs both load."""
    out = {}
    for path in sorted(root.glob("*year/*/summary.json")):
        summary = json.loads(path.read_text())
        out[f"{path.parts[-3]}/{path.parts[-2]}"] = summary
    return out


def _predictions(root, year, variant):
    ys, ps, ids = [], [], []
    for p in sorted((root / f"{year}year" / variant).glob("fold_*/predictions.npz")):
        x = np.load(p); ys.append(x["target"]); ps.append(x["probability"]); ids.append(x["sample_index"])
    if not ys: raise FileNotFoundError
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ids)


def _auc(y, p):
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else math.nan


def _plots(output, assets, root, reproduced, per_cancer, counts, pretrain, ig_dir, sweep_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    made = []; variants = VARIANT_NAMES; years = np.asarray(YEARS)
    colors = dict(zip(variants, ("#4c78a8", "#f58518", "#54a24b", "#e45756")))

    def save(fig, name):
        fig.tight_layout(); fig.savefig(output / name, dpi=210, bbox_inches="tight")
        plt.close(fig); made.append(name)

    fig, ax = plt.subplots(figsize=(8, 5))
    for v in variants:
        ax.plot(years, [reproduced.get(f"{y}year/{v}", {}).get("mean_auc", math.nan) for y in years],
                marker="o", label=DISPLAY[v], color=colors[v])
    ax.set(xlabel="Verification year", ylabel="ROC-AUC", xticks=years, ylim=(.5, .85),
           title="Table 1 reproduced ROC-AUC"); ax.legend(fontsize=8); ax.grid(alpha=.25)
    save(fig, "table1_auc_by_year.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in variants:
        d = []
        for y in years:
            value = reproduced.get(f"{y}year/{v}", {}).get("mean_auc", math.nan)
            paper = PAPER_TABLE1[int(y)][variants.index(v)]
            d.append(value - paper if math.isfinite(value) else math.nan)
        ax.plot(years, d, marker="o", label=DISPLAY[v], color=colors[v])
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="Verification year", ylabel="Reproduced - paper ROC-AUC", xticks=years,
           title="Deviation from published Table 1"); ax.legend(fontsize=8); ax.grid(alpha=.25)
    save(fig, "table1_auc_delta.png")
    values = []; positions = []; pos = 0
    for y in years:
        for v in variants:
            fold = reproduced.get(f"{y}year/{v}", {}).get("fold_auc", [])
            if fold: positions.append(pos); values.append(fold)
            pos += 1
        pos += 1
    fig, ax = plt.subplots(figsize=(10, 5))
    if values: ax.boxplot(values, positions=positions, widths=.7, showmeans=True)
    ax.set(xlabel="Year / model condition", ylabel="Fold ROC-AUC",
           title="Five-fold ROC-AUC distributions"); ax.grid(axis="y", alpha=.25)
    save(fig, "table1_fold_auc_boxplot.png")
    totals = [sum(x[0] for x in counts[int(y)].values()) for y in years]  # noqa: F841
    deaths = [sum(x[1] for x in counts[int(y)].values()) for y in years]
    surv = [sum(x[2] for x in counts[int(y)].values()) for y in years]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(years, surv, label="survival", color="#54a24b")
    ax.bar(years, deaths, bottom=surv, label="death", color="#e45756")
    ax.set(xlabel="Verification year", ylabel="Samples", xticks=years,
           title="Supplementary Table 1 sample composition"); ax.legend()
    save(fig, "supplementary_table1_sample_counts.png")
    if "3year/dnn" in per_cancer and "3year/gnn_dnn" in per_cancer:
        xs = []; ys = []; ss = []; cs = []; labs = []
        for name in CANCER_TYPES:
            x = per_cancer["3year/dnn"][name]; y = per_cancer["3year/gnn_dnn"][name]
            if math.isfinite(x) and math.isfinite(y):
                total, death, _ = counts[3][name]
                xs.append(x); ys.append(y); ss.append(max(15, total / 2))
                cs.append(death / total); labs.append(name)
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(xs, ys, s=ss, c=cs, cmap="viridis", alpha=.8)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        for x, y, n in zip(xs, ys, labs): ax.annotate(n, (x, y), fontsize=7)
        fig.colorbar(sc, ax=ax, label="Death ratio")
        ax.set(xlabel="DNN ROC-AUC", ylabel="GNN + DNN ROC-AUC", xlim=(0, 1), ylim=(0, 1),
               title="Figure 3a reproduction")
        save(fig, "figure3a_per_cancer_auc.png")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, v in zip(axes, ("dnn", "gnn_dnn")):
        for name in CANCER_TYPES:
            vals = [per_cancer.get(f"{y}year/{v}", {}).get(name, math.nan) for y in years]
            if any(math.isfinite(x) for x in vals): ax.plot(years, vals, lw=.8, alpha=.75, label=name)
        ax.set(title=DISPLAY[v], xlabel="Verification year", xticks=years, ylim=(0, 1)); ax.grid(alpha=.2)
    axes[0].set_ylabel("Per-cancer ROC-AUC")
    axes[1].legend(ncol=3, fontsize=5, loc="center left", bbox_to_anchor=(1, .5))
    fig.suptitle("Figure 3b reproduction")
    save(fig, "figure3b_per_cancer_auc_transition.png")
    sweep = []
    for epoch in SWEEP_EPOCHS:
        for mode in ("end_to_end", "frozen"):
            path = sweep_root / f"epoch_{epoch}" / mode / "5year" / "gnn_dnn" / "summary.json"
            if path.exists():
                item = json.loads(path.read_text())
                sweep.append((epoch, mode, item["mean_auc"], item["std_auc"]))
    if sweep:
        fig, ax = plt.subplots(figsize=(8, 5))
        for mode, color in (("end_to_end", "#4c78a8"), ("frozen", "#f58518")):
            selected = [x for x in sweep if x[1] == mode]
            ax.errorbar([x[0] for x in selected], [x[2] for x in selected],
                        yerr=[x[3] for x in selected], marker="o", capsize=3,
                        label=mode.replace("_", "-"), color=color)
        ax.set(xlabel="Graph pre-training epoch", ylabel="5-year ROC-AUC",
               title="Figure 2 reproduction"); ax.legend(); ax.grid(alpha=.25)
        save(fig, "figure2_pretraining_sweep.png")
    histories = []
    for p in sorted(root.glob("*year/*/fold_*/metrics.json")):
        x = json.loads(p.read_text())
        histories.append((int(p.parts[-4].replace("year", "")), p.parts[-3], x["history"]))
    if histories:
        fig, axes = plt.subplots(5, 4, figsize=(16, 18), sharex=True)
        for row, y in enumerate(years):
            for col, v in enumerate(variants):
                ax = axes[row, col]
                for yy, vv, h in histories:
                    if yy == y and vv == v:
                        ax.plot([x["epoch"] for x in h], [x["test_auc"] for x in h], alpha=.45, lw=.8)
                ax.set_title(f"{y}y {DISPLAY[v]}", fontsize=8); ax.grid(alpha=.2)
        fig.supylabel("Held-out fold ROC-AUC"); fig.supxlabel("Fine-tuning epoch")
        fig.suptitle("All Table 1 training curves")
        save(fig, "table1_training_curves.png")
    if pretrain.exists():
        h = json.loads(pretrain.read_text())
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([x["epoch"] for x in h], [x["loss"] for x in h])
        ax2 = ax.twinx(); ax2.plot([x["epoch"] for x in h], [x["accuracy"] for x in h], color="#e45756")
        ax.set(xlabel="Pre-training epoch", ylabel="DistMult loss",
               title="Graph pre-training diagnostics"); ax2.set_ylabel("Pairwise accuracy")
        save(fig, "pretraining_diagnostics.png")
    ig = next(iter(sorted(ig_dir.glob("**/attributions.npz"))), None)
    if ig:
        z = np.load(ig)
        if "degree" in z and "graph_score" in z:
            degree = z["degree"]; score = z["graph_score"]
            top = np.argsort(score)[-1500:]; mask = np.ones(score.size, dtype=bool); mask[top] = False
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(degree[mask], score[mask], s=3, alpha=.25)
            ax.scatter(degree[top], score[top], s=5, alpha=.6, color="red")
            r = np.corrcoef(degree, score)[0, 1]
            ax.set(xlabel="Degree centrality", ylabel="Graph-node IG L2",
                   title=f"Figure 4 reproduction (Pearson r={r:.3f})")
            save(fig, "figure4_degree_vs_ig.png")
    assets.mkdir(parents=True, exist_ok=True)
    for name in made: shutil.copy2(output / name, assets / name)
    return made


def run_cancer_report(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["run_dir"]); output = Path(cfg["output_dir"])
    dataset = GraphDataset.open(cfg["dataset"]["dir"], cfg["dataset"].get("name"))
    docs = Path(cfg.get("docs_dir", "docs")); assets = docs / "cancer_reproduction_assets"
    sweep_root = Path(cfg.get("sweep_dir", "outputs/cancer/pretraining_sweep"))
    output.mkdir(parents=True, exist_ok=True)
    reproduced = _summaries(root); variants = VARIANT_NAMES
    header = ["year"]
    for v in variants: header.extend((f"paper_{v}", f"reproduced_{v}", f"delta_{v}"))
    rows = []; fold_rows = []
    for y in YEARS:
        row: list[Any] = [y]
        for i, v in enumerate(variants):
            paper = PAPER_TABLE1[y][i]; item = reproduced.get(f"{y}year/{v}")
            value = float(item["mean_auc"]) if item else math.nan
            row.extend((paper, value, value - paper if math.isfinite(value) else math.nan))
            if item: fold_rows.append([y, v, *item["fold_auc"], item["mean_auc"], item["std_auc"]])
        rows.append(row)
    tsv(output / "table1_comparison.tsv", header, rows)
    tsv(output / "table1_fold_auc.tsv",
         ["year", "variant", "fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "mean", "std"], fold_rows)
    (output / "table1_comparison.md").write_text("# Table 1 reproduction\n\n" + mdtable(header, rows) + "\n")
    # Table 1 is ROC-AUC because that is what the manuscript reports; the same
    # folds scored at a 0.5 decision threshold go into their own table.
    threshold_header = ["year", "variant", "mean_auc", "mean_accuracy", "mean_precision",
                        "mean_recall", "mean_f1"]
    threshold_rows = []
    for y in YEARS:
        for v in variants:
            item = reproduced.get(f"{y}year/{v}")
            if item:
                threshold_rows.append([
                    y, v, item["mean_auc"],
                    *[item.get(f"mean_{m}", math.nan)
                      for m in ("accuracy", "precision", "recall", "f1")],
                ])
    tsv(output / "table1_threshold_metrics.tsv", threshold_header, threshold_rows)
    counts = {}; count_rows = []
    group_codes = {}
    for y in YEARS:
        task = dataset.task(f"{y}year")
        labels = task.labels(); codes = np.asarray(task.groups()); group_codes[y] = codes
        counts[y] = {}
        for code, name in enumerate(CANCER_TYPES):
            selected = labels[codes == code]
            death = int((selected == 0).sum()); survival = int((selected == 1).sum())
            counts[y][name] = (len(selected), death, survival)
            count_rows.append([name, y, len(selected), death, survival])
    tsv(output / "supplementary_table1_sample_counts.tsv",
         ["cancer_type", "year", "total", "death", "survival"], count_rows)
    per = {}; per_rows = []
    for y in YEARS:
        codes = group_codes[y]
        for v in variants:
            try: target, prob, indices = _predictions(root, y, v)
            except FileNotFoundError: continue
            vals = {}
            for code, name in enumerate(CANCER_TYPES):
                mask = codes[indices] == code
                vals[name] = _auc(target[mask], prob[mask]) if mask.any() else math.nan
                per_rows.append([y, v, name, int(mask.sum()), vals[name]])
            per[f"{y}year/{v}"] = vals
    (output / "per_cancer_auc.json").write_text(json.dumps(per, indent=2, allow_nan=True))
    tsv(output / "per_cancer_auc.tsv", ["year", "variant", "cancer_type", "samples", "roc_auc"], per_rows)
    sweep_rows = []
    for path in sorted(sweep_root.glob("epoch_*/*/5year/gnn_dnn/summary.json")):
        item = json.loads(path.read_text())
        sweep_rows.append([int(path.parts[-5].replace("epoch_", "")), path.parts[-4],
                           *item["fold_auc"], item["mean_auc"], item["std_auc"]])
    tsv(output / "figure2_pretraining_sweep.tsv",
         ["pretrain_epoch", "mode", "fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "mean", "std"],
         sweep_rows)
    made = _plots(output, assets, root, reproduced, per, counts,
                  Path(cfg.get("pretrain_history", "outputs/cancer/pretrain_50/history.json")),
                  Path(cfg.get("ig_dir", "outputs/cancer/ig")), sweep_root)
    status = "complete" if len(reproduced) == 20 else f"incomplete ({len(reproduced)}/20 conditions)"
    summary_counts = [
        [y, sum(v[0] for v in counts[y].values()), sum(v[1] for v in counts[y].values()),
         sum(v[2] for v in counts[y].values())] for y in YEARS
    ]
    figure_block = figures(made, "cancer_reproduction_assets")
    num_genes = dataset.manifest["source"].get("num_genes")
    md = f"""# Inoue et al. cancer prognosis reproduction

## Reproduction status

Table 1 grid status: **{status}**. The workflow covers five verification years,
four model variants, five stratified folds, 150 fine-tuning epochs, and the
manuscript final-epoch evaluation protocol. Completed fold artifacts are reused.

## Table 1: published and reproduced ROC-AUC

{mdtable(header, rows)}

NA means that a full condition has not completed. Fold values are exported to
outputs/cancer/report/table1_fold_auc.tsv.

## Threshold metrics at 0.5

{mdtable(threshold_header, threshold_rows)}

Table 1 compares ROC-AUC because that is the metric the manuscript reports. These are the same
folds and the same held-out predictions scored at a fixed 0.5 decision threshold, so they describe
the operating point rather than the ranking. The labels are imbalanced and shift with the
verification year (88.6% survival at 1 year, 34.9% at 5), so accuracy is not comparable across
years; also exported to outputs/cancer/report/table1_threshold_metrics.tsv.

## Supplementary Table 1: data audit

{mdtable(["year", "total", "death", "survival"], summary_counts)}

The prepared dataset contains {num_genes:,} expression features, {len(CANCER_TYPES)} cancer
types, and a graph with {dataset.num_nodes:,} stored nodes,
{dataset.manifest['num_edges']:,} directed edges and {dataset.num_relations} relations.
Cancer-level counts are in outputs/cancer/report/supplementary_table1_sample_counts.tsv.

## Reproduction plots

{figure_block}

## Exact commands

    conda activate gnn
    bash scripts/cancer/reproduce_paper.sh prepare
    bash scripts/cancer/reproduce_paper.sh pretrain
    python scripts/cancer/reproduce_table1.py --gpus 0,1,2
    bash scripts/cancer/reproduce_paper.sh figure2
    bash scripts/cancer/reproduce_paper.sh ig
    bash scripts/cancer/reproduce_paper.sh report

Preprocessing is a separate step: `pathwaygnn-data cancer-prepare` writes the
generic dataset under data_cancer/prepared, and every `pathwaygnn` command then
selects it through the `dataset:` block of configs/cancer/dataset.yaml. The
Table 1 runner schedules all 20 conditions over GPUs and resumes at fold level.
selection: final_epoch follows the manuscript. best_test_auc is only for
public-code compatibility and uses the held-out fold for selection.

## Ensembl-to-HGNC conversion boundary

counts_gene.tsv contains 11,285 expression columns but no Ensembl identifier
row or column. A separately supplied ordered Ensembl ID file can be mapped by
`pathwaygnn-data cancer-map-ids` through MyGene.info, with version stripping,
ambiguity flags, and local cache files. Without that missing list and historical
MSigDB/LM22 snapshots, the supplied {num_genes:,}-gene matrices are the exact
compatibility input.

## Interpretation scope

The workflow generates Table 1, Supplementary Table 1, Figure 2 sweep, fold and cancer AUC
tables, Figure 3 panels, training diagnostics, and Figure 4 when attribution
arrays exist. Historical DAVID enrichment p-values depend on an external
database release; ranked gene lists are exported, but exact historical
p-values are not asserted.
"""
    markdown_path, html_path = write_document(docs, "cancer_reproduction", md, "Cancer reproduction")
    result = {
        "status": status,
        "conditions": len(reproduced),
        "tables": sorted(str(p) for p in output.glob("*.tsv")),
        "plots": [str(output / n) for n in made],
        "markdown": str(markdown_path),
        "html": str(html_path),
    }
    (output / "report.json").write_text(json.dumps(result, indent=2))
    return result
