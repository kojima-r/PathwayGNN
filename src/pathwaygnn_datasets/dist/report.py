"""Reporting for the graph-partitioning benchmark.

Renders what ``pathwaygnn dist-benchmark`` measured — partitioning cost, per-step
time, and peak memory across the ``num_parts`` x ``parts_per_batch`` grid — into
tables, figures and one document.

Unlike the corpus reports next door this presentation is not dataset-specific:
it describes the *engine's* behaviour on whichever graphs the benchmark was
pointed at. It lives here anyway because this is where document rendering
(:mod:`pathwaygnn_datasets.document`) and every generated document under ``docs/``
belong, and because the engine may not import from this package.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from pathwaygnn_datasets.document import mdtable, tsv, write_document

TITLE = "PathwayGNN graph-partitioning benchmark"
ROW_FIELDS = (
    "dataset", "mode", "num_parts", "parts_per_batch", "nodes_per_step", "edges_per_step",
    "edges_per_step_min", "edges_per_step_max", "edges_per_pass", "edge_coverage",
    "steps_per_pass", "load_ms", "step_ms", "peak_mib", "pass_seconds",
)
PARTITION_FIELDS = (
    "dataset", "num_parts", "seconds", "disk_mib", "nodes_per_part_min", "nodes_per_part_max",
    "nodes_per_part_mean", "internal_edge_fraction", "edgeless_parts",
)
COLORS = ("#4c78a8", "#54a24b", "#f58518", "#e45756", "#b279a2", "#9d755d", "#72b7b2")


def _number(value: Any) -> float:
    """``None`` becomes nan so `document.fmt` renders it as ``NA``."""
    return math.nan if value is None else float(value)


def _rows_for(results: dict[str, Any], dataset: str, mode: str) -> list[dict[str, Any]]:
    return [
        row for row in results["rows"] if row["dataset"] == dataset and row["mode"] == mode
    ]


def _baseline(results: dict[str, Any], dataset: str) -> dict[str, Any] | None:
    rows = _rows_for(results, dataset, "full_graph")
    return rows[0] if rows else None


def _grid(results: dict[str, Any], dataset: str) -> tuple[list[int], list[int]]:
    """The (num_parts, parts_per_batch) values actually measured for this dataset."""
    rows = _rows_for(results, dataset, "partitioned")
    return (
        sorted({int(row["num_parts"]) for row in rows}),
        sorted({int(row["parts_per_batch"]) for row in rows}),
    )


def _pivot(
    results: dict[str, Any], dataset: str, field: str, integer: bool = False,
    percent: bool = False,
) -> tuple[list[str], list[list[Any]]]:
    """``num_parts`` down the rows, ``parts_per_batch`` across the columns.

    ``integer`` keeps counts out of `document.fmt`'s four-decimal float format and
    ``percent`` scales a fraction to a percentage; an unmeasured cell stays nan
    either way so it renders as ``NA``.
    """
    parts, batches = _grid(results, dataset)
    lookup = {
        (int(row["num_parts"]), int(row["parts_per_batch"])): row
        for row in _rows_for(results, dataset, "partitioned")
    }
    header = ["num_parts"] + [f"x{batch}" for batch in batches]
    rows: list[list[Any]] = []
    for count in parts:
        line: list[Any] = [count]
        for batch in batches:
            row = lookup.get((count, batch))
            if row is None or row[field] is None:
                line.append(math.nan)
            else:
                value = float(row[field]) * (100.0 if percent else 1.0)
                line.append(int(value) if integer else value)
        rows.append(line)
    return header, rows


def _figure_line(text: str, name: str, assets: str) -> str:
    """A figure with an explicit caption.

    ``document.figures`` derives captions from file names, which is right for the
    corpus reports; here each figure is a five-panel composite that needs a
    sentence naming the graph and the quantities plotted.
    """
    return f"![{text}]({assets}/{name})"


def _headline(results: dict[str, Any], dataset: str, baseline: dict[str, Any]) -> str:
    """State the trade-off at its extreme, since that is what decides feasibility."""
    rows = [row for row in _rows_for(results, dataset, "partitioned") if row["peak_mib"]]
    if not rows:
        return ""
    cheapest = min(rows, key=lambda row: row["peak_mib"])
    memory_factor = _number(baseline["peak_mib"]) / _number(cheapest["peak_mib"])
    time_factor = _number(cheapest["pass_seconds"]) / _number(baseline["pass_seconds"])
    return (
        f"At the far end of the grid (`num_parts: {cheapest['num_parts']}`, "
        f"`parts_per_batch: {cheapest['parts_per_batch']}`) a step needs "
        f"{_number(cheapest['peak_mib']):,.0f} MiB, i.e. **{memory_factor:,.0f}x less memory** "
        f"than the full graph -- but a pass over the graph takes {time_factor:,.0f}x longer "
        f"({_number(cheapest['pass_seconds']):,.1f} s against "
        f"{_number(baseline['pass_seconds']):,.2f} s) and sees only "
        f"{_number(cheapest['edge_coverage']) * 100:,.1f}% of its edges. That is the trade the "
        f"mode makes: memory, paid for in wall time and fidelity."
    )


def _plots(
    output: Path, assets: Path, results: dict[str, Any], labels: dict[str, Any]
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

    names = [item["name"] for item in results["datasets"]]
    has_memory = any(row["peak_mib"] is not None for row in results["rows"])

    for dataset in names:
        parts, batches = _grid(results, dataset)
        if not parts:
            continue
        rows = _rows_for(results, dataset, "partitioned")
        lookup = {(int(r["num_parts"]), int(r["parts_per_batch"])): r for r in rows}
        baseline = _baseline(results, dataset)
        panels = ["peak_mib", "step_ms", "pass_seconds", "edge_coverage"]
        if not has_memory:
            panels.remove("peak_mib")
        axis_labels = {
            "peak_mib": "Peak memory per step (MiB)",
            "step_ms": "Median step time (ms)",
            "pass_seconds": "Derived time for one pass (s)",
            "edge_coverage": "Edges a pass sees (% of graph)",
        }
        scales = {"edge_coverage": 100.0}
        fig, axes = plt.subplots(1, len(panels) + 1, figsize=(5.6 * (len(panels) + 1), 4.4))
        for axis, field in zip(axes, panels):
            for position, batch in enumerate(batches):
                scale = scales.get(field, 1.0)
                series = [(p, lookup[(p, batch)][field]) for p in parts if (p, batch) in lookup]
                axis.plot(
                    [p for p, _ in series], [_number(v) * scale for _, v in series],
                    marker="o", color=COLORS[position % len(COLORS)],
                    label=f"parts_per_batch={batch}",
                )
            if baseline is not None and field != "pass_seconds":
                axis.axhline(
                    _number(baseline[field]) * scales.get(field, 1.0), ls="--", color="black",
                    lw=1, label="full graph",
                )
            # Anchored at zero: the step-time panel spans a few percent, and an
            # autoscaled axis would render that timing noise as structure.
            axis.set(xscale="log", xticks=parts, xlabel="num_parts", ylabel=axis_labels[field],
                     title=f"{dataset}: {axis_labels[field]}", ylim=(0, None))
            axis.set_xticklabels([str(p) for p in parts])
            axis.grid(alpha=.3)
            axis.legend(fontsize=7)
        # The mechanism itself: cost is set by the subgraph a step sees, not by num_parts.
        axis = axes[-1]
        field = "peak_mib" if has_memory else "step_ms"
        axis.scatter(
            [row["nodes_per_step"] for row in rows], [_number(row[field]) for row in rows],
            color=COLORS[0], label="partitioned", zorder=3,
        )
        if baseline is not None:
            axis.scatter(
                [baseline["nodes_per_step"]], [_number(baseline[field])],
                color=COLORS[3], marker="*", s=180, label="full graph", zorder=3,
            )
        axis.set(xlabel="Nodes per step", ylabel=axis_labels[field],
                 title=f"{dataset}: {axis_labels[field]} vs subgraph size")
        axis.grid(alpha=.3)
        axis.legend(fontsize=8)
        save(fig, f"cost_{dataset}.png")

    # Partitioning cost and how much of the graph survives the cut.
    partitioning = results["partitioning"]
    if partitioning:
        fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.4))
        for position, dataset in enumerate(names):
            series = [item for item in partitioning if item["dataset"] == dataset]
            if not series:
                continue
            counts = [item["num_parts"] for item in series]
            color = COLORS[position % len(COLORS)]
            title = labels.get(dataset, {}).get("title")
            legend = f"{dataset} ({title})" if title else dataset
            axes[0].plot(counts, [item["seconds"] for item in series], marker="o",
                         color=color, label=legend)
            axes[1].plot(counts, [item["disk_mib"] for item in series], marker="o",
                         color=color, label=legend)
            axes[2].plot(counts, [item["internal_edge_fraction"] * 100 for item in series],
                         marker="o", color=color, label=legend)
        for axis, ylabel, title in (
            (axes[0], "METIS wall time (s)", "Cost of cutting the graph"),
            (axes[1], "Partition files (MiB)", "On-disk size"),
            (axes[2], "Edges inside a partition (%)", "How much of the graph survives the cut"),
        ):
            counts = sorted({item["num_parts"] for item in partitioning})
            axis.set(xscale="log", xticks=counts, xlabel="num_parts", ylabel=ylabel, title=title)
            axis.set_xticklabels([str(count) for count in counts])
            axis.grid(alpha=.3)
            axis.legend(fontsize=8)
        save(fig, "partitioning_cost.png")

    assets.mkdir(parents=True, exist_ok=True)
    for name in made:
        shutil.copy2(output / name, assets / name)
    return made


def run_dist_report(cfg: dict[str, Any]) -> dict[str, Any]:
    results_path = Path(cfg.get("results", "outputs/dist/benchmark/results.json"))
    if not results_path.is_file():
        raise FileNotFoundError(
            f"{results_path} is missing; run `pathwaygnn dist-benchmark --config "
            "configs/dist/benchmark.yaml` first"
        )
    results = json.loads(results_path.read_text())
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    document_name = cfg.get("document", "dist_report")
    assets_name = f"{document_name}_assets"
    docs = Path(cfg.get("docs_dir", "docs"))

    tsv(output / "configurations.tsv", ROW_FIELDS,
        [[row.get(field) for field in ROW_FIELDS] for row in results["rows"]])
    tsv(output / "partitioning.tsv", PARTITION_FIELDS,
        [[item.get(field) for field in PARTITION_FIELDS] for item in results["partitioning"]])

    # Human-readable names come from this config, not from the measurement.
    labels = cfg.get("labels") or {}
    made = _plots(output, docs / assets_name, results, labels)

    environment = results["environment"]
    settings = results["settings"]
    names = [item["name"] for item in results["datasets"]]
    has_memory = any(row["peak_mib"] is not None for row in results["rows"])

    # --- the graphs themselves, as one table rather than a line of prose each ----
    summary_header = [
        "Dataset", "Graph", "Nodes", "Relations", "Edges", "Encoder parameters",
        "Full-graph step (ms)", "Full-graph peak (MiB)",
    ]
    summary_rows = []
    for info in results["datasets"]:
        baseline = _baseline(results, info["name"])
        summary_rows.append([
            f"`{info['name']}`",
            labels.get(info["name"], {}).get("title") or info["name"],
            f"{info['num_nodes']:,}",
            info["num_relations"],
            f"{info['num_edges']:,}",
            f"{info['num_parameters']:,}",
            "NA" if baseline is None else f"{_number(baseline['step_ms']):,.1f}",
            "NA" if baseline is None or not has_memory
            else f"{_number(baseline['peak_mib']):,.0f}",
        ])
    graphs_block = ["## Graphs under test", "", mdtable(summary_header, summary_rows), ""]
    for info in results["datasets"]:
        label = labels.get(info["name"], {})
        if label.get("note"):
            title = label.get("title") or info["name"]
            graphs_block += [f"**`{info['name']}` — {title}.** {label['note']}", ""]
    graphs_block += [
        "The last two columns are the un-partitioned baseline every partitioned number below is "
        "read against: one step over the whole graph, on the same model and optimizer. They are "
        "what decides whether partitioning is needed at all.",
        "",
    ]

    # --- cutting the graph: one cross-graph section, with its figure beside it ---
    cut_block = ["## Cutting the graph with METIS", ""]
    cut_block += [
        "This is the one-off cost, paid by `pathwaygnn partition` before training. "
        "`edges inside a partition` is the share of edges whose two endpoints land in the same "
        "partition; the rest are only seen when their two partitions share a batch, which is what "
        "the coverage tables further down measure.",
        "",
    ]
    if "partitioning_cost.png" in made:
        cut_block += [
            _figure_line(
                "METIS cut cost and quality for both graphs: wall time, the size of the "
                "partition files on disk, and the share of edges that stay inside a partition",
                "partitioning_cost.png", assets_name,
            ),
            "",
        ]
    cut_block += [
        mdtable(
            ["Dataset", "num_parts", "METIS s", "disk MiB", "nodes/part (mean)", "min", "max",
             "edges inside a partition (%)", "edgeless parts"],
            [
                [f"`{item['dataset']}`", item["num_parts"], item["seconds"], item["disk_mib"],
                 item["nodes_per_part_mean"], item["nodes_per_part_min"],
                 item["nodes_per_part_max"], item["internal_edge_fraction"] * 100,
                 item["edgeless_parts"]]
                for item in results["partitioning"]
            ],
        ),
        "",
    ]

    budget_sentence = (
        "**Peak memory per step** fixes what fits,"
        if has_memory
        else "peak memory was not measurable on this device, so start from the step cost;"
    )

    sections: list[str] = []
    for dataset in names:
        parts, batches = _grid(results, dataset)
        if not parts:
            continue
        baseline = _baseline(results, dataset)
        info = next(item for item in results["datasets"] if item["name"] == dataset)
        title = labels.get(dataset, {}).get("title") or dataset
        block = [f"## Dataset `{dataset}` — {title}", ""]
        if baseline is not None:
            memory = (
                f"{_number(baseline['peak_mib']):,.0f} MiB peak" if has_memory
                else "peak memory unavailable on CPU"
            )
            block += [
                f"**Full-graph baseline** (no `training.partition` block): one step sees all "
                f"{baseline['nodes_per_step']:,} nodes and {baseline['edges_per_step']:,} edges, "
                f"costs {_number(baseline['step_ms']):,.1f} ms and {memory}.",
                "",
            ]
            if has_memory:
                block += [_headline(results, dataset, baseline), ""]
        if f"cost_{dataset}.png" in made:
            block += [
                f"The four line panels below plot the `{dataset}` tables in this section — peak "
                "memory, step time, time per pass and edge coverage, each against `num_parts` with "
                "one line per `parts_per_batch`, and the full-graph baseline as the dashed line. "
                "The fifth panel drops `num_parts` and plots cost directly against the number of "
                "nodes a step sees, which is what the two knobs actually control.",
                "",
                _figure_line(
                    f"`{dataset}` ({title}): peak memory, step time, time per pass and edge "
                    "coverage against num_parts, one line per parts_per_batch, with the "
                    "full-graph baseline dashed; and cost against the nodes a step sees",
                    f"cost_{dataset}.png", assets_name,
                ),
                "",
            ]
        block += [
            "### What a step sees",
            "",
            "Columns are `parts_per_batch`.",
            "",
            "Nodes per step (mean over a pass):",
            "",
        ]
        header, rows = _pivot(results, dataset, "nodes_per_step", integer=True)
        block += [mdtable(header, rows), "", "Edges per step (mean over a pass; METIS "
                  "balances nodes, not edges, so single batches vary widely -- the TSV carries "
                  "the min and max):", ""]
        header, rows = _pivot(results, dataset, "edges_per_step")
        block += [
            mdtable(header, rows), "",
            "Fraction of the graph's edges a pass sees at all (%). An edge is only visible "
            "when both endpoints land in the same batch, so this is the fidelity the "
            "configuration trains at; `shuffle: true` varies which edges those are per epoch:",
            "",
        ]
        header, rows = _pivot(results, dataset, "edge_coverage", percent=True)
        block += [mdtable(header, rows), "", "Steps per pass over all partitions:", ""]
        header, rows = _pivot(results, dataset, "steps_per_pass", integer=True)
        block += [mdtable(header, rows), ""]
        if has_memory:
            block += ["### Peak memory per step (MiB)", ""]
            header, rows = _pivot(results, dataset, "peak_mib")
            block += [mdtable(header, rows), ""]
        block += ["### Median step time (ms)", ""]
        header, rows = _pivot(results, dataset, "step_ms")
        block += [mdtable(header, rows), "", "Median batch-load time (ms), i.e. reading the "
                  "partition files and building the subgraph:", ""]
        header, rows = _pivot(results, dataset, "load_ms")
        block += [mdtable(header, rows), "", "### Derived time for one pass over the graph (s)", "",
                  "`steps_per_pass x (step + load)`, single process. Under `torchrun` the steps "
                  "are divided across ranks.", ""]
        header, rows = _pivot(results, dataset, "pass_seconds")
        block += [mdtable(header, rows), ""]
        sections.append("\n".join(block))

    md = f"""# {TITLE}

Graph partitioning (`training.partition` in `configs/*/pretrain_partitioned.yaml`) trades
fidelity for memory: more partitions mean smaller subgraphs per step and therefore less memory,
but also more edges cut and more steps per pass over the graph. This document is the measurement
of that trade-off, produced by `pathwaygnn dist-benchmark` and rendered by
`pathwaygnn-data dist-report`. It is generated -- edit the modules, not this file.

## How it was measured

| Setting | Value |
|---|---|
| Device | {environment['device']} ({environment['device_name']}) |
| PyTorch / Python | {environment['torch']} / {environment['python']} |
| Model | hidden_dim {settings['model']['hidden_dim']}, num_layers {settings['model']['num_layers']}, dropout {settings['model']['dropout']} |
| Positive edges per step | {settings['batch_size']} |
| Timed steps per configuration | {settings['steps']} (median), after {settings['warmup']} warmup steps |
| DataLoader workers | {settings['num_workers']} |
| Memory metric | {environment['memory_metric']} |
| Grid | num_parts {settings['num_parts']} x parts_per_batch {settings['parts_per_batch']} |

Each timed step is one forward+backward+optimizer step of the real pre-training path, and the
peak memory is measured with parameters, gradients and optimizer state already resident, so it is
what has to fit rather than activations alone. Configurations with
`parts_per_batch > num_parts` are not measurable and are listed as skipped
({len(results['skipped'])} of them). One pass over the graph is **derived** from the median
per-step cost rather than timed end to end, so that the whole grid stays cheap to re-run.

{chr(10).join(graphs_block)}
{chr(10).join(cut_block)}
{"".join(f"{chr(10)}{section}{chr(10)}" for section in sections)}
## Reading the numbers

- **Memory tracks the subgraph a step sees, not `num_parts` itself.** `num_parts` and
  `parts_per_batch` only matter through their ratio: doubling `num_parts` at fixed
  `parts_per_batch` halves the nodes per step, and doubling `parts_per_batch` puts them back.
  The rightmost panel of each graph's figure is that relationship, with the full-graph point on
  the same axes.
- **Per-step time falls much less than memory.** One `RelationalGIN` layer runs one GINConv per
  relation whatever the subgraph size, so a graph with many relations pays a fixed per-step cost
  that shrinking the subgraph cannot remove. Time per *pass over the graph* therefore rises with
  `num_parts` even as memory falls -- partitioning buys memory, and buys it with wall time.
- **Cutting more finely cuts more edges.** The `edges inside a partition` column is the fraction
  of edges both of whose endpoints land in the same partition. Edges between two partitions are
  still trained on whenever those partitions share a batch, which is why `shuffle: true` matters:
  it varies the pairings across epochs.
- **Batch-load time is real and grows with `parts_per_batch`.** It is measured with
  `num_workers: 0` so it shows up; set `training.partition.num_workers` above zero in a real run
  and it overlaps with compute.
- **The partitioned path carries a small fixed overhead**, so where a batch covers most of the
  graph it can need slightly *more* memory than the full-graph path -- visible wherever a line
  sits above the dashed baseline. It gathers the subgraph's rows out of the embedding table, and
  that gather plus its gradient is an extra pair of tensors the full-graph path does not
  materialize. It is only worth paying where the ratio actually shrinks the subgraph.

## Choosing a configuration

Work from each graph's own section: {budget_sentence} **Fraction of the graph's edges a pass
sees** says what that costs in fidelity, and **Derived time for one pass** says what it costs in
wall time. `Graphs under test` holds the un-partitioned upper bound on both fidelity and memory. A configuration is only worth using when the full-graph row does not fit -- the
partitioned objective is not the same objective, so it is a mechanism for graphs that do not fit,
not a speedup for graphs that do. Note how differently the two corpora here land: the graph with
356 relations needs 14 GiB un-partitioned and is the case the mode exists for, while the one with
13 relations already fits in 1.5 GiB and has nothing to gain.

Under `torchrun` the steps of a pass are divided across ranks
(`DistributedSampler`), so wall time per epoch falls with the rank count while the per-step
memory above stays as measured.
"""
    markdown_path, html_path = write_document(docs, document_name, md, TITLE)
    return {
        "markdown": str(markdown_path),
        "html": str(html_path),
        "figures": made,
        "tables": [str(output / "configurations.tsv"), str(output / "partitioning.tsv")],
        "datasets": names,
        "configurations": len(results["rows"]),
    }
