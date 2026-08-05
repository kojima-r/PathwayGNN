"""The partition-count benchmark and the document it renders."""

import json
from pathlib import Path

import pytest

from pathwaygnn.data.format import GraphDataset
from pathwaygnn.training.dist_benchmark import run_dist_benchmark
from pathwaygnn_datasets.dist.report import run_dist_report

NUM_PARTS = [2, 4]
PARTS_PER_BATCH = [1, 2, 8]


@pytest.fixture
def results(dataset: GraphDataset, tmp_path: Path) -> dict:
    summary = run_dist_benchmark({
        "datasets": [{"name": dataset.name, "dir": str(dataset.root)}],
        "num_parts": NUM_PARTS,
        "parts_per_batch": PARTS_PER_BATCH,
        "model": {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0},
        "batch_size": 16,
        "steps": 2,
        "warmup": 1,
        "device": "cpu",
        "partition_root": str(tmp_path / "parts"),
        "output_dir": str(tmp_path / "benchmark"),
    })
    return json.loads(Path(summary["output"]).read_text())


def test_benchmark_covers_the_grid_and_reports_what_it_skipped(
    dataset: GraphDataset, results: dict
):
    partitioned = [row for row in results["rows"] if row["mode"] == "partitioned"]
    assert [row["num_parts"] for row in results["partitioning"]] == NUM_PARTS
    # parts_per_batch: 8 exceeds both partition counts, so it is skipped, not silently dropped.
    assert len(partitioned) == 2 + 2
    assert {entry["parts_per_batch"] for entry in results["skipped"]} == {8}
    baseline = [row for row in results["rows"] if row["mode"] == "full_graph"]
    assert len(baseline) == 1
    assert baseline[0]["nodes_per_step"] == dataset.num_nodes
    assert baseline[0]["edges_per_step"] == int(dataset.manifest["num_edges"])


def test_measured_subgraph_sizes_are_consistent(results: dict):
    for row in results["rows"]:
        assert row["step_ms"] > 0
        assert row["steps_per_pass"] >= 1
        if row["mode"] != "partitioned":
            continue
        # Selecting every partition of a cut reproduces the whole graph, so that
        # configuration must land exactly on the full-graph row.
        if row["num_parts"] == row["parts_per_batch"]:
            baseline = next(r for r in results["rows"] if r["mode"] == "full_graph")
            assert row["nodes_per_step"] == baseline["nodes_per_step"]
            assert row["edges_per_step"] == baseline["edges_per_step"]
            assert row["edge_coverage"] == 1.0
        else:
            # Cutting loses edges; a pass can never see more than the whole graph.
            assert 0.0 < row["edge_coverage"] <= 1.0
        assert row["steps_per_pass"] == -(-row["num_parts"] // row["parts_per_batch"])
        assert row["edges_per_step_min"] <= row["edges_per_step"] <= row["edges_per_step_max"]
    # No CUDA in the test environment, so the memory column is absent, not zero.
    assert all(row["peak_mib"] is None for row in results["rows"])


def test_report_writes_markdown_and_matching_html(results: dict, tmp_path: Path):
    docs = tmp_path / "docs"
    summary = run_dist_report({
        "results": str(tmp_path / "benchmark" / "results.json"),
        "labels": {"toy": {"title": "A toy graph", "note": "Synthetic, for the test suite."}},
        "output_dir": str(tmp_path / "report"),
        "docs_dir": str(docs),
        "document": "dist_report",
    })
    markdown = Path(summary["markdown"]).read_text()
    html = Path(summary["html"]).read_text()
    assert "# PathwayGNN graph-partitioning benchmark" in markdown
    assert "<h1>PathwayGNN graph-partitioning benchmark</h1>" in html
    # Every table in the Markdown has to survive into the HTML.
    assert markdown.count("| num_parts |") == html.count("<th>num_parts</th>")
    # The memory sections drop out when the benchmark ran on CPU rather than
    # rendering a column of zeros.
    assert "### Peak memory per step (MiB)" not in markdown
    assert "not measurable on this device" in markdown
    assert "unavailable on CPU" in markdown
    for name in summary["figures"]:
        assert (docs / "dist_report_assets" / name).is_file()
        # Referenced relative to the document, so docs/ stays movable.
        assert f"](dist_report_assets/{name})" in markdown
    for table in summary["tables"]:
        assert Path(table).is_file()
    assert results["settings"]["num_parts"] == NUM_PARTS


def test_report_names_the_graphs_and_places_figures_with_them(results: dict, tmp_path: Path):
    """The document has to say what a dataset key is, and show each figure in context."""
    docs = tmp_path / "docs"
    summary = run_dist_report({
        "results": str(tmp_path / "benchmark" / "results.json"),
        "labels": {"toy": {"title": "A toy graph", "note": "Synthetic, for the test suite."}},
        "output_dir": str(tmp_path / "report2"),
        "docs_dir": str(docs),
        "document": "named",
    })
    markdown = Path(summary["markdown"]).read_text()
    # The basic properties of every graph, collected in one table up front.
    assert "## Graphs under test" in markdown
    assert "| Dataset | Graph | Nodes | Relations | Edges |" in markdown
    assert "| `toy` | A toy graph |" in markdown
    assert "Synthetic, for the test suite." in markdown
    # The dataset section names the graph, not just its key.
    assert "## Dataset `toy` — A toy graph" in markdown
    # Each figure sits inside the section it belongs to, not in a trailing dump.
    cut = markdown.index("## Cutting the graph with METIS")
    section = markdown.index("## Dataset `toy`")
    reading = markdown.index("## Reading the numbers")
    assert cut < markdown.index("](named_assets/partitioning_cost.png)") < section
    assert section < markdown.index("](named_assets/cost_toy.png)") < reading
    # ... so nothing is left over after the closing discussion.
    assert "](named_assets/" not in markdown[reading:]


def test_report_falls_back_to_the_bare_dataset_key(results: dict, tmp_path: Path):
    summary = run_dist_report({
        "results": str(tmp_path / "benchmark" / "results.json"),
        "output_dir": str(tmp_path / "report3"),
        "docs_dir": str(tmp_path / "docs3"),
        "document": "unlabelled",
    })
    markdown = Path(summary["markdown"]).read_text()
    assert "## Dataset `toy` — toy" in markdown
    assert "| `toy` | toy |" in markdown


def test_report_refuses_to_render_without_measurements(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="dist-benchmark"):
        run_dist_report({
            "results": str(tmp_path / "missing.json"),
            "output_dir": str(tmp_path / "report"),
            "docs_dir": str(tmp_path / "docs"),
        })
