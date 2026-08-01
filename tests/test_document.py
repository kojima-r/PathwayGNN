from pathlib import Path

import numpy as np

from pathwaygnn_datasets.document import fmt, markdown_document, mdtable, tsv, write_document


def test_fmt_rounds_floats_and_marks_missing() -> None:
    assert fmt(0.123456) == "0.1235"
    assert fmt(np.float32(1)) == "1.0000"
    assert fmt(float("nan")) == "NA"
    assert fmt(float("inf")) == "NA"
    assert fmt(7) == "7" and fmt("kd_inh") == "kd_inh"


def test_tables_render_in_both_formats(tmp_path: Path) -> None:
    header = ["task", "auc"]
    rows = [["kd_inh", 0.5059], ["oe_act", float("nan")]]
    markdown = mdtable(header, rows)
    assert markdown.splitlines() == [
        "| task | auc |",
        "|---|---|",
        "| kd_inh | 0.5059 |",
        "| oe_act | NA |",
    ]
    tsv(tmp_path / "t.tsv", header, rows)
    assert (tmp_path / "t.tsv").read_text().splitlines()[1] == "kd_inh\t0.5059"

    page = markdown_document(f"# Title\n\n## Section\n\n{markdown}\n", "Report")
    assert "<title>Report</title>" in page
    assert "<h1>Title</h1>" in page and "<h2>Section</h2>" in page
    assert page.count("<table>") == 1 and "<td>0.5059</td>" in page
    assert "|" not in page.split("<body>")[1]


def test_document_renders_code_figures_and_emphasis(tmp_path: Path) -> None:
    text = (
        "# T\n\nA **bold** claim with <escaped> markup.\n\n"
        "    conda activate gnn\n    pathwaygnn cv --config x.yaml\n\n"
        "![plot.png](assets/plot.png)\n"
    )
    page = markdown_document(text, "T")
    assert "<strong>bold</strong>" in page
    assert "&lt;escaped&gt;" in page
    assert "<pre><code>conda activate gnn\npathwaygnn cv --config x.yaml</code></pre>" in page
    assert '<img src="assets/plot.png"' in page and "<figcaption>plot.png</figcaption>" in page


def test_write_document_emits_both_files(tmp_path: Path) -> None:
    markdown_path, html_path = write_document(tmp_path / "docs", "tr_report", "# T\n\nbody\n", "T")
    assert markdown_path == tmp_path / "docs" / "tr_report.md"
    assert html_path == tmp_path / "docs" / "tr_report.html"
    assert markdown_path.read_text() == "# T\n\nbody\n"
    assert "<p>body</p>" in html_path.read_text()
