import struct
import zlib
from pathlib import Path

import numpy as np

from pathwaygnn_datasets.document import (
    caption,
    figures,
    fmt,
    markdown_document,
    mdtable,
    repo_path,
    tsv,
    write_document,
)


def _png(path: Path, width: int, height: int) -> None:
    """A valid 1-colour PNG of a given size, so figure placement can be tested."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    raw = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


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
    assert "<h1>Title</h1>" in page and '<h2 id="section">Section</h2>' in page
    assert page.count("<table>") == 1 and "<td>0.5059</td>" in page
    # Tables are wrapped so a wide one scrolls instead of squeezing the page.
    assert '<div class="tablewrap">' in page
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
    # Every figure links to the original file, which is the only way to read a
    # multi-panel PNG at full resolution.
    assert '<a href="assets/plot.png" title="Open at full size">' in page


def test_inline_markdown_does_not_leak(tmp_path: Path) -> None:
    """The reports write links, code spans, italics and lists; all must render."""
    text = (
        "# T\n\n## S\n\n### Deeper\n\n"
        "See [the config](README_config.md) for `training.partition` and *emphasis*.\n\n"
        "- **First** item that wraps\n  onto a second line\n"
        "- Second item with `a_code_span` and **bold**\n\n"
        "1. Step one\n2. Step two\n"
    )
    page = markdown_document(text, "T")
    body = page.split("<body>")[1]
    assert '<h3 id="deeper">Deeper</h3>' in body
    assert '<a href="README_config.md">the config</a>' in body
    assert "<code>training.partition</code>" in body
    assert "<em>emphasis</em>" in body
    assert body.count("<li>") == 4 and "<ul>" in body and "<ol>" in body
    assert "onto a second line" in body  # the continuation joined its item
    # An underscore inside a code span is not emphasis.
    assert "<code>a_code_span</code>" in body
    for leak in ("###", "](", "**", "- **", "1. "):
        assert leak not in body


def test_emphasis_pairs_across_a_code_span() -> None:
    """The reports write bold claims with a config key inside them."""
    page = markdown_document(
        "# T\n\n- **Memory tracks `num_parts` itself.** The rest is `parts_per_batch`.\n", "T"
    )
    body = page.split("<body>")[1]
    assert (
        "<strong>Memory tracks <code>num_parts</code> itself.</strong>" in body
    ), body
    assert "**" not in body
    # ... and asterisks inside a code span stay literal.
    assert "<code>a*b</code>" in markdown_document("# T\n\nx `a*b` y\n", "T")


def test_figures_are_placed_by_aspect_ratio(tmp_path: Path) -> None:
    assets = tmp_path / "r_assets"
    assets.mkdir()
    _png(assets / "tall.png", 800, 1000)      # 0.8  -> prose column
    _png(assets / "medium.png", 2000, 800)    # 2.5  -> breaks out wide
    _png(assets / "strip.png", 5800, 900)     # 6.4  -> wide and pans sideways
    text = "# T\n\n" + figures(["tall.png", "medium.png", "strip.png"], "r_assets") + "\n"
    page = markdown_document(text, "T", assets_root=tmp_path)
    tall, medium, strip = (
        page.split(f'<img src="r_assets/{name}.png"')[0].rsplit("<figure", 1)[1]
        for name in ("tall", "medium", "strip")
    )
    assert "class=" not in tall
    assert 'class="wide"' in medium
    assert 'class="wide pan"' in strip
    # Intrinsic dimensions come from the PNG header, so the page does not reflow.
    assert 'width="5800" height="900"' in page
    assert "scroll sideways" in page  # only the panning figure says so
    assert page.count("scroll sideways") == 1


def test_captions_come_from_file_names() -> None:
    assert caption("cv_training_curves.png") == "CV training curves"
    assert caption("ig_top_nodes.png") == "IG top nodes"
    assert caption("per_site_auc_scatter.png") == "Per site AUC scatter"
    assert caption("table1_auc_by_year.png") == "Table 1 AUC by year"
    assert caption("figure3b_per_cancer_auc_transition.png") == "Figure 3b per cancer AUC transition"
    assert figures(["cost_cdr.png"], "a") == "![Cost CDR](a/cost_cdr.png)"


def test_long_documents_get_a_table_of_contents() -> None:
    short = markdown_document("# T\n\n## A\n\nbody\n", "T")
    assert 'class="toc"' not in short
    long = markdown_document("# T\n\n## A\n\nx\n\n## B\n\ny\n\n### C\n\nz\n", "T")
    assert 'class="toc"' in long
    assert '<a href="#a">A</a>' in long and '<li class="sub"><a href="#c">C</a></li>' in long
    # It sits directly after the title, before the first section.
    assert long.index('class="toc"') < long.index('id="a"')


def test_absolute_source_paths_are_made_repo_relative(tmp_path: Path) -> None:
    """A report gets handed to someone whose machine has no /home/me/checkout."""
    assert repo_path(Path.cwd() / "data_tr" / "processed") == "data_tr/processed"
    # Somewhere outside the working directory has to stay as it is.
    assert repo_path("/etc/hosts") == "/etc/hosts"
    assert repo_path("data_cdr/prepared") == "data_cdr/prepared"


def test_write_document_emits_both_files(tmp_path: Path) -> None:
    markdown_path, html_path = write_document(tmp_path / "docs", "tr_report", "# T\n\nbody\n", "T")
    assert markdown_path == tmp_path / "docs" / "tr_report.md"
    assert html_path == tmp_path / "docs" / "tr_report.html"
    assert markdown_path.read_text() == "# T\n\nbody\n"
    assert "<p>body</p>" in html_path.read_text()
