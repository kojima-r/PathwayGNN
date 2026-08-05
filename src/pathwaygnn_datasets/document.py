"""Shared table and document rendering for the dataset reports.

Both report modules build a Markdown document and a matching standalone HTML
page from the same source text, so the two files can never disagree.

The HTML side is meant to be handed to someone as the report — opened in a
browser, printed, or attached — so it carries its own stylesheet and needs
nothing but its ``<name>_assets/`` directory alongside it. Three decisions in
here exist for that reason:

* **Figures get placed by shape.** The reports' figures are matplotlib panels
  ranging from taller-than-wide to 6.5:1 strips. A single ``max-width:100%``
  would squeeze a 6.5:1 strip into an illegible ribbon, so the renderer reads
  each PNG's real dimensions and lets wide figures break out of the prose
  column, while the widest keep a legible height and pan sideways. Every figure
  also links to the original file at full resolution.
* **Prose keeps a readable measure; tables take the room they need.** Tables here
  run to fourteen columns, so a wide one may use the full page width and scroll
  beyond it, while paragraphs stay capped near 70 characters either way.
* **Light only.** Every figure is a PNG with a white canvas, so a dark theme
  would frame white rectangles.
"""

from __future__ import annotations

import csv
import html
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Aspect ratio (width / height) above which a figure breaks out of the prose
# column, and above which it instead keeps its height and pans sideways.
WIDE_ASPECT = 2.2
PAN_ASPECT = 3.5
# Acronyms to restore when a file name is turned back into a caption.
ACRONYMS = {
    "auc", "roc", "ig", "cv", "cdr", "tr", "gnn", "mlp", "tcga", "gdsc", "ic50",
    "id", "ids", "hgnc", "lm22", "msigdb", "metis", "ddp", "kd", "oe", "inh", "act",
}

STYLE = """
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth}
body{
  margin:0;padding:3.25rem 1.5rem 6rem;background:#fff;color:#1b1f24;
  font:400 16.5px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  display:grid;grid-template-columns:1fr min(62rem,100%) 1fr;
}
body>*{grid-column:2}
h1,h2,h3,h4{line-height:1.25;font-weight:650;letter-spacing:-.011em;margin:0;scroll-margin-top:1rem}
h1{font-size:2.05rem;padding-bottom:.9rem;border-bottom:2px solid #1b1f24;margin-bottom:1.6rem}
h2{font-size:1.45rem;margin:3rem 0 1rem;padding-top:1.1rem;border-top:1px solid #e3e7eb}
h3{font-size:1.12rem;margin:2.1rem 0 .7rem;color:#2c3239}
h4{font-size:1rem;margin:1.6rem 0 .5rem;color:#454c54}
p{margin:0 0 1.05rem}
p,li{max-width:44rem}
ul,ol{margin:0 0 1.3rem;padding-left:1.35rem}
li{margin:0 0 .5rem}
strong{font-weight:650;color:#11161b}
a{color:#0b5cad;text-underline-offset:2px}
code{
  font:500 .875em/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background:#f2f4f7;border:1px solid #e6eaee;border-radius:4px;padding:.08em .32em;
  overflow-wrap:break-word;
}
pre{
  background:#f7f9fb;border:1px solid #e3e7eb;border-left:3px solid #98a2ac;border-radius:6px;
  padding:.95rem 1.1rem;overflow-x:auto;margin:0 0 1.5rem;
}
pre code{background:none;border:0;padding:0;font-size:.855rem}

/* fit-content, so a narrow numeric table stays compact instead of being stretched
   across the band; max-width then turns an over-wide one into a scroller. */
.tablewrap{
  width:fit-content;max-width:100%;overflow-x:auto;
  margin:0 auto 1.7rem;border:1px solid #dfe4e9;border-radius:8px;
}
table{border-collapse:collapse;width:auto;font-size:.895rem;font-variant-numeric:tabular-nums}
th,td{padding:.48rem .8rem;text-align:right;border-bottom:1px solid #edf0f3;white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
thead th{background:#f2f4f7;border-bottom:2px solid #d3d9df;font-size:.82rem;font-weight:650}
tbody tr:nth-child(even){background:#fafbfc}
tbody tr:hover{background:#f0f6fc}
tbody tr:last-child td{border-bottom:0}

figure{margin:2.1rem 0 2.3rem}
figure a{display:block;text-decoration:none}
figure img{
  display:block;margin:0 auto;max-width:100%;height:auto;
  border:1px solid #e3e7eb;border-radius:6px;background:#fff;
}
figcaption{margin-top:.6rem;font-size:.82rem;color:#5b636d;text-align:center}
figure.wide{grid-column:1/-1;width:min(100rem,calc(100% - 3rem));margin-inline:auto}
.tablewrap.wide{grid-column:1/-1;justify-self:center;max-width:min(100rem,calc(100% - 3rem))}
figure.pan{overflow-x:auto;padding-bottom:.35rem}
figure.pan img{max-width:none;width:auto;height:clamp(18rem,42vh,32rem)}

.toc{margin:0 0 2.75rem;padding:1.05rem 1.35rem;background:#f7f9fb;border:1px solid #e3e7eb;border-radius:8px}
.toc-title{font-size:.76rem;font-weight:650;letter-spacing:.09em;text-transform:uppercase;color:#5b636d;margin:0 0 .65rem}
.toc ul{list-style:none;margin:0;padding:0;columns:2;column-gap:2.5rem}
.toc li{max-width:none;margin:0 0 .32rem;break-inside:avoid;font-size:.895rem}
.toc li.sub{padding-left:1rem;font-size:.845rem}
.toc a{color:#2c3239;text-decoration:none}
.toc a:hover{color:#0b5cad;text-decoration:underline}

@media (max-width:52rem){
  body{padding:2rem 1.05rem 4rem}
  h1{font-size:1.6rem}h2{font-size:1.28rem}
  .toc ul{columns:1}
  figure.wide{width:100%}
  .tablewrap.wide{max-width:100%}
}
@media print{
  body{padding:0;font-size:10pt;grid-template-columns:0 100% 0}
  a{color:inherit;text-decoration:none}
  h2,h3{break-after:avoid}
  figure,.tablewrap,tr{break-inside:avoid}
  .tablewrap,figure.pan{overflow:visible;max-width:100%}
  table{font-size:8.5pt}
  figure.pan img{width:auto;height:auto;max-width:100%}
  thead{display:table-header-group}
  tbody tr:hover{background:none}
  .toc{break-after:page}
}
"""


def fmt(value: Any) -> str:
    """Four decimals for floats, ``NA`` for nan/inf, everything else verbatim."""
    if isinstance(value, (float, np.floating)):
        return "NA" if not math.isfinite(float(value)) else f"{float(value):.4f}"
    return str(value)


def tsv(path: Path, header: Sequence[Any], rows: Iterable[Sequence[Any]]) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def mdtable(header: Sequence[Any], rows: Iterable[Sequence[Any]]) -> str:
    header = [str(item) for item in header]
    return "\n".join(
        ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        + ["| " + " | ".join(fmt(x) for x in row) + " |" for row in rows]
    )


def htable(header: Sequence[Any], rows: Iterable[Sequence[Any]]) -> str:
    """Cells go through the inline renderer, so a `code span` in a header works."""
    head = "".join(f"<th>{_inline(str(x))}</th>" for x in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{_inline(fmt(x))}</td>" for x in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def repo_path(path: Any) -> str:
    """Show a path relative to the working directory when it lives under it.

    Some ``dataset.json`` manifests record absolute source directories, and a
    report is meant to be handed to someone else — whose machine has no
    ``/home/somebody/checkout`` in it.
    """
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except (ValueError, OSError, TypeError):
        return str(path)


def caption(name: str) -> str:
    """``cv_training_curves.png`` -> ``CV training curves``.

    The figures are named after what they plot, so the file name is the caption
    once the separators and the shouted acronyms are put back.
    """
    words = Path(name).stem.replace("-", "_").split("_")
    spelled = []
    for word in words:
        if word.lower() in ACRONYMS:
            spelled.append(word.upper())
        else:
            # `table1` -> `table 1`, `figure3b` -> `figure 3b`.
            spelled.append(re.sub(r"^([a-z]+)(\d+[a-z]?)$", r"\1 \2", word))
    text = " ".join(word for word in spelled if word)
    return text[:1].upper() + text[1:] if text else name


def figures(names: Iterable[str], assets: str) -> str:
    return "\n\n".join(f"![{caption(name)}]({assets}/{name})" for name in names)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _png_size(path: Path) -> tuple[int, int] | None:
    """Width and height straight out of the PNG header, no image library needed."""
    try:
        header = path.read_bytes()[:24]
    except OSError:
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def _inline(text: str) -> str:
    """Escape, then render the inline Markdown the reports use.

    Code spans are set aside as placeholders rather than rendered in place: the
    reports routinely write ``**a claim about `some_key`.**``, and emphasis has to
    pair across the code span while the span's own underscores and asterisks stay
    literal.
    """
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(match.group(1))
        return f"\x00{len(spans) - 1}\x00"

    stashed = html.escape(re.sub(r"`([^`]+)`", stash, text))
    stashed = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', stashed)
    stashed = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", stashed)
    stashed = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", stashed)
    return re.sub(
        r"\x00(\d+)\x00",
        lambda match: f"<code>{html.escape(spans[int(match.group(1))])}</code>",
        stashed,
    )


def _figure(alt: str, source: str, assets_root: Path | None) -> str:
    """One figure, placed according to the real shape of the image."""
    size = _png_size(assets_root / source) if assets_root is not None else None
    classes, dimensions = [], ""
    if size:
        width, height = size
        dimensions = f' width="{width}" height="{height}"'
        aspect = width / height if height else 1.0
        if aspect >= WIDE_ASPECT:
            classes.append("wide")
        if aspect >= PAN_ASPECT:
            classes.append("pan")
    attribute = f' class="{" ".join(classes)}"' if classes else ""
    hint = " (scroll sideways, or click for full size)" if "pan" in classes else ""
    return (
        f"<figure{attribute}>"
        f'<a href="{html.escape(source)}" title="Open at full size">'
        f'<img src="{html.escape(source)}" alt="{html.escape(alt)}"{dimensions} loading="lazy">'
        f"</a>"
        f"<figcaption>{_inline(alt)}{hint}</figcaption>"
        f"</figure>"
    )


def _contents(headings: list[tuple[int, str, str]]) -> str:
    items = []
    for level, text, slug in headings:
        attribute = ' class="sub"' if level > 2 else ""
        items.append(f'<li{attribute}><a href="#{slug}">{_inline(text)}</a></li>')
    return (
        '<nav class="toc"><p class="toc-title">Contents</p><ul>'
        + "".join(items)
        + "</ul></nav>"
    )


def markdown_document(
    markdown_text: str, title: str, assets_root: str | Path | None = None
) -> str:
    """Render the subset of Markdown the reports emit into a standalone page.

    ``assets_root`` is the directory the figure paths are relative to; given it,
    figures are placed by their real aspect ratio (see the module docstring).
    """
    root = Path(assets_root) if assets_root is not None else None
    lines = markdown_text.splitlines()
    body: list[str] = []
    paragraph: list[str] = []
    headings: list[tuple[int, str, str]] = []
    contents_at: int | None = None
    index = 0

    def flush() -> None:
        if paragraph:
            body.append(f"<p>{_inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        heading = re.match(r"(#{1,4}) +(.*)", line)
        if heading:
            flush()
            level, text = len(heading.group(1)), heading.group(2).strip()
            if level == 1:
                body.append(f"<h1>{_inline(text)}</h1>")
                contents_at = len(body)
            else:
                slug = _slug(text)
                headings.append((level, text, slug))
                body.append(f'<h{level} id="{slug}">{_inline(text)}</h{level}>')
            index += 1
            continue
        if line.startswith("    "):
            flush()
            code = []
            while index < len(lines) and (lines[index].startswith("    ") or not lines[index]):
                code.append(lines[index][4:] if lines[index].startswith("    ") else "")
                index += 1
            body.append("<pre><code>" + html.escape("\n".join(code).rstrip()) + "</code></pre>")
            continue
        if line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|---"):
            flush()
            header = [item.strip() for item in line.strip("|").split("|")]
            index += 2
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                rows.append([item.strip() for item in lines[index].strip("|").split("|")])
                index += 1
            # A table this wide needs more room than the prose column has.
            classes = "tablewrap wide" if len(header) > 6 else "tablewrap"
            body.append(f'<div class="{classes}">{htable(header, rows)}</div>')
            continue
        figure = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
        if figure:
            flush()
            body.append(_figure(figure.group(1), figure.group(2), root))
            index += 1
            continue
        bullet = re.match(r"([-*]|\d+\.) +(.*)", line)
        if bullet:
            flush()
            tag = "ul" if bullet.group(1) in "-*" else "ol"
            items: list[str] = []
            while index < len(lines):
                match = re.match(r"([-*]|\d+\.) +(.*)", lines[index])
                if match:
                    items.append(match.group(2).strip())
                    index += 1
                    continue
                # A continuation line is indented, but not by the four spaces
                # that would make it a code block.
                if items and re.match(r"  {0,1}\S", lines[index]):
                    items[-1] += " " + lines[index].strip()
                    index += 1
                    continue
                break
            body.append(
                f"<{tag}>" + "".join(f"<li>{_inline(item)}</li>" for item in items) + f"</{tag}>"
            )
            continue
        if not line:
            flush()
        else:
            paragraph.append(line)
        index += 1
    flush()

    if contents_at is not None and len(headings) >= 3:
        body.insert(contents_at, _contents(headings))
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>"
        + "\n".join(body)
        + "</body></html>"
    )


def write_document(
    docs_dir: str | Path, name: str, markdown_text: str, title: str
) -> tuple[Path, Path]:
    docs = Path(docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    markdown_path = docs / f"{name}.md"
    html_path = docs / f"{name}.html"
    markdown_path.write_text(markdown_text)
    html_path.write_text(markdown_document(markdown_text, title, assets_root=docs))
    return markdown_path, html_path
