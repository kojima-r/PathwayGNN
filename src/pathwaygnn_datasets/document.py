"""Shared table and document rendering for the dataset reports.

Both report modules build a Markdown document and a matching standalone HTML
page from the same source text, so the two files can never disagree.
"""

from __future__ import annotations

import csv
import html
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

STYLE = (
    "body{font:16px/1.55 system-ui;max-width:1180px;margin:auto;padding:2rem}"
    "table{border-collapse:collapse;display:block;overflow:auto}"
    "th,td{border:1px solid #bbb;padding:.35rem .55rem;text-align:right}"
    "th:first-child,td:first-child{text-align:left}img{max-width:100%}"
    "pre,code{background:#f4f4f4}pre{padding:1rem;overflow:auto}"
)


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
    head = "".join(f"<th>{html.escape(str(x))}</th>" for x in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(fmt(x))}</td>" for x in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def figures(names: Iterable[str], assets: str) -> str:
    return "\n\n".join(f"![{name}]({assets}/{name})" for name in names)


def markdown_document(markdown_text: str, title: str) -> str:
    """Render the subset of Markdown the reports emit into a standalone page."""
    lines = markdown_text.splitlines()
    body: list[str] = []
    paragraph: list[str] = []
    index = 0

    def flush():
        if paragraph:
            text = " ".join(paragraph)
            text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))
            body.append(f"<p>{text}</p>")
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        if line.startswith("# ") or line.startswith("## "):
            flush()
            level = 1 if line.startswith("# ") else 2
            body.append(f"<h{level}>{html.escape(line[level + 1:])}</h{level}>")
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
            body.append(htable(header, rows))
            continue
        match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
        if match:
            flush()
            body.append(
                f"<figure><img src=\"{html.escape(match.group(2))}\" "
                f"alt=\"{html.escape(match.group(1))}\">"
                f"<figcaption>{html.escape(match.group(1))}</figcaption></figure>"
            )
            index += 1
            continue
        if not line:
            flush()
        else:
            paragraph.append(line)
        index += 1
    flush()
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>" + STYLE + "</style></head><body>"
        + "\n".join(body) + "</body></html>"
    )


def write_document(
    docs_dir: str | Path, name: str, markdown_text: str, title: str
) -> tuple[Path, Path]:
    docs = Path(docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    markdown_path = docs / f"{name}.md"
    html_path = docs / f"{name}.html"
    markdown_path.write_text(markdown_text)
    html_path.write_text(markdown_document(markdown_text, title))
    return markdown_path, html_path
