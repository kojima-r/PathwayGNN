#!/usr/bin/env python3
"""PathwayCommons SIF の gene symbol に Entrez ID / HGNC ID / Ensembl ID を対応づける.

埋め込み表 (`build_node_embeddings.py`) はノード名を gene symbol で書くが、
下流のコーパスが同じ表記とは限らない。

    data_tr             : gene symbol    (A1BG) -> そのまま一致する
    data_cancer/data_cdr: 数値 HGNC ID   (5)    -> この表の hgnc_id で一致させる
    その他               : Entrez ID      (1)    -> この表の entrez_id で一致させる

データソース: HGNC complete set (承認シンボル・旧シンボル・別名シンボルと、
その entrez_id / ensembl_gene_id が 1 ファイルに入っている)。標準ライブラリのみで動く。

使い方:
    python scripts/build_gene_ids.py
    python scripts/build_gene_ids.py --all --out data/gene_ids.tsv

出力 (TSV):
    gene_symbol  hgnc_id  entrez_id  ensembl_gene_id  match_type
    match_type は primary (承認シンボル一致) / prev (旧シンボル) / alias (別名)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import (  # noqa: E402
    DEFAULT_SIF,
    collect_participants,
    download,
    log,
    split_participants,
)

HGNC_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
)

csv.field_size_limit(1 << 30)


def split_field(value):
    """HGNC の複数値フィールド ("A|B" または '"A|B"') を分解する."""
    return [item.strip() for item in (value or "").strip('"').split("|") if item.strip()]


def index_symbols(path):
    """symbol (大文字) -> (match_type, record) の索引.

    承認シンボルを最優先し、旧シンボル・別名シンボルは承認シンボルと衝突しない場合だけ
    採用する (同じ別名を複数の遺伝子が持つことがあるため、衝突した別名は捨てる)。
    """
    primary, secondary, ambiguous = {}, {}, set()
    n = 0
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            symbol = (row.get("symbol") or "").strip()
            hgnc = (row.get("hgnc_id") or "").strip()
            if not symbol or not hgnc:
                continue
            n += 1
            record = {
                # data_cancer の vertices_dic.tsv は "HGNC:" を落とした数値表記なので、
                # 数値部分を出力する (突き合わせ側で prefix を足すのは容易)。
                "hgnc_id": hgnc.split(":")[-1],
                "entrez_id": (row.get("entrez_id") or "").strip(),
                "ensembl_gene_id": (row.get("ensembl_gene_id") or "").strip(),
            }
            primary[symbol.upper()] = record
            for kind in ("prev_symbol", "alias_symbol"):
                for other in split_field(row.get(kind)):
                    key = other.upper()
                    label = "prev" if kind == "prev_symbol" else "alias"
                    if key in secondary and secondary[key][1] != record:
                        ambiguous.add(key)
                    secondary.setdefault(key, (label, record))
    for key in ambiguous:
        secondary.pop(key, None)
    log(
        "HGNC %d 行を索引化 (承認シンボル %d / 旧・別名 %d、衝突により除外 %d)"
        % (n, len(primary), len(secondary), len(ambiguous))
    )
    return primary, secondary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sif", default=DEFAULT_SIF, help="PathwayCommons SIF ファイル")
    ap.add_argument("--out", default="data/gene_ids.tsv", help="出力 TSV")
    ap.add_argument("--cache-dir", default="raw/ext", help="HGNC ファイルの保存先")
    ap.add_argument(
        "--all", action="store_true", help="SIF に出現するものだけでなく HGNC 全遺伝子を出力"
    )
    ap.add_argument("--force-download", action="store_true", help="キャッシュを無視して再取得")
    args = ap.parse_args(argv)

    hgnc_path = download(
        HGNC_URL, os.path.join(args.cache_dir, "hgnc_complete_set.txt"), args.force_download
    )
    primary, secondary = index_symbols(hgnc_path)

    if args.all:
        symbols = sorted(primary)
        # 承認シンボルが最後に来るように重ねる (旧・別名より優先)。
        lookup = dict(secondary)
        lookup.update({key: ("primary", record) for key, record in primary.items()})
    else:
        log("SIF から gene symbol 候補を抽出: %s" % args.sif)
        participants = collect_participants(args.sif)
        chebi_ids, others = split_participants(participants)
        log("非 ChEBI participant: %d 件 (ChEBI: %d 件)" % (len(others), len(chebi_ids)))
        symbols = sorted(others)
        lookup = None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    missing_path = os.path.splitext(args.out)[0] + ".missing.txt"
    counts = {"primary": 0, "prev": 0, "alias": 0}
    n_missing = 0
    with open(args.out, "w", encoding="utf-8", newline="") as out, open(
        missing_path, "w", encoding="utf-8"
    ) as miss:
        out.write("gene_symbol\thgnc_id\tentrez_id\tensembl_gene_id\tmatch_type\n")
        for symbol in symbols:
            key = symbol.upper()
            if lookup is not None:
                match_type, record = lookup[key]
            elif key in primary:
                match_type, record = "primary", primary[key]
            elif key in secondary:
                match_type, record = secondary[key]
            else:
                miss.write(symbol + "\n")
                n_missing += 1
                continue
            counts[match_type] += 1
            out.write(
                "%s\t%s\t%s\t%s\t%s\n"
                % (
                    symbol,
                    record["hgnc_id"],
                    record["entrez_id"],
                    record["ensembl_gene_id"],
                    match_type,
                )
            )
    log(
        "出力: %s (%d 件: primary %d / prev %d / alias %d)"
        % (args.out, sum(counts.values()), counts["primary"], counts["prev"], counts["alias"])
    )
    log("HGNC に対応が無い participant: %d 件 -> %s" % (n_missing, missing_path))


if __name__ == "__main__":
    main()
