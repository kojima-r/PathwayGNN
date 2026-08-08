#!/usr/bin/env python3
"""PathwayCommons SIF に含まれる gene symbol と UniProt ID / アミノ酸配列の対応表を作成する.

データソース: UniProtKB REST API (https://rest.uniprot.org/)
  ヒト (organism_id:9606) の Swiss-Prot エントリを取得し、
  gene symbol (primary / synonym) で SIF の participant と突き合わせる。

使い方:
    python scripts/build_gene_uniprot_seq.py
    python scripts/build_gene_uniprot_seq.py --include-unreviewed --one-per-gene
    python scripts/build_gene_uniprot_seq.py --fasta data/gene_uniprot.fasta

出力 (TSV):
    gene_symbol  uniprot_id  entry_name  reviewed  match_type  protein_name  length  sequence
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import DEFAULT_SIF, collect_participants, log, split_participants  # noqa: E402

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,id,reviewed,gene_primary,gene_synonym,protein_name,length,sequence"
PAGE_SIZE = 500

csv.field_size_limit(1 << 30)


def fetch_uniprot(query, dest, force=False, retries=3, timeout=180):
    """UniProt REST をページングしながら取得し TSV.gz として保存する."""
    if os.path.exists(dest) and not force:
        log("cache hit: %s" % dest)
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    url = "%s?%s" % (
        UNIPROT_SEARCH,
        urllib.parse.urlencode(
            {"query": query, "fields": FIELDS, "format": "tsv", "size": PAGE_SIZE}
        ),
    )
    tmp = dest + ".part"
    n_rows = 0
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as out:
        first = True
        while url:
            body = None
            next_url = None
            for attempt in range(1, retries + 1):
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "PathwayCommonEmb/1.0"}
                    )
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        body = resp.read().decode("utf-8")
                        link = resp.headers.get("Link", "")
                        if 'rel="next"' in link:
                            next_url = link.split("<", 1)[1].split(">", 1)[0]
                    break
                except (urllib.error.URLError, OSError) as e:
                    log("  取得失敗 (%d/%d): %s" % (attempt, retries, e))
                    if attempt == retries:
                        raise
                    time.sleep(5 * attempt)
            lines = body.splitlines()
            if not first and lines:
                lines = lines[1:]  # 2 ページ目以降のヘッダを除去
            first = False
            for line in lines:
                out.write(line + "\n")
            n_rows += len(lines)
            if n_rows % 5000 < PAGE_SIZE:
                log("  UniProt %d 行取得" % n_rows)
            url = next_url
    os.replace(tmp, dest)
    log("UniProt 取得完了: %s (%d 行)" % (dest, n_rows))
    return dest


def index_by_symbol(uniprot_tsv_gz):
    """gene symbol (大文字化) -> [(match_type, record), ...]."""
    index = {}
    with gzip.open(uniprot_tsv_gz, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        n = 0
        for row in reader:
            n += 1
            rec = {
                "accession": row.get("Entry", ""),
                "entry_name": row.get("Entry Name", ""),
                "reviewed": row.get("Reviewed", ""),
                "primary": (row.get("Gene Names (primary)") or "").strip(),
                "protein_name": (row.get("Protein names") or "").replace("\t", " "),
                "length": row.get("Length", ""),
                "sequence": (row.get("Sequence") or "").strip(),
            }
            if not rec["accession"]:
                continue
            # primary は複数返る場合がある (";" 区切り)
            primaries = [s.strip() for s in rec["primary"].replace(";", " ").split() if s.strip()]
            synonyms = [
                s.strip()
                for s in (row.get("Gene Names (synonym)") or "").replace(";", " ").split()
                if s.strip()
            ]
            for sym in primaries:
                index.setdefault(sym.upper(), []).append(("primary", rec))
            for sym in synonyms:
                if sym.upper() not in {p.upper() for p in primaries}:
                    index.setdefault(sym.upper(), []).append(("synonym", rec))
    log("UniProt エントリ %d 件, symbol %d 種をインデックス化" % (n, len(index)))
    return index


def pick_best(hits):
    """1 遺伝子 1 レコードに絞る: primary 一致 > reviewed > 配列長."""
    def key(h):
        match_type, rec = h
        return (
            1 if match_type == "primary" else 0,
            1 if rec["reviewed"].lower().startswith("reviewed") else 0,
            len(rec["sequence"]),
        )

    return max(hits, key=key)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sif", default=DEFAULT_SIF, help="PathwayCommons SIF ファイル")
    ap.add_argument("--out", default="data/gene_uniprot_seq.tsv", help="出力 TSV")
    ap.add_argument("--cache-dir", default="raw/ext", help="UniProt ダウンロードの保存先")
    ap.add_argument("--organism", default="9606", help="NCBI taxonomy ID (既定: ヒト)")
    ap.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="TrEMBL (unreviewed) も含める。取得量が大きく増える",
    )
    ap.add_argument(
        "--one-per-gene",
        action="store_true",
        help="gene symbol ごとに 1 エントリだけ出力する (既定は全候補を出力)",
    )
    ap.add_argument("--fasta", default=None, help="配列を FASTA でも出力する場合のパス")
    ap.add_argument("--force-download", action="store_true", help="キャッシュを無視して再取得")
    args = ap.parse_args(argv)

    log("SIF から gene symbol 候補を抽出: %s" % args.sif)
    participants = collect_participants(args.sif)
    chebi_ids, symbols = split_participants(participants)
    log("非 ChEBI participant: %d 件 (ChEBI: %d 件)" % (len(symbols), len(chebi_ids)))

    query = "(organism_id:%s)" % args.organism
    if not args.include_unreviewed:
        query += " AND (reviewed:true)"
    tag = "%s_%s" % (args.organism, "all" if args.include_unreviewed else "sp")
    uniprot_path = fetch_uniprot(
        query, os.path.join(args.cache_dir, "uniprot_%s.tsv.gz" % tag), args.force_download
    )
    index = index_by_symbol(uniprot_path)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    missing_path = os.path.splitext(args.out)[0] + ".missing.txt"
    fasta_fh = open(args.fasta, "w", encoding="utf-8") if args.fasta else None

    n_genes = n_rows = n_missing = 0
    with open(args.out, "w", encoding="utf-8", newline="") as out, open(
        missing_path, "w", encoding="utf-8"
    ) as miss:
        out.write(
            "gene_symbol\tuniprot_id\tentry_name\treviewed\tmatch_type\tprotein_name\tlength\tsequence\n"
        )
        for sym in sorted(symbols):
            hits = index.get(sym.upper())
            if not hits:
                miss.write(sym + "\n")
                n_missing += 1
                continue
            n_genes += 1
            selected = [pick_best(hits)] if args.one_per_gene else hits
            seen_acc = set()
            for match_type, rec in selected:
                if rec["accession"] in seen_acc:
                    continue
                seen_acc.add(rec["accession"])
                out.write(
                    "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
                    % (
                        sym,
                        rec["accession"],
                        rec["entry_name"],
                        rec["reviewed"],
                        match_type,
                        rec["protein_name"],
                        rec["length"],
                        rec["sequence"],
                    )
                )
                n_rows += 1
                if fasta_fh and rec["sequence"]:
                    fasta_fh.write(">%s|%s|%s\n" % (sym, rec["accession"], rec["entry_name"]))
                    seq = rec["sequence"]
                    for i in range(0, len(seq), 60):
                        fasta_fh.write(seq[i : i + 60] + "\n")

    if fasta_fh:
        fasta_fh.close()
        log("FASTA: %s" % args.fasta)
    log("出力: %s (%d 行 / %d symbol)" % (args.out, n_rows, n_genes))
    log(
        "UniProt 未対応の participant: %d 件 -> %s (低分子名などを含む)"
        % (n_missing, missing_path)
    )


if __name__ == "__main__":
    main()
