#!/usr/bin/env python3
"""PathwayCommons SIF に含まれる ChEBI ID と SMILES の対応表を作成する.

データソース: ChEBI flat files (https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files/)
  - compounds.tsv.gz     : ChEBI ID と名前
  - secondary_ids.tsv.gz : 廃止 (secondary) ID -> primary ID
  - structures.tsv.gz    : SMILES / InChI / InChIKey

使い方:
    python scripts/build_chebi_smiles.py
    python scripts/build_chebi_smiles.py --out data/chebi_smiles.tsv --all

出力 (TSV):
    chebi_id  primary_chebi_id  name  smiles  inchi  inchikey
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
    smart_open,
    split_participants,
)

CHEBI_FTP = "https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files"
COMPOUNDS_URL = "%s/compounds.tsv.gz" % CHEBI_FTP
SECONDARY_URL = "%s/secondary_ids.tsv.gz" % CHEBI_FTP
STRUCTURES_URL = "%s/structures.tsv.gz" % CHEBI_FTP

NULLS = ("", "null", "NULL", "\\N")

# structures.tsv の molfile 列は非常に長い (クォート内に改行を含む)
csv.field_size_limit(1 << 30)


def _val(x):
    x = (x or "").strip()
    return "" if x in NULLS else x


def load_compounds(path):
    """compounds.tsv から数値 ID -> 名前の辞書を作る."""
    id2name = {}
    with smart_open(path) as fh:
        reader = csv.reader(fh, delimiter="\t", quotechar='"')
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        for row in reader:
            if len(row) < len(header):
                continue
            cid = _val(row[idx["id"]])
            if not cid:
                continue
            name = _val(row[idx["name"]]) or _val(row[idx.get("ascii_name", idx["name"])])
            if name:
                id2name[cid] = name
    log("compounds: %d 件" % len(id2name))
    return id2name


def load_secondary_ids(path):
    """secondary_ids.tsv から secondary ID -> primary ID / primary -> [secondary] を作る."""
    sec2pri = {}
    pri2sec = {}
    with smart_open(path) as fh:
        reader = csv.reader(fh, delimiter="\t", quotechar='"')
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        for row in reader:
            if len(row) < len(header):
                continue
            pri = _val(row[idx["compound_id"]])
            sec = _val(row[idx["secondary_id"]])
            if not pri or not sec or pri == sec:
                continue
            sec2pri[sec] = pri
            pri2sec.setdefault(pri, []).append(sec)
    log("secondary_ids: %d 件" % len(sec2pri))
    return sec2pri, pri2sec


def load_structures(path, wanted=None):
    """compound_id -> dict(smiles, inchi, inchikey).

    1 化合物に複数構造があるため default_structure=true を優先する。
    wanted (数値 ID の集合) を渡すとその分だけ保持してメモリを節約する。
    """
    out = {}
    with smart_open(path) as fh:
        reader = csv.reader(fh, delimiter="\t", quotechar='"')
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        n = 0
        for row in reader:
            n += 1
            if n % 500_000 == 0:
                log("  structures %d 行" % n)
            if len(row) < len(header):
                continue
            cid = _val(row[idx["compound_id"]])
            if not cid or (wanted is not None and cid not in wanted):
                continue
            smiles = _val(row[idx["smiles"]])
            inchi = _val(row[idx["standard_inchi"]])
            key = _val(row[idx["standard_inchi_key"]])
            if not (smiles or inchi or key):
                continue
            is_default = _val(row[idx["default_structure"]]).lower() in ("true", "t", "1", "y")
            rec = {
                "smiles": smiles,
                "inchi": inchi,
                "inchikey": key,
                # SMILES を持つ default 構造 > default 構造 > SMILES を持つ構造 > その他
                "score": (2 if is_default else 0) + (1 if smiles else 0),
            }
            prev = out.get(cid)
            if prev is None:
                out[cid] = rec
            elif rec["score"] > prev["score"]:
                for k in ("smiles", "inchi", "inchikey"):
                    if not rec[k] and prev[k]:
                        rec[k] = prev[k]  # 採用しない側の情報も欠損補完に使う
                out[cid] = rec
            else:
                for k in ("smiles", "inchi", "inchikey"):
                    if not prev[k] and rec[k]:
                        prev[k] = rec[k]
    log("structures: %d 化合物" % len(out))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sif", default=DEFAULT_SIF, help="PathwayCommons SIF ファイル")
    ap.add_argument("--out", default="data/chebi_smiles.tsv", help="出力 TSV")
    ap.add_argument("--cache-dir", default="raw/ext", help="ChEBI flat file の保存先")
    ap.add_argument("--all", action="store_true", help="SIF に限らず ChEBI 全化合物を出力する")
    ap.add_argument("--force-download", action="store_true", help="キャッシュを無視して再取得")
    args = ap.parse_args(argv)

    wanted_num = None
    chebi_ids = None
    if not args.all:
        log("SIF から ChEBI ID を抽出: %s" % args.sif)
        participants = collect_participants(args.sif)
        chebi_ids, others = split_participants(participants)
        log("ChEBI ID: %d 件 (非 ChEBI participant: %d 件)" % (len(chebi_ids), len(others)))
        wanted_num = {c.split(":", 1)[1] for c in chebi_ids}

    compounds_path = download(
        COMPOUNDS_URL, os.path.join(args.cache_dir, "compounds.tsv.gz"), args.force_download
    )
    secondary_path = download(
        SECONDARY_URL, os.path.join(args.cache_dir, "secondary_ids.tsv.gz"), args.force_download
    )
    structures_path = download(
        STRUCTURES_URL, os.path.join(args.cache_dir, "structures.tsv.gz"), args.force_download
    )

    id2name = load_compounds(compounds_path)
    sec2pri, pri2sec = load_secondary_ids(secondary_path)

    if wanted_num is not None:
        # 構造は primary / secondary のどちらに登録されている場合もあるため両方を対象にする
        expanded = set(wanted_num)
        for cid in wanted_num:
            pri = sec2pri.get(cid, cid)
            expanded.add(pri)
            expanded.update(pri2sec.get(pri, ()))
        struct_wanted = expanded
    else:
        struct_wanted = None
        chebi_ids = {"CHEBI:%s" % cid for cid in id2name}

    structures = load_structures(structures_path, struct_wanted)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    missing_path = os.path.splitext(args.out)[0] + ".missing.txt"
    n_out = n_smiles = 0
    with open(args.out, "w", encoding="utf-8", newline="") as out, open(
        missing_path, "w", encoding="utf-8"
    ) as miss:
        out.write("chebi_id\tprimary_chebi_id\tname\tsmiles\tinchi\tinchikey\n")
        for chebi_id in sorted(chebi_ids, key=lambda x: int(x.split(":")[1]) if x.split(":")[1].isdigit() else 0):
            num = chebi_id.split(":", 1)[1]
            pri = sec2pri.get(num, num)
            rec = structures.get(num) or structures.get(pri) or {}
            if not rec:
                for sec in pri2sec.get(pri, ()):
                    if sec in structures:
                        rec = structures[sec]
                        break
            name = id2name.get(num) or id2name.get(pri, "")
            smiles = rec.get("smiles", "")
            if not (smiles or rec.get("inchi")):
                miss.write("%s\t%s\n" % (chebi_id, name))
                if not name and pri == num and num not in id2name:
                    continue  # ChEBI に存在しない ID
            out.write(
                "%s\t%s\t%s\t%s\t%s\t%s\n"
                % (
                    chebi_id,
                    "CHEBI:%s" % pri,
                    name.replace("\t", " "),
                    smiles,
                    rec.get("inchi", ""),
                    rec.get("inchikey", ""),
                )
            )
            n_out += 1
            if smiles:
                n_smiles += 1
    log("出力: %s (%d 行, SMILES 有り %d 件)" % (args.out, n_out, n_smiles))
    log("SMILES/InChI 無し: %s" % missing_path)


if __name__ == "__main__":
    main()
