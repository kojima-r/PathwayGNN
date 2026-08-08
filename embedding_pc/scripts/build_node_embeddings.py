#!/usr/bin/env python3
"""タンパク質 / 化合物の埋め込みを束ねて processed/node_embeddings.npz を作る.

入力:
    processed/protein_emb.shard*.npz   (embed_proteins.py の出力, 2560 次元)
    processed/chemical_emb.npz         (embed_chemicals.py の出力,  512 次元)
    data/gene_ids.tsv                  (build_gene_ids.py の出力, 任意)

出力:
    processed/node_embeddings.npz      種別ごとの (names, embeddings) と別名 (実運用の読み込み用)
    processed/node_embeddings.json     {ノード名: 埋め込みベクトル}
    processed/node_embeddings.meta.json  ノードごとの種別・次元・別名表とサマリ

ノード名は PathwayCommons SIF の participant 表記に合わせ、
タンパク質は gene symbol (例 A1BG)、化合物は ChEBI ID (例 CHEBI:36) を使う。

`data/gene_ids.tsv` があると、タンパク質に **別名 (alias)** を付ける。
別名はベクトルを複製せず「別名 -> 行番号」の対応として持つので、ファイルはほとんど太らない。
これにより、ノード名が gene symbol でないコーパス
(data_cancer = 数値 HGNC ID、Entrez ID を使うもの) でも同じ表が使える。

使い方:
    python scripts/build_node_embeddings.py
    python scripts/build_node_embeddings.py --no-gene-ids     # 別名を付けない
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_DIR = os.path.join(ROOT, "processed")
DEFAULT_GENE_IDS = os.path.join(ROOT, "data", "gene_ids.tsv")
# gene_ids.tsv のどの列を別名として使うか (列名 -> 別名の種類)
ALIAS_COLUMNS = ("hgnc_id", "entrez_id", "ensembl_gene_id")


def log(msg):
    print("[build_node_embeddings] %s" % msg, flush=True)


def load_npz_list(paths):
    """複数 npz から (names, embeddings) を連結して返す."""
    names, embs = [], []
    for path in sorted(paths):
        data = np.load(path, allow_pickle=True)
        n = [str(x) for x in data["names"]]
        e = data["embeddings"]
        if len(n) != len(e):
            raise ValueError("%s: names/embeddings length mismatch" % path)
        names.extend(n)
        embs.append(e)
        log("  %s -> %s" % (os.path.basename(path), e.shape))
    if not embs:
        return [], np.zeros((0, 0), dtype=np.float32)
    return names, np.concatenate(embs, axis=0)


def load_aliases(path, names, reserved=(), columns=ALIAS_COLUMNS):
    """gene symbol の別名 (HGNC ID / Entrez ID / Ensembl ID) を ID 体系ごとに作る.

    **体系ごとに分ける**のが要点。HGNC ID も Entrez ID も裸の数値なので、混ぜると
    「5」が A1BG (HGNC:5) と別の遺伝子 (Entrez 5) の両方を指してしまう。
    どちらで突き合わせるかは、コーパスのノード名を知っている下流が選ぶ
    (`model.node_embeddings.aliases:`)。

    Args:
        names: タンパク質の行名 (別名はこの行を指す)。
        reserved: 別名として使ってはいけない名前 (化合物の ChEBI ID など)。

    Returns:
        {列名: {"names": [...], "rows": int64[...], "dropped": {...}}}
        names[i] が rows[i] 行目のベクトルを指す。同じ体系の中で2つの symbol が同じ ID を
        指す場合と、既にノード名として使われている ID は捨てる (黙ってどちらかに寄せない)。
    """
    row_of = {name: row for row, name in enumerate(names)}
    taken = set(names) | set(reserved)
    hits = {column: {} for column in columns}
    dropped = {
        column: {"ambiguous": 0, "clashes_with_name": 0} for column in columns
    }
    unknown_symbols = 0
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for record in csv.DictReader(fh, delimiter="\t"):
            symbol = (record.get("gene_symbol") or "").strip()
            row = row_of.get(symbol)
            if row is None:
                unknown_symbols += 1
                continue
            for column in columns:
                alias = (record.get(column) or "").strip()
                if not alias:
                    continue
                if alias in taken:
                    dropped[column]["clashes_with_name"] += 1
                    continue
                bucket = hits[column]
                if alias not in bucket:
                    bucket[alias] = row
                elif bucket[alias] is not None and bucket[alias] != row:
                    bucket[alias] = None  # 曖昧: 後段で落とす
    result = {}
    for column in columns:
        bucket = hits[column]
        for alias, row in list(bucket.items()):
            if row is None:
                del bucket[alias]
                dropped[column]["ambiguous"] += 1
        alias_names = sorted(bucket)
        result[column] = {
            "names": alias_names,
            "rows": np.array([bucket[alias] for alias in alias_names], dtype=np.int64),
            "dropped": dropped[column],
        }
    result["_unknown_symbols"] = unknown_symbols
    return result


def dump_json_stream(fh, items, decimals):
    """{name: [floats]} をメモリに全部載せずに書き出す."""
    fh.write("{\n")
    first = True
    for name, vec in items:
        if not first:
            fh.write(",\n")
        first = False
        v = np.round(np.asarray(vec, dtype=np.float64), decimals)
        body = ",".join(repr(float(x)) for x in v)
        fh.write("%s: [%s]" % (json.dumps(name, ensure_ascii=False), body))
    fh.write("\n}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument(
        "--decimals", type=int, default=5, help="JSON に書く小数点以下の桁数"
    )
    ap.add_argument(
        "--gene-ids",
        default=DEFAULT_GENE_IDS,
        help="build_gene_ids.py の出力。存在すればタンパク質に別名を付ける",
    )
    ap.add_argument(
        "--no-gene-ids", action="store_true", help="別名を付けない (gene symbol のみ)"
    )
    args = ap.parse_args()

    prot_paths = glob.glob(os.path.join(args.dir, "protein_emb.shard*.npz"))
    chem_path = os.path.join(args.dir, "chemical_emb.npz")

    log("protein shards: %d" % len(prot_paths))
    prot_names, prot_emb = load_npz_list(prot_paths)
    log("chemical:")
    chem_names, chem_emb = load_npz_list([chem_path] if os.path.exists(chem_path) else [])

    # 重複チェック (shard 間の取りこぼし / 二重計上を検出する)
    for label, names in (("protein", prot_names), ("chemical", chem_names)):
        if len(set(names)) != len(names):
            raise ValueError("%s: duplicated node names" % label)
    overlap = set(prot_names) & set(chem_names)
    if overlap:
        raise ValueError(
            "gene symbol と ChEBI ID が衝突: %s" % sorted(overlap)[:10]
        )

    prot_dim = int(prot_emb.shape[1]) if len(prot_emb) else 0
    chem_dim = int(chem_emb.shape[1]) if len(chem_emb) else 0
    log("protein: %d nodes x %d dim" % (len(prot_names), prot_dim))
    log("chemical: %d nodes x %d dim" % (len(chem_names), chem_dim))

    # NaN / Inf は下流の学習を壊すので落とす
    def clean(names, emb, label):
        if not len(emb):
            return names, emb
        good = np.isfinite(emb).all(axis=1)
        n_bad = int((~good).sum())
        if n_bad:
            log("%s: NaN/Inf を含む %d ノードを除外" % (label, n_bad))
            names = [n for n, g in zip(names, good) if g]
            emb = emb[good]
        return names, emb

    prot_names, prot_emb = clean(prot_names, prot_emb, "protein")
    chem_names, chem_emb = clean(chem_names, chem_emb, "chemical")

    # ---- 別名 (HGNC ID / Entrez ID / Ensembl ID) ----
    aliases = {}
    if not args.no_gene_ids and prot_names:
        if os.path.exists(args.gene_ids):
            found = load_aliases(args.gene_ids, prot_names, reserved=chem_names)
            found.pop("_unknown_symbols", None)
            aliases = {column: entry for column, entry in found.items() if len(entry["names"])}
            log(
                "別名を %s から作成: %s"
                % (
                    args.gene_ids,
                    ", ".join(
                        "%s %d 件" % (column, len(entry["names"]))
                        for column, entry in aliases.items()
                    ),
                )
            )
        else:
            log(
                "%s が無いので別名を付けない (scripts/build_gene_ids.py で作成できる)"
                % args.gene_ids
            )

    # ---- JSON (別名は含めない。meta.json の alias_to_name で引く) ----
    json_path = os.path.join(args.dir, "node_embeddings.json")
    items = list(zip(prot_names, prot_emb)) + list(zip(chem_names, chem_emb))
    with open(json_path, "w", encoding="utf-8") as fh:
        dump_json_stream(fh, items, args.decimals)
    log(
        "saved %s  (%d nodes, %.1f MB)"
        % (json_path, len(items), os.path.getsize(json_path) / 1e6)
    )

    # ---- メタ情報 ----
    meta = {
        "n_nodes": len(items),
        "node_types": {
            "protein": {
                "n": len(prot_names),
                "dim": prot_dim,
                "model": "biohub/ESMC-6B",
                "source": "data/gene_uniprot_seq.tsv",
                "key": "gene_symbol",
                "pooling": "残基方向 mean (special token 除外, 4096 残基超は窓分割の残基数重み付き平均)",
            },
            "chemical": {
                "n": len(chem_names),
                "dim": chem_dim,
                "model": "unimolv1 (unimol_tools UniMolRepr, cls_repr)",
                "source": "data/chebi_smiles.tsv",
                "key": "chebi_id",
                "pooling": "cls_repr (分子単位)",
            },
        },
        "node_type": (
            {n: "protein" for n in prot_names} | {n: "chemical" for n in chem_names}
        ),
        # ID 体系ごとの「別名 -> 正式なノード名」。ノード名が gene symbol でない
        # コーパス (data_cancer = 数値 HGNC ID など) 用。どれを使うかは下流が選ぶ。
        "aliases": {
            column: {"n": len(entry["names"]), "dropped": entry["dropped"]}
            for column, entry in aliases.items()
        },
        "alias_to_name": {
            column: {
                alias: prot_names[int(row)]
                for alias, row in zip(entry["names"], entry["rows"])
            }
            for column, entry in aliases.items()
        },
    }
    meta_path = os.path.join(args.dir, "node_embeddings.meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    log("saved %s" % meta_path)

    # ---- npz (実運用の読み込み用) ----
    npz_path = os.path.join(args.dir, "node_embeddings.npz")
    arrays = {
        "protein_names": np.array(prot_names, dtype=object),
        "protein_embeddings": prot_emb.astype(np.float32),
        "chemical_names": np.array(chem_names, dtype=object),
        "chemical_embeddings": chem_emb.astype(np.float32),
    }
    for column, entry in aliases.items():
        # ベクトルは複製せず、別名 -> 行番号 で持つ (pathwaygnn 側が展開する)
        arrays["protein_alias_%s_names" % column] = np.array(entry["names"], dtype=object)
        arrays["protein_alias_%s_rows" % column] = entry["rows"]
    np.savez_compressed(npz_path, **arrays)
    log("saved %s (%.1f MB)" % (npz_path, os.path.getsize(npz_path) / 1e6))


if __name__ == "__main__":
    sys.exit(main())
