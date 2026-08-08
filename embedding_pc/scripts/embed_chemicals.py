#!/usr/bin/env python3
"""chebi_smiles.tsv の SMILES を Uni-Mol で埋め込む.

unimol_test.py と同じ設定 (unimolv1 / remove_hs=False / cls_repr) で、
1 ChEBI ID あたり 1 本の [512] ベクトルを作る。

unimol_tools にそのまま SMILES を渡すと 2 点で詰まるため、前処理を自前で行う:

1. 推論時は不正 SMILES で ValueError を投げてバッチごと落ちる。
   ChEBI の SMILES には kekulize できないものが 1 割弱あるので、
   同じ行の InChI から RDKit 正準 SMILES を再生成して救済する。

2. コンフォマー生成 (AllChem.EmbedMolecule) が一部の分子で事実上停止する。
   unimol_tools 内部の呼び出しには時間制限が無いので、ここで
   ETKDG の timeout を設定して自前で 3D 座標を作り、
   atoms / coordinates を直接 UniMolRepr に渡す
   (unimol_tools 側の inner_smi2coords と同じ手順を再現している)。

使い方:
    python scripts/embed_chemicals.py

出力: processed/chemical_emb.npz  (names, embeddings, conformer_ok)
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np

csv.field_size_limit(1 << 30)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_TSV = os.path.join(ROOT, "data", "chebi_smiles.tsv")
DEFAULT_OUTDIR = os.path.join(ROOT, "processed")

# ETKDG の 1 分子あたりの制限時間 (秒)
CONF_TIMEOUT = 20.0
CONF_SEED = 42


def log(msg):
    print("[embed_chemicals] %s" % msg, flush=True)


def load_smiles(path):
    """(chebi_id, smiles, inchi) のリストを返す. SMILES が空の行は落とす."""
    rows = []
    seen = set()
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            name = (row.get("chebi_id") or "").strip()
            smi = (row.get("smiles") or "").strip()
            if not name or not smi:
                continue
            if name in seen:
                continue
            seen.add(name)
            rows.append((name, smi, (row.get("inchi") or "").strip()))
    return rows


def filter_valid(rows):
    """RDKit でパースできる SMILES だけ残す (壊れていれば InChI から復元)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    ok, bad = [], []
    n_rescued = 0
    for name, smi, inchi in rows:
        if Chem.MolFromSmiles(smi) is not None:
            ok.append((name, smi))
            continue

        mol = Chem.MolFromInchi(inchi) if inchi else None
        if mol is not None:
            canon = Chem.MolToSmiles(mol)
            if canon and Chem.MolFromSmiles(canon) is not None:
                ok.append((name, canon))
                n_rescued += 1
                continue
        bad.append((name, smi))
    return ok, bad, n_rescued


def smi2coords(smi):
    """SMILES -> (atoms, coordinates, conformer_ok).

    unimol_tools の inner_smi2coords (remove_hs=False) と同じ流れ:
      AddHs -> EmbedMolecule -> MMFF 最適化、失敗したら 2D 座標、
      それも駄目ならゼロ座標。EmbedMolecule にのみ時間制限を足している。
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None, False
    mol = AllChem.AddHs(mol)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    if not atoms:
        return None, None, False

    params = AllChem.ETKDGv3()
    params.randomSeed = CONF_SEED
    params.timeout = int(CONF_TIMEOUT)

    ok = False
    try:
        res = AllChem.EmbedMolecule(mol, params)
        if res == 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:  # noqa: BLE001  MMFF が使えない分子はそのまま
                pass
            coords = mol.GetConformer().GetPositions().astype(np.float32)
            ok = True
        else:
            AllChem.Compute2DCoords(mol)
            coords = mol.GetConformer().GetPositions().astype(np.float32)
    except Exception:  # noqa: BLE001
        coords = np.zeros((len(atoms), 3), dtype=np.float32)

    if len(coords) != len(atoms):
        return None, None, False
    return atoms, coords, ok


def _worker(item):
    name, smi = item
    try:
        atoms, coords, ok = smi2coords(smi)
    except Exception:  # noqa: BLE001
        return name, None, None, False
    return name, atoms, coords, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=DEFAULT_TSV)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--model-name", default="unimolv1")
    ap.add_argument("--remove-hs", action="store_true", help="水素を除いて埋め込む")
    ap.add_argument("--cpu", action="store_true", help="GPU を使わない")
    ap.add_argument("--workers", type=int, default=16, help="コンフォマー生成の並列数")
    ap.add_argument("--limit", type=int, default=0, help="デバッグ用: 先頭 N 件のみ")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "chemical_emb.npz")
    bad_path = os.path.join(args.outdir, "chemical_emb.failed.txt")

    rows = load_smiles(args.tsv)
    if args.limit:
        rows = rows[: args.limit]
    log("rows with SMILES: %d" % len(rows))

    rows, invalid, n_rescued = filter_valid(rows)
    log(
        "RDKit valid: %d (InChI から救済 %d)  invalid: %d"
        % (len(rows), n_rescued, len(invalid))
    )
    failed = list(invalid)

    # ---- コンフォマー生成 (時間制限つき / 並列) ----
    t0 = time.time()
    prepared = []  # (name, atoms, coords, conformer_ok)
    with mp.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap(_worker, rows, chunksize=8), start=1):
            name, atoms, coords, ok = res
            if atoms is None:
                failed.append((name, "conformer generation failed"))
            else:
                prepared.append((name, atoms, coords, ok))
            if i % 1000 == 0:
                log("conformers %d/%d  %.1fs" % (i, len(rows), time.time() - t0))
    n_ok3d = sum(1 for _, _, _, ok in prepared if ok)
    log(
        "conformers done: %d (3D 成功 %d / 2D フォールバック %d)  %.1fs"
        % (len(prepared), n_ok3d, len(prepared) - n_ok3d, time.time() - t0)
    )

    # ---- Uni-Mol 埋め込み ----
    from unimol_tools import UniMolRepr

    encoder = UniMolRepr(
        data_type="molecule",
        model_name=args.model_name,
        remove_hs=args.remove_hs,
        use_cuda=not args.cpu,
    )

    names, vecs, conf_ok = [], [], []
    t1 = time.time()
    for start in range(0, len(prepared), args.batch_size):
        batch = prepared[start : start + args.batch_size]
        payload = {
            "atoms": [list(a) for _, a, _, _ in batch],
            "coordinates": [np.asarray(c, dtype=np.float32) for _, _, c, _ in batch],
        }
        try:
            res = encoder.get_repr(payload, return_atomic_reprs=False)
            cls = np.asarray(res["cls_repr"] if isinstance(res, dict) else res)
            assert len(cls) == len(batch), "cls_repr length mismatch"
            for (name, _, _, ok), v in zip(batch, cls):
                names.append(name)
                vecs.append(np.asarray(v, dtype=np.float32))
                conf_ok.append(ok)
        except Exception as exc:  # noqa: BLE001
            log("batch at %d failed (%s) -> per-molecule retry" % (start, exc))
            for name, atoms, coords, ok in batch:
                try:
                    one = {
                        "atoms": [list(atoms)],
                        "coordinates": [np.asarray(coords, dtype=np.float32)],
                    }
                    res = encoder.get_repr(one, return_atomic_reprs=False)
                    cls = np.asarray(res["cls_repr"] if isinstance(res, dict) else res)
                    names.append(name)
                    vecs.append(np.asarray(cls[0], dtype=np.float32))
                    conf_ok.append(ok)
                except Exception as exc2:  # noqa: BLE001
                    failed.append((name, "unimol: %s" % exc2))
                    log("  failed %s: %s" % (name, exc2))

        el = time.time() - t1
        log(
            "%d/%d  %.1fs  %.1f mol/s"
            % (len(names), len(prepared), el, len(names) / max(el, 1e-9))
        )

    dim = len(vecs[0]) if vecs else 512
    emb = (
        np.stack(vecs).astype(np.float32)
        if vecs
        else np.zeros((0, dim), dtype=np.float32)
    )
    np.savez(
        out_path,
        names=np.array(names, dtype=object),
        embeddings=emb,
        conformer_ok=np.array(conf_ok, dtype=bool),
    )
    log("saved %s  shape=%s" % (out_path, emb.shape))

    with open(bad_path, "w", encoding="utf-8") as fh:
        for name, why in failed:
            fh.write("%s\t%s\n" % (name, why))
    log("failed: %d (%s)" % (len(failed), bad_path))


if __name__ == "__main__":
    sys.exit(main())
