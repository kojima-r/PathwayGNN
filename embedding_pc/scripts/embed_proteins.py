#!/usr/bin/env python3
"""gene_uniprot_seq.tsv のアミノ酸配列を ESMC-6B で埋め込む.

esm_test.py と同じ手順 (最終 layer の hidden state -> special token 除外 ->
残基方向 mean pooling) で、1 gene symbol あたり 1 本の [2560] ベクトルを作る。

MAX_WINDOW を超える配列は窓分割して各窓を埋め込み、残基数で重み付き平均する
(全残基にわたる平均という意味を保ったまま attention の O(L^2) を回避する)。

使い方 (GPU 3 枚に分散する例):
    for i in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$i python scripts/embed_proteins.py \
          --shard $i --num-shards 3 &
    done; wait

出力: processed/protein_emb.shard{i}.npz  (names, embeddings, lengths)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "biohub/ESMC-6B"
HIDDEN_DIM = 2560

# 1 回の forward に流す最大残基数。これを超える配列は窓分割する。
MAX_WINDOW = 4096

csv.field_size_limit(1 << 30)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_TSV = os.path.join(ROOT, "data", "gene_uniprot_seq.tsv")
DEFAULT_OUTDIR = os.path.join(ROOT, "processed")


def log(msg):
    print("[embed_proteins] %s" % msg, flush=True)


def load_sequences(path):
    """(gene_symbol, sequence) のリストを返す. 配列が空の行は落とす."""
    rows = []
    seen = set()
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            name = (row.get("gene_symbol") or "").strip()
            seq = "".join((row.get("sequence") or "").split()).upper()
            if not name or not seq:
                continue
            if name in seen:
                continue
            seen.add(name)
            rows.append((name, seq))
    return rows


def load_model(dtype=torch.bfloat16):
    log("loading %s" % MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model = model.cuda()
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def _embed_window(sequence, tokenizer, model, special_ids):
    """1 窓分の配列を埋め込み (残基平均ベクトル, 残基数) を返す."""
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # 最終 Transformer layer: [token_length, hidden_dim]
    hidden = outputs.hidden_states[-1][0]
    input_ids = inputs["input_ids"][0]

    # BOS / EOS / PAD などを除外
    residue_mask = torch.tensor(
        [int(t) not in special_ids for t in input_ids],
        dtype=torch.bool,
        device=input_ids.device,
    )
    residue = hidden[residue_mask]
    n = int(residue.shape[0])
    if n == 0:
        return None, 0

    vec = residue.float().mean(dim=0).cpu().numpy()
    return vec, n


def embed_sequence(sequence, tokenizer, model, special_ids, max_window=MAX_WINDOW):
    """配列全体の残基平均ベクトル [hidden_dim] を返す.

    max_window を超える場合は窓分割し、窓内残基数で重み付き平均する。
    """
    if len(sequence) <= max_window:
        vec, n = _embed_window(sequence, tokenizer, model, special_ids)
        return vec

    total = np.zeros(HIDDEN_DIM, dtype=np.float64)
    total_n = 0
    for start in range(0, len(sequence), max_window):
        chunk = sequence[start : start + max_window]
        vec, n = _embed_window(chunk, tokenizer, model, special_ids)
        if vec is None:
            continue
        total += vec.astype(np.float64) * n
        total_n += n
    if total_n == 0:
        return None
    return (total / total_n).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=DEFAULT_TSV)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-window", type=int, default=MAX_WINDOW)
    ap.add_argument("--limit", type=int, default=0, help="デバッグ用: 先頭 N 件のみ")
    ap.add_argument(
        "--checkpoint-every", type=int, default=1000, help="途中保存の間隔 (0 で無効)"
    )
    ap.add_argument("--resume", action="store_true", help="途中保存から再開する")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "protein_emb.shard%d.npz" % args.shard)
    ckpt_path = out_path + ".partial.npz"

    rows = load_sequences(args.tsv)
    if args.limit:
        rows = rows[: args.limit]
    log("total sequences: %d" % len(rows))

    # 長い配列ほど時間がかかるので、shard 間の負荷が偏らないよう
    # 長さ順に並べてラウンドロビンで割り当てる
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i][1]))
    mine = [rows[i] for k, i in enumerate(order) if k % args.num_shards == args.shard]
    log("shard %d/%d -> %d sequences" % (args.shard, args.num_shards, len(mine)))

    names, vecs, lengths = [], [], []
    done = set()
    if args.resume and os.path.exists(ckpt_path):
        ck = np.load(ckpt_path, allow_pickle=True)
        names = list(ck["names"])
        vecs = list(ck["embeddings"])
        lengths = list(ck["lengths"])
        done = set(names)
        log("resumed from checkpoint: %d done" % len(done))

    tokenizer, model = load_model()
    special_ids = set(tokenizer.all_special_ids)

    t0 = time.time()
    n_new = 0
    for idx, (name, seq) in enumerate(mine):
        if name in done:
            continue
        try:
            vec = embed_sequence(
                seq, tokenizer, model, special_ids, max_window=args.max_window
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log("OOM on %s (len=%d) -> retry with window 1024" % (name, len(seq)))
            vec = embed_sequence(seq, tokenizer, model, special_ids, max_window=1024)

        if vec is None:
            log("skip %s: no residue token" % name)
            continue

        names.append(name)
        vecs.append(vec.astype(np.float32))
        lengths.append(len(seq))
        n_new += 1

        if args.checkpoint_every and n_new % args.checkpoint_every == 0:
            np.savez(
                ckpt_path,
                names=np.array(names, dtype=object),
                embeddings=np.stack(vecs).astype(np.float32),
                lengths=np.array(lengths, dtype=np.int64),
            )
            el = time.time() - t0
            log(
                "%d/%d  %.1fs  %.2f seq/s  (checkpoint)"
                % (len(names), len(mine), el, n_new / el)
            )

    emb = (
        np.stack(vecs).astype(np.float32)
        if vecs
        else np.zeros((0, HIDDEN_DIM), dtype=np.float32)
    )
    np.savez(
        out_path,
        names=np.array(names, dtype=object),
        embeddings=emb,
        lengths=np.array(lengths, dtype=np.int64),
    )
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    log("saved %s  shape=%s  elapsed=%.1fs" % (out_path, emb.shape, time.time() - t0))


if __name__ == "__main__":
    sys.exit(main())
