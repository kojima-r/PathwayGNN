"""PathwayCommons SIF ファイル共通ユーティリティ.

PathwayCommons13.All.hgnc.txt は以下のタブ区切り形式:
    PARTICIPANT_A  INTERACTION_TYPE  PARTICIPANT_B  INTERACTION_DATA_SOURCE
    INTERACTION_PUBMED_ID  PATHWAY_NAMES  MEDIATOR_IDS

PARTICIPANT_A / PARTICIPANT_B には
  - HGNC gene symbol (例: A1BG)
  - ChEBI ID (例: CHEBI:17775)
  - 低分子の名前そのもの (例: (+)-artemisinin)  ← CTD 由来など
が混在する。
"""

from __future__ import annotations

import gzip
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_SIF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "raw",
    "PathwayCommons13.All.hgnc.txt",
)

PARTICIPANT_COLS = (0, 2)


def iter_participants(sif_path):
    """SIF の PARTICIPANT_A / PARTICIPANT_B を順に yield する."""
    with open(sif_path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
        if not header.startswith("PARTICIPANT_A"):
            # ヘッダ無しファイルにも対応する
            fh.seek(0)
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            for i in PARTICIPANT_COLS:
                p = cols[i].strip()
                if p:
                    yield p


def collect_participants(sif_path, verbose=True):
    """ユニークな participant の集合を返す."""
    seen = set()
    n = 0
    for p in iter_participants(sif_path):
        n += 1
        seen.add(p)
        if verbose and n % 5_000_000 == 0:
            log("  %d 行分の participant を走査 (unique=%d)" % (n // 2, len(seen)))
    if verbose:
        log("participant 総数=%d, unique=%d" % (n, len(seen)))
    return seen


def split_participants(participants):
    """(chebi_ids, others) に分割する. chebi は 'CHEBI:12345' 形式."""
    chebi = set()
    others = set()
    for p in participants:
        if p.upper().startswith("CHEBI:"):
            chebi.add("CHEBI:" + p.split(":", 1)[1].strip())
        else:
            others.add(p)
    return chebi, others


def log(msg):
    sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stderr.flush()


def download(url, dest, force=False, retries=3, timeout=120):
    """dest が無ければ url をダウンロードする (簡易キャッシュ)."""
    if os.path.exists(dest) and not force:
        log("cache hit: %s" % dest)
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    for attempt in range(1, retries + 1):
        try:
            log("download: %s -> %s" % (url, dest))
            req = urllib.request.Request(url, headers={"User-Agent": "PathwayCommonEmb/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
                total = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    total += len(chunk)
            os.replace(tmp, dest)
            log("  done (%.1f MB)" % (total / 1e6))
            return dest
        except (urllib.error.URLError, OSError) as e:
            log("  失敗 (%d/%d): %s" % (attempt, retries, e))
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
    return dest


def smart_open(path, mode="rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="replace", newline="")
    return open(path, mode, encoding="utf-8", errors="replace", newline="")
