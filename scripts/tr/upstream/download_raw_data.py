#!/usr/bin/env python3
"""Download the public sources of the target-repositioning corpus into data_tr/raw.

Standard library only, so the corpus can be acquired without the training
environment. Every transfer is resumable: the LINCS Level 5 matrix is 21.3 GB and
NCBI resets long-running connections, so each file is fetched into a ``.part``
alongside its destination and continued with a Range request until the announced
length is reached.

Two of the sources the reference preprocessing used are gone and are replaced
here by equivalents that are still served (both recorded in the manifest):

* KEGG DISEASE -> OMIM: LinkDB's ``omim_disease.list`` is no longer published;
  the same links come from the KEGG REST endpoint ``/link/omim/ds``.
* OMIM -> DOID: ``src/DOreports/xrefs_in_DO.tsv`` was removed from the Disease
  Ontology repository; the OMIM cross-references are read out of ``HumanDO.obo``.

The gene-symbol table is the current HGNC complete set rather than the reference
implementation's 2023-10-29 custom export, so symbol renaming follows today's
nomenclature.

The therapeutic-target labels come from this repository's release mirror: the
Kyutech group's ``target_disease_data.zip`` is no longer served (see the
``labels`` entry below).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data_tr" / "raw"
# The GEO download CGI ignores Range requests, so a reset transfer would have to
# start over; the FTP mirror answers 206 and is therefore what a 21 GB file needs.
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"

# name -> (url, destination, decompress?)
DOWNLOADS: dict[str, tuple[str, Path, bool]] = {
    # LINCS L1000 (GSE92742). Level 5 = one moderated z-score profile per
    # signature; the build stage reads only the 978 landmark rows, but GCTX
    # stores the whole 473,647 x 12,328 matrix in one HDF5 file.
    "level5": (
        GEO + "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz",
        RAW / "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
        True,
    ),
    "sig_info": (
        GEO + "GSE92742_Broad_LINCS_sig_info.txt.gz",
        RAW / "GSE92742_Broad_LINCS_sig_info.txt",
        True,
    ),
    "gene_info": (
        GEO + "GSE92742_Broad_LINCS_gene_info.txt.gz",
        RAW / "GSE92742_Broad_LINCS_gene_info.txt",
        True,
    ),
    # CREEDS manual disease signatures (up/down gene lists per GEO study).
    "creeds": (
        "https://maayanlab.cloud/CREEDS/download/disease_signatures-v1.0.json",
        RAW / "disease_signatures-v1.0.json",
        False,
    ),
    # KEGG DISEASE -> OMIM, the replacement for LinkDB's omim_disease.list.
    "kegg_omim": (
        "https://rest.genome.jp/link/omim/ds",
        RAW / "kegg_disease_omim.list",
        False,
    ),
    # Disease Ontology, read for its OMIM xrefs (OMIM -> DOID).
    "do": (
        "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/HumanDO.obo",
        RAW / "HumanDO.obo",
        False,
    ),
    # HGNC: approved symbols plus previous/alias symbols, used to map every gene
    # name in the corpus onto one approved symbol.
    "hgnc": (
        "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
        RAW / "hgnc_complete_set.txt",
        False,
    ),
    # Pathway Commons v12, the knowledge graph. Byte-identical to the copy under
    # data_cancer/; kept here so data_tr/raw is self-contained.
    "graph": (
        "https://download.baderlab.org/PathwayCommons/PC2/v12/PathwayCommons12.All.hgnc.sif.gz",
        RAW / "PathwayCommons12.All.hgnc.sif",
        True,
    ),
    # The therapeutic-target labels. Originally published by the Kyutech group
    # (labo.bio.kyutech.ac.jp/~yamani/target_repositioning/target_disease_data.zip),
    # which no longer resolves and has no archived copy; this release mirrors the
    # DOID-keyed tables. Unpacks to {inhibitory,activatory}_target_disease.tsv.
    "labels": (
        "https://github.com/kojima-r/PathwayGNN/releases/download/v1/data_tr__target_disease.zip",
        RAW / "data_tr__target_disease.zip",
        False,
    ),
}
LABEL_MEMBERS = ("inhibitory_target_disease.tsv", "activatory_target_disease.tsv")
AGENT = {"User-Agent": "PathwayGNN/1"}


def _remote_length(url: str) -> int | None:
    request = urllib.request.Request(url, headers=AGENT)
    try:
        with urllib.request.urlopen(request) as response:
            length = response.headers.get("Content-Length")
    except urllib.error.URLError:
        return None
    return int(length) if length and length.isdigit() else None


def download(url: str, destination: Path, decompress: bool, force: bool, attempts: int) -> None:
    """Fetch ``url``, resuming a partial transfer, and optionally gunzip it."""
    if destination.exists() and not force:
        print(f"exists: {destination.relative_to(ROOT)}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".gz.part" if decompress else ".part"
    partial = destination.with_suffix(destination.suffix + suffix)
    if force:
        partial.unlink(missing_ok=True)
    total = _remote_length(url)
    print(f"download: {url}" + (f" ({total / 2**30:.1f} GiB)" if total else ""))
    for attempt in range(1, attempts + 1):
        have = partial.stat().st_size if partial.exists() else 0
        if total is not None and have >= total:
            break
        headers = dict(AGENT)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request) as response, partial.open("ab" if have else "wb") as output:
                if have and response.status != 206:
                    # The server ignored the Range request; start over.
                    output.seek(0)
                    output.truncate()
                shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as error:
            got = partial.stat().st_size if partial.exists() else 0
            if attempt == attempts or got <= have:
                raise
            print(f"  retry {attempt}/{attempts} at {got:,} bytes after {error}")
            time.sleep(min(30, 2 * attempt))
            continue
        if total is None:
            break
    got = partial.stat().st_size if partial.exists() else 0
    if total is not None and got != total:
        raise RuntimeError(f"{destination.name}: got {got:,} of {total:,} bytes")
    if decompress:
        print(f"  gunzip -> {destination.name}")
        with gzip.open(partial, "rb") as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
        partial.unlink()
    else:
        partial.replace(destination)


def extract_labels(archive: Path, force: bool) -> None:
    """Unpack the two label tables next to the archive."""
    if all((archive.parent / name).exists() for name in LABEL_MEMBERS) and not force:
        print(f"exists: {', '.join(LABEL_MEMBERS)}")
        return
    with zipfile.ZipFile(archive) as bundle:
        for name in LABEL_MEMBERS:
            member = next((item for item in bundle.namelist() if Path(item).name == name), None)
            if member is None:
                raise RuntimeError(f"{archive.name} does not contain {name}")
            with bundle.open(member) as source, (archive.parent / name).open("wb") as output:
                shutil.copyfileobj(source, output)
            print(f"  unzip -> {name}")


def manifest() -> dict[str, str]:
    """Hash every file in data_tr/raw, newest content wins, and write SHA256SUMS."""
    destination = RAW / "SHA256SUMS"
    digests: dict[str, str] = {}
    for path in sorted(item for item in RAW.rglob("*") if item.is_file() and item != destination):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        digests[str(path.relative_to(RAW))] = digest.hexdigest()
    with destination.open("w", encoding="utf-8") as output:
        for name, digest in digests.items():
            output.write(f"{digest}  {name}\n")
    return digests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(DOWNLOADS), help="fetch a subset")
    parser.add_argument("--force-download", action="store_true", help="ignore existing files")
    parser.add_argument("--attempts", type=int, default=100, help="resume attempts per file")
    parser.add_argument("--no-manifest", action="store_true", help="skip writing SHA256SUMS")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    selected = args.only or list(DOWNLOADS)
    for name in selected:
        url, destination, decompress = DOWNLOADS[name]
        download(url, destination, decompress, args.force_download, args.attempts)
        if name == "labels":
            extract_labels(destination, args.force_download)
    missing = [str(DOWNLOADS[name][1]) for name in selected if not DOWNLOADS[name][1].exists()]
    if missing:
        print("missing:", *missing, sep="\n  ", file=sys.stderr)
        return 2
    if not args.no_manifest:
        print(json.dumps(manifest(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
