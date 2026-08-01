#!/usr/bin/env python3
"""Download current public inputs and adapt them to GraphCDRScan's legacy names.

COSMIC's Cancer Gene Census now requires login. Supply its v104 TSV/TAR with
--cosmic-cgc; all other inputs are downloaded without credentials. Either the
GRCh37 or the GRCh38 archive works, since only the gene symbols are read and
those are identical in both builds.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data_cdr" / "raw"
SOURCES = RAW / "sources"
DOWNLOADS = {
    "mutations": ("https://cog.sanger.ac.uk/cmp/download/mutations_all_latest.csv.gz", SOURCES / "mutations_all_latest.csv.gz"),
    "models": ("https://cog.sanger.ac.uk/cmp/download/model_list_latest.csv.gz", SOURCES / "model_list_latest.csv.gz"),
    "compounds": ("https://cmp.cog.sanger.ac.uk/download/screened_compounds_rel_8.5.csv", SOURCES / "screened_compounds_rel_8.5.csv"),
    "gdsc1": ("https://cmp.cog.sanger.ac.uk/download/GDSC1_fitted_dose_response_27Oct23.xlsx", SOURCES / "GDSC1_fitted_dose_response_27Oct23.xlsx"),
    "hgnc": ("https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt", SOURCES / "hgnc_complete_set.txt"),
    # Cell Model Passports reports variant coordinates on GRCh38, so the
    # mutational-context lookup must use hg38 (verified: reference alleles match
    # hg38 for 100% of sampled SNVs versus 24.5% for hg19).
    "hg38": ("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit", RAW / "hg38.2bit"),
    "reactome": ("https://reactome.org/download/tools/ReactomeFIs/FIsInGene_04142025_with_annotations.txt.zip", SOURCES / "FIsInGene_04142025_with_annotations.txt.zip"),
    "supplement": ("https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-018-27214-6/MediaObjects/41598_2018_27214_MOESM1_ESM.pdf", SOURCES / "CDRscan_supplementary_information.pdf"),
}


def download(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"exists: {destination.relative_to(ROOT)}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"download: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "GraphCDRScan/1"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(destination)


def hgnc_maps() -> tuple[dict[str, str], dict[str, str]]:
    ensembl, symbols = {}, {}
    with DOWNLOADS["hgnc"][1].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            hgnc_id = row["hgnc_id"].removeprefix("HGNC:")
            if row.get("ensembl_gene_id"):
                ensembl[row["ensembl_gene_id"]] = hgnc_id
            if row.get("symbol"):
                symbols[row["symbol"]] = hgnc_id
    return ensembl, symbols


def write_hgnc() -> None:
    with DOWNLOADS["hgnc"][1].open(newline="", encoding="utf-8") as source, (RAW / "EnsemblToHGNC.tsv").open("w", newline="", encoding="utf-8") as output:
        fields = ["HGNC ID", "Approved symbol", "Ensembl gene ID"]
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in csv.DictReader(source, delimiter="\t"):
            if row.get("ensembl_gene_id"):
                writer.writerow({"HGNC ID": row["hgnc_id"], "Approved symbol": row["symbol"], "Ensembl gene ID": row["ensembl_gene_id"]})


def paper_lists() -> set[str]:
    executable = shutil.which("pdftotext")
    if not executable:
        raise RuntimeError("pdftotext is required")
    with tempfile.TemporaryDirectory(prefix="graphcdrscan-") as temp:
        text_path = Path(temp) / "supplement.txt"
        subprocess.run([executable, "-layout", str(DOWNLOADS["supplement"][1]), str(text_path)], check=True)
        text = text_path.read_text(encoding="utf-8")
    table2 = text[text.index("Supplementary Table S2"):text.index("Supplementary Table S3")]
    cells = [(int(number), cosmic_id) for number, cosmic_id in re.findall(r"(?m)(?:^|\s)(\d{1,3})\s+(\d{6,7})\s+", table2)]
    if len(cells) != 787 or {row[0] for row in cells} != set(range(1, 788)):
        raise RuntimeError("failed to extract 787 cell lines from paper Table S2")
    cells.sort()
    with (RAW / "used_cell_lines.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["cosmic_id"])
        writer.writerows([[cosmic_id] for _, cosmic_id in cells])
    compounds = []
    for line in text[text.index("Supplementary Table S3"):].splitlines():
        match = re.match(r"^\s*(\d{1,3})\s+(.*?)\s{2,}", line)
        if not match or not 1 <= int(match.group(1)) <= 229:
            continue
        number, name = int(match.group(1)), match.group(2).strip()
        tokens = re.findall(r"(?i)(?:^|\s)(\d+|none|several)(?=\s|$)", line)
        sample_size = ""
        if tokens and tokens[-1].isdigit() and int(tokens[-1]) <= 1000:
            sample_size = tokens.pop()
        pubchem_id = tokens[-1] if tokens else ""
        if pubchem_id.lower() in {"none", "several"}:
            pubchem_id = ""
        name = {
            "681640": "Wee1 Inhibitor",
            "BX796": "BX795",
            "Lestauritinib": "Lestaurtinib",
            "SB-505124": "SB505124",
        }.get(name, name)
        compounds.append((number, name, pubchem_id, sample_size))
    if len(compounds) != 229 or {row[0] for row in compounds} != set(range(1, 230)):
        raise RuntimeError("failed to extract 229 compounds from paper Table S3")
    compounds.sort()
    with (RAW / "used_compounds.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["name", "pubchem_id", "sample_size"])
        writer.writerows([row[1:] for row in compounds])
    return {cosmic_id for _, cosmic_id in cells}


def model_maps() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    models, cosmic = {}, {}
    with gzip.open(DOWNLOADS["models"][1], "rt", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            cosmic_id = row.get("COSMIC_ID", "").strip().removesuffix(".0")
            models[row["model_id"]] = row
            if cosmic_id:
                cosmic[row["model_id"]] = cosmic_id
    return models, cosmic


def write_compounds() -> None:
    fields = ["Drug ID", "Screening Site", "Drug Name", "Synonyms", "Target", "Target Pathway"]
    mapping = dict(zip(fields, ["DRUG_ID", "SCREENING_SITE", "DRUG_NAME", "SYNONYMS", "TARGET", "TARGET_PATHWAY"]))
    with DOWNLOADS["compounds"][1].open(newline="", encoding="utf-8-sig") as source, (RAW / "Screened_Compounds.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in csv.DictReader(source):
            writer.writerow({new: row[old] for new, old in mapping.items()})


def xlsx_to_csv(source: Path, output_dir: Path) -> Path:
    office = shutil.which("libreoffice") or shutil.which("soffice")
    if not office:
        raise RuntimeError("LibreOffice is required to convert GDSC XLSX")
    subprocess.run([office, "--headless", "--convert-to", "csv", "--outdir", str(output_dir), str(source)], check=True, stdout=subprocess.DEVNULL)
    result = output_dir / (source.stem + ".csv")
    if not result.exists():
        raise RuntimeError("LibreOffice did not produce CSV")
    return result


def write_dose_response(used: set[str], cosmic_by_model: dict[str, str]) -> None:
    with tempfile.TemporaryDirectory(prefix="graphcdrscan-") as temp:
        converted = xlsx_to_csv(DOWNLOADS["gdsc1"][1], Path(temp))
        with converted.open(newline="", encoding="utf-8-sig") as source, (RAW / "v17_fitted_dose_response.csv").open("w", newline="", encoding="utf-8") as output:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(output, fieldnames=list(reader.fieldnames or []) + ["COSMIC_ID"])
            writer.writeheader()
            for row in reader:
                cosmic_id = cosmic_by_model.get(row["SANGER_MODEL_ID"], "")
                if cosmic_id in used:
                    row["COSMIC_ID"] = cosmic_id
                    writer.writerow(row)


def write_mutations(used: set[str], models: dict[str, dict[str, str]], ensembl: dict[str, str]) -> None:
    fields = ["Gene name", "ID_sample", "HGNC ID", "Mutation genome position", "Mutation Description", "Primary site", "ID_tumour", "Mutation CDS"]
    valid_chromosomes = {str(number) for number in range(1, 25)}
    with gzip.open(DOWNLOADS["mutations"][1], "rt", newline="", encoding="utf-8-sig") as source, (RAW / "CosmicCLP_MutantExport.tsv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in csv.DictReader(source):
            model = models.get(row["model_id"])
            if not model:
                continue
            cosmic_id = model.get("COSMIC_ID", "").removesuffix(".0")
            hgnc_id = ensembl.get(row["ensembl_gene_id"])
            chromosome = row["chromosome"].removeprefix("chr")
            chromosome = {"X": "23", "Y": "24"}.get(chromosome, chromosome)
            position = row["position"].removesuffix(".0")
            reference, alternative = row["reference"], row["alternative"]
            if cosmic_id not in used or not hgnc_id or chromosome not in valid_chromosomes or not position.isdigit() or not reference or not alternative:
                continue
            start = int(position)
            end = start + max(len(reference), 1) - 1
            kind = "Insertion" if row["type"].lower() == "insertion" or len(alternative) > len(reference) else "Deletion" if row["type"].lower() == "deletion" or len(reference) > len(alternative) else "Substitution"
            writer.writerow({"Gene name": row["gene_symbol"], "ID_sample": cosmic_id, "HGNC ID": hgnc_id, "Mutation genome position": f"{chromosome}:{start}-{end}", "Mutation Description": kind, "Primary site": model.get("tissue", ""), "ID_tumour": cosmic_id, "Mutation CDS": f"c.{start}{reference}>{alternative}"})


def write_reactome(symbols: dict[str, str]) -> None:
    with zipfile.ZipFile(DOWNLOADS["reactome"][1]) as archive:
        with archive.open(archive.namelist()[0]) as binary, io.TextIOWrapper(binary, encoding="utf-8") as source, (RAW / "reactome_rev2.graph.tsv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            for row in csv.DictReader(source, delimiter="\t"):
                gene1, gene2 = symbols.get(row["Gene1"]), symbols.get(row["Gene2"])
                if gene1 and gene2:
                    writer.writerow([gene1, row["Annotation"], gene2])


def write_cgc(path: Path) -> None:
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith((".tsv", ".csv", ".tsv.gz", ".csv.gz")) and "readme" not in member.name.lower()]
            if not members:
                raise RuntimeError("CGC table not found in TAR")
            handle = archive.extractfile(members[0])
            if handle is None:
                raise RuntimeError("CGC table is unreadable")
            data = handle.read()
            if members[0].name.endswith(".gz"):
                data = gzip.decompress(data)
            text = data.decode("utf-8-sig")
    else:
        text = path.read_text(encoding="utf-8-sig")
    delimiter = "\t" if text[:8192].count("\t") > text[:8192].count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    candidates = ("Gene Symbol", "GENE_SYMBOL", "Gene symbol", "gene_symbol")
    column = next((candidate for candidate in candidates if candidate in (reader.fieldnames or [])), None)
    if not column:
        raise RuntimeError(f"CGC gene-symbol column not found: {reader.fieldnames}")
    with (RAW / "cancer_gene_census.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["Gene Symbol"])
        writer.writerows([row[column]] for row in reader if row.get(column))


def manifest() -> None:
    destination = RAW / "SHA256SUMS"
    with destination.open("w", encoding="utf-8") as output:
        for path in sorted(item for item in RAW.rglob("*") if item.is_file() and item != destination):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            output.write(f"{digest.hexdigest()}  {path.relative_to(RAW)}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmic-cgc", type=Path)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir(parents=True, exist_ok=True)
    if not args.no_download:
        for url, destination in DOWNLOADS.values():
            download(url, destination, args.force_download)
    missing = [path for _, path in DOWNLOADS.values() if not path.exists()]
    if missing:
        print("missing source files:", *missing, sep="\n  ", file=sys.stderr)
        return 2
    used = paper_lists()
    ensembl, symbols = hgnc_maps()
    models, cosmic_by_model = model_maps()
    write_hgnc()
    write_compounds()
    write_dose_response(used, cosmic_by_model)
    write_mutations(used, models, ensembl)
    write_reactome(symbols)
    if args.cosmic_cgc:
        write_cgc(args.cosmic_cgc.resolve())
    elif not (RAW / "cancer_gene_census.csv").exists():
        print("notice: cancer_gene_census.csv requires a logged-in COSMIC v104 download; rerun with --cosmic-cgc", file=sys.stderr)
    manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
