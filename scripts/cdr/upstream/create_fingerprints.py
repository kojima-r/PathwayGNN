#!/usr/bin/env python3
"""Create a PaDEL-compatible 3072-bit table using RDKit and PubChem CIDs."""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors


def fetch_smiles(cids: list[int]) -> dict[int, str]:
    result: dict[int, str] = {}
    for start in range(0, len(cids), 100):
        batch = cids[start : start + 100]
        query = ",".join(str(cid) for cid in batch)
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{urllib.parse.quote(query)}/property/ConnectivitySMILES/JSON"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "GraphCDRScan/1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
        for row in payload.get("PropertyTable", {}).get("Properties", []):
            if row.get("ConnectivitySMILES"):
                result[int(row["CID"])] = row["ConnectivitySMILES"]
        if start + 100 < len(cids):
            time.sleep(0.2)
    return result


def bits(molecule) -> list[int]:
    vectors = [
        AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=1024),
        Chem.RDKFingerprint(molecule, fpSize=1024),
        rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(molecule, nBits=1024),
    ]
    values: list[int] = []
    for vector in vectors:
        values.extend([int(value) for value in vector])
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compounds", default="data_cdr/raw/Screened_Compounds.csv")
    parser.add_argument("--used-compounds", default="data_cdr/raw/used_compounds.csv")
    parser.add_argument("--output", default="data_cdr/raw/fingerprints.csv")
    args = parser.parse_args()

    with open(args.used_compounds, newline="", encoding="utf-8-sig") as handle:
        used = list(csv.DictReader(handle))
    with open(args.compounds, newline="", encoding="utf-8-sig") as handle:
        annotations = {row["Drug Name"]: row for row in csv.DictReader(handle)}

    records = []
    missing_names = []
    for row in used:
        if not row.get("pubchem_id"):
            continue
        name = row["name"]
        annotation = annotations.get(name)
        if annotation is None:
            missing_names.append(name)
            continue
        records.append((int(annotation["Drug ID"]), int(row["pubchem_id"]), name))
    if missing_names:
        raise RuntimeError("No current compound annotation for: " + ", ".join(missing_names))

    smiles = fetch_smiles(sorted({cid for _, cid, _ in records}))
    missing_cids = sorted({cid for _, cid, _ in records} - smiles.keys())
    if missing_cids:
        raise RuntimeError(f"PubChem returned no ConnectivitySMILES for CIDs: {missing_cids}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Name"] + [f"RDKit_{index:04d}" for index in range(3072)])
        written = set()
        for drug_id, cid, _ in records:
            if drug_id in written:
                continue
            molecule = Chem.MolFromSmiles(smiles[cid])
            if molecule is None:
                raise RuntimeError(f"RDKit could not parse PubChem CID {cid}")
            writer.writerow([drug_id] + bits(molecule))
            written.add(drug_id)
    print(f"wrote {len(written)} RDKit fingerprints to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
