from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

def _normalize(identifier: str) -> str:
    return identifier.strip().split(".", 1)[0]

def map_ensembl_ids(
    input_path: str | Path,
    output_path: str | Path,
    cache_path: str | Path,
    species: str = "human",
    batch_size: int = 1000,
) -> dict[str, Any]:
    """Map an ordered Ensembl list to HGNC IDs using a persistent MyGene snapshot."""
    source, output, cache = Path(input_path), Path(output_path), Path(cache_path)
    identifiers = [_normalize(line.split("\t")[0]) for line in source.read_text().splitlines() if line.strip()]
    digest = hashlib.sha256(("\n".join(identifiers)+"\n").encode()).hexdigest()
    if cache.exists():
        payload = json.loads(cache.read_text())
        if payload.get("input_sha256") != digest:
            raise ValueError("The mapping cache belongs to a different ordered Ensembl ID list")
        results = payload["results"]
        reused = True
    else:
        try:
            import mygene
        except ImportError as error:
            raise RuntimeError("Install the paper extra or mygene before cancer-map-ids") from error
        client = mygene.MyGeneInfo()
        results = []
        for start in range(0, len(identifiers), batch_size):
            results.extend(client.querymany(
                identifiers[start : start + batch_size],
                scopes="ensembl.gene",
                fields="symbol,hgnc,entrezgene,ensembl.gene",
                species=species,
                as_dataframe=False,
                returnall=False,
                verbose=False,
            ))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({
            "provider": "MyGene.info",
            "species": species,
            "scope": "ensembl.gene",
            "fields": ["symbol", "hgnc", "entrezgene", "ensembl.gene"],
            "input_sha256": digest,
            "results": results,
        }, indent=2))
        reused = False
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(_normalize(str(result.get("query", ""))), []).append(result)
    rows=[]; mapped=ambiguous=0
    for position,(original,identifier) in enumerate(zip([x.split("\t")[0].strip() for x in source.read_text().splitlines() if x.strip()],identifiers)):
        hits=[x for x in grouped.get(identifier,[]) if not x.get("notfound")]
        if hits: mapped+=1
        if len(hits)>1: ambiguous+=1
        if not hits:
            rows.append([position,original,identifier,"","","","not_found",0])
        else:
            for hit in hits:
                hgnc=str(hit.get("hgnc",""))
                hgnc_numeric=hgnc.split(":",1)[-1] if hgnc else ""
                rows.append([position,original,identifier,hgnc_numeric,hit.get("symbol",""),hit.get("entrezgene",""),"ambiguous" if len(hits)>1 else "mapped",len(hits)])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w",newline="") as handle:
        writer=csv.writer(handle,delimiter="\t")
        writer.writerow(["position","input_id","ensembl_gene","hgnc_id","symbol","entrezgene","status","hit_count"])
        writer.writerows(rows)
    summary={"input":str(source),"output":str(output),"cache":str(cache),"cache_reused":reused,"total":len(identifiers),"mapped":mapped,"unmapped":len(identifiers)-mapped,"ambiguous":ambiguous,"input_sha256":digest}
    (output.with_suffix(output.suffix+".metadata.json")).write_text(json.dumps(summary,indent=2))
    return summary
