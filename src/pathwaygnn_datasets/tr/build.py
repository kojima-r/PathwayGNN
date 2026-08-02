"""Build ``data_tr/processed`` from the public sources in ``data_tr/raw``.

This is the stage before :mod:`pathwaygnn_datasets.tr.prepare`, and it follows
the reference preprocessing notebooks of ``target-repositioning-share``:

1. **gene symbols** — one converter from the HGNC complete set: every previous
   or alias symbol maps onto its approved symbol, ambiguous synonyms are
   resolved the way the reference does, and every gene name in the corpus is
   passed through it and upper-cased.
2. **graph** — the Pathway Commons SIF with both endpoints renamed.
3. **perturbation signatures** — LINCS L1000 (GSE92742) Level 5, landmark genes
   only, ``trt_sh.cgs`` for knockdown and ``trt_oe`` for overexpression,
   averaged over replicates, time points and doses.
4. **disease signatures** — CREEDS manual disease signatures, exploded into one
   row per (disease, gene) and averaged across studies of the same disease.
5. **labels** — KEGG DISEASE ids converted to DOID through OMIM, then the
   gene x disease grid outside the positive set becomes the negatives.

Two decisions are recorded in ``build_manifest.json`` because they change the
corpus rather than just its encoding:

* ``per_cell_line`` — ``(pert_iname, cell_id)`` is the unit of a perturbation
  profile. Setting it false averages the cell lines away, leaving one row per
  perturbed gene.
* ``human_only`` — keep only ``organism == "human"`` disease signatures.

The labels were published by the Kyutech group as ``target_disease_data.zip``
(``labo.bio.kyutech.ac.jp/~yamani/target_repositioning/``). That URL no longer
resolves, so the downloader takes the DOID-keyed tables from this repository's
release mirror and step 5 only re-renames their gene column. Point
``kegg_labels`` at the KEGG-keyed ``.txt`` originals if you have them, and the
KEGG -> OMIM -> DOID conversion runs instead.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

LANDMARK_COLUMN = "pr_is_lm"
PERT_TYPES = {"knockdown": "trt_sh.cgs", "overexpression": "trt_oe"}
GCTX_MATRIX = "0/DATA/0/matrix"
GCTX_ROW_IDS = "0/META/ROW/id"
GCTX_COL_IDS = "0/META/COL/id"


# --------------------------------------------------------------------------- #
# gene symbols
# --------------------------------------------------------------------------- #
def _split_symbols(field: str) -> list[str]:
    """HGNC packs multiple symbols into one field; the exports differ in separator."""
    return [part.strip() for part in re.split(r"[|,]", field) if part.strip()]


def build_symbol_converter(hgnc_path: Path) -> dict[str, str]:
    """Map every previous/alias symbol onto an approved symbol.

    Mirrors ``04_convert_to_approved_symbol``: synonyms that are themselves
    approved symbols are dropped; a synonym claimed by several approved symbols
    prefers the alias rows over the previous rows, and if that still leaves more
    than one candidate the reference keeps the ambiguity visible by renaming to
    ``"<synonym>:<approved>/<approved>"`` rather than picking a winner.
    """
    approved: set[str] = set()
    claims: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"Previous": set(), "Alias": set()})
    with hgnc_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("status") != "Approved":
                continue
            symbol = row["symbol"].strip()
            approved.add(symbol)
            for kind, column in (("Previous", "prev_symbol"), ("Alias", "alias_symbol")):
                for synonym in _split_symbols(row.get(column) or ""):
                    claims[synonym][kind].add(symbol)
    converter: dict[str, str] = {}
    for synonym, kinds in claims.items():
        if synonym in approved:
            continue
        candidates = kinds["Alias"] if kinds["Alias"] and kinds["Previous"] else kinds["Alias"] | kinds["Previous"]
        if len(candidates) == 1:
            converter[synonym] = next(iter(candidates))
        else:
            converter[synonym] = f"{synonym}:{'/'.join(sorted(candidates))}"
    return converter


def rename(symbol: str, converter: dict[str, str]) -> str:
    return converter.get(symbol, symbol).upper()


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #
def build_graph(sif_path: Path, converter: dict[str, str], output: Path) -> dict[str, int]:
    nodes: set[str] = set()
    relations: set[str] = set()
    rows = 0
    with sif_path.open(encoding="utf-8", newline="") as source, output.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for row in csv.reader(source, delimiter="\t"):
            if len(row) < 3 or row[1] == "INTERACTION_TYPE":
                continue
            source_node, relation, target_node = row[0], row[1], row[2]
            source_node, target_node = rename(source_node, converter), rename(target_node, converter)
            writer.writerow([source_node, relation, target_node])
            nodes.update((source_node, target_node))
            relations.add(relation)
            rows += 1
    return {"graph_rows": rows, "graph_nodes": len(nodes), "graph_relations": len(relations)}


# --------------------------------------------------------------------------- #
# LINCS L1000 perturbation signatures
# --------------------------------------------------------------------------- #
def _landmark_genes(gene_info_path: Path) -> tuple[list[str], list[str]]:
    """Return the landmark gene ids and their symbols, in file order."""
    ids, symbols = [], []
    with gene_info_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row[LANDMARK_COLUMN] == "1":
                ids.append(row["pr_gene_id"])
                symbols.append(row["pr_gene_symbol"])
    return ids, symbols


def _signature_index(sig_info_path: Path, pert_type: str) -> dict[str, tuple[str, str]]:
    """sig_id -> (pert_iname, cell_id) for one perturbation type."""
    index: dict[str, tuple[str, str]] = {}
    with sig_info_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["pert_type"] == pert_type:
                index[row["sig_id"]] = (row["pert_iname"], row["cell_id"])
    return index


def _read_gctx_columns(
    gctx_path: Path, gene_ids: list[str], sig_ids: set[str], chunk: int = 8192
) -> tuple[list[str], np.ndarray]:
    """Read the landmark rows of the selected signatures out of the GCTX matrix.

    GCTX stores the matrix transposed: ``matrix[signature, gene]``. h5py allows a
    fancy index on one axis only, so signatures are read in contiguous blocks and
    the landmark genes are selected as the (sorted) second index.
    """
    try:
        import h5py
    except ModuleNotFoundError as error:  # pragma: no cover - dependency hint
        raise ModuleNotFoundError(
            "reading the LINCS GCTX matrix needs h5py: pip install -e '.[tr-upstream]'"
        ) from error

    with h5py.File(gctx_path, "r") as handle:
        row_ids = [value.decode() if isinstance(value, bytes) else str(value) for value in handle[GCTX_ROW_IDS][:]]
        col_ids = [value.decode() if isinstance(value, bytes) else str(value) for value in handle[GCTX_COL_IDS][:]]
        gene_position = {name: index for index, name in enumerate(row_ids)}
        missing = [gene for gene in gene_ids if gene not in gene_position]
        if missing:
            raise KeyError(f"{len(missing)} landmark genes are absent from the matrix: {missing[:5]}")
        gene_columns = np.asarray([gene_position[gene] for gene in gene_ids], dtype=np.int64)
        order = np.argsort(gene_columns)
        sorted_columns, restore = gene_columns[order], np.argsort(order)

        selected = np.asarray([index for index, name in enumerate(col_ids) if name in sig_ids], dtype=np.int64)
        if selected.size != len(sig_ids):
            raise KeyError(f"matrix holds {selected.size} of the {len(sig_ids)} requested signatures")
        matrix = handle[GCTX_MATRIX]
        values = np.empty((selected.size, len(gene_ids)), dtype=np.float32)
        cursor = 0
        for start in range(0, matrix.shape[0], chunk):
            stop = min(start + chunk, matrix.shape[0])
            take = selected[(selected >= start) & (selected < stop)]
            if not take.size:
                continue
            block = matrix[start:stop, sorted_columns]
            values[cursor : cursor + take.size] = block[take - start][:, restore]
            cursor += take.size
        return [col_ids[index] for index in selected], values


def build_signature(
    gctx_path: Path,
    sig_info_path: Path,
    gene_info_path: Path,
    pert_type: str,
    converter: dict[str, str],
    output: Path,
    per_cell_line: bool,
    profiles: tuple[list[str], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Average the Level 5 profiles of one perturbation type and write them out.

    ``profiles`` is an already-read ``(sig_ids, values)`` pair covering at least
    this perturbation type, so that one pass over the 23 GB matrix can serve both
    knockdown and overexpression.
    """
    gene_ids, gene_symbols = _landmark_genes(gene_info_path)
    index = _signature_index(sig_info_path, pert_type)
    if profiles is None:
        sig_ids, values = _read_gctx_columns(gctx_path, gene_ids, set(index))
    else:
        keep = [position for position, sig_id in enumerate(profiles[0]) if sig_id in index]
        sig_ids, values = [profiles[0][position] for position in keep], profiles[1][keep]

    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for position, sig_id in enumerate(sig_ids):
        pert_iname, cell_id = index[sig_id]
        groups[(pert_iname, cell_id) if per_cell_line else (pert_iname,)].append(position)

    columns = [rename(symbol, converter) for symbol in gene_symbols]
    header = (["pert_iname", "cell_id"] if per_cell_line else ["pert_iname"]) + columns
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for key in sorted(groups):
            mean = values[groups[key]].mean(axis=0)
            writer.writerow(list(key) + [f"{value:.7g}" for value in mean])
    return {
        "profiles": len(sig_ids),
        "rows": len(groups),
        "genes": len(columns),
        "pert_type": pert_type,
    }


# --------------------------------------------------------------------------- #
# CREEDS disease signatures
# --------------------------------------------------------------------------- #
def build_disease_signature(
    creeds_path: Path, converter: dict[str, str], output: Path, human_only: bool
) -> dict[str, Any]:
    records = json.loads(creeds_path.read_text(encoding="utf-8"))
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    used = 0
    for record in records:
        if not record.get("do_id"):
            continue
        if human_only and record.get("organism") != "human":
            continue
        used += 1
        for key in ("down_genes", "up_genes"):
            for gene, value in record.get(key) or []:
                totals[(record["do_id"], rename(gene, converter))].append(float(value))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["do_id", "gene_name", "expression"])
        for do_id, gene in sorted(totals):
            writer.writerow([do_id, gene, f"{float(np.mean(totals[(do_id, gene)])):.17g}"])
    return {
        "signatures_used": used,
        "diseases": len({do_id for do_id, _ in totals}),
        "rows": len(totals),
    }


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def kegg_to_doid(kegg_omim_path: Path, obo_path: Path) -> dict[str, set[str]]:
    """KEGG DISEASE -> DOID, through OMIM.

    ``kegg_disease_omim.list`` is ``ds:H00001<TAB>omim:104300<TAB>relation`` and
    the cross-references come out of the ontology itself, since the
    ``xrefs_in_DO.tsv`` report the reference implementation used was removed.
    Current releases write OMIM as ``xref: MIM:104300`` (``MIM:PS...`` for a
    phenotypic series); older ones used ``OMIM:``, and both are accepted. The
    handful of direct ``xref: KEGG:H#####`` links are used as well — the numeric
    ``KEGG:#####`` xrefs are pathway ids, not diseases, and are ignored.
    """
    omim_to_kegg: dict[str, set[str]] = defaultdict(set)
    with kegg_omim_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 2:
                continue
            first, second = row[0], row[1]
            kegg, omim = (first, second) if first.startswith("ds:") else (second, first)
            omim_to_kegg[omim.replace("omim:", "OMIM:")].add(kegg.replace("ds:", ""))

    mapping: dict[str, set[str]] = defaultdict(set)
    doid = None
    for line in obo_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("id: DOID:"):
            doid = line[4:].strip()
        elif line.startswith("[") and not line.startswith("[Term]"):
            doid = None
        elif not doid or not line.startswith("xref: "):
            continue
        else:
            xref = line[6:].strip()
            if xref.startswith(("MIM:", "OMIM:")):
                omim = "OMIM:" + xref.split(":", 1)[1].removeprefix("PS")
                for kegg in omim_to_kegg.get(omim, ()):
                    mapping[kegg].add(doid)
            elif xref.startswith("KEGG:H"):
                mapping[xref.removeprefix("KEGG:")].add(doid)
    return mapping


def _label_grid(pairs: set[tuple[str, str]]) -> list[tuple[str, str, int]]:
    """Positives plus every other gene x disease combination as a negative."""
    genes = sorted({gene for gene, _ in pairs})
    diseases = sorted({disease for _, disease in pairs})
    return sorted(
        (gene, disease, int((gene, disease) in pairs)) for gene in genes for disease in diseases
    )


def build_labels(
    kegg_path: Path | None,
    converted_path: Path | None,
    converter: dict[str, str],
    kegg_map: dict[str, set[str]],
    output: Path,
) -> dict[str, Any]:
    """Write ``gene<TAB>doid<TAB>label``.

    ``kegg_path`` is the KEGG-keyed table as the Kyutech group published it, and
    is converted through OMIM to DOID. ``converted_path`` is the DOID-keyed table
    of this repository's release mirror, whose gene column is only re-renamed
    with the current HGNC converter. Either way the positives define the grid and
    everything outside it becomes a negative.
    """
    pairs: set[tuple[str, str]] = set()
    unmapped: set[str] = set()
    if kegg_path is not None and kegg_path.is_file():
        source = "kegg"
        with kegg_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                gene = rename(row["gene"], converter)
                kegg_id = row.get("disease_id") or row.get("kegg", "")
                doids = kegg_map.get(kegg_id, set())
                if not doids:
                    unmapped.add(kegg_id)
                for doid in doids:
                    pairs.add((gene, doid))
        rows = _label_grid(pairs)
    elif converted_path is not None and converted_path.is_file():
        source = "converted"
        with converted_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if int(float(row["label"])) == 1:
                    pairs.add((rename(row["gene"], converter), row["doid"]))
        rows = _label_grid(pairs)
    else:
        raise FileNotFoundError(
            f"no label source for {output.name}: fetch data_tr__target_disease.zip with "
            "scripts/tr/upstream/download_raw_data.py, or point kegg_labels/labels_dir at one"
        )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["gene", "doid", "label"])
        writer.writerows(rows)
    return {
        "source": source,
        "rows": len(rows),
        "positives": len(pairs),
        "genes": len({gene for gene, _ in pairs}),
        "diseases": len({disease for _, disease in pairs}),
        "kegg_ids_unmapped": sorted(unmapped),
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def build_tr_processed(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path.cwd()
    raw_dir = _resolve(root, cfg.get("raw_dir", "data_tr/raw"))
    output_dir = _resolve(root, cfg.get("output_dir", "data_tr/processed"))
    labels_dir = _resolve(root, cfg.get("labels_dir", raw_dir))
    kegg_labels = {key: _resolve(root, value) for key, value in (cfg.get("kegg_labels") or {}).items()}
    per_cell_line = bool(cfg.get("per_cell_line", True))
    human_only = bool(cfg.get("human_only", True))
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = build_symbol_converter(raw_dir / cfg.get("hgnc", "hgnc_complete_set.txt"))
    manifest: dict[str, Any] = {
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "per_cell_line": per_cell_line,
        "human_only": human_only,
        "symbol_converter_entries": len(converter),
    }
    manifest.update(
        build_graph(
            raw_dir / cfg.get("graph_sif", "PathwayCommons12.All.hgnc.sif"),
            converter,
            output_dir / "graph.tsv",
        )
    )

    gctx = raw_dir / cfg.get("gctx", "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx")
    sig_info = raw_dir / cfg.get("sig_info", "GSE92742_Broad_LINCS_sig_info.txt")
    gene_info = raw_dir / cfg.get("gene_info", "GSE92742_Broad_LINCS_gene_info.txt")
    # One pass over the matrix for both perturbation types: the landmark rows of
    # every signature we need are ~230 MB, while the file itself is 23 GB.
    wanted = {sig_id for pert_type in PERT_TYPES.values() for sig_id in _signature_index(sig_info, pert_type)}
    profiles = _read_gctx_columns(gctx, _landmark_genes(gene_info)[0], wanted)
    for name, pert_type in PERT_TYPES.items():
        manifest[f"{name}_signature"] = build_signature(
            gctx, sig_info, gene_info, pert_type, converter,
            output_dir / f"{name}_signature.tsv", per_cell_line, profiles,
        )

    manifest["disease_signature"] = build_disease_signature(
        raw_dir / cfg.get("creeds", "disease_signatures-v1.0.json"),
        converter,
        output_dir / "disease_specific_signature.tsv",
        human_only,
    )

    kegg_map = kegg_to_doid(
        raw_dir / cfg.get("kegg_omim", "kegg_disease_omim.list"),
        raw_dir / cfg.get("disease_ontology", "HumanDO.obo"),
    )
    manifest["kegg_to_doid_entries"] = len(kegg_map)
    for name in ("inhibitory", "activatory"):
        manifest[f"{name}_labels"] = build_labels(
            kegg_labels.get(name),
            labels_dir / f"{name}_target_disease.tsv",
            converter,
            kegg_map,
            output_dir / f"{name}_target_disease.tsv",
        )

    (output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return manifest
