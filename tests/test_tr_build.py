"""`tr-build-processed`: raw LINCS/CREEDS/HGNC/PathwayCommons -> data_tr/processed."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pathwaygnn_datasets.tr.build import (
    build_disease_signature,
    build_labels,
    build_signature,
    build_symbol_converter,
    build_tr_processed,
    kegg_to_doid,
)

HGNC_HEADER = ["hgnc_id", "symbol", "status", "prev_symbol", "alias_symbol"]
HGNC_ROWS = [
    ["HGNC:1", "AAA", "Approved", "OLD1", "ALIAS1"],
    ["HGNC:2", "BBB", "Approved", "OLD2|SHARED", "AAA"],  # AAA is approved -> not a synonym
    ["HGNC:3", "CCC", "Approved", "SHARED", ""],  # SHARED is claimed twice
    ["HGNC:4", "DDD", "Entry Withdrawn", "OLD4", ""],  # withdrawn rows are ignored
]


def _write(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, delimiter="\t").writerows(rows)


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    _write(root / "hgnc_complete_set.txt", [HGNC_HEADER, *HGNC_ROWS])
    return root


def test_symbol_converter_follows_the_reference_rules(raw: Path) -> None:
    converter = build_symbol_converter(raw / "hgnc_complete_set.txt")
    assert converter["OLD1"] == "AAA" and converter["ALIAS1"] == "AAA"
    assert "AAA" not in converter  # a synonym that is itself an approved symbol
    assert "OLD4" not in converter  # only Approved rows contribute
    # Claimed as a previous symbol by two genes: the ambiguity is kept visible.
    assert converter["SHARED"] == "SHARED:BBB/CCC"


def test_disease_signature_filters_by_organism_and_averages(raw: Path, tmp_path: Path) -> None:
    records = [
        {"do_id": "DOID:1", "organism": "human", "up_genes": [["OLD1", 1.0]], "down_genes": []},
        {"do_id": "DOID:1", "organism": "human", "up_genes": [["AAA", 3.0]], "down_genes": []},
        {"do_id": "DOID:2", "organism": "mouse", "up_genes": [["BBB", 1.0]], "down_genes": []},
        {"do_id": None, "organism": "human", "up_genes": [["BBB", 1.0]], "down_genes": []},
    ]
    creeds = raw / "disease_signatures-v1.0.json"
    creeds.write_text(json.dumps(records), encoding="utf-8")
    converter = build_symbol_converter(raw / "hgnc_complete_set.txt")
    output = tmp_path / "disease.tsv"

    stats = build_disease_signature(creeds, converter, output, human_only=True)
    assert stats == {"signatures_used": 2, "diseases": 1, "rows": 1}
    rows = list(csv.DictReader(output.open(encoding="utf-8"), delimiter="\t"))
    # OLD1 is renamed to AAA, so the two studies average into one value.
    assert rows[0]["gene_name"] == "AAA" and float(rows[0]["expression"]) == 2.0

    stats = build_disease_signature(creeds, converter, output, human_only=False)
    assert stats["signatures_used"] == 3 and stats["diseases"] == 2


def test_kegg_to_doid_reads_the_ontology_xrefs(raw: Path) -> None:
    _write(raw / "kegg.list", [["ds:H00001", "omim:100100", "equivalent"], ["ds:H00002", "omim:200200", "x"]])
    (raw / "HumanDO.obo").write_text(
        "\n".join(
            [
                # Current releases write MIM:, older ones OMIM:; both are read.
                "[Term]", "id: DOID:11", "xref: MIM:100100", "",
                "[Term]", "id: DOID:12", "xref: OMIM:PS200200", "",
                # A direct KEGG DISEASE link; the numeric xref is a pathway id.
                "[Term]", "id: DOID:13", "xref: KEGG:H00003", "xref: KEGG:04950", "",
                "[Typedef]", "id: is_a", "xref: MIM:100100", "",
            ]
        ),
        encoding="utf-8",
    )
    mapping = kegg_to_doid(raw / "kegg.list", raw / "HumanDO.obo")
    # MIM:PS200200 is the phenotypic-series form of OMIM:200200.
    assert mapping == {"H00001": {"DOID:11"}, "H00002": {"DOID:12"}, "H00003": {"DOID:13"}}


def test_labels_fill_the_grid_with_negatives(raw: Path, tmp_path: Path) -> None:
    """The DOID-keyed table of the release mirror: only the gene column is renamed."""
    _write(
        raw / "inhibitory_target_disease.tsv",
        [["gene", "doid", "label"], ["OLD1", "DOID:11", 1], ["BBB", "DOID:12", 1], ["BBB", "DOID:11", 0]],
    )
    converter = build_symbol_converter(raw / "hgnc_complete_set.txt")
    output = tmp_path / "inhibitory_target_disease.tsv"
    stats = build_labels(None, raw / "inhibitory_target_disease.tsv", converter, {}, output)

    assert stats["source"] == "converted" and stats["positives"] == 2
    rows = list(csv.DictReader(output.open(encoding="utf-8"), delimiter="\t"))
    assert len(rows) == 4  # 2 genes x 2 diseases
    assert {(row["gene"], row["doid"]) for row in rows if row["label"] == "1"} == {
        ("AAA", "DOID:11"),  # OLD1 renamed
        ("BBB", "DOID:12"),
    }


def test_build_signature_averages_replicates_per_cell_line(raw: Path, tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    _write(
        raw / "gene_info.txt",
        [
            ["pr_gene_id", "pr_gene_symbol", "pr_gene_title", "pr_is_lm", "pr_is_bing"],
            ["10", "OLD1", "landmark", "1", "1"],
            ["20", "BBB", "landmark", "1", "1"],
            ["30", "CCC", "inferred", "0", "1"],
        ],
    )
    _write(
        raw / "sig_info.txt",
        [
            ["sig_id", "pert_id", "pert_iname", "pert_type", "cell_id"],
            ["S1", "P", "GENE1", "trt_sh.cgs", "MCF7"],
            ["S2", "P", "GENE1", "trt_sh.cgs", "MCF7"],  # replicate of S1
            ["S3", "P", "GENE1", "trt_sh.cgs", "PC3"],
            ["S4", "P", "GENE2", "trt_oe", "MCF7"],  # a different perturbation type
        ],
    )
    gctx = raw / "level5.gctx"
    with h5py.File(gctx, "w") as handle:
        # GCTX stores the matrix transposed: [signature, gene].
        handle.create_dataset("0/DATA/0/matrix", data=np.array(
            [[1.0, 10.0, 0.0], [3.0, 20.0, 0.0], [5.0, 30.0, 0.0], [7.0, 40.0, 0.0]], dtype=np.float32
        ))
        handle.create_dataset("0/META/ROW/id", data=np.array([b"10", b"20", b"30"]))
        handle.create_dataset("0/META/COL/id", data=np.array([b"S1", b"S2", b"S3", b"S4"]))

    converter = build_symbol_converter(raw / "hgnc_complete_set.txt")
    output = tmp_path / "knockdown_signature.tsv"
    stats = build_signature(
        gctx, raw / "sig_info.txt", raw / "gene_info.txt", "trt_sh.cgs", converter, output, True
    )
    assert stats == {"profiles": 3, "rows": 2, "genes": 2, "pert_type": "trt_sh.cgs"}
    rows = list(csv.DictReader(output.open(encoding="utf-8"), delimiter="\t"))
    assert [row["cell_id"] for row in rows] == ["MCF7", "PC3"]
    # OLD1 is renamed, only landmark genes are kept, and S1/S2 are averaged.
    assert list(rows[0]) == ["pert_iname", "cell_id", "AAA", "BBB"]
    assert (float(rows[0]["AAA"]), float(rows[0]["BBB"])) == (2.0, 15.0)
    assert (float(rows[1]["AAA"]), float(rows[1]["BBB"])) == (5.0, 30.0)

    # per_cell_line=False averages the cell lines away.
    flat = tmp_path / "flat.tsv"
    build_signature(
        gctx, raw / "sig_info.txt", raw / "gene_info.txt", "trt_sh.cgs", converter, flat, False
    )
    row = next(csv.DictReader(flat.open(encoding="utf-8"), delimiter="\t"))
    assert "cell_id" not in row and float(row["AAA"]) == 3.0


def test_build_writes_a_bundle_and_manifest(raw: Path, tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    _write(raw / "PathwayCommons12.All.hgnc.sif", [["OLD1", "interacts-with", "BBB"]])
    _write(
        raw / "GSE92742_Broad_LINCS_gene_info.txt",
        [["pr_gene_id", "pr_gene_symbol", "pr_is_lm"], ["10", "OLD1", "1"]],
    )
    _write(
        raw / "GSE92742_Broad_LINCS_sig_info.txt",
        [
            ["sig_id", "pert_iname", "pert_type", "cell_id"],
            ["S1", "AAA", "trt_sh.cgs", "MCF7"],
            ["S2", "AAA", "trt_oe", "MCF7"],
        ],
    )
    with h5py.File(raw / "level5.gctx", "w") as handle:
        handle.create_dataset("0/DATA/0/matrix", data=np.array([[1.0], [2.0]], dtype=np.float32))
        handle.create_dataset("0/META/ROW/id", data=np.array([b"10"]))
        handle.create_dataset("0/META/COL/id", data=np.array([b"S1", b"S2"]))
    (raw / "disease_signatures-v1.0.json").write_text(
        json.dumps([{"do_id": "DOID:11", "organism": "human", "up_genes": [["OLD1", 1.0]], "down_genes": []}]),
        encoding="utf-8",
    )
    _write(raw / "kegg_disease_omim.list", [["ds:H00001", "omim:100100", "equivalent"]])
    (raw / "HumanDO.obo").write_text("[Term]\nid: DOID:11\nxref: MIM:100100\n", encoding="utf-8")
    for name in ("inhibitory", "activatory"):
        _write(raw / f"{name}_target_disease.tsv", [["gene", "doid", "label"], ["AAA", "DOID:11", 1]])

    processed = tmp_path / "processed"
    manifest = build_tr_processed(
        {"raw_dir": str(raw), "output_dir": str(processed), "gctx": "level5.gctx"}
    )
    assert manifest["per_cell_line"] is True and manifest["human_only"] is True
    assert manifest["graph_rows"] == 1 and manifest["graph_nodes"] == 2
    assert manifest["knockdown_signature"]["rows"] == manifest["overexpression_signature"]["rows"] == 1
    assert manifest["kegg_to_doid_entries"] == 1
    assert manifest["inhibitory_labels"]["source"] == "converted"
    assert json.loads((processed / "build_manifest.json").read_text()) == manifest
    for name in (
        "graph.tsv",
        "knockdown_signature.tsv",
        "overexpression_signature.tsv",
        "disease_specific_signature.tsv",
        "inhibitory_target_disease.tsv",
        "activatory_target_disease.tsv",
    ):
        assert (processed / name).is_file()
    # The graph is renamed with the same converter as the signatures.
    assert next(csv.reader((processed / "graph.tsv").open(), delimiter="\t")) == [
        "AAA", "interacts-with", "BBB"
    ]
