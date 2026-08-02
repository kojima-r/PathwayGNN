"""`cancer-build-processed` builds the upstream bundle `cancer-prepare` reads.

The fixture is a miniature TCGA: three genes on a four-node graph, six recount2
columns belonging to five patients, and clinical records chosen so that every
branch of the published sample selection fires.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pathwaygnn.data.format import GraphDataset
from pathwaygnn_datasets.cancer.build import build_cancer_processed, select_samples
from pathwaygnn_datasets.cancer.paper import CANCER_TYPES
from pathwaygnn_datasets.cancer.prepare import prepare_cancer_dataset

# gene symbol -> HGNC id -> Ensembl id; TP53 is deliberately absent from the graph.
GENES = [("BRCA1", "1100", "ENSG00000012048"), ("EGFR", "3236", "ENSG00000146648"),
         ("MYC", "7553", "ENSG00000136997"), ("TP53", "11998", "ENSG00000141510")]
# (file id, patient, cancer type, vital status, days)
SAMPLES = [
    ("f1", "TCGA-AA-0001", "BRCA", "Alive", 900),    # censored late, survives year 1 and 2
    ("f2", "TCGA-AA-0002", "LUAD", "Dead", 200),     # dies inside year 1
    ("f3", "TCGA-AA-0003", "BRCA", "Alive", 100),    # censored inside year 1 -> dropped there
    ("f4", "TCGA-AA-0004", "LUAD", "Dead", 800),     # survives year 1, dies in year 2
    ("f5", "TCGA-AA-0005", "BRCA", "Alive", 9000),   # beyond the long-survival cutoff
    ("f6", "TCGA-AA-0001", "BRCA", "Alive", 900),    # a second file for patient 1
]


def _write(path: Path, rows, delimiter="\t") -> None:
    with path.open("w", newline="") as handle:
        csv.writer(handle, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL).writerows(rows)


def make_raw(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    _write(root / "graph.sif", [
        ["BRCA1", "interacts-with", "EGFR"],
        ["EGFR", "controls-expression-of", "MYC"],
        ["MYC", "interacts-with", "CHEBI:1234"],
    ])
    _write(root / "hgnc.tsv", [["HGNC ID", "Approved symbol", "Ensembl gene ID"],
                               *[[f"HGNC:{i}", s, e] for s, i, e in GENES]])
    # counts: one row per gene, one column per file, header = file ids upper-cased
    counts = [[item[0].upper() for item in SAMPLES]]
    for position, _ in enumerate(GENES):
        counts.append([float(100 * (position + 1) + index) for index in range(len(SAMPLES))])
    _write(root / "counts.tsv", counts)
    _write(root / "gene_ids.txt", [[gene[2], 1000 * (index + 1)] for index, gene in enumerate(GENES)])
    _write(root / "genes.gmt", [["CANCER_SET", "a curated set", "BRCA1", "EGFR", "MYC", "TP53"]])
    _write(root / "metadata.tsv",
           [["gdc_file_id", "gdc_cases.submitter_id", "gdc_cases.project.project_id"],
            *[[f, p, f"TCGA-{t}"] for f, p, t, _, _ in SAMPLES]])
    _write(root / "clinical.csv",
           [["bcr_patient_barcode", "type", "vital_status", "last_contact_days_to", "death_days_to"],
            *[[p, t, v, d if v == "Alive" else "", d if v == "Dead" else ""]
              for _, p, t, v, d in {s[1]: s for s in SAMPLES}.values()]],
           delimiter=",")
    return {
        "graph_sif": str(root / "graph.sif"),
        "hgnc_table": str(root / "hgnc.tsv"),
        "expression": str(root / "counts.tsv"),
        "gene_ids": str(root / "gene_ids.txt"),
        "metadata": str(root / "metadata.tsv"),
        "clinical": str(root / "clinical.csv"),
        "gene_sets": [str(root / "genes.gmt")],
        "years": [1, 2],
        "max_survival_days": 3595,
    }


def test_build_writes_the_upstream_bundle(tmp_path: Path) -> None:
    cfg = make_raw(tmp_path / "raw")
    cfg["output_dir"] = str(tmp_path / "processed")
    manifest = build_cancer_processed(cfg)
    processed = Path(cfg["output_dir"])

    # The SIF is symmetrised and its symbols become HGNC ids; CHEBI stays put.
    assert manifest["graph"] == {
        "num_nodes": 4, "num_relations": 2, "num_edges": 6, "nodes_renamed_to_hgnc": 3,
    }
    vertices = dict(row.split("\t") for row in
                    (processed / "vertices_dic.tsv").read_text().splitlines())
    assert sorted(vertices) == sorted(["1100", "3236", "7553", "CHEBI:1234"])
    assert sorted(vertices.values(), key=int) == ["0", "1", "2", "3"]
    # Sorted encoding, so the ids do not depend on set iteration order.
    assert [name for name, _ in sorted(vertices.items(), key=lambda kv: int(kv[1]))] == sorted(vertices)

    # TP53 is in the gene set but not in the graph, so it is dropped.
    assert manifest["genes"]["genes_selected"] == 3
    assert manifest["samples"]["remaining_samples"] == 5  # f5 exceeds the cutoff

    # Year 1 drops the sample censored inside the year; year 2 drops it too.
    assert manifest["samples"]["per_year"]["1"] == {"samples": 4, "survival": 3, "death": 1}
    labels = np.loadtxt(processed / "1years_labels.tsv", delimiter="\t")
    assert labels[:, 0].tolist() == [0, 1, 2, 3] and labels[:, 1].tolist() == [1, 0, 1, 1]

    samples = np.loadtxt(processed / "1years_sample.tsv", delimiter="\t")
    assert samples.shape == (4, 35)
    assert samples[0, 1] == CANCER_TYPES.index("BRCA")
    assert samples[0, 2 + CANCER_TYPES.index("BRCA")] == 1 and samples[0, 2:].sum() == 1
    # No serialized-header row, unlike the shipped bundle.
    assert not np.array_equal(samples[0], np.arange(35))

    node_input = np.loadtxt(processed / "1years_node_input.tsv", delimiter="\t")
    assert node_input.shape == (4 * 3, 3)
    assert node_input[:, 0].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    # Every sample lists the genes in the same order, which prepare then checks.
    assert node_input[0:3, 1].tolist() == node_input[3:6, 1].tolist()
    # log1p of the raw counts: gene 0 of file f1 is 100.
    assert node_input[0, 2] == pytest.approx(np.log1p(100.0))


def test_built_bundle_feeds_cancer_prepare(tmp_path: Path) -> None:
    cfg = make_raw(tmp_path / "raw")
    cfg["output_dir"] = str(tmp_path / "source" / "processed")
    build_cancer_processed(cfg)

    # The published per-year counts do not apply to a miniature corpus.
    with pytest.raises(ValueError, match="Supplementary Table 1"):
        prepare_cancer_dataset(tmp_path / "source", tmp_path / "prepared", [1], 3)
    manifest = prepare_cancer_dataset(
        tmp_path / "source", tmp_path / "prepared", [1], 3, strict_sample_counts=False
    )
    assert manifest["name"] == "cancer" and manifest["num_nodes"] == 4
    dataset = GraphDataset.open(tmp_path / "prepared", "cancer")
    task = dataset.task("1year")
    assert task.num_samples == 4 and task.covariate_dim == 33
    assert task.group_names == tuple(CANCER_TYPES)
    assert json.loads((tmp_path / "prepared/tasks/1year/task.json").read_text())["num_positive"] == 3


def test_long_survival_cutoff_follows_the_published_rule(tmp_path: Path) -> None:
    """With `max_survival_days` unset the cutoff is derived, not assumed."""
    cfg = make_raw(tmp_path / "raw")
    root = tmp_path / "raw"
    _, stats = select_samples(
        [item[0].upper() for item in SAMPLES], root / "metadata.tsv", root / "clinical.csv",
        [1], None,
    )
    # Censored times are 900, 100, 9000, 900 -> median 900; the pooled deceased
    # and above-median censored times are 200, 800, 9000 -> 95th percentile 9000.
    assert stats["censored_median_days"] == 900.0
    assert stats["long_survival_cutoff_days"] == 9000.0
    assert stats["excluded_beyond_cutoff"] == 0
    assert stats["usable_columns"] == len(SAMPLES)
