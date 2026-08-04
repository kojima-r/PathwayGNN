"""Constants published by Inoue et al., used for preprocessing checks and reporting."""

from __future__ import annotations

CANCER_TYPES = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
]
PAPER_SAMPLE_COUNTS = {1: 9484, 2: 7308, 3: 5915, 4: 5036, 5: 4492}
PAPER_TABLE1 = {
    1: [0.6312, 0.7425, 0.7382, 0.7585],
    2: [0.6168, 0.7672, 0.7678, 0.7596],
    3: [0.6188, 0.7624, 0.7733, 0.7890],
    4: [0.6173, 0.7674, 0.7714, 0.7900],
    5: [0.5971, 0.7644, 0.7581, 0.7850],
}
# ``seed_index`` pins the per-variant seed offset so that a run of a single
# condition reproduces the seed it would get inside the full grid.
PAPER_VARIANTS = [
    {"name": "dnn", "use_graph": False, "use_sample_features": False, "seed_index": 0},
    {"name": "dnn_cancer", "use_graph": False, "use_sample_features": True, "seed_index": 1},
    {"name": "gnn_dnn", "use_graph": True, "use_sample_features": False, "seed_index": 2},
    {"name": "gnn_dnn_cancer", "use_graph": True, "use_sample_features": True, "seed_index": 3},
]
VARIANT_NAMES = [item["name"] for item in PAPER_VARIANTS]
DISPLAY = {
    "dnn": "DNN",
    "dnn_cancer": "DNN + cancer types",
    "gnn_dnn": "GNN + DNN",
    "gnn_dnn_cancer": "GNN + DNN + cancer types",
}
YEARS = (1, 2, 3, 4, 5)
