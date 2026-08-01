"""Dataset-specific preprocessing and reporting.

Everything here knows about one particular corpus: file names, column layouts,
identifier conventions, published reference numbers. Its whole job is to finish
before training starts and leave a dataset in the generic
:mod:`pathwaygnn.data.format` layout, which the ``pathwaygnn`` CLI then consumes
without any dataset knowledge of its own.

    pathwaygnn-data tr-prepare      --config configs/tr/prepare.yaml
    pathwaygnn-data cancer-prepare  --config configs/cancer/prepare.yaml
    pathwaygnn-data cancer-map-ids  --config configs/cancer/id_mapping.yaml
    pathwaygnn-data cdr-prepare     --config configs/cdr/prepare.yaml
    pathwaygnn-data cancer-report   --config configs/cancer/report.yaml
    pathwaygnn-data tr-report       --config configs/tr/report.yaml
    pathwaygnn-data cdr-report      --config configs/cdr/report.yaml
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from pathwaygnn.config import load_config

HELP = {
    "tr-prepare": "PathwayCommons + perturbation/disease signatures -> dataset `tr`",
    "cancer-prepare": "TCGA survival bundle -> dataset `cancer`",
    "cancer-map-ids": "map an ordered Ensembl ID list to HGNC IDs via MyGene.info",
    "cdr-prepare": "GraphCDRScan GDSC/CCLP bundle -> dataset `cdr`",
    "cancer-report": "render the Inoue et al. comparison tables, figures and document",
    "tr-report": "render the target-repositioning tables, figures and document",
    "cdr-report": "render the drug-response tables, figures and document",
}


def _run(command: str, cfg: dict[str, Any]) -> Any:
    if command == "tr-prepare":
        from pathwaygnn_datasets.tr.prepare import prepare_tr_dataset

        return prepare_tr_dataset(
            cfg["raw_dir"], cfg["output_dir"], float(cfg.get("cutoff", 1e-7))
        )
    if command == "cancer-prepare":
        from pathwaygnn_datasets.cancer.prepare import prepare_cancer_dataset

        return prepare_cancer_dataset(
            cfg["source_dir"],
            cfg["output_dir"],
            cfg.get("years", [1, 2, 3, 4, 5]),
            int(cfg.get("num_genes", 4448)),
        )
    if command == "cancer-map-ids":
        from pathwaygnn_datasets.cancer.gene_mapping import map_ensembl_ids

        return map_ensembl_ids(
            cfg["input_path"],
            cfg["output_path"],
            cfg["cache_path"],
            cfg.get("species", "human"),
            int(cfg.get("batch_size", 1000)),
        )
    if command == "cdr-prepare":
        from pathwaygnn_datasets.cdr.prepare import prepare_cdr_dataset

        return prepare_cdr_dataset(
            cfg["source_dir"],
            cfg["output_dir"],
            bool(cfg.get("binary_mutations", False)),
            tuple(cfg.get("tasks", ("sensitive_drugwise", "sensitive_global"))),
        )
    if command == "cancer-report":
        from pathwaygnn_datasets.cancer.report import run_cancer_report

        return run_cancer_report(cfg)
    if command == "cdr-report":
        from pathwaygnn_datasets.cdr.report import run_cdr_report

        return run_cdr_report(cfg)
    from pathwaygnn_datasets.tr.report import run_tr_report

    return run_tr_report(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(prog="pathwaygnn-data", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in HELP.items():
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True)
    args = parser.parse_args()
    result = _run(args.command, load_config(args.config))
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=True, default=str))


if __name__ == "__main__":
    main()
