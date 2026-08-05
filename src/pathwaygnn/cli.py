"""The training CLI. Every command reads one prepared dataset and one config.

Dataset-specific preprocessing is *not* part of this CLI; run it first with
``pathwaygnn-data`` so that the dataset directory follows
:mod:`pathwaygnn.data.format`.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from pathwaygnn.config import load_config

COMMANDS: dict[str, tuple[str, str]] = {
    "partition": ("pathwaygnn.data.partition", "run_partitioning"),
    "pretrain": ("pathwaygnn.training.pretrain", "run_pretraining"),
    "finetune": ("pathwaygnn.training.finetune", "run_finetuning"),
    "cv": ("pathwaygnn.training.cv", "run_cv"),
    "ig": ("pathwaygnn.training.ig", "run_ig"),
    "benchmark": ("pathwaygnn.training.benchmark", "run_benchmark"),
    "dist-benchmark": ("pathwaygnn.training.dist_benchmark", "run_dist_benchmark"),
}
HELP = {
    "partition": "cut the graph into METIS partitions for memory-bounded pre-training",
    "pretrain": "pre-train the relational graph encoder on edge prediction",
    "finetune": "train one train/validation/test split of a task",
    "cv": "stratified k-fold cross-validation over a grid of variants and tasks",
    "ig": "Integrated Gradients attribution for one cross-validation fold",
    "benchmark": "graph-free baselines on the same features",
    "dist-benchmark": "sweep the partition count and record step time and peak memory",
}


def _runner(command: str) -> Callable[[dict[str, Any]], Any]:
    module_name, function = COMMANDS[command]
    module = __import__(module_name, fromlist=[function])
    return getattr(module, function)


def main() -> None:
    parser = argparse.ArgumentParser(prog="pathwaygnn", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = commands.add_parser(name, help=HELP[name])
        command.add_argument("--config", required=True)
    args = parser.parse_args()
    result = _runner(args.command)(load_config(args.config))
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=True, default=str))


if __name__ == "__main__":
    main()
