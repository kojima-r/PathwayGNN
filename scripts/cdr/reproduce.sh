#!/usr/bin/env bash
# Everything docs/cdr_report.md reports, in order. Fold-level resume makes a
# re-run cheap: delete an output directory to force that piece to recompute.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

RUN=(python -m pathwaygnn.cli)
DATA=(python -m pathwaygnn_datasets.cli)

"${DATA[@]}" cdr-prepare --config configs/cdr/prepare.yaml

"${RUN[@]}" pretrain  --config configs/cdr/pretrain.yaml
# One process per (task, variant); drop to `pathwaygnn cv --config
# configs/cdr/cv.yaml` to run the same 40 folds serially.
python scripts/cdr/run_cv.py
"${RUN[@]}" finetune  --config configs/cdr/finetune_drugwise.yaml
"${RUN[@]}" finetune  --config configs/cdr/finetune_global.yaml
"${RUN[@]}" benchmark --config configs/cdr/benchmark_drugwise.yaml
"${RUN[@]}" benchmark --config configs/cdr/benchmark_global.yaml
"${RUN[@]}" ig        --config configs/cdr/ig_drugwise.yaml
"${RUN[@]}" ig        --config configs/cdr/ig_global.yaml

"${DATA[@]}" cdr-report --config configs/cdr/report.yaml
