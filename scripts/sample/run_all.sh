#!/usr/bin/env bash
# The whole tutorial in one command: data_sample/raw -> prepared -> pre-training
# -> cross-validation -> single split -> graph-free baselines -> attribution
# -> prediction table.
#
# CPU only, about 3 minutes end to end. Everything it writes lives under
# data_sample/prepared/ and outputs/sample/, both of which are .gitignored, so
# `rm -rf data_sample/prepared outputs/sample` puts the repository back.
#
# The final `summarize.py` prints the tables README.md quotes.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

RUN=(python -m pathwaygnn.cli)
DATA=(python -m pathwaygnn_datasets.cli)

"${DATA[@]}" sample-prepare --config configs/sample/prepare.yaml

"${RUN[@]}" pretrain  --config configs/sample/pretrain.yaml
"${RUN[@]}" cv        --config configs/sample/cv.yaml
"${RUN[@]}" finetune  --config configs/sample/finetune.yaml
# Baselines need the extra dependencies: pip install -e '.[benchmark]'
"${RUN[@]}" benchmark --config configs/sample/benchmark.yaml
"${RUN[@]}" ig        --config configs/sample/ig.yaml
# Scores the corpus itself, because the repository ships no external data.
"${RUN[@]}" pred      --config configs/sample/pred.yaml

python scripts/sample/summarize.py
