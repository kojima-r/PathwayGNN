#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-$(python -c 'import torch; print(max(torch.cuda.device_count(), 1))')}"
torchrun --standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE}" \
  -m pathwaygnn.cli pretrain --config configs/tr/pretrain.yaml
