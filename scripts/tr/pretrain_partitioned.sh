#!/usr/bin/env bash
# Partitioned distributed pre-training: cut the graph once, then let every rank
# train on its own share of the partitions. Unlike scripts/tr/pretrain_distributed.sh
# (which replicates the whole graph on every rank), peak memory here is set by
# `training.partition.parts_per_batch`, not by the graph size.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${CONFIG:-configs/tr/pretrain_partitioned.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(python -c 'import torch; print(max(torch.cuda.device_count(), 1))')}"

# Cutting the graph is the one step that needs it in memory, so it runs once, in a
# single process. It is a no-op when the partitions are already there and current.
python -m pathwaygnn.cli partition --config "${CONFIG}"

torchrun --standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE}" \
  -m pathwaygnn.cli pretrain --config "${CONFIG}"
