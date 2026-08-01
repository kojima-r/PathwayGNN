#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
STAGE="${1:-full}"

run_stage() {
  local name="$1"
  case "${name}" in
    prepare)
      pathwaygnn-data cancer-prepare --config configs/cancer/prepare.yaml
      ;;
    map-ids)
      pathwaygnn-data cancer-map-ids --config configs/cancer/id_mapping.yaml
      ;;
    pretrain)
      if [[ -s outputs/cancer/pretrain_50/best.pt && "${FORCE_PRETRAIN:-0}" != "1" ]]; then
        echo "Reusing outputs/cancer/pretrain_50/best.pt"
      else
        torchrun --standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE:-3}" -m pathwaygnn.cli pretrain --config configs/cancer/pretrain.yaml
      fi
      ;;
    figure2-pretrain)
      torchrun --standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE:-3}" -m pathwaygnn.cli pretrain --config configs/cancer/pretrain_sweep.yaml
      ;;
    figure2)
      if [[ ! -s outputs/cancer/pretrain_sweep/epoch_50.pt ]]; then run_stage figure2-pretrain; fi
      python scripts/cancer/reproduce_figure2.py --gpus "${TABLE1_GPUS:-0,1,2}"
      ;;
    table1|cv)
      python scripts/cancer/reproduce_table1.py --gpus "${TABLE1_GPUS:-0,1,2}" --jobs-per-gpu "${JOBS_PER_GPU:-1}"
      ;;
    report)
      pathwaygnn-data cancer-report --config configs/cancer/report.yaml
      ;;
    ig)
      pathwaygnn ig --config configs/cancer/ig.yaml
      pathwaygnn-data cancer-report --config configs/cancer/report.yaml
      ;;
    *)
      echo "Unknown stage: ${name}" >&2
      exit 2
      ;;
  esac
}

if [[ "${STAGE}" == "full" || "${STAGE}" == "all" ]]; then
  for stage in prepare pretrain table1 figure2 ig report; do
    run_stage "${stage}"
  done
else
  run_stage "${STAGE}"
fi
