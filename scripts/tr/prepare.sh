#!/usr/bin/env bash
# Preprocessing only: data_tr/processed -> data_tr/prepared.
#
# The stage before it (public sources -> the bundle) is separate and has its own
# dependency (h5py, for the LINCS GCTX matrix):
#   python -m scripts.tr.upstream.download_raw_data
#   bash scripts/tr/build_processed.sh
# Skip it when data_tr/processed/ is already present.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

pathwaygnn-data tr-prepare --config configs/tr/prepare.yaml "$@"
