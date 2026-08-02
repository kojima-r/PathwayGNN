#!/usr/bin/env bash
# Rebuild the bundle: data_tr/raw -> data_tr/processed.
#
# Fetch the raw sources first (standard library only, ~21.5 GB):
#   python -m scripts.tr.upstream.download_raw_data
# This stage needs h5py for the LINCS GCTX matrix:
#   pip install -e '.[tr-upstream]'
#
# Skip it entirely when data_tr/processed/ is already present and go straight to
# scripts/tr/prepare.sh.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

pathwaygnn-data tr-build-processed --config configs/tr/build_processed.yaml "$@"
