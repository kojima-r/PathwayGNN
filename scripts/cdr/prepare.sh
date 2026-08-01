#!/usr/bin/env bash
# Preprocessing only: data_cdr/processed/full_features -> data_cdr/prepared.
#
# The upstream GraphCDRScan stage (raw GDSC/CCLP/Reactome files -> the processed
# bundle) is separate, expensive, and has its own dependencies
# (`pip install -e '.[cdr-upstream]'`, plus pdftotext and LibreOffice):
#   python -m scripts.cdr.upstream.download_raw_data
#   python -m scripts.cdr.upstream.prepare_data --config configs/cdr/upstream.json
# It is skipped here because data_cdr/processed/ is already present.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

pathwaygnn-data cdr-prepare --config configs/cdr/prepare.yaml "$@"
