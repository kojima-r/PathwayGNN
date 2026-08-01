#!/usr/bin/env bash
# Preprocessing only: raw target-repositioning files -> data_tr/prepared.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

pathwaygnn-data tr-prepare --config configs/tr/prepare.yaml "$@"
