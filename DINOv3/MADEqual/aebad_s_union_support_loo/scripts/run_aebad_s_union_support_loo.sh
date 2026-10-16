#!/usr/bin/env bash
# Default is filesystem preflight. Capture/analyze require an explicit mode.
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/DINO-NVS}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
cd "$PROJECT_ROOT"
if [ "$#" -eq 0 ]; then
    set -- preflight
elif [ "$1" != preflight ] && [ "$1" != capture ] && [ "$1" != analyze ]; then
    set -- preflight "$@"
fi
exec "$PYTHON" -u -m DINOv3.MADEqual.aebad_s_union_support_loo.runner "$@"
