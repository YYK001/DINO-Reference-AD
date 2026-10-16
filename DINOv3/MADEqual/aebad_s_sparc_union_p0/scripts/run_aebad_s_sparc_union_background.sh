#!/usr/bin/env bash
# No network/install/shutdown. Default mode is read-only preflight.
set -u
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/DINO-NVS}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
BASE="$PROJECT_ROOT/DINOv3/outputs/mad_equal"
MODE="${1:-preflight}"
if [ "$#" -gt 0 ]; then shift; fi
cd "$PROJECT_ROOT" || exit 1
exec "$PYTHON" -B -u -m DINOv3.MADEqual.aebad_s_sparc_union_p0.runner "$MODE" \
  --project-root "$PROJECT_ROOT" \
  --sparc-base-dir "${SPARC_BASE_DIR:-$BASE/aebad_s_sparc_background_mixed_20261009_095419}" \
  --output-dir "${OUTPUT_DIR:-$BASE/aebad_s_sparc_union_k8_seed42_v1}" \
  --cpu-workers "${CPU_WORKERS:-8}" \
  --device "${DEVICE:-cuda:0}" "$@"
