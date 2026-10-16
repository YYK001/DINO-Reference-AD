#!/usr/bin/env bash
# Default read-only preflight. No network, installs, Source rebuilding or shutdown.
set -u
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/DINO-NVS}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
BASE="$PROJECT_ROOT/DINOv3/outputs/mad_equal"
MODE="${1:-preflight}"
if [ "$#" -gt 0 ]; then shift; fi
cd "$PROJECT_ROOT" || exit 1
exec "$PYTHON" -B -u -m DINOv3.MADEqual.aebad_s_fastref_k8_p0.runner "$MODE" \
  --project-root "$PROJECT_ROOT" \
  --results-dir "${RESULTS_DIR:-$BASE/aebad_s_target_fewshot_seed42_v1}" \
  --capture-dir "${CAPTURE_DIR:-$BASE/aebad_s_union_support_loo_B_20261007_230639}" \
  --support-cache-dir "${SUPPORT_CACHE_DIR:-$BASE/aebad_s_union_background_replay12_seed42_v1/support_features/target_matched/background}" \
  --data "${DATA_DIR:-/root/autodl-tmp/datasets/AeBAD/AeBAD_S}" \
  --output-dir "${OUTPUT_DIR:-$BASE/aebad_s_fastref_target512_source_bounded_density_seed42_v3}" \
  --cpu-workers "${CPU_WORKERS:-4}" --device "${DEVICE:-cuda:0}" "$@"
