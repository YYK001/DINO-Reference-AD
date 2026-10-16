#!/usr/bin/env bash
# Default is read-only preflight. No installation, network or shutdown.
set -u
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/DINO-NVS}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
BASE="$PROJECT_ROOT/DINOv3/outputs/mad_equal"
MODE="${1:-preflight}"
if [ "$#" -gt 0 ]; then shift; fi
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/aebad_s_sparc_background_seed42_v1}"
cd "$PROJECT_ROOT" || exit 1
args=(
  --project-root "$PROJECT_ROOT"
  --results-dir "${RESULTS_DIR:-$BASE/aebad_s_target_fewshot_seed42_v1}"
  --capture-dir "${CAPTURE_DIR:-$BASE/aebad_s_union_support_loo_B_20261007_230639}"
  --support-cache-dir "${SUPPORT_CACHE_DIR:-$BASE/aebad_s_union_background_replay12_seed42_v1/support_features/target_matched/background}"
  --data "${DATA_DIR:-/root/autodl-tmp/datasets/AeBAD/AeBAD_S}"
  --official-root "${OFFICIAL_ROOT:-$PROJECT_ROOT/external_baselines/SuperADD}"
  --dinov3-root "${DINOV3_ROOT:-/root/autodl-tmp/dinov3}"
  --checkpoint "${CHECKPOINT:-$PROJECT_ROOT/external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}"
  --output-dir "$OUTPUT_DIR"
)
if [ "$MODE" = preflight ]; then args+=(--no-write); fi
exec "$PYTHON" -B -u -m DINOv3.MADEqual.aebad_s_sparc_p0.runner "$MODE" "${args[@]}" "$@"
