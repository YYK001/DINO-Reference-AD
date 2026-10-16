#!/usr/bin/env bash
set -euo pipefail

# Preparation only: this script does not connect, upload, download or diagnose.
# Default mode is preflight; replay requires the user to invoke it explicitly.
MODE=${1:-preflight}
if [ "$#" -gt 0 ]; then shift; fi
case "$MODE" in
  preflight|replay) ;;
  --help|-h) MODE=--help ;;
  *) echo 'Usage: bash run_aebad_s_background_replay12.sh [preflight|replay] [CLI overrides]' >&2; exit 2 ;;
esac

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/DINO-NVS}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
AEBAD_ROOT=${AEBAD_ROOT:-/root/autodl-tmp/datasets/AeBAD/AeBAD_S}
RESULTS_DIR=${RESULTS_DIR:-$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$PROJECT_ROOT/external_baselines/SuperADD}
DINOV3_ROOT=${DINOV3_ROOT:-/root/autodl-tmp/dinov3}
CHECKPOINT=${CHECKPOINT:-$OFFICIAL_ROOT/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}
SOURCE_MEMORY=${SOURCE_MEMORY:-$RESULTS_DIR/official_memory.pt}
SOURCE_MAD=${SOURCE_MAD:-$RESULTS_DIR/source_mad_state.npz}
SUPPORT_CACHE_DIR=${SUPPORT_CACHE_DIR:-$RESULTS_DIR/support_features}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_union_background_replay12_seed42_v1}
DEVICE=${DEVICE:-cuda:0}

# Known-valid values, avoiding inherited invalid libgomp thread settings.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

cd "$PROJECT_ROOT"
if [ "$MODE" = '--help' ]; then
  exec "$PYTHON" -m DINOv3.MADEqual.aebad_s_target_fewshot.replay12 --help
fi
exec "$PYTHON" -u -m DINOv3.MADEqual.aebad_s_target_fewshot.replay12 "$MODE" \
  --project-root "$PROJECT_ROOT" \
  --results-dir "$RESULTS_DIR" \
  --aebad-root "$AEBAD_ROOT" \
  --official-root "$OFFICIAL_ROOT" \
  --dinov3-root "$DINOV3_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --memory "$SOURCE_MEMORY" \
  --mad "$SOURCE_MAD" \
  --support-cache-dir "$SUPPORT_CACHE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" "$@"
