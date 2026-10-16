#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/DINO-NVS}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
AEBAD_ROOT=${AEBAD_ROOT:-/root/autodl-tmp/datasets/AeBAD/AeBAD_S}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$PROJECT_ROOT/external_baselines/SuperADD}
DINOV3_ROOT=${DINOV3_ROOT:-/root/autodl-tmp/dinov3}
CHECKPOINT=${CHECKPOINT:-$OFFICIAL_ROOT/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_synthetic_fewshot_seed42_v1}
: "${BASE_RUN:?Set BASE_RUN to the completed AeBAD-S target-fewshot formal output directory}"

cd "$PROJECT_ROOT"
"$PYTHON" -u -m DINOv3.MADEqual.aebad_s_synthetic_fewshot.runner \
  --aebad-root "$AEBAD_ROOT" \
  --base-run "$BASE_RUN" \
  --project-root "$PROJECT_ROOT" \
  --official-root "$OFFICIAL_ROOT" \
  --dinov3-root "$DINOV3_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0

