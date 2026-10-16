#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/DINO-NVS}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
ROBUSTAD_ROOT=${ROBUSTAD_ROOT:-/root/autodl-tmp/datasets/RobustAD}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$PROJECT_ROOT/external_baselines/SuperADD}
DINOV3_ROOT=${DINOV3_ROOT:-/root/autodl-tmp/dinov3}
CHECKPOINT=${CHECKPOINT:-$OFFICIAL_ROOT/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/DINOv3/outputs/mad_equal/robustad_target_fewshot_pcb_seed42_p0_v1}
SOURCE_OUTPUT_DIR=${SOURCE_OUTPUT_DIR:-$PROJECT_ROOT/outputs/relation_reliability/seed42_v6}
RELATION_CACHE_ROOT=${RELATION_CACHE_ROOT:-$PROJECT_ROOT/outputs/relation_oracle_cache/mvtec_robustad_seed42_v2}
DEVICE=${DEVICE:-cuda:0}

mkdir -p "$(dirname "$OUTPUT_DIR")"
cd "$PROJECT_ROOT"

exec "$PYTHON" -u -m DINOv3.MADEqual.robustad_target_fewshot_p0.runner \
  --category PCB \
  --robustad-root "$ROBUSTAD_ROOT" \
  --project-root "$PROJECT_ROOT" \
  --official-root "$OFFICIAL_ROOT" \
  --dinov3-root "$DINOV3_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --source-output-dir "$SOURCE_OUTPUT_DIR" \
  --relation-cache-root "$RELATION_CACHE_ROOT" \
  --device "$DEVICE"
