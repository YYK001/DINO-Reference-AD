#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/DINO-NVS}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
AEBAD_ROOT=${AEBAD_ROOT:-/root/autodl-tmp/datasets/AeBAD/AeBAD_S}
BASELINE_OUTPUT=${BASELINE_OUTPUT:?set BASELINE_OUTPUT to the completed AeBAD-S Full P0 output}
BROAD6_OUTPUT=${BROAD6_OUTPUT:?set BROAD6_OUTPUT to the completed Broad6 output}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_mmr_compatible_eval_seed42_v1}
REPORT_PATH=${REPORT_PATH:-$OUTPUT_DIR/mmr_compatible_objective_report.md}

cd "$PROJECT_ROOT"

ARGS=(
  --aebad-root "$AEBAD_ROOT"
  --baseline-output "$BASELINE_OUTPUT"
  --broad6-output "$BROAD6_OUTPUT"
  --output-dir "$OUTPUT_DIR"
  --report-path "$REPORT_PATH"
)

if [ -n "${PHYS4_OUTPUT:-}" ]; then
  ARGS+=(--phys4-output "$PHYS4_OUTPUT")
fi

exec "$PYTHON" -u -m DINOv3.MADEqual.aebad_s_broad6_compose2.mmr_compatible_eval "${ARGS[@]}"
