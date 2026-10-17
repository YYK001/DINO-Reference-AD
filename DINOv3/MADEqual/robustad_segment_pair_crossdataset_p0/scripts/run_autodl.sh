#!/usr/bin/env bash
# Manual entry only. Default is read-only preflight; failed stages never shut down.
set -Eeuo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONDONTWRITEBYTECODE=1
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/DINO-NVS}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
ROBUSTAD_ROOT="${ROBUSTAD_ROOT:-/root/autodl-tmp/datasets/RobustAD}"
SOURCE_ROOT="${SOURCE_ROOT:-$PROJECT_ROOT/outputs/relation_reliability/seed42_v6}"
FORMAL_METALPARTS="${FORMAL_METALPARTS:-$PROJECT_ROOT/DINOv3/outputs/mad_equal/robustad_target_fewshot_metalparts_seed42_p0_v1}"
FORMAL_PCB="${FORMAL_PCB:-$PROJECT_ROOT/DINOv3/outputs/mad_equal/robustad_target_fewshot_pcb_seed42_p0_v1}"
DINOV3_ROOT="${DINOV3_ROOT:-/root/autodl-tmp/dinov3}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/DINOv3/outputs/mad_equal/robustad_segment_pair_$(date +%Y%m%d_%H%M%S)}"
CPU_WORKERS="${CPU_WORKERS:-8}"
DEVICE="${DEVICE:-cuda:0}"
EVALUATION_TAG="${EVALUATION_TAG:-v1}"
SUMMARY_DIR="${SUMMARY_DIR:-$OUTPUT_ROOT/summary_$EVALUATION_TAG}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
stage="${1:-preflight}"
if (( $# )); then shift; fi
cd "$PROJECT_ROOT"
module=DINOv3.MADEqual.robustad_segment_pair_crossdataset_p0.runner
args=(--project-root "$PROJECT_ROOT" --data "$ROBUSTAD_ROOT"
  --source-root "$SOURCE_ROOT" --formal-metalparts "$FORMAL_METALPARTS" --formal-pcb "$FORMAL_PCB"
  --dinov3 "$DINOV3_ROOT" --checkpoint "$CHECKPOINT" --output-root "$OUTPUT_ROOT"
  --device "$DEVICE" --cpu-workers "$CPU_WORKERS" --evaluation-tag "$EVALUATION_TAG")
if [[ -n "${SOURCE_RAW_ROOT:-}" ]]; then args+=(--source-raw-root "$SOURCE_RAW_ROOT"); fi
if [[ -n "${FEATURE_ROOT:-}" ]]; then args+=(--feature-root "$FEATURE_ROOT"); fi
if [[ -n "${IMAGE_STATE_ROOT:-}" ]]; then args+=(--image-state-root "$IMAGE_STATE_ROOT"); fi
if [[ "$stage" != all ]]; then
  exec "$PYTHON" -B -u -m "$module" "$stage" "${args[@]}" "$@"
fi
mkdir -p "$OUTPUT_ROOT"
log="$OUTPUT_ROOT/pipeline.log"
trap 'rc=$?; printf "\nPIPELINE_EXIT_CODE=%s; failed, outputs retained; no shutdown\n" "$rc" | tee -a "$log"; exit "$rc"' ERR
started=$(date +%s)
for step in preflight recover-calibration capture predict evaluate evaluate-aupro; do
  printf '\nSTAGE=%s START=%s\n' "$step" "$(date --iso-8601=seconds)" | tee -a "$log"
  "$PYTHON" -B -u -m "$module" "$step" "${args[@]}" "$@" 2>&1 | tee -a "$log"
  printf 'STAGE=%s EXIT_CODE=0\n' "$step" | tee -a "$log"
done
"$PYTHON" -B -u -m "$module" summarize "${args[@]}" --summary-dir "$SUMMARY_DIR" 2>&1 | tee -a "$log"
"$PYTHON" -B -u -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s.get("all_requested_metrics_complete") is True else "metrics or ARD pending; no shutdown")' "$SUMMARY_DIR/status.json"
"$PYTHON" -B -u -m "$module" package "${args[@]}" --summary-dir "$SUMMARY_DIR" --archive "${OUTPUT_ROOT}_review.tar.gz" 2>&1 | tee -a "$log"
printf '\nPIPELINE_EXIT_CODE=0\nELAPSED_SECONDS=%s\nREVIEW_ARCHIVE=%s\n' "$(( $(date +%s)-started ))" "${OUTPUT_ROOT}_review.tar.gz" | tee -a "$log"
if [[ "$AUTO_SHUTDOWN" == 1 ]]; then
  printf 'All requested metrics, ARD and packaging succeeded; shutting down in 10 seconds.\n' | tee -a "$log"
  sleep 10
  shutdown -h now
fi
