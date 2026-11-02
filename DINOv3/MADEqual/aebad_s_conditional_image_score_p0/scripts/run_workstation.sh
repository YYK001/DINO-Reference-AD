#!/usr/bin/env bash
set -euo pipefail
work=${WORK_ROOT:-/home/ubun/YYK}
repo=${PROJECT_ROOT:-$work/DINO-Reference-AD}
PY=${PYTHON:-$work/conda_envs/yyk/bin/python}
stage=${1:-preflight}
if [ "$#" -gt 0 ]; then shift; fi
tag=${RUN_TAG:-aebad_conditional_image_score_v1}
run=${RUN_ROOT:-$work/runs/$tag}
cache=${CACHE_ROOT:-$work/storage/conditional_image_score_cache/$tag}
device=${DEVICE:-cpu}
if [ -n "${GPU_INDEX:-}" ]; then
  [[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || { echo 'GPU_INDEX must select one physical GPU'; exit 1; }
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  device=cuda:0
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$work/.cache" TMPDIR="$work/tmp" MPLBACKEND=Agg
mkdir -p "$work/runs" "$work/tmp" "$work/.cache"
cd "$repo"
extra=()
for name in STATISTICS_PROVIDER RESPONSE_PROVIDER RETRIEVAL_PROVIDER FEATURE_MANIFEST SOURCE_BANK; do
  value=${!name:-}
  if [ -n "$value" ]; then flag=${name,,};flag=${flag//_/-};extra+=("--$flag" "$value"); fi
done
exec "$PY" -B -u -m DINOv3.MADEqual.aebad_s_conditional_image_score_p0.runner "$stage" \
  --parent-run "${PARENT_RUN:-$work/runs/aebad_target_loo_two_K8_20261030_013433}" \
  --memory42-root "${MEMORY42_ROOT:-$work/storage/sacp_cache/aebad_s_sacp_seed42_20261028_011454/predictions}" \
  --memory43-root "${MEMORY43_ROOT:-$work/storage/two_K8_cache/aebad_target_loo_two_K8_20261030_013433}" \
  --raw-cache "${RAW_CACHE:-$work/storage/reference_residual_cache/aebad_reference_residual_compact_v1_20261027_002907}" \
  --sacp-run "${SACP_RUN:-$work/runs/aebad_s_sacp_seed42_20261028_011454}" \
  --support-feature-root "${SUPPORT_FEATURE_ROOT:-$work/storage/joint_response_cache/aebad_joint_reference_response_v1}" \
  --data "${DATA:-$work/storage/datasets/AeBAD.zip}" \
  --checkpoint "${CHECKPOINT:-$work/storage/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}" \
  --official-root "$repo/external_baselines/SuperADD" --dinov3-root "$repo/external_baselines/dinov3" \
  --run-root "$run" --cache-root "$cache" --device "$device" --cpu-workers 8 "${extra[@]}" "$@"
