#!/usr/bin/env bash
set -euo pipefail
work=${WORK_ROOT:-/home/ubun/YYK}
repo=${PROJECT_ROOT:-$work/DINO-Reference-AD}
PY=${PYTHON:-$work/conda_envs/yyk/bin/python}
stage=${1:-preflight}
if [ "$#" -gt 0 ]; then shift; fi
tag=${RUN_TAG:-aebad_shared_residual_translation_v1}
run=${RUN_ROOT:-$work/runs/$tag}
cache=${CACHE_ROOT:-$work/storage/shared_residual_cache/$tag}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$work/.cache" TMPDIR="$work/tmp" MPLBACKEND=Agg
device=cpu
if [ "$stage" != preflight ] && [ "$stage" != summarize ] && [ "$stage" != pack ] && [ "$stage" != evaluate ]; then
  : "${GPU_INDEX:?Select exactly one physical GPU explicitly before GPU stages}"
  [[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || { echo 'GPU_INDEX must be one integer'; exit 1; }
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  device=cuda:0
elif [ -n "${GPU_INDEX:-}" ]; then
  [[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || exit 1
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  device=cuda:0
fi
mkdir -p "$work/tmp" "$work/runs" "$work/.cache"
cd "$repo"
exec "$PY" -B -u -m DINOv3.MADEqual.aebad_s_shared_residual_translation_p0.runner "$stage" \
  --parent-run "$work/runs/aebad_target_loo_two_K8_20261030_013433" \
  --sacp-run "$work/runs/aebad_s_sacp_seed42_20261028_011454" \
  --raw-cache "$work/storage/reference_residual_cache/aebad_reference_residual_compact_v1_20261027_002907" \
  --memory42-root "$work/storage/sacp_cache/aebad_s_sacp_seed42_20261028_011454/predictions" \
  --support-feature-root "$work/storage/joint_response_cache/aebad_joint_reference_response_v1" \
  --data "$work/storage/datasets/AeBAD.zip" \
  --checkpoint "$work/storage/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth" \
  --run-root "$run" --cache-root "$cache" --evaluation-scratch "$work/tmp/${tag}_pixels" \
  --official-root "$repo/external_baselines/SuperADD" --dinov3-root "$repo/external_baselines/dinov3" \
  --device "$device" --cpu-workers 8 "$@"
