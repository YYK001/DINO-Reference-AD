#!/usr/bin/env bash
set -euo pipefail
work=${WORK_ROOT:-/home/ubun/YYK}
repo=${PROJECT_ROOT:-$work/DINO-Reference-AD}
py=${PYTHON:-$work/conda_envs/yyk/bin/python}
stage=${1:-preflight}; if [ "$#" -gt 0 ]; then shift; fi
run=${RUN_ROOT:-$work/runs/aebad_target_loo_weight_seed42_v1}
tag=$(basename "$run")
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$work/.cache" MPLBACKEND=Agg TMPDIR="$work/tmp"
if [ "$stage" = aupro ] || [ "$stage" = all ] || [[ " $* " == *" --supplement-retrieval "* ]]; then
  : "${GPU_INDEX:?Set one physical GPU explicitly}"
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
fi
cd "$repo"
exec "$py" -B -u -m DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.runner "$stage" \
 --run-root "$run" --cache-root "${CACHE_ROOT:-$work/storage/layer_weight_cache/$tag}" \
 --evaluation-scratch "${EVALUATION_SCRATCH:-$work/tmp/${tag}_pixels}" \
 --parent-run "${PARENT_RUN:-$work/runs/aebad_s_sacp_seed42_20261028_011454}" \
 --parent-cache "${PARENT_CACHE:-$work/storage/sacp_cache/aebad_s_sacp_seed42_20261028_011454}" \
 --raw-run "${RAW_RUN:-$work/runs/aebad_reference_residual_compact_v1_20261027_002907}" \
 --raw-cache "${RAW_CACHE:-$work/storage/reference_residual_cache/aebad_reference_residual_compact_v1_20261027_002907}" \
 --data "${DATA:-$work/storage/datasets/AeBAD.zip}" \
 --checkpoint "${CHECKPOINT:-$work/storage/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}" \
 --official-root "$repo/external_baselines/SuperADD" --dinov3-root "$repo/external_baselines/dinov3" \
 --device cuda:0 --cpu-workers 8 "$@"
