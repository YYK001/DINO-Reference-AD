#!/usr/bin/env bash
set -euo pipefail
work=${WORK_ROOT:-/home/ubun/YYK}
repo=${PROJECT_ROOT:-$work/DINO-Reference-AD}
py=${PYTHON:-$work/conda_envs/yyk/bin/python}
run=${RUN_ROOT:-$work/runs/aebad_joint_reference_response_v1}
cache=${CACHE_ROOT:-$work/storage/joint_response_cache/$(basename "$run")}
stage=${1:-preflight};if [ "$#" -gt 0 ]; then shift;fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
export XDG_CACHE_HOME="$work/.cache" TMPDIR="$work/tmp"
mkdir -p "$work/tmp" "$work/.cache"
case "$stage" in capture|capture-normals|smoke|predict|aupro|all|all-cpu)
  : "${GPU_INDEX:?Set the single authorized physical GPU index explicitly}"
  case "$GPU_INDEX" in *[!0-9]*|'') echo 'One integer GPU_INDEX required';exit 2;;esac
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX";;
esac
args=(--parent-run "${PARENT_RUN:-$work/runs/aebad_target_loo_two_K8_20261030_013433}"
 --sacp-run "${SACP_RUN:-$work/runs/aebad_s_sacp_seed42_20261028_011454}"
 --memory42-root "${MEMORY42_ROOT:-$work/storage/sacp_cache/aebad_s_sacp_seed42_20261028_011454/predictions}"
 --memory43-root "${MEMORY43_ROOT:-$work/storage/two_K8_cache/aebad_target_loo_two_K8_20261030_013433}"
 --raw-cache "${RAW_CACHE:-$work/storage/reference_residual_cache/aebad_reference_residual_compact_v1_20261027_002907}"
 --run-root "$run" --cache-root "$cache" --evaluation-scratch "${EVALUATION_SCRATCH:-$work/tmp/$(basename "$run")_pixels}"
 --data "${DATA:-$work/storage/datasets/AeBAD.zip}" --device cuda:0 --cpu-workers 8
 --checkpoint "${CHECKPOINT:-$work/storage/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}"
 --official-root "$repo/external_baselines/SuperADD" --dinov3-root "$repo/external_baselines/dinov3")
for name in SOURCE_BANK SUPPORT_FEATURE_ROOT FEATURE_MANIFEST; do
  if [ -n "${!name:-}" ]; then opt=${name,,};opt=${opt//_/-};args+=("--$opt" "${!name}");fi
done
cd "$repo"
exec "$py" -B -u -m DINOv3.MADEqual.aebad_s_joint_reference_response_p0.runner "$stage" "${args[@]}" "$@"
