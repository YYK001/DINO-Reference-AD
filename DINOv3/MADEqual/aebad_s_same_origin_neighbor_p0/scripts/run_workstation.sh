#!/usr/bin/env bash
# Existing yyk only. Default is inventory; no installs, network or shutdown.
set -euo pipefail
work="${WORK_ROOT:-/home/ubun/YYK}"
repo="${PROJECT_ROOT:-$work/DINO-Reference-AD}"
py="${PYTHON:-$work/conda_envs/yyk/bin/python}"
stage="${1:-preflight}"
if [ "$#" -gt 0 ]; then shift; fi
run="${RUN_ROOT:-$work/runs/aebad_same_origin_neighbor_seed42_v1}"
tag="$(basename "$run")"
cd "$repo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg XDG_CACHE_HOME="$work/.cache" TMPDIR="$work/tmp"
if [ "$stage" != preflight ] && [ "$stage" != pack ] && [ "$stage" != summarize ]; then
  : "${GPU_INDEX:?Set the single physical GPU explicitly, e.g. GPU_INDEX=1}"
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
fi
args=()
for name in PARENT_RUN PARENT_CACHE RAW_CACHE FEATURE_ROOT FEATURE_MANIFEST SOURCE_BANK SELECTED_INDICES SOURCE_NEIGHBORHOOD AUGMENTED_GRIDS CATALOG; do
  if [ -n "${!name:-}" ]; then flag="$(printf '%s' "$name" | tr '[:upper:]_' '[:lower:]-')"; args+=("--$flag" "${!name}"); fi
done
exec "$py" -B -u -m DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0.runner "$stage" \
  --run-root "$run" --cache-root "${CACHE_ROOT:-$work/storage/same_origin_cache/$tag}" \
  --evaluation-scratch "${EVALUATION_SCRATCH:-$work/tmp/${tag}_pixels}" \
  --build-scratch "${BUILD_SCRATCH:-$work/storage/same_origin_build/$tag}" \
  --data "${DATA:-$work/storage/datasets/AeBAD.zip}" \
  --checkpoint "${CHECKPOINT:-$work/storage/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}" \
  --official-root "$repo/external_baselines/SuperADD" --dinov3-root "$repo/external_baselines/dinov3" \
  --device cuda:0 --cpu-workers "${CPU_WORKERS:-8}" "${args[@]}" "$@"
