#!/usr/bin/env bash
# Default inventory only. No installation, global environment changes or shutdown.
set -euo pipefail
work="${WORK_ROOT:-/home/ubun/YYK}"
repo="${PROJECT_ROOT:-$work/DINO-Reference-AD}"
python="${PYTHON:-$work/conda_envs/yyk/bin/python}"
stage="${1:-preflight}"
if [ "$#" -gt 0 ]; then shift; fi
run="${RUN_ROOT:-$work/runs/aebad_s_blind_context_seed42_v1}"
tag="$(basename "$run")"
cd "$repo"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg XDG_CACHE_HOME="$work/.cache" TMPDIR="$work/tmp"
exec "$python" -B -u -m DINOv3.MADEqual.aebad_s_blind_context_p0.runner "$stage" \
  --parent-run "${PARENT_RUN:-$work/runs/aebad_s_sacp_seed42_20261028_011454}" \
  --parent-cache "${PARENT_CACHE:-$work/storage/sacp_cache/aebad_s_sacp_seed42_20261028_011454}" \
  --raw-run "${RAW_RUN:-$work/runs/aebad_reference_residual_compact_v1_20261027_002907}" \
  --raw-cache "${RAW_CACHE:-$work/storage/reference_residual_cache/aebad_reference_residual_compact_v1_20261027_002907}" \
  --run-root "$run" --cache-root "${CACHE_ROOT:-$work/storage/blind_context_cache/$tag}" \
  --evaluation-scratch "${EVALUATION_SCRATCH:-$work/tmp/${tag}_pixels}" \
  --data "${DATA:-$work/storage/datasets/AeBAD.zip}" \
  --checkpoint "${CHECKPOINT:-$work/storage/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}" \
  --official-root "$repo/external_baselines/SuperADD" --dinov3-root "$repo/external_baselines/dinov3" \
  --device cuda:0 --cpu-workers "${CPU_WORKERS:-8}" "$@"
