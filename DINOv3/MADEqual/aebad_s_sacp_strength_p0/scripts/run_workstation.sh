#!/usr/bin/env bash
# Cache replay only. The sole GPU phase is explicit evaluate-aupro.
set -euo pipefail
work="${WORK_ROOT:-/home/ubun/YYK}"
repo="${PROJECT_ROOT:-$work/DINO-Reference-AD}"
python="${PYTHON:-$work/conda_envs/yyk/bin/python}"
stage="${1:-preflight}"
if [ "$#" -gt 0 ]; then shift; fi
run="${RUN_ROOT:-$work/runs/aebad_s_sacp_strength_l1_v1}"
tag="$(basename "$run")"
parent="${PARENT_RUN:-$work/runs/aebad_s_sacp_seed42_20261028_011454}"
parent_cache="${PARENT_CACHE:-$work/storage/sacp_cache/aebad_s_sacp_seed42_20261028_011454}"
cd "$repo"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg XDG_CACHE_HOME="$work/.cache" TMPDIR="$work/tmp"
exec "$python" -B -u -m DINOv3.MADEqual.aebad_s_sacp_strength_p0.runner "$stage" \
  --parent-run "$parent" --parent-cache "$parent_cache" \
  --run-root "$run" --cache-root "${CACHE_ROOT:-$work/storage/sacp_strength_cache/$tag}" \
  --evaluation-scratch "${EVALUATION_SCRATCH:-$work/tmp/${tag}_pixels}" \
  --data "${DATA:-$work/storage/datasets/AeBAD.zip}" \
  --cpu-workers "${CPU_WORKERS:-8}" --device cuda:0 "$@"
