#!/usr/bin/env bash
# No activation, installation, GPU selection or remote access. Default inventory.
set -euo pipefail
work=${WORK_ROOT:-/home/ubun/YYK}
repo=${REPO_ROOT:-$work/DINO-Reference-AD}
PYTHON=${PYTHON:-$work/conda_envs/yyk/bin/python}
stage=${1:-preflight}
export OMP_NUM_THREADS=${CPU_WORKERS:-8} OPENBLAS_NUM_THREADS=${CPU_WORKERS:-8}
export MKL_NUM_THREADS=${CPU_WORKERS:-8} NUMEXPR_NUM_THREADS=${CPU_WORKERS:-8}
export XDG_CACHE_HOME="$work/.cache" TMPDIR="$work/tmp"
cd "$repo"
args=(--run-root "${RUN_ROOT:-$work/runs/aebad_annular_ordinal_v1}"
  --cache-root "${CACHE_ROOT:-$work/storage/annular_ordinal_cache/aebad_annular_ordinal_v1}"
  --parent-run "${PARENT_RUN:-$work/runs/aebad_target_loo_two_K8_20261030_013433}"
  --conditional-run "${CONDITIONAL_RUN:-$work/runs/aebad_conditional_image_score_20261102_224335}"
  --raw-run "${RAW_RUN:-$work/runs/aebad_reference_residual_compact_v1_20261027_002907}"
  --data "${DATA:-$work/storage/datasets/AeBAD.zip}" --cpu-workers "${CPU_WORKERS:-8}")
if [[ ${ALLOW_OVER_12H:-0} == 1 ]]; then args+=(--allow-over-12h); fi
exec "$PYTHON" -B -u -m DINOv3.MADEqual.aebad_s_annular_ordinal_p0.runner "$stage" "${args[@]}"
