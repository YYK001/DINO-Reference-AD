#!/usr/bin/env bash
# No installation or remote access. Default inventory; CUDA uses the user's one selected GPU.
set -euo pipefail
work=${WORK_ROOT:-/home/ubun/YYK}
repo=${REPO_ROOT:-$work/DINO-Reference-AD}
PYTHON=${PYTHON:-$work/conda_envs/yyk/bin/python}
stage=${1:-preflight}
backend=${BACKEND:-cuda}
if [[ "$backend" == cuda ]]; then
  # Identity includes visibility even for stages which never initialize CUDA.
  if [[ -n ${GPU_INDEX:-} ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  fi
  if [[ -z ${CUDA_VISIBLE_DEVICES:-} && "$stage" != preflight && "$stage" != evaluate && "$stage" != report && "$stage" != pack ]]; then
    printf 'Set GPU_INDEX to the one GPU selected by the user. No CUDA/CPU fallback.\n' >&2
    exit 2
  fi
  if [[ ${CUDA_VISIBLE_DEVICES:-} == *,* ]]; then printf 'Exactly one visible GPU is required.\n' >&2; exit 2; fi
fi
export OMP_NUM_THREADS=${CPU_WORKERS:-8} OPENBLAS_NUM_THREADS=${CPU_WORKERS:-8}
export MKL_NUM_THREADS=${CPU_WORKERS:-8} NUMEXPR_NUM_THREADS=${CPU_WORKERS:-8}
export XDG_CACHE_HOME="$work/.cache" TMPDIR="$work/tmp"
cd "$repo"
default_run=$work/runs/aebad_annular_ordinal_v1
if [[ "$backend" == cuda ]]; then default_run=$work/runs/aebad_annular_ordinal_cuda_v1; fi
args=(--run-root "${RUN_ROOT:-$default_run}"
  --cache-root "${CACHE_ROOT:-$work/storage/annular_ordinal_cache/aebad_annular_ordinal_v1}"
  --parent-run "${PARENT_RUN:-$work/runs/aebad_target_loo_two_K8_20261030_013433}"
  --conditional-run "${CONDITIONAL_RUN:-$work/runs/aebad_conditional_image_score_20261102_224335}"
  --raw-run "${RAW_RUN:-$work/runs/aebad_reference_residual_compact_v1_20261027_002907}"
  --data "${DATA:-$work/storage/datasets/AeBAD.zip}" --cpu-workers "${CPU_WORKERS:-8}"
  --backend "$backend" --device "${DEVICE:-cuda:0}"
  --window-batch "${WINDOW_BATCH:-64}" --index-shard-rows "${INDEX_SHARD_ROWS:-16777216}"
  --candidate-tile "${CANDIDATE_TILE:-262144}" --index-cache-gib "${INDEX_CACHE_GIB:-24}"
  --image-cache-gib "${IMAGE_CACHE_GIB:-4}" --geometry-groups "${GEOMETRY_GROUPS:-8}"
  --hypothesis-batch "${HYPOTHESIS_BATCH:-16}" --writer-depth "${WRITER_DEPTH:-2}" --prefetch-depth "${PREFETCH_DEPTH:-2}")
if [[ ${ALLOW_OVER_12H:-0} == 1 ]]; then args+=(--allow-over-12h); fi
exec "$PYTHON" -B -u -m DINOv3.MADEqual.aebad_s_annular_ordinal_p0.runner "$stage" "${args[@]}"
