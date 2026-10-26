#!/usr/bin/env bash
# Manual entry only. Uses the existing private yyk environment; installs nothing.
set -euo pipefail
stage=${1:-preflight}
profile=${PROFILE:-compact_v1}
work=${WORK_ROOT:-/home/ubun/YYK}
project=${PROJECT_ROOT:-$work/DINO-Reference-AD}
python=${PYTHON:-$work/conda_envs/yyk/bin/python}
tag=${RUN_TAG:-aebad_reference_residual_${profile}_$(date +%Y%m%d_%H%M%S)}
run=${RUN_ROOT:-$work/runs/$tag}
cache=${CACHE_ROOT:-$work/storage/reference_residual_cache/$tag}
scratch=${EVALUATION_SCRATCH:-$work/tmp/$tag}
export CUDA_VISIBLE_DEVICES=${GPU_INDEX:-1}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg XDG_CACHE_HOME=$work/.cache PYTHONDONTWRITEBYTECODE=1
args=(--profile "$profile" --run-root "$run" --cache-root "$cache" --evaluation-scratch "$scratch" --sampling-scratch "${SAMPLING_SCRATCH:-$work/tmp/${tag}_sampling}"
  --data "${DATA:-$work/storage/datasets/AeBAD.zip}"
  --checkpoint "${CHECKPOINT:-$work/storage/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}"
  --official-root "$project/external_baselines/SuperADD"
  --dinov3-root "$project/external_baselines/dinov3"
  --device cuda:0 --cpu-workers "${CPU_WORKERS:-8}" --query-mode "${QUERY_MODE:-stream}"
  --reference-cache-gib "${REFERENCE_CACHE_GIB:-4}")
if [[ $stage == smoke ]]; then args+=(--smoke-mode "${SMOKE_MODE:-constructed}" --smoke-updates "${SMOKE_UPDATES:-30}"); fi
if [[ -n ${RECENT_RUN:-} ]]; then args+=(--recent-run "$RECENT_RUN"); fi
if [[ -n ${SECOND_RECENT_RUN:-} ]]; then args+=(--recent-run "$SECOND_RECENT_RUN"); fi
if [[ -n ${RAW_ASSET_MANIFEST:-} ]]; then args+=(--asset-manifest "$RAW_ASSET_MANIFEST"); fi
cd "$project"
printf 'STAGE=%s\nPROFILE=%s\nRUN_ROOT=%s\nCACHE_ROOT=%s\nGPU_PHYSICAL=%s\n' "$stage" "$profile" "$run" "$cache" "$CUDA_VISIBLE_DEVICES"
"$python" -B -u -m DINOv3.MADEqual.aebad_s_reference_residual_segmentation.runner "$stage" "${args[@]}"
