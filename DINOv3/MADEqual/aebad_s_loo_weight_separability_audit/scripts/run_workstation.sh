#!/usr/bin/env bash
set -euo pipefail
work=${WORK_ROOT:-/home/ubun/YYK}
repo=${PROJECT_ROOT:-$work/DINO-Reference-AD}
py=${PYTHON:-$work/conda_envs/yyk/bin/python}
stage=${1:-preflight}; if [ "$#" -gt 0 ]; then shift; fi
run=${RUN_ROOT:-$work/runs/aebad_loo_weight_cross_audit_v1}
tag=$(basename "$run")
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$work/.cache" TMPDIR="$work/tmp" MPLBACKEND=Agg
mkdir -p "$work/tmp" "$work/.cache" "$work/runs"
if [ "$stage" = aupro ] || [ "$stage" = all ]; then
 : "${GPU_INDEX:?Set one physical GPU for explicit ADEval execution}"
 export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
fi
extra=()
if [ -n "${RESPONSE_ROOT:-}" ]; then extra+=(--response-root "$RESPONSE_ROOT"); fi
if [ -n "${EXPORT_ROOT:-}" ]; then extra+=(--export-root "$EXPORT_ROOT"); fi
cd "$repo"
exec "$py" -B -u -m DINOv3.MADEqual.aebad_s_loo_weight_separability_audit.runner "$stage" \
 --parent-run "${PARENT_RUN:-$work/runs/aebad_target_loo_two_K8_20261030_013433}" \
 --output-root "$run" --evaluation-scratch "${EVALUATION_SCRATCH:-$work/tmp/${tag}_pixels}" \
 --memory42-root "${MEMORY42_ROOT:-$work/storage/sacp_cache/aebad_s_sacp_seed42_20261028_011454/predictions}" \
 --memory43-root "${MEMORY43_ROOT:-$work/storage/two_K8_cache/aebad_target_loo_two_K8_20261030_013433}" \
 --data "${DATA:-$work/storage/datasets/AeBAD.zip}" --device cuda:0 --cpu-workers 8 "${extra[@]}" "$@"
