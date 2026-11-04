#!/usr/bin/env bash
set -euo pipefail
work=${WORK_ROOT:-/home/ubun/YYK}
repo=${PROJECT_ROOT:-$work/DINO-Reference-AD}
PY=${PYTHON:-$work/conda_envs/yyk/bin/python}
stage=${1:-preflight}
if [ "$#" -gt 0 ]; then shift; fi
profile=${LP_PROFILE:-mixed_background_v2}
if [ "$stage" = final-check ]; then tag=${RUN_TAG:-aebad_reverse_capacity_final_single_v1};else tag=${RUN_TAG:-aebad_reverse_capacity_background_v2};fi
run=${RUN_ROOT:-$work/runs/$tag}
cache=${CACHE_ROOT:-$work/storage/reverse_capacity_cache/$tag}
device=${DEVICE:-cpu}
if [ -n "${GPU_INDEX:-}" ]; then
  [[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || { echo 'GPU_INDEX selects one physical GPU'; exit 1; }
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX";device=cuda:0
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$work/.cache" TMPDIR="$work/tmp" MPLBACKEND=Agg
export CUDA_CACHE_PATH="$work/.cache/cuda" CUPY_CACHE_DIR="$work/.cache/cupy" NUMBA_CACHE_DIR="$work/.cache/numba"
mkdir -p "$work/runs" "$work/tmp" "$work/.cache" "$CUDA_CACHE_PATH" "$CUPY_CACHE_DIR" "$NUMBA_CACHE_DIR"
cd "$repo"
if [ "$stage" = final-check ]; then
  coverage_parent=${COVERAGE_PARENT:-$(cat "$work/runs/latest_reverse_capacity_run.txt")}
  inventory_run="${run}_assets"
  printf 'STAGE=existing-asset-inventory LOG=%s\n' "${inventory_run}_launcher.log"
  if [ ! -f "$inventory_run/inventory.json" ]; then
    if RUN_ROOT="$inventory_run" CACHE_ROOT="${cache}_assets" LP_PROFILE=mixed_background_v2 \
      bash "$0" preflight > "${inventory_run}_launcher.log" 2>&1;then :
    else
      rc=$?
      if [ "$rc" -ne 2 ] || [ ! -f "$inventory_run/inventory.json" ];then cat "${inventory_run}_launcher.log";exit "$rc";fi
    fi
  fi
  printf 'STAGE=final-single-check OUTPUT=%s\n' "$run"
  exec "${LP_PYTHON:-$work/conda_envs/yyk_lp/bin/python}" -B -u \
    -m DINOv3.MADEqual.aebad_s_reverse_capacity_coverage_p0.lp_v2 \
    --final-check --problem-root "${DISTANCE_EXPORT:-${coverage_parent}_gpu_lp_fp32}" \
    --output-root "$run" --inventory-run "$inventory_run" --coverage-parent "$coverage_parent" "$@"
fi
extra=()
if [ "$profile" = mixed_background_v2 ]; then
  coverage_parent=${COVERAGE_PARENT:-$(cat "$work/runs/latest_reverse_capacity_run.txt")}
  extra+=(--lp-profile mixed_background_v2 --resume-from "$coverage_parent"
    --lp-python "${LP_PYTHON:-$work/conda_envs/yyk_lp/bin/python}"
    --distance-export "${DISTANCE_EXPORT:-${coverage_parent}_gpu_lp_fp32}"
    --legacy-trial "${coverage_parent}_gpu_lp_fp32" --legacy-trial "${coverage_parent}_gpu_lp_mixed")
elif [ "$profile" = cpu_v1 ]; then
  extra+=(--lp-profile cpu_v1)
else
  echo "Unknown LP_PROFILE: $profile";exit 1
fi
if [ -n "${FEATURE_MANIFEST:-}" ]; then extra+=(--feature-manifest "$FEATURE_MANIFEST");fi
if [ -n "${RMS_STAT:-}" ]; then extra+=(--rms-stat "$RMS_STAT");fi
exec "$PY" -B -u -m DINOv3.MADEqual.aebad_s_reverse_capacity_coverage_p0.runner "$stage" \
  --run-root "$run" --cache-root "$cache" --device "$device" --cpu-workers "${CPU_WORKERS:-8}" \
  --parent-run "${PARENT_RUN:-$work/runs/aebad_target_loo_two_K8_20261030_013433}" \
  --conditional-run "${CONDITIONAL_RUN:-$work/runs/aebad_conditional_image_score_20261102_224335}" \
  --raw-run "${RAW_RUN:-$work/runs/aebad_reference_residual_compact_v1_20261027_002907}" \
  --asset-run "$work/runs/aebad_s_sacp_seed42_20261028_011454" \
  --asset-run "$work/runs/aebad_correspondence_20261025_201937" \
  --asset-run "$work/runs/aebad_reference_residual_20261026_204836" \
  --feature-root "$work/storage/joint_response_cache/aebad_joint_reference_response_v1" \
  --feature-root "$work/storage/two_K8_cache/aebad_target_loo_two_K8_20261030_013433" \
  --feature-root "$work/storage/conditional_image_score_cache/aebad_conditional_image_score_20261102_224335" \
  --data "$work/storage/datasets/AeBAD.zip" \
  --checkpoint "$work/storage/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth" \
  --official-root "$repo/external_baselines/SuperADD" --dinov3-root "$repo/external_baselines/dinov3" \
  "${extra[@]}" "$@"
