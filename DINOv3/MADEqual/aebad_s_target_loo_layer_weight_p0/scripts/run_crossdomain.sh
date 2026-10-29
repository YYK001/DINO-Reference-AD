#!/usr/bin/env bash
# Default is inventory. All stages and GPU work require explicit invocation.
set -euo pipefail
work=${WORK_ROOT:-/home/ubun/YYK}
repo=${PROJECT_ROOT:-$work/DINO-Reference-AD}
py=${PYTHON:-$work/conda_envs/yyk/bin/python}
stage=${1:-preflight}; if [ "$#" -gt 0 ]; then shift; fi
root=${RUN_ROOT:-$work/runs/aebad_target_loo_crossdomain_seed42_v1}
tag=$(basename "$root")
cache=${CACHE_ROOT:-$work/storage/layer_weight_cache/$tag}
background=${SOURCE_STATS_RUN:-$work/runs/aebad_target_loo_weight_seed42_20261029_182735}
script="$repo/DINOv3/MADEqual/aebad_s_target_loo_layer_weight_p0/scripts/run_workstation.sh"
cd "$repo"
rc=0
if [ "$stage" != aggregate ] && [ "$stage" != pack ]; then
  for domain in illumination view; do
    peer=view; if [ "$domain" = view ]; then peer=illumination; fi
    printf '\nDOMAIN=%s STAGE=%s\n' "$domain" "$stage"
    if env DOMAIN="$domain" RUN_ROOT="$root/$domain" CACHE_ROOT="$cache/$domain" \
      EVALUATION_SCRATCH="$work/tmp/${tag}_${domain}_pixels" SOURCE_STATS_RUN="$background" \
      bash "$script" "$stage" --budget-peer-run "$root/$peer" "$@"; then :
    else
      code=$?
      if [ "$stage" != preflight ]; then exit "$code"; fi
      rc=$code
    fi
  done
fi
if [ "$stage" = all ] || [ "$stage" = aggregate ]; then
  "$py" -B -u -m DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.runner aggregate \
    --run-root "$root/summary" --background-run "$background" \
    --illumination-run "$root/illumination" --view-run "$root/view"
fi
if [ "$stage" = all ] || [ "$stage" = aggregate ] || [ "$stage" = pack ]; then
  "$py" -B -u -m DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.runner pack --run-root "$root"
fi
exit "$rc"
