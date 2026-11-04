#!/usr/bin/env bash
set -euo pipefail
work=${WORK_ROOT:-/home/ubun/YYK}
repo=${PROJECT_ROOT:-$work/DINO-Reference-AD}
script="$repo/DINOv3/MADEqual/aebad_s_reverse_capacity_coverage_p0/scripts/run_workstation.sh"
stage=${1:-preflight}
if [ "$#" -gt 0 ]; then shift; fi
if [ "$stage" = preflight ]; then
  export LP_PROFILE=mixed_background_v2
  exec bash "$script" preflight "$@"
fi
[[ "$stage" = all || "$stage" = smoke || "$stage" = final-check ]] || { echo 'Use preflight, smoke, all or final-check';exit 1; }
: "${GPU_INDEX:?Set GPU_INDEX to the agreed single physical GPU}"
if [ "$stage" = final-check ];then tag=aebad_reverse_capacity_final_single_v1;else tag=aebad_reverse_capacity_background_v2_$(date +%Y%m%d_%H%M%S);fi
run=${RUN_ROOT:-$work/runs/$tag}
cache=${CACHE_ROOT:-$work/storage/reverse_capacity_cache/$(basename "$run")}
parent=${COVERAGE_PARENT:-$(cat "$work/runs/latest_reverse_capacity_run.txt")}
session=${SESSION:-${tag}_$(date +%Y%m%d_%H%M%S)}
launcher="${run}_launch_${session}.sh"
log="${run}_launcher.log"
run_resolved=$(realpath -m "$run")
parent_resolved=$(realpath -m "$parent")
if [[ "$run_resolved" = "$parent_resolved" || "$run_resolved" = "$parent_resolved/"* || "$parent_resolved" = "$run_resolved/"* ]]; then
  echo 'RUN_ROOT overlaps the original run; select a new phase1 directory.';exit 1
fi
mkdir -p "$work/runs"
{
  printf '#!/usr/bin/env bash\nset -uo pipefail\n'
  printf 'export WORK_ROOT=%q PROJECT_ROOT=%q RUN_ROOT=%q CACHE_ROOT=%q COVERAGE_PARENT=%q LP_PROFILE=mixed_background_v2 GPU_INDEX=%q CPU_WORKERS=%q\n' \
    "$work" "$repo" "$run" "$cache" "$parent" "$GPU_INDEX" "${CPU_WORKERS:-8}"
  printf 'cd %q || exit 1\n' "$repo"
  if [ "$stage" = final-check ];then
    printf 'flock -n %q bash %q %q' "${run}.lock" "$script" "$stage"
  else
    printf 'bash %q %q --extract-missing' "$script" "$stage"
  fi
  if [ "$#" -gt 0 ]; then printf ' %q' "$@";fi
  printf '\nrc=$?\nprintf "%%s\\n" "$rc" > %q\nprintf "FINAL_EXIT_CODE=%%s\\n" "$rc"\nexit "$rc"\n' "${run}.exitcode"
} > "$launcher"
touch "$log"
tmux new-session -d -s "$session" "bash $(printf '%q' "$launcher") >> $(printf '%q' "$log") 2>&1"
printf '%s\n' "$run" > "$work/runs/latest_reverse_capacity_phase1_run.txt"
printf 'SESSION=%s\nRUN=%s\nLOG=%s\n' "$session" "$run" "$log"
tail -n 40 -F "$log"
