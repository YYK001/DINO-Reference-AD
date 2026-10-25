#!/usr/bin/env bash
# Run inside tmux. No installation, shutdown, remote connection or resource admission.
set -euo pipefail
: "${WORK:=/home/ubun/YYK}"
: "${PY:=$WORK/conda_envs/yyk/bin/python}"
: "${REPO:=$WORK/DINO-Reference-AD}"
: "${RUN:?set a NEW run directory, or the same directory to resume}"
: "${CACHE:?set a separate mechanical-disk cache directory}"
if [[ "${1:-preflight}" != preflight ]]; then
  : "${GPU:?set the single physical GPU index explicitly}"
  case "$GPU" in *','*) printf 'Only one GPU is allowed\n'; exit 2;; esac
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg XDG_CACHE_HOME="$WORK/.cache" TMPDIR="$WORK/tmp"
[[ "${1:-preflight}" == preflight ]] || mkdir -p "$TMPDIR"
cd "$REPO"
args=(--run-root "$RUN" --cache-root "$CACHE"
  --data "${DATA:-$WORK/storage/datasets/AeBAD.zip}"
  --checkpoint "${CHECKPOINT:-$WORK/storage/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}"
  --official-root "$REPO/external_baselines/SuperADD" --dinov3-root "$REPO/external_baselines/dinov3"
  --device cuda:0 --cpu-workers 8 --query-mode "${QUERY_MODE:-stream}"
  --evaluation-scratch "$WORK/tmp/$(basename "$RUN")/pixel_scratch")
[[ -z "${RECENT_WITNESS:-}" ]] || args+=(--recent-run "$RECENT_WITNESS")
[[ -z "${RECENT_GRAPH:-}" ]] || args+=(--recent-run "$RECENT_GRAPH")
"$PY" -B -u -m DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runner "${1:-preflight}" "${args[@]}"
