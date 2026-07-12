#!/usr/bin/env bash
# Stuttering BINARY fluent-vs-disfluent eval (Task 8b): frozen 256px T-SSL V-JEPA2
# encoder + attentive probe, leave-one-speaker-out. OpenCV dataloader (pal8-safe;
# do NOT use the decord-based eval_disfluency on this corpus). Writes a result JSON.
#   bash scripts/20_eval_stutter_binary.sh                                   # tssl256 LOSO
#   bash scripts/20_eval_stutter_binary.sh --checkpoint $ARTI_OUT/runs/tssl_vitl_256/latest.pt --tag tssl_latest
#   bash scripts/20_eval_stutter_binary.sh --probe mean                      # cheaper probe
#   bash scripts/20_eval_stutter_binary.sh --split fixed --test-speaker PWS10 --val-speaker PWS7
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"

# --- resource caps: 1 GPU, <= 8 CPU cores -------------------------------------
# Single GPU: expose only one device (override e.g. CUDA_VISIBLE_DEVICES=1 ... to pick another).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Belt-and-suspenders BLAS/OMP caps (Python also pins per-worker threads at runtime).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CFG="${DEV_DIR}/configs/eval_stutter_binary.yaml"
# CPU budget is num_workers (6) + cpu_threads (2) = 8 cores; tune with the flags below.
python -m artijepa.eval_stutter_binary --config "${CFG}" "$@"
