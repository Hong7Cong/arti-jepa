#!/usr/bin/env bash
# Stuttering BINARY fluent-vs-disfluent — DYNAMIC (variable-length) eval.
# Samples each event at a target FPS, tiles into in-distribution 32f windows (K ~
# duration), spatial-pools each window, and runs a masked sequence probe over the
# variable-length temporal sequence. 1-GPU + <=8-core caps; cache on /data1.
#   bash scripts/21_eval_stutter_binary_dynamic.sh                          # seq_attentive, 25 fps
#   bash scripts/21_eval_stutter_binary_dynamic.sh --probe seq_lstm
#   bash scripts/21_eval_stutter_binary_dynamic.sh --sample-fps native
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CFG="${DEV_DIR}/configs/eval_stutter_binary_dynamic.yaml"
python -m artijepa.eval_stutter_binary_dynamic --config "${CFG}" "$@"
