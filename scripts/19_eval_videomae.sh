#!/usr/bin/env bash
# VideoMAE-L video baseline on usc_lss gold phoneme, 256->native 224px, ATTENTIVE
# spatial probe only, 3 seeds, WITH saved probe weights. The generic video-SSL
# competitor to Arti-JEPA's video-JEPA (vs the per-frame image baselines).
# float16 (V100, no bf16); HF weights cached on /scratch1 (no /home quota).
set -uo pipefail
DEV=/project2/shrikann_35/hongn/vjepa2/dev_artiJEPA
source "${DEV}/scripts/_env.sh"
export HF_HOME=/scratch1/hongn/huggingface_checkpoints
export HF_HUB_CACHE=/scratch1/hongn/huggingface_checkpoints/hub
export HF_HUB_OFFLINE=0
CFG="${DEV}/configs/eval_phoneme_usc_lss_videomae.yaml"

run() {  # <desc> <args...>
  local desc="$1"; shift
  echo "===== ${desc}  ($(date +%H:%M:%S)) ====="
  python -m artijepa.eval_phoneme "$@" --probe attentive --dtype float16 || \
    echo "!!!!! FAILED: ${desc}"
}

echo "########## EVAL-VIDEOMAE START $(date) on $(hostname) ##########"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
for s in 0 1 2; do
  run "videomae | attentive | s${s}" --config "${CFG}" --model videomae --tag videomae --seed "${s}"
done
echo "########## EVAL-VIDEOMAE DONE $(date) ##########"
echo "results + weights -> /scratch1/hongn/artijepa/eval/phoneme_usc_lss_videomae*_attentive_ce_s{0,1,2}.{json,pt}"
