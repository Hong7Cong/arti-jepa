#!/usr/bin/env bash
# Finish the MISSING image-baseline attentive spatial probes WITH saved weights
# (eval_phoneme.py writes <stem>.pt beside <stem>.json). Headline 256px attentive
# spatial probe, 3 seeds, for image baselines clip/siglip/dinov2/vitl/resnet
# (native res, spatial grid). clip/siglip hit their existing *sp caches; vitl (the
# key sup-ViT-L/16 competitor), dinov2, resnet have NO spatial cache yet -> fresh
# extraction. float16 = the baselines' cache dtype (matches base_*sp caches).
#   NOTE: the pretrained256 + combined-ckpt eval moved to 18_eval_comb100.sh
#         (bfloat16, apples-to-apples with the tssl256/pretrained256 caches).
# Serial; +e so one failure won't abort.
set -uo pipefail
DEV=/project2/shrikann_35/hongn/vjepa2/dev_artiJEPA
source "${DEV}/scripts/_env.sh"
CFG256="${DEV}/configs/eval_phoneme_usc_lss_256.yaml"
CFGBASE="${DEV}/configs/eval_phoneme_usc_lss_baseline.yaml"
HEAD=attentive
SEEDS="0 1 2"
DT=float16

run() {  # <desc> <args...>
  local desc="$1"; shift
  echo "===== ${desc}  ($(date +%H:%M:%S)) ====="
  python -m artijepa.eval_phoneme "$@" --probe "${HEAD}" --dtype "${DT}" || \
    echo "!!!!! FAILED: ${desc}"
}

echo "########## PROBE-WEIGHTS-256 START $(date) on $(hostname) ##########"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

# image baselines -- native res spatial grid; dinov2@518 needs a tiny clip batch.
# clip/siglip: cache hit (base_clipsp/base_siglipsp). vitl/dinov2/resnet: fresh
# spatial extraction (no *sp cache exists yet).
for m in clip siglip dinov2 vitl resnet; do
  BATCH=""; [ "$m" = "dinov2" ] && BATCH="--batch 2"
  for s in ${SEEDS}; do
    run "base_${m} | ${HEAD} | s${s}" --config "${CFGBASE}" \
        --model "${m}" --tag "base_${m}" --seed "${s}" ${BATCH}
  done
done

echo "########## PROBE-WEIGHTS-256 DONE $(date) ##########"
echo "results + weights -> /scratch1/hongn/artijepa/eval/*_attentive_ce_s{0,1,2}.{json,pt}"
