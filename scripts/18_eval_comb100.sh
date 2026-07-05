#!/usr/bin/env bash
# Eval the combined (+longitudinal) T-SSL checkpoint ckpt_100 on the usc_lss gold
# OOD phoneme task @256px, WITH saved probe weights (eval_phoneme writes
# <stem>.pt beside <stem>.json; save_probe defaults on).
#
# dtype = bfloat16 to MATCH the existing headline caches (tssl256 / pretrained256)
# so the combined-vs-75only-vs-pretrained lift is apples-to-apples (V100 handles
# bf16). Fresh tag `tssl256comb100` => features extracted once (seed 0), cache-hit
# after. Also regenerates the pretrained256 + tssl256(75-only) attentive probes so
# their reproducible .pt weights exist too (feature caches hit; only probe trains).
# Seeds 0,1,2. Headline probe = attentive; tcn_spatial added for comb100 (same
# spatial cache -> ~free).
set -uo pipefail
DEV=/project2/shrikann_35/hongn/vjepa2/dev_artiJEPA
source "${DEV}/scripts/_env.sh"
CFG=${DEV}/configs/eval_phoneme_usc_lss_256.yaml
DT=bfloat16
SEEDS="0 1 2"
RUNS=${ARTI_OUT}/runs
CKPT100=${RUNS}/tssl_vitl_256_combined/ckpt_100.pt
TSSL256=${RUNS}/tssl_vitl_256/latest.pt

run() {  # <desc> <args...>
  local desc="$1"; shift
  echo "===== ${desc}  ($(date +%H:%M:%S)) ====="
  python -m artijepa.eval_phoneme "$@" --dtype "${DT}" || echo "!!!!! FAILED: ${desc}"
}

echo "########## EVAL-COMB100 START $(date) on $(hostname) ##########"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
[ -f "${CKPT100}" ] || { echo "MISSING ckpt_100: ${CKPT100}"; exit 1; }

# 1) combined ckpt_100 -- fresh tag => extraction on first call, cache-hit after
for pr in attentive tcn_spatial; do
  for s in ${SEEDS}; do
    run "tssl256comb100 | ${pr} | s${s}" --config "${CFG}" \
        --encoder "${CKPT100}" --tag tssl256comb100 --probe "${pr}" --seed "${s}"
  done
done

# 2) references -- regenerate probe .pt (bf16 feature caches hit; probe re-trains)
for s in ${SEEDS}; do
  run "pretrained256 | attentive | s${s}" --config "${CFG}" \
      --tag pretrained256 --probe attentive --seed "${s}"
done
for s in ${SEEDS}; do
  run "tssl256 | attentive | s${s}" --config "${CFG}" \
      --encoder "${TSSL256}" --tag tssl256 --probe attentive --seed "${s}"
done

echo "########## EVAL-COMB100 DONE $(date) ##########"
echo "results + weights -> ${ARTI_OUT}/eval/phoneme_usc_lss_tssl256comb100*_attentive_ce_s{0,1,2}.{json,pt}"
