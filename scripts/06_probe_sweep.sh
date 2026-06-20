#!/usr/bin/env bash
# Probe-head x loss x encoder ablation (Plan B.4) on the CACHED 128px features of
# the V-JEPA pretrained + T-SSL encoders. Heads: tcn/lstm/transformer; loss: CE
# (kappa+PER) / CTC (PER only). Features are a cache-hit, so each run is just a
# fast probe train. One result JSON per combo (…_<head>_<loss>.json).
#   bash dev_artiJEPA/scripts/06_probe_sweep.sh [head ...]   # default: tcn lstm transformer
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"
CFG="${DEV_DIR}/configs/eval_phoneme_usc_lss.yaml"
TSSL=/scratch1/hongn/artijepa/runs/tssl_vitl_128/latest.pt

HEADS=("$@")
[ ${#HEADS[@]} -eq 0 ] && HEADS=(tcn lstm transformer)

for enc in pretrained tssl; do
  if [ "$enc" = pretrained ]; then EARGS=(); TAG=pretrained128
  else EARGS=(--encoder "$TSSL"); TAG=tssl128; fi
  for head in "${HEADS[@]}"; do
    for loss in ce ctc; do
      echo "==================== ${TAG} | ${head} | ${loss} ===================="
      python -m artijepa.eval_phoneme --config "$CFG" "${EARGS[@]}" \
             --tag "$TAG" --probe "$head" --loss "$loss"
    done
  done
done
