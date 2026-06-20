#!/usr/bin/env bash
# Spatial-aware probe ablation (Plan B.4 probe axis): does keeping the un-pooled
# [B,T',S',D] token grid beat the default mean-over-S' pooling for phonemes?
# Heads (CE only): tcn_spatial (learned attn-pool over S' + TCN over t) and
# attentive (V-JEPA AttentivePooler over S' per t). On the V-JEPA pretrained +
# T-SSL encoders. The un-pooled feature cache is built once per encoder (shared by
# both heads), then each probe train is fast. One result JSON per combo.
#   bash dev_artiJEPA/scripts/07_probe_spatial.sh [config.yaml] [head ...]
# default config = 128px usc_lss; default heads = tcn_spatial attentive
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"

CFG="${DEV_DIR}/configs/eval_phoneme_usc_lss.yaml"
if [[ "${1:-}" == *.yaml ]]; then CFG="$1"; shift; fi
# T-SSL checkpoint: pick 256 if the config is the 256px one, else 128
if [[ "$CFG" == *_256* ]]; then
  TSSL=/scratch1/hongn/artijepa/runs/tssl_vitl_256/latest.pt; RES=256
else
  TSSL=/scratch1/hongn/artijepa/runs/tssl_vitl_128/latest.pt; RES=128
fi

HEADS=("$@")
[ ${#HEADS[@]} -eq 0 ] && HEADS=(tcn_spatial attentive)

for enc in pretrained tssl; do
  if [ "$enc" = pretrained ]; then EARGS=(); TAG="pretrained${RES}"
  else EARGS=(--encoder "$TSSL"); TAG="tssl${RES}"; fi
  for head in "${HEADS[@]}"; do
    echo "==================== ${TAG} | ${head} ===================="
    python -m artijepa.eval_phoneme --config "$CFG" "${EARGS[@]}" \
           --tag "$TAG" --probe "$head"
  done
done
