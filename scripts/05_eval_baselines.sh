#!/usr/bin/env bash
# Image-encoder baselines for the gold/OOD phoneme eval (Plan Part C):
# frozen CLIP / SigLIP / DINOv2 / supervised ViT-L / ResNet, same probe + labels
# as the V-JEPA rows. Writes one result JSON per model to .../eval/.
#   bash dev_artiJEPA/scripts/05_eval_baselines.sh                 # all five
#   bash dev_artiJEPA/scripts/05_eval_baselines.sh clip siglip     # subset
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"
CFG="${DEV_DIR}/configs/eval_phoneme_usc_lss_baseline.yaml"

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(clip siglip dinov2 vitl resnet)

for m in "${MODELS[@]}"; do
  # DINOv2's native 518px is heavy -> smaller extraction batch to respect the
  # 16 GB host-RAM cap; the others fit at the config default.
  BATCH=""
  [ "$m" = "dinov2" ] && BATCH="--batch 2"
  echo "============================================================"
  echo "== baseline: ${m}  ${BATCH}"
  echo "============================================================"
  python -m artijepa.eval_phoneme --config "${CFG}" --model "${m}" --tag "base_${m}" ${BATCH}
done
