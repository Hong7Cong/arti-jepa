#!/usr/bin/env bash
# Disfluency-type classification eval (Task 8): frozen encoder / VideoMAE baseline,
# attentive probe @ 256px, leave-one-speaker-out. Writes one result JSON per run.
#   bash scripts/16_eval_disfluency.sh                              # frozen V-JEPA2 baseline
#   bash scripts/16_eval_disfluency.sh --encoder $ARTI_OUT/runs/tssl_vitl_256/latest.pt --tag tssl256
#   bash scripts/16_eval_disfluency.sh --model videomae --tag vmae_frozen
#   bash scripts/16_eval_disfluency.sh --mode finetune --model videomae --tag vmae_ft
#   bash scripts/16_eval_disfluency.sh --model vitl --tag base_vitl        # image baseline
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"
CFG="${DEV_DIR}/configs/eval_disfluency.yaml"
python -m artijepa.eval_disfluency --config "${CFG}" "$@"
