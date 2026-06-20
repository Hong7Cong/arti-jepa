#!/usr/bin/env bash
# Phoneme-prediction eval (Plan T0/C): frozen encoder -> per-token phoneme probe,
# report Cohen's kappa + PER. "with vs without T-SSL" = swap --encoder.
# Usage:
#   bash dev_artiJEPA/scripts/04_eval_phoneme.sh                              # Task2 gold, pretrained
#   bash dev_artiJEPA/scripts/04_eval_phoneme.sh <config> [--encoder ... --tag ... --probe ...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"
CONFIG="${1:-dev_artiJEPA/configs/eval_phoneme_usc_lss.yaml}"
shift || true
python -m artijepa.eval_phoneme --config "${CONFIG}" "$@"
