#!/usr/bin/env bash
# Launch single-GPU Articulator-Conditioned JEPA training (frozen encoder + arti-
# conditioned predictor, A=6). aucjepa_plans_new.md §2a / M1.
#   bash scripts/12_train_acjepa.sh [config.yaml] [--max-steps N] [--resume ckpt]
# Default = the 128px smoke config (fits a P100). Use acjepa_arti6_256.yaml on a 32 GB GPU.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"

CONFIG="${1:-${DEV_DIR}/configs/acjepa_arti6_128.yaml}"
shift || true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
echo "== acjepa train | config=$(basename "${CONFIG}") =="
python -m artijepa.acjepa_train --config "${CONFIG}" "$@"
