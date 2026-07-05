#!/usr/bin/env bash
# Launch AC-JEPA-audio training (frozen encoder + audio-conditioned predictor).
#   bash scripts/09_train_aucjepa.sh [config.yaml] [--max-steps N] [--resume ckpt]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"

CONFIG="${1:-${DEV_DIR}/configs/aucjepa_vitl_128.yaml}"
shift || true
echo "== nvidia-smi (check the GPU is free before a heavy run) =="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv || true

# -u: unbuffered stdout so the redirected log is live (Python block-buffers stdout
# to a file otherwise -> progress lines lag far behind the flushed train_log.csv).
python -u -m artijepa.aai_train --config "${CONFIG}" "$@"
