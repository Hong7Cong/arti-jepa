#!/usr/bin/env bash
# Launch domain-adaptive T-SSL. Default = primary ViT-L @ 256px config.
#   bash scripts/03_train_tssl.sh [config.yaml] [--max-steps N]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"

CONFIG="${1:-${DEV_DIR}/configs/tssl_vitl_256.yaml}"
shift || true
echo "== nvidia-smi (check the GPU is free before a heavy run) =="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv || true

python -m artijepa.tssl_train --config "${CONFIG}" "$@"
