#!/usr/bin/env bash
# Launch domain-adaptive T-SSL. Default = primary ViT-L @ 256px config.
#   bash scripts/03_train_tssl.sh [config.yaml] [--max-steps N]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"

CONFIG="${1:-${DEV_DIR}/configs/tssl_vitl_256.yaml}"
shift || true

# Default to expandable CUDA segments to avoid fragmentation OOMs at 256px (the
# allocator otherwise strands reserved-but-unallocated blocks). Override by
# exporting PYTORCH_CUDA_ALLOC_CONF before calling. Applies to launches AND resumes.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "== PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} =="

echo "== nvidia-smi (check the GPU is free before a heavy run) =="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv || true

python -m artijepa.tssl_train --config "${CONFIG}" "$@"
