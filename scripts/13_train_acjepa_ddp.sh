#!/usr/bin/env bash
# Launch MULTI-GPU (DDP) Articulator-Conditioned JEPA training via torchrun -- one
# process per visible GPU. aucjepa_plans_new.md §2a.
#   bash scripts/13_train_acjepa_ddp.sh [config.yaml] [--max-steps N] [--resume ckpt]
# Default = the 256px DDP config.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"

CONFIG="${1:-${DEV_DIR}/configs/acjepa_arti6_256_ddp.yaml}"
shift || true

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NGPU="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
else
  NGPU="$(nvidia-smi -L | wc -l)"
fi
NGPU="${NGPU:-1}"

echo "== launching acjepa DDP on ${NGPU} GPU(s) | config=$(basename "${CONFIG}") =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1
torchrun --standalone --nnodes=1 --nproc_per_node="${NGPU}" \
  -m artijepa.acjepa_train_ddp --config "${CONFIG}" "$@"
