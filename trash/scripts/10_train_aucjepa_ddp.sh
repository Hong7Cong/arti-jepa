#!/usr/bin/env bash
# Launch MULTI-GPU (DDP) AC-JEPA-audio training (frozen encoder + audio-conditioned
# predictor) via torchrun -- one process per visible GPU.
#   bash scripts/10_train_aucjepa_ddp.sh [config.yaml] [--max-steps N] [--resume ckpt]
# Default config = the 256px DDP config (rtMRI-75 + WavLM).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"

CONFIG="${1:-${DEV_DIR}/configs/aucjepa_vitl_256_ddp.yaml}"
shift || true

# Number of GPUs: honour CUDA_VISIBLE_DEVICES if set, else count all GPUs.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NGPU="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
else
  NGPU="$(nvidia-smi -L | wc -l)"
fi
NGPU="${NGPU:-1}"

echo "== launching DDP on ${NGPU} GPU(s) | config=$(basename "${CONFIG}") =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true

# expandable_segments curbs fragmentation (the 256px graph is large); OMP threads
# split across the ranks; P100s have no NVLink so NCCL runs over PCIe (set
# NCCL_P2P_DISABLE=1 only if peer-to-peer misbehaves).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# --standalone: single-node rendezvous on a free localhost port (no MASTER_ADDR
# wrangling). -u via PYTHONUNBUFFERED so the tee'd log is live.
export PYTHONUNBUFFERED=1
torchrun --standalone --nnodes=1 --nproc_per_node="${NGPU}" \
  -m artijepa.aucjepa_train_ddp --config "${CONFIG}" "$@"
