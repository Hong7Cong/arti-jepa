#!/usr/bin/env bash
# Offline audio-feature cache for AC-JEPA-audio (decoupled batch step).
# Runs in the `his-extract` env (torch 2.6+cu124 + transformers 4.56) so it can
# load WavLM AND drive the P100/V100 -- the `artijepa` env's transformers (5.x)
# needs torch>=2.7 and hangs. Training reads only the .npy + meta.json.
#
#   bash scripts/08_build_audio_feats.sh [--limit 20] [--manifest ...] [--layer -1]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/project2/shrikann_35/hongn/vjepa2"
DEV_DIR="${REPO_ROOT}/dev_artiJEPA"
export PYTHONPATH="${REPO_ROOT}:${DEV_DIR}:${PYTHONPATH:-}"

source /apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh
conda activate /scratch1/hongn/conda/envs/his-extract
cd "${REPO_ROOT}"

MANIFEST="${MANIFEST:-/scratch1/hongn/artijepa/manifest_alltrain.csv}"
OUT="${OUT:-/scratch1/hongn/artijepa/audio_feats/wavlm_base_plus}"

echo "== audio feature extraction (env: his-extract) =="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true
python -m artijepa.build_audio_features --manifest "${MANIFEST}" --out "${OUT}" "$@"
