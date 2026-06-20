#!/usr/bin/env bash
# Create the `artijepa` conda env with V100-compatible (CUDA 12.x) PyTorch.
#
# Why a new env: the existing vjepa2-312 ships torch 2.12+cu130, which cannot
# drive this node's Tesla V100 (driver 575.x => CUDA 12.9). We pin torch 2.6.0
# +cu124 (compute capability 7.0 supported) so the V100 is usable for T-SSL.
#
# Usage:  bash dev_artiJEPA/setup_env.sh
set -euo pipefail

CONDA_SH="/apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh"
ENV_PREFIX="/scratch1/hongn/conda/envs/artijepa"   # keep envs off /home1 quota
PY_VER="3.11"

source "${CONDA_SH}"

if [ ! -d "${ENV_PREFIX}" ]; then
  echo "[setup] creating env at ${ENV_PREFIX} (python ${PY_VER})"
  conda create -y -p "${ENV_PREFIX}" "python=${PY_VER}" pip
fi

conda activate "${ENV_PREFIX}"
python -m pip install --upgrade pip

echo "[setup] installing torch 2.6.0 + torchvision 0.21.0 (cu124, V100-ready)"
python -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

echo "[setup] installing project dependencies"
python -m pip install \
  "numpy<2.3" decord==0.6.0 einops timm pandas pyyaml \
  opencv-python-headless scipy huggingface_hub "transformers>=4.44" \
  iopath beartype psutil tqdm

echo "[setup] sanity check"
python - <<'PY'
import torch, torchvision, decord, einops, timm, numpy, cv2, scipy, yaml
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0),
          "| capability:", torch.cuda.get_device_capability(0))
print("decord", decord.__version__, "| numpy", numpy.__version__, "| cv2", cv2.__version__)
PY

echo "[setup] DONE. Activate with:"
echo "  source ${CONDA_SH} && conda activate ${ENV_PREFIX}"
