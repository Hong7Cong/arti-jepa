#!/usr/bin/env bash
# Create the `artijepa` conda env with CUDA 12.x PyTorch.
#
# This server has 4x NVIDIA RTX 6000 Ada (compute capability 8.9, driver
# CUDA 13.0). torch 2.6.0+cu124 supports sm_89 and runs against the cu13
# driver, so we keep the same pin the project was validated with.
#
# Usage:  bash setup_env.sh
set -euo pipefail

CONDA_SH="/data2/hongn/miniconda3/etc/profile.d/conda.sh"
ENV_PREFIX="/data2/hongn/miniconda3/envs/artijepa"   # default envs dir -> `conda activate artijepa`
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
