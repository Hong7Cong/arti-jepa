# Shared environment setup. `source` this from the other scripts.
# Puts the V-JEPA2 repo root AND the arti-jepa dir on PYTHONPATH so both
# `src.*`/`app.*` (parent repo) and `artijepa.*` (this project) import.
export REPO_ROOT="/data2/hongn/vjepa2"              # parent facebookresearch/vjepa2 clone
export DEV_DIR="/data2/hongn/arti-jepa"             # this project (standalone)
export ARTI_OUT="/data2/hongn/artijepa"            # manifests / stats / runs
export ARTI_DATA_ROOT="/data1/span_data/rtmri75s"          # 75-speaker corpus
export ARTI_LONGI_ROOT="/data1/span_data/longitudinal"     # longitudinal corpus
export PYTHONPATH="${REPO_ROOT}:${DEV_DIR}:${PYTHONPATH:-}"

source /data2/hongn/miniconda3/etc/profile.d/conda.sh
conda activate /data2/hongn/miniconda3/envs/artijepa
cd "${DEV_DIR}"
