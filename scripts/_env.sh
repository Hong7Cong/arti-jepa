# Shared environment setup. `source` this from the other scripts.
# Puts the V-JEPA2 repo root AND dev_artiJEPA on PYTHONPATH so both
# `src.*`/`app.*` (parent repo) and `artijepa.*` (this project) import.
export REPO_ROOT="/project2/shrikann_35/hongn/vjepa2"
export DEV_DIR="${REPO_ROOT}/dev_artiJEPA"
export ARTI_OUT="/scratch1/hongn/artijepa"          # all artifacts off /project2
export PYTHONPATH="${REPO_ROOT}:${DEV_DIR}:${PYTHONPATH:-}"

source /apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh
conda activate /scratch1/hongn/conda/envs/artijepa
cd "${REPO_ROOT}"
