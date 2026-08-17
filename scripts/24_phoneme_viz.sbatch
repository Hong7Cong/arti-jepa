#!/bin/bash
#SBATCH --job-name=ph_viz
#SBATCH --account=shrikann_35
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/eval/viz/slurm_%x_%j.log
#
# Cross-model phoneme VISUALS: t-SNE + confusion matrices over the Phase-1
# frozen encoders, reusing the un-pooled spatial feature caches + saved attentive
# probes (docs/phonePred.md 4.2). Four views each:
#   t-SNE   : phoneme (manner) | vowel (identity) | consonant (manner) | place
#   confmat : groups (manner)  | vowels | consonants | place (of articulation)
# Runs on a compute node (NOT the login node -- the caches are large and sklearn/
# torch spawn many threads; the login node's process cap kills it).
#
# HOW TO RUN:
#   sbatch dev_artiJEPA/scripts/24_phoneme_viz.sbatch                 # both splits
#   sbatch dev_artiJEPA/scripts/24_phoneme_viz.sbatch test_lss        # x-domain only
#   sbatch dev_artiJEPA/scripts/24_phoneme_viz.sbatch test            # in-domain only
# OUTPUTS:
#   t-SNE   -> /scratch1/hongn/artijepa/eval/tsne/compare_<view>_repA_tsne_<split>_s0.png
#              + per-encoder tsne_<enc>_<split>_s0.png
#   confmat -> /scratch1/hongn/artijepa/eval/confmat/confmat_<mtype>_compare_<split>_s0.png
#              + per-encoder confmat_<enc>_<split>_s0.png + confmat_values_<split>_s0.json
set -euo pipefail

source /project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/scripts/_env.sh   # conda + PYTHONPATH + cd REPO_ROOT
# a compute node tolerates a few BLAS threads; keep it modest to stay friendly
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

# best -> worst so the comparison panels read left-to-right by cross-domain kappa
ENCODERS="tssl256comb100,videomae_rtmri,pretrained,videomae,dinov2,vitl"
SEED=0
SPLITS="${1:-test_lss test}"     # default: cross-domain first, then in-domain

echo "########## phoneme viz: encoders=$ENCODERS seed=$SEED splits='$SPLITS' job ${SLURM_JOB_ID:-local} $(date) on $(hostname) ##########"

for SPLIT in $SPLITS; do
  echo "===== [$SPLIT] t-SNE (4 views, rep B, cross-model) $(date) ====="
  python -m artijepa.tsne_phonemes --encoder "$ENCODERS" --seed "$SEED" \
      --split "$SPLIT" --view all --rep A --method tsne --cap 200

  echo "===== [$SPLIT] confusion matrices (4 granularities, cross-model) $(date) ====="
  python -m artijepa.confmat_phonemes --encoder "$ENCODERS" --seed "$SEED" \
      --split "$SPLIT" --which all
done

echo "########## phoneme viz DONE $(date) ##########"
