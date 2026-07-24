#!/bin/bash
#SBATCH --job-name=phgplot
#SBATCH --account=shrikann_35
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G          # memmap reads only (no preload); tSNE is the RAM peak
#SBATCH --time=6:00:00     # ~1 h per encoder (4 forwards + 8 t-SNEs), 3 encoders + margin
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/eval/phgroups/slurm_%x_%j.log
#
# Phase-3 FIGURES: confusion matrices + t-SNE for the saved phoneme-CLIP probes.
#
# Runs artijepa/plot_phgroups.py once per encoder tag.  For each tag it reloads the
# saved phgroups_<tag>_s0_<task>.pt probes, re-runs them over the CACHED test_lss
# features (no re-extraction), and writes:
#     figs/phgroups_confmat_<tag>_<split>_s0.png   4 task panels, row-normalized=recall
#     figs/phgroups_tsneA_<tag>_<split>_s0.png     rep A = phoneme vector (probe space)
#     figs/phgroups_tsneB_<tag>_<split>_s0.png     rep B = raw pooled tokens (no probe)
#     figs/phgroups_confmat_<tag>_<split>_s0.json  plotted matrices + re-scored macro-F1
#
# `place` drops the Glottal class by default: Glottal-true clips are excluded AND the
# Glottal logit is masked, so predictions re-argmax over the 7 survivors (rows sum to
# support).  Its macro-F1 therefore differs from the 8-class value in the eval JSON.
# Pass --keep-place-glottal in EXTRA to keep 8 classes.
#
# A tag with a missing probe is SKIPPED for that task only (the figure gets fewer
# panels) -- e.g. `pretrained` has no place probe until job 10528776 lands.
#
# HOW TO RUN:
#   cd /project2/shrikann_35/hongn/vjepa2
#   sbatch dev_artiJEPA/scripts/30_phgroups_plots.sbatch videomae_rtmri videomae pretrained
#   # split/seed via env:  SPLIT=test SEED=0 sbatch ... <tags>
#   # extra flags via env: EXTRA="--keep-place-glottal --per-class 300" sbatch ... <tags>
# MONITOR:
#   tail -f /scratch1/hongn/artijepa/eval/phgroups/slurm_phgplot_<jobid>.log

set -uo pipefail
[ $# -ge 1 ] || { echo "usage: sbatch 30_phgroups_plots.sbatch <tag> [tag ...]"; exit 2; }
SPLIT="${SPLIT:-test_lss}"
SEED="${SEED:-0}"
EXTRA="${EXTRA:-}"

source /project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/scripts/_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8      # t-SNE/PCA are the CPU cost

echo "########## phgroups plots: tags='$*' split=$SPLIT seed=$SEED extra='$EXTRA' job ${SLURM_JOB_ID:-local} $(date) on $(hostname) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

rc=0
for TAG in "$@"; do
  echo "===== PLOTTING $TAG ($(date +%T)) ====="
  srun --ntasks=1 python -u -m artijepa.plot_phgroups \
      --tag "$TAG" --split "$SPLIT" --seed "$SEED" \
      --per-class 600 --workers 8 $EXTRA || { echo "!! $TAG FAILED (rc=$?)"; rc=1; }
done
echo "########## phgroups plots done rc=$rc $(date) ##########"
exit $rc
