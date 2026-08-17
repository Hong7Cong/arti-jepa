#!/bin/bash
#SBATCH --job-name=gloss_tsne
#SBATCH --account=shrikann_35
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/eval/gloss/slurm_%x_%j.log
#
# Cross-speaker per-CLASS t-SNE for the glossectomy transfer eval (RESULTS_gloss.md).
# One panel per class, points colored BY SPEAKER (spk1/spk2/spk3 + usc_s1 healthy ref),
# marker = condition (o pre, ^ post, s usc). Three label spaces in one pass:
#     vowels (15 panels) | place (8) | manner (5)
# -- the same label spaces as the segment-level Phase-3 ladder, so these panels are the
# geometry behind those macro-F1 / kappa numbers. Each panel title carries sil(spk), the
# speaker-silhouette of that class (positive = speaker-specific, ~0 = transfers).
#
# CPU-ONLY: reads the frozen feature caches eval_gloss already wrote, no encoder forward,
# no GPU. Runs on a compute node because rep A pushes every cached token through the
# probe's AttentivePooler on CPU (~40 min) and sklearn/t-SNE spawn many threads -- the
# login node's process cap kills it.
#
# HOW TO RUN:
#   sbatch dev_artiJEPA/scripts/33_gloss_tsne_classes.sbatch                       # A+B
#   sbatch dev_artiJEPA/scripts/33_gloss_tsne_classes.sbatch tssl256comb215 attentive_lstm 0 B
#   sbatch dev_artiJEPA/scripts/33_gloss_tsne_classes.sbatch tssl256comb215 attentive_lstm 0 "A B" mincap
# ARGS: [encoder] [head] [seed] [reps] [fig_suffix]
#   fig_suffix lands in the filename before _s<seed>; use it whenever a restyled figure
#   should sit BESIDE the previous version rather than overwrite it.
# OUTPUTS -> /scratch1/hongn/artijepa/eval/gloss/
#   tsne_gloss_{vowels,place,manner}_rep{A,B}_segment_tsne_<enc>_<head>_uscref[_<sfx>]_s<seed>.png
set -euo pipefail

source /project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/scripts/_env.sh   # conda + PYTHONPATH + cd REPO_ROOT
# a compute node tolerates a few BLAS threads; keep it modest to stay friendly
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export SCRATCHPAD="${SCRATCHPAD:-/scratch1/$USER/tmp}"; mkdir -p "$SCRATCHPAD"

ENC="${1:-tssl256comb215}"
HEAD="${2:-attentive_lstm}"
SEED="${3:-0}"
REPS="${4:-A B}"          # A = probe-q space (headline), B = raw encoder geometry
FIGSFX="${5:-}"           # extra filename tag, so a restyle does not clobber the old figure

echo "########## gloss per-class t-SNE: enc=$ENC head=$HEAD seed=$SEED reps='$REPS' figsfx='${FIGSFX:-none}' job ${SLURM_JOB_ID:-local} $(date) on $(hostname) ##########"

for REP in $REPS; do
  echo "===== rep $REP  (vowels + place + manner, segment level, usc_s1 overlay) $(date) ====="
  python -m artijepa.tsne_gloss_vowels \
      --encoder "$ENC" --head "$HEAD" --seed "$SEED" \
      --train-set annot16 --rep "$REP" --per segment \
      --label-space vowels,place,manner --include-usc --cap 200 \
      --fig-suffix "$FIGSFX"
done

echo "########## gloss per-class t-SNE DONE $(date) ##########"
