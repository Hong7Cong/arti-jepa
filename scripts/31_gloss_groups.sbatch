#!/bin/bash
#SBATCH --job-name=glossphg
#SBATCH --account=shrikann_35
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # ragged gloss cache is ~18 GB; preload + page cache headroom
#SBATCH --time=4:00:00     # gloss utts are long connected speech (rainbow/caterpillar/
                           # spont, 16-31s @100fps) and the extractor decodes the whole
                           # video per utt -> ~75min pre + ~85min post ≈ 2.7h extraction
                           # (NOT the ~5min Annot-16 rate); pre cache-hits on resubmit.
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/eval/gloss/slurm_%x_%j.log
#
# Test 4 -- SEGMENT-level gloss transfer (docs/GLOSS.md §12).
#
# WHAT: the Annot-16-trained Phase-3 clip probes (vowels/consonants/manner/place) are
#   applied -- NO retraining -- to pre- and post-glossectomy phoneme segments. One gold
#   phoneme = one sample = one label. Reports macro/weighted F1 + P + R + per-class +
#   confusion + kappa, pooled and per speaker x session (spk2: post1/post2), plus a
#   clustering readout (silhouette by phoneme/speaker/condition, k-means ARI/NMI, and a
#   pre->post nearest-centroid drift matrix vs a leave-one-utterance-out baseline).
#
# WHY not eval_gloss.py: that one is frame-level (sliding window, kappa over 40 ms
#   tokens). Gold boundaries here make it an identity-only read, and the label spaces
#   have no `sil` class for the OOD smear to hide in.
#
# HOW TO RUN:
#   cd /project2/shrikann_35/hongn/vjepa2
#   sbatch dev_artiJEPA/scripts/31_gloss_groups.sbatch                 # tssl256comb215 s0
#   sbatch dev_artiJEPA/scripts/31_gloss_groups.sbatch tssl256comb215 1
#   # DEBUG (own cache, ~1 min):  3rd arg = segment limit per condition
#   sbatch dev_artiJEPA/scripts/31_gloss_groups.sbatch tssl256comb215 0 400
# PREREQ: the Phase-3 probes must exist ->
#   /scratch1/hongn/artijepa/eval/phgroups/phgroups_<ENC>_s<SEED>_{vowels,consonants,manner,place}.pt
#   (scripts/28_phoneme_groups.sbatch <ENC> <SEED>)
# MONITOR:
#   squeue -u $USER -o "%.12i %.14j %.8T %.10M %R" | grep glossphg
#   tail -f /scratch1/hongn/artijepa/eval/gloss/slurm_glossphg_<jobid>.log
# OUTPUTS (/scratch1/hongn/artijepa/eval/gloss):
#   glossphg_<ENC>_s<SEED>.json          metrics + per speaker/session + clusters + drift
#   confmat_glossphg_<task>[_perspk]_*.png   tsne_glossphg_<task>_rep{A,B}_*.png
#   drift_glossphg_<task>_<spk>_rep{A,B}_*.png

set -uo pipefail
ENC="${1:-tssl256comb215}"
SEED="${2:-0}"
LIMIT="${3:-}"
# 4th arg = which tasks get the per-speaker confusion grid + rep A/B t-SNE (the pooled
# confusion for all 4 is always drawn). Default manner,place (the 25x25 consonants
# per-speaker CONFMAT is unwieldy; its t-SNE is fine). A re-run cache-hits the feature
# extraction, so passing "vowels,consonants,manner,place" just redraws the figures cheaply.
FIGS_TASKS="${4:-}"

source /project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/scripts/_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=dev_artiJEPA/configs/eval_gloss_groups.yaml
mkdir -p /scratch1/hongn/artijepa/eval/gloss

LIM=()
[ -n "$LIMIT" ] && LIM=(--limit "$LIMIT")
FIGT=()
[ -n "$FIGS_TASKS" ] && FIGT=(--figs-tasks "$FIGS_TASKS")

echo "########## gloss-groups (segment-level): $ENC seed=$SEED limit='${LIMIT:-none}' figs_tasks='${FIGS_TASKS:-manner,place}' job ${SLURM_JOB_ID:-local} $(date) on $(hostname) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun python -u -m artijepa.eval_gloss_groups \
    --config "$CONFIG" --encoder "$ENC" --seed "$SEED" \
    --spk3-repaired none --cluster --figs "${LIM[@]}" "${FIGT[@]}"
echo "########## $ENC done $(date) ##########"
