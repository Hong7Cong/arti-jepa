#!/bin/bash
#SBATCH --job-name=phgroups
#SBATCH --account=shrikann_35
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G          # mmap page-cache headroom over the ragged clip cache (~130 GB)
#SBATCH --time=32:00:00    # 256px: ~4.6 h extract (once, cached) + ~1.5-5 h x 5 tasks.
                           # 12 h was too tight and the 2 V-JEPA jobs died mid-task;
                           # raised 24 -> 32 h when `vowcons` made it 5 tasks (2026-08-07).
                           # A CACHED subset run needs far less -- override on the CLI,
                           # e.g.  sbatch --time=8:00:00 $S <enc> 0 "" vowcons
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/eval/phgroups/slurm_%x_%j.log
#
# Phase 3 -- PHONEME-CLIP classification (RESULTS_phonepred.md "Phase 3").
#
# WHAT: one sample = ONE phoneme cut by start-end alignment, classified with a
#   SINGLE label at five granularities: vowcons(2) / vowels(15) / consonants(25) /
#   manner(5) / place(8).  vowcons is the binary vowel-vs-consonant cut over ALL
#   non-sil phonemes and additionally reports roc_auc / average_precision /
#   f1_binary (Consonant = positive).
#   Head = attentive_lstm CLIP mode (AttentivePooler over S' per temporal
#   step -> bi-LSTM over the phoneme's variable-length token sequence -> LAST hidden
#   = the phoneme vector -> linear).  FPS = 50.  Metrics: macro/weighted F1 +
#   precision + recall + per-class + confusion matrix.
#
# DATA: same combined Annot-16 + usc_lss manifest as Phase 1-2.  Train on 14 Annot-16
#   speakers, model-select on val (sub030), report test (in-domain sub043) AND
#   test_lss (cross-domain usc_s1).  The clip feature cache is TASK-AGNOSTIC (all
#   non-sil phonemes) -> extracted ONCE per encoder x split, shared by all 4 tasks.
#
# HOW TO RUN (one job per encoder):
#   cd /project2/shrikann_35/hongn/vjepa2
#   S=dev_artiJEPA/scripts/28_phoneme_groups.sbatch
#   for ENC in tssl256comb215 pretrained videomae videomae_rtmri; do sbatch $S $ENC; done
#   # optional 2nd arg = seed (default 0);  DEBUG: 3rd arg = clip limit/split (e.g. 400)
#   # optional 4th arg = comma-separated task subset (default: all 5).  Pass "" for the
#   # limit when you only want tasks:   sbatch $S pretrained 0 "" place
#   # A task-subset run MERGES into the existing phgroups_<tag>_s<seed>.json: the named
#   # tasks are recomputed, previously-written ones are carried over.  So adding only the
#   # binary task on top of the finished 4-task files is just:
#   #   for ENC in tssl256comb215 pretrained videomae videomae_rtmri; do
#   #       sbatch $S $ENC 0 "" vowcons; done
#   # (the feature cache is task-agnostic -> these skip extraction and reuse it).
# MONITOR:
#   squeue -u $USER -o "%.12i %.14j %.8T %.10M %R" | grep phgroups
#   tail -f /scratch1/hongn/artijepa/eval/phgroups/slurm_phgroups_<jobid>.log
# OUTPUTS (meta.out = /scratch1/hongn/artijepa/eval/phgroups):
#   phgroups_<tag>_s<seed>.json   (all 4 tasks: metrics + confusion matrices)

set -uo pipefail
ENC="${1:?usage: sbatch 28_phoneme_groups.sbatch <tssl256comb215|pretrained|videomae|videomae_rtmri> [seed] [limit]}"
SEED="${2:-0}"
LIMIT="${3:-}"
TASKS="${4:-}"

source /project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/scripts/_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=dev_artiJEPA/configs/eval_phoneme_groups.yaml
TSSL_DIR=/scratch1/hongn/artijepa/runs/tssl_vitl_256_combined
VMAE_RTMRI=/scratch1/hongn/videomae_ct/runs/vitl_rtmri_combined_ct/checkpoint-214.pth
mkdir -p /scratch1/hongn/artijepa/eval/phgroups

case "$ENC" in
  tssl256comb215)  SEL=(--encoder "$TSSL_DIR/ckpt_215.pt");           TAG=tssl256comb215 ;;
  pretrained)      SEL=();                                            TAG=pretrained ;;
  videomae)        SEL=(--model videomae);                            TAG=videomae ;;
  # rt-MRI VideoMAE: repo-format ckpt-214 -> encoder.checkpoint (VideoMAEEncoder loads it)
  videomae_rtmri)  SEL=(--model videomae --checkpoint "$VMAE_RTMRI"); TAG=videomae_rtmri ;;
  *) echo "unknown encoder key: $ENC"; exit 2 ;;
esac
LIM=()
[ -n "$LIMIT" ] && LIM=(--limit "$LIMIT")
TSK=()
[ -n "$TASKS" ] && TSK=(--tasks "$TASKS")

echo "########## phoneme-groups: $ENC (tag=$TAG) seed=$SEED limit='${LIMIT:-none}' tasks='${TASKS:-all}' job ${SLURM_JOB_ID:-local} $(date) on $(hostname) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun python -u -m artijepa.eval_phoneme_groups \
    --config "$CONFIG" --dtype float16 \
    "${SEL[@]}" --tag "$TAG" --seed "$SEED" "${LIM[@]}" "${TSK[@]}"
echo "########## $ENC done $(date) ##########"
