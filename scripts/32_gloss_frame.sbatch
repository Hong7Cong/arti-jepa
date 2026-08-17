#!/bin/bash
#SBATCH --job-name=glossframe
#SBATCH --account=shrikann_35
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G          # 256px un-pooled spatial cache for 38 pre + 48 post utts
#SBATCH --time=8:00:00     # first seed extracts both condition caches (~2-3 h); seeds
                           # 1-2 cache-hit and are ~10 min each. One encoder per job.
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/eval/gloss/slurm_%x_%j.log
#
# FRAME-level gloss transfer (docs/GLOSS.md §11) -- the launcher the frame-level results
# never had (they were run by hand from the RESULTS "Reproduce" block).
#
# WHAT: the Annot-16-trained phoneme probe is applied -- NO retraining -- to pre/post
#   glossectomy features; kappa/PER over 40 ms tokens, pooled + per speaker x session,
#   with the in-domain and healthy-OOD (usc_lss) anchors carried in the JSON.
#
# CORPUS (rebuilt 2026-08-06): manifests point at the NATIVE ~99.4 fps source videos
#   (`spk*/<cond>/{video,avi}`), NOT the old uniform-100-fps `resampled_video/` re-encode.
#   The dataloader resamples native -> target_fps 50 by interpolation, so the extra 100 fps
#   intermediate was lossy and unnecessary. Sub-20-fps acquisitions are excluded by
#   `gloss.build_manifest --min-fps 20`. pre n=38, post n=48 (spk3/post = 16 after the
#   filename correction). => ALL caches are new; nothing cache-hits from before this date.
#
# HOW TO RUN:
#   cd /project2/shrikann_35/hongn/vjepa2
#   sbatch dev_artiJEPA/scripts/32_gloss_frame.sbatch tssl256comb215 attentive_lstm 0
#   sbatch dev_artiJEPA/scripts/32_gloss_frame.sbatch tssl256comb100 attentive      0
#   # seeds 1,2 after seed 0 has written the cache (they then cache-hit):
#   sbatch dev_artiJEPA/scripts/32_gloss_frame.sbatch tssl256comb215 attentive_lstm 1
# PREREQ: the matching Annot-16 probes must exist ->
#   /scratch1/hongn/artijepa/eval/phoneme/*_<HEAD>_ce_s<SEED>.pt
# MONITOR:
#   squeue -u $USER -o "%.12i %.14j %.8T %.10M %R" | grep glossframe
#   tail -f /scratch1/hongn/artijepa/eval/gloss/slurm_glossframe_<jobid>.log
# OUTPUTS (/scratch1/hongn/artijepa/eval/gloss):
#   gloss_transfer_<ENC>_annot16[_<HEAD>]_s<SEED>.json   metrics + per-speaker + anchors
#   confmat_gloss_*_s<SEED>.png   tsne_gloss_*_s<SEED>.png

set -uo pipefail
ENC="${1:-tssl256comb215}"
HEAD="${2:-attentive_lstm}"
SEED="${3:-0}"
EXTRA="${4:-}"          # e.g. --per-speaker-figs

source /project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/scripts/_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

XTRA=()
[ -n "$EXTRA" ] && read -r -a XTRA <<< "$EXTRA"

echo "########## gloss-frame: $ENC head=$HEAD seed=$SEED job ${SLURM_JOB_ID:-local} $(date) on $(hostname) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun python -u -m artijepa.eval_gloss \
    --encoder "$ENC" --train-set annot16 --head "$HEAD" --loss ce --seed "$SEED" \
    "${XTRA[@]}"
echo "########## $ENC/$HEAD s$SEED done $(date) ##########"
