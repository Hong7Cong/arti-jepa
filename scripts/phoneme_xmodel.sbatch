#!/bin/bash
#SBATCH --job-name=ph_xmodel
#SBATCH --account=shrikann_35
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/eval/xmodel/slurm_%x_%j.log
#
# Phase 1 -- CROSS-MODEL phoneme headline: attentive x CE x 256 x seeds{0,1,2}.
# One SPATIAL feature extraction per encoder (~79 min, cached under
# .../feat_cache/phoneme/<tag>sp_<hash>/), then 3 fast streaming-loader probes that
# reuse that cache (frozen features are seed-independent). Comparable to
# RESULTS_usclss.md headline. Fixes for the 256px spatial cache (madvise OOM +
# _FeatStream I/O) live in artijepa/eval_phoneme.py; see docs/phonePred.md §4.1.
#
# HOW TO RUN  (one encoder per submission):
#   sbatch dev_artiJEPA/scripts/phoneme_xmodel.sbatch tssl256comb100   # T-SSL combined (ckpt_100), ADAPTED
#   sbatch dev_artiJEPA/scripts/phoneme_xmodel.sbatch pretrained       # V-JEPA2 stock  (s0 already done -> skipped)
#   sbatch dev_artiJEPA/scripts/phoneme_xmodel.sbatch videomae         # off-the-shelf 3-D VideoMAE (native 224/16f)
#   sbatch dev_artiJEPA/scripts/phoneme_xmodel.sbatch vitl             # ImageNet-supervised ViT-L (per-frame)
#   sbatch dev_artiJEPA/scripts/phoneme_xmodel.sbatch dinov2           # DINOv2 ViT-L (per-frame)
#   # videomae_rtmri (ckpt-214) needs a repo->HF conversion first -- NOT wired here (see docs/phonePred.md §5).
# MONITOR:
#   squeue -u $USER
#   tail -f /scratch1/hongn/artijepa/eval/xmodel/slurm_ph_xmodel_<jobid>.log
# OUTPUTS (meta.out = /scratch1/hongn/artijepa/eval):
#   metrics : phoneme_usc_lss_<tag>sp_<hash>_attentive_ce_s<seed>.json
#   probe   : phoneme_usc_lss_<tag>sp_<hash>_attentive_ce_s<seed>.pt   (best-val weights)
#   feats   : .../feat_cache/phoneme/<tag>sp_<hash>/{train,val,test,test_lss}.*.npy

set -uo pipefail
ENC="${1:?usage: sbatch phoneme_xmodel.sbatch <tssl256comb100|tssl256comb215|pretrained|videomae|vitl|dinov2>}"

source /project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/scripts/_env.sh   # conda + PYTHONPATH + cd REPO_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=dev_artiJEPA/configs/eval_phoneme_annot16_combined.yaml
OUTDIR=/scratch1/hongn/artijepa/eval
TSSL=/scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/ckpt_100.pt   # FIXED ckpt_100 snapshot
TSSL215=/scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/ckpt_215.pt # FINAL ckpt (epoch 215, = pinned copy of latest.pt)

case "$ENC" in
  tssl256comb100) SEL=(--encoder "$TSSL");    TAG=tssl256comb100 ;;
  tssl256comb215) SEL=(--encoder "$TSSL215"); TAG=tssl256comb215 ;;
  pretrained)     SEL=();                     TAG=pretrained ;;
  videomae)       SEL=(--model videomae);     TAG=videomae ;;
  vitl)           SEL=(--model vitl);         TAG=vitl ;;
  dinov2)         SEL=(--model dinov2);       TAG=dinov2 ;;
  *) echo "unknown encoder key: $ENC (want tssl256comb100|tssl256comb215|pretrained|videomae|vitl|dinov2)"; exit 2 ;;
esac

# optional 2nd arg = seed list (default all 3), e.g. "sbatch phoneme_xmodel.sbatch tssl256comb100 2"
# runs only that/those seed(s) -- use to finish seeds a TIMEOUT killed (features cache-hit).
SEEDS="${2:-0 1 2}"

echo "########## phoneme x-model: $ENC (tag=$TAG) seeds='$SEEDS' job ${SLURM_JOB_ID:-local} $(date) on $(hostname) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for SEED in $SEEDS; do
  # NOTE: no tag-glob skip here -- meta.out is shared with the usc_lss-only headline,
  # whose runs use the SAME tag prefix but a DIFFERENT manifest hash, so a wildcard
  # <tag>sp_*_s<seed>.json match would falsely skip THIS (combined-manifest) run.
  # eval_phoneme cache-hits the frozen features across seeds anyway, so re-running a
  # seed only re-trains the cheap probe; deterministic -> just overwrites its JSON.
  echo "===== [$ENC] seed $SEED START $(date) ====="
  srun python -u -m artijepa.eval_phoneme \
      --config "$CONFIG" --dtype float16 \
      "${SEL[@]}" --tag "$TAG" --seed "$SEED"
  echo "===== [$ENC] seed $SEED END   $(date) ====="
done
echo "########## $ENC done $(date) ##########"
