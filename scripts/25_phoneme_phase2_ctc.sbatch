#!/bin/bash
#SBATCH --job-name=ph2_loss
#SBATCH --account=shrikann_35
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G          # was 64G: Jul-15 eval-loader OOM on test_lss. Code fix bounds the
                           # eval loaders (2 workers x prefetch 1); this is added headroom for
                           # the mmap page cache over the ~149 GiB spatial cache.
#SBATCH --time=30:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/eval/phase2/slurm_%x_%j.log
#
# Phase 2 -- LOSS LANDSCAPE (CE vs CTC) for phoneme prediction on the combined
# Annot-16 + usc_lss cross-domain split.  RESULTS_phonepred.md "Phase 2".
#
# WHAT / WHY
#   Question: does CTC fix the OOD insertion blow-up (Phase 1 `pretrained`
#   test_lss PER-micro = 1.040 > 1.0, i.e. CE-argmax emits more phonemes than the
#   reference has)?
#
#   Head = `attentive_lstm` (NEW): the V-JEPA AttentivePooler pools the 256 spatial
#   tokens per temporal step (learned, NOT a mean -- *where* in the vocal tract the
#   signal sits is phonetic), then a bi-LSTM over time, then linear.
#
#   Both losses run through the SAME head in UTTERANCE mode (whole utterances, not
#   1.28 s chunks), so this comparison isolates the LOSS.  Running CTC on whole
#   utterances against Phase 1's chunk-mode `attentive` x CE would have confounded
#   loss with temporal context (816 tokens vs 16), so CE is re-run here too.
#   CE reports kappa + PER (kappa ties back to the Phase 1 table); CTC is PER-only
#   (kappa is undefined without a forced alignment).
#
#   Note the plain `attentive` head remains CE-only/chunk-mode and is untouched --
#   the Phase 1 headline stays reproducible.
#
# COST: the frozen spatial caches ALREADY EXIST from Phase 1 (~200 GB each) and the
#   cache hash ignores probe type -> ZERO re-extraction, cache-hit straight to the
#   probe.  Each job = 3 seeds x 30 epochs over the ~149 GiB train cache.
#     tssl256comb100 -> feat_cache/phoneme/tssl256comb100sp_c8eaee90f7/
#     pretrained     -> feat_cache/phoneme/pretrainedsp_589c9438e5/
#
# HOW TO RUN (one job per encoder x loss = 6 jobs; loss in {ce, ctc, focal}):
#   cd /project2/shrikann_35/hongn/vjepa2
#   S=dev_artiJEPA/scripts/25_phoneme_phase2_ctc.sbatch
#   for ENC in tssl256comb100 pretrained; do for L in ce ctc focal; do sbatch $S $ENC $L; done; done
#   # focal = CE reweighted by (1-p_t)^gamma (probe.focal_gamma, default 2.0): the CE-family
#   #   angle on the OOD over-prediction (test_lss PERµ>1.0), same head/context as CE (kappa+PER).
#   # optional 3rd arg = seed list (default "0 1 2"), e.g.  sbatch $S pretrained focal 2
# MONITOR:
#   squeue -u $USER -o "%.12i %.14j %.8T %.10M %R" | grep ph2_loss
#   tail -f /scratch1/hongn/artijepa/eval/phase2/slurm_ph2_loss_<jobid>.log
# OUTPUTS (meta.out = /scratch1/hongn/artijepa/eval):
#   phoneme_usc_lss_<tag>sp_<hash>_attentive_lstm_<loss>_s<seed>.json   metrics
#   phoneme_usc_lss_<tag>sp_<hash>_attentive_lstm_<loss>_s<seed>.pt     best-val probe

set -uo pipefail
ENC="${1:?usage: sbatch 25_phoneme_phase2_ctc.sbatch <tssl256comb100|tssl256comb215|pretrained> <ce|ctc|focal> [seeds]}"
LOSS="${2:?usage: ... <ce|ctc|focal> -- the Phase 2 loss axis}"

source /project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/scripts/_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=dev_artiJEPA/configs/eval_phoneme_annot16_combined.yaml
TSSL_DIR=/scratch1/hongn/artijepa/runs/tssl_vitl_256_combined
mkdir -p /scratch1/hongn/artijepa/eval/phase2

# NOTE: the feature-cache hash (eval_phoneme._tag) keys on spec/size/fps/manifest,
# NOT the ckpt epoch -- so each ckpt MUST get a distinct TAG or it cache-hits stale
# features from another epoch. Tag encodes the epoch (comb100 vs comb215) to keep
# caches AND output JSONs separate.
case "$ENC" in
  tssl256comb100) SEL=(--encoder "$TSSL_DIR/ckpt_100.pt"); TAG=tssl256comb100 ;;  # Phase 1 pinned ckpt
  tssl256comb215) SEL=(--encoder "$TSSL_DIR/ckpt_215.pt"); TAG=tssl256comb215 ;;
  pretrained)     SEL=();                                  TAG=pretrained ;;
  *) echo "unknown encoder key: $ENC (want tssl256comb100|tssl256comb215|pretrained)"; exit 2 ;;
esac
case "$LOSS" in
  ce|ctc|focal) ;;
  *) echo "unknown loss: $LOSS (want ce|ctc|focal)"; exit 2 ;;
esac

SEEDS="${3:-0 1 2}"

echo "########## phoneme phase2: $ENC x $LOSS (tag=$TAG) seeds='$SEEDS' job ${SLURM_JOB_ID:-local} $(date) on $(hostname) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for SEED in $SEEDS; do
  echo "===== [$ENC/$LOSS] seed $SEED START $(date) ====="
  srun python -u -m artijepa.eval_phoneme \
      --config "$CONFIG" --dtype float16 \
      "${SEL[@]}" --tag "$TAG" --probe attentive_lstm --loss "$LOSS" --seed "$SEED"
  echo "===== [$ENC/$LOSS] seed $SEED END   $(date) ====="
done
echo "########## $ENC x $LOSS done $(date) ##########"
