Initinal Settings:
The environment for building/running Arti-JEPA is the conda env `artijepa`
(torch 2.6+cu124, V100/L40S-compatible) — activate via
`source dev_artiJEPA/scripts/_env.sh`. (The stock `vjepa2-312` env ships
torch 2.12+cu130 which can't drive this node's GPU.)
src code and development is at /project2/shrikann_35/hongn/vjepa2/dev_artiJEPA, (JEPA source code for reference is at /project2/shrikann_35/hongn/vjepa2. no change in this folder only make changes in dev_artiJEPA folder)
75-speaker rtMRI video dataset (unlabeled, T-SSL pretraining) is at /scratch1/hongn/speaker75

Downstream evaluation = PHONEME PREDICTION (the old stimulus-group probe was
dropped as not meaningful). Two tasks, both probing frozen encoder features with
a small per-token classifier; metrics = frame-level Cohen's kappa + Phoneme Error
Rate (PER). The "with vs without T-SSL" lift on these is the headline result.
  * Task 1 — PSEUDO phonemes from an audio phoneme model (wav2vec2 / WavLM CTC)
    run on the 75-speaker corpus's paired audio (22.05 kHz). Self-consistent
    transfer probe.
  * Task 2 — GOLD phonemes with timestamps for one held-out OOD speaker at
    /scratch1/hongn/usc_lss (104x104 video @ 99 fps, 16 kHz audio, 684 utts,
    ARPABET). OOD in speaker + resolution + frame rate.
Temporal alignment is done in SECONDS so any fps works: clips are resampled to
target_fps (25), the audio phoneme stream is ~50 Hz (20 ms), so 2 audio units ≈
1 video frame and 4 ≈ 1 JEPA token (tubelet 2 -> 80 ms). The 99-fps OOD speaker
needs no special-casing because labels live in seconds, not native frames.
