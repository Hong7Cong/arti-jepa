# Arti-JEPA: adapting V-JEPA 2 to real-time MRI (rtMRI) vocal-tract video.
#
# This package implements the preprocessing pipeline (Part A) and the
# domain-adaptive self-supervised pre-training track (T-SSL, Part B) described
# in Arti-JEPA-Plans.md. It deliberately reuses the parent V-JEPA 2 repo
# (src.*, app.vjepa.*) for the encoder/predictor/mask machinery and only adds
# the rtMRI-specific data engineering + a single-process T-SSL trainer.
__version__ = "0.1.0"
