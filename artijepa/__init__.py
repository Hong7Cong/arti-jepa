# Arti-JEPA: adapting V-JEPA 2 to real-time MRI (rtMRI) vocal-tract video.
#
# This package implements the preprocessing pipeline (Part A) and the
# domain-adaptive self-supervised pre-training track (T-SSL, Part B) described
# in Arti-JEPA-Plans.md. It reuses V-JEPA 2's encoder/predictor/mask machinery
# and only adds the rtMRI-specific data engineering + a single-process T-SSL
# trainer. The V-JEPA 2 subset it needs is vendored under `artijepa._vendor`
# (see that package's docstring), so no vjepa2 checkout on PYTHONPATH is
# required.
__version__ = "0.1.0"

