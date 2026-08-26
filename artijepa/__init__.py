# Arti-JEPA: adapting V-JEPA 2 to real-time MRI (rtMRI) vocal-tract video.
#
# This package implements the preprocessing pipeline (Part A) and the
# domain-adaptive self-supervised pre-training track (T-SSL, Part B) described
# in Arti-JEPA-Plans.md. It deliberately reuses the parent V-JEPA 2 repo
# (src.*, app.vjepa.*) for the encoder/predictor/mask machinery and only adds
# the rtMRI-specific data engineering + a single-process T-SSL trainer.
__version__ = "0.1.0"


# Several modules lazily import the parent V-JEPA 2 repo (`src.*`, `app.*`) --
# e.g. src.models.attentive_pooler, src.utils.schedulers. That repo is not
# pip-installable under those names (its setup.py exports `models`, `utils`,
# ... not `src.*`), so when arti-jepa is installed as a package rather than run
# from inside a vjepa2 checkout, point VJEPA2_REPO at the checkout root:
#
#   export VJEPA2_REPO=/path/to/vjepa2
#
# Without it, plain `import artijepa.*` still works; only the V-JEPA-2-backed
# code paths (encoder/predictor/attentive-pooler construction) will fail.
def _add_vjepa2_to_path() -> None:
    import os
    import sys

    repo = os.environ.get("VJEPA2_REPO")
    if repo and os.path.isdir(os.path.join(repo, "src")) and repo not in sys.path:
        sys.path.insert(0, repo)


_add_vjepa2_to_path()
