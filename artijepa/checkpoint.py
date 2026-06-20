"""Locate / download V-JEPA 2 pretrained checkpoints.

The parent repo's ``src/hub/backbones.py`` points at a localhost stub, so we
fetch the real weights from the public mirror and cache them under /scratch1.
"""

import os

import torch

FBAI_BASE = "https://dl.fbaipublicfiles.com/vjepa2"
DEFAULT_CACHE = "/scratch1/hongn/artijepa/checkpoints"

# model_name -> remote filename
FILES = {
    "vit_large": "vitl.pt",
    "vit_huge": "vith.pt",
    "vit_giant": "vitg.pt",
    "vit_giant_384": "vitg-384.pt",
}


def resolve_checkpoint(model_name="vit_large", path=None, cache_dir=DEFAULT_CACHE):
    """Return a local path to the checkpoint, downloading it if necessary.

    If ``path`` is given and exists it is returned as-is.
    """
    if path and os.path.exists(path):
        return path
    if model_name not in FILES:
        raise ValueError(f"unknown model_name {model_name}; known: {list(FILES)}")
    os.makedirs(cache_dir, exist_ok=True)
    fname = FILES[model_name]
    local = os.path.join(cache_dir, fname)
    if not os.path.exists(local):
        url = f"{FBAI_BASE}/{fname}"
        print(f"[checkpoint] downloading {url} -> {local}")
        torch.hub.download_url_to_file(url, local, progress=True)
    return local


def clean_backbone_key(state_dict):
    """Strip 'module.' / 'backbone.' prefixes -> bare ViT / predictor keys."""
    out = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        out[k] = v
    return out


def filtered_load(module, state_dict):
    """Load only keys whose shapes match (avoids size-mismatch raises).

    Returns (n_loaded, missing, skipped) for logging.
    """
    msd = module.state_dict()
    keep = {k: v for k, v in state_dict.items()
            if k in msd and tuple(v.shape) == tuple(msd[k].shape)}
    module.load_state_dict(keep, strict=False)
    missing = [k for k in msd if k not in keep]
    skipped = [k for k in state_dict if k not in keep]
    return len(keep), missing, skipped
