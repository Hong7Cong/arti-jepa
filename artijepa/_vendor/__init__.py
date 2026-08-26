"""Vendored subset of V-JEPA 2 (https://github.com/facebookresearch/vjepa2).

Only the transitive closure actually reached from `artijepa` is copied here --
15 modules of the upstream `src/` and `app/` trees -- so that arti-jepa installs
and runs without a vjepa2 checkout on PYTHONPATH.

Upstream is MIT-licensed (Copyright (c) Meta Platforms, Inc. and affiliates);
see the LICENSE file next to this one. The files are byte-for-byte upstream
except that `src.*` / `app.*` imports are rewritten to `artijepa._vendor.*`.

Vendored from vjepa2 @ v0.0.2 on 2026-08-25. To refresh, re-copy from upstream
and re-apply the import rewrite.
"""
