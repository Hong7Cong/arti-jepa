"""Smoke test for the full-grid dynamic stutter path (no GPU, no real data)."""
import numpy as np
import torch

import artijepa.eval_stutter_binary_dynamic as E

rng = np.random.default_rng(0)
L = rng.integers(16, 401, 40)
off = np.zeros(41, dtype=np.int64)
off[1:] = np.cumsum(L)
S_, D = 8, 32
feats = rng.standard_normal((int(off[-1]), S_, D)).astype(np.float16)
y = (rng.random(40) < 0.5).astype(np.int64)

cfg = {"meta": {"seed": 0},
       "data": {"pool_mode": "none", "num_workers": 0},
       "probe": {"type": "seq_attentive_lstm", "max_tokens": 1024, "chunk": 8,
                 "checkpoint": True, "eval_workers": 0, "lstm_hidden": 16, "heads": 4,
                 "epochs": 2, "warmup": 1, "class_weight": "balanced"}}

# 1) token budget respects the PADDED bound
ds = E._RaggedDS(feats, off, np.arange(40), y)
sam = E._TokenBudget(ds.lengths, 1024, shuffle=True, seed=0)
batches = list(iter(sam))
worst = max(len(b) * max(ds.lengths[i] for i in b) for b in batches)
print(f"batches={len(batches)} worst padded tokens={worst} "
      f"(budget 1024, Lmax={L.max()})")
assert worst <= max(1024, int(L.max())), worst

# 2) probe forward on a real padded batch
ld = E._ragged_loader(ds, cfg, shuffle=False, eval_mode=True)
x, ln, yy, pos = next(iter(ld))
clf = E.DynamicSeqProbe(D, 2, kind="seq_attentive_lstm", heads=4, lstm_hidden=16,
                        chunk=8, checkpoint=True)
o = clf(x.float(), ln)
print("fwd", tuple(x.shape), "->", tuple(o.shape))
assert o.shape == (x.shape[0], 2)

# 3) ORDER SAFETY: preds must return in idx order despite reordered batches
for j in range(40):
    feats[off[j], 0, 0] = 1.0 if y[j] else -1.0


class Ident(torch.nn.Module):
    """logit driven by the clip's own first feature -> misalignment shows immediately."""

    def forward(self, x, lengths):
        v = x.reshape(x.shape[0], -1)[:, 0]
        return torch.stack([-v, v], -1)


p, _ = E._infer(Ident(), feats, off, np.arange(40), y, torch.device("cpu"), cfg)
print("order-safe preds match labels:", bool((p == y).all()),
      "| any unfilled:", bool((p < 0).any()))
assert (p == y).all() and not (p < 0).any()

# 4) spatial-pooled path keeps fixed-clip batching (unchanged behaviour)
cfg2 = {"meta": {"seed": 0}, "data": {"pool_mode": "spatial", "num_workers": 0},
        "probe": dict(cfg["probe"], type="seq_lstm", batch_size=8)}
f2 = feats.mean(1)
ds2 = E._RaggedDS(f2, off, np.arange(40), y)
b2 = next(iter(E._ragged_loader(ds2, cfg2, shuffle=False, eval_mode=True)))
print("pooled path batch:", tuple(b2[0].shape), "fixed bs=8 ->", b2[0].shape[0] == 8)
assert b2[0].shape[0] == 8 and b2[0].ndim == 3

print("pool_mode_for_probe:",
      {k: E.pool_mode_for_probe(k)
       for k in ["seq_attentive", "seq_lstm", "seq_attentive_lstm"]})
print("ALL OK")
