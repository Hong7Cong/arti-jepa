"""Verify probe weight save/load round-trips (no GPU, synthetic full-grid cache)."""
import numpy as np
import torch

import artijepa.eval_stutter_binary_dynamic as E

rng = np.random.default_rng(0)
N, S_, D = 24, 8, 32
L = rng.integers(16, 60, N)
off = np.zeros(N + 1, dtype=np.int64)
off[1:] = np.cumsum(L)
feats = rng.standard_normal((int(off[-1]), S_, D)).astype(np.float16)
y = (rng.random(N) < 0.5).astype(np.int64)

cfg = {"meta": {"seed": 0}, "data": {"pool_mode": "none", "num_workers": 0},
       "probe": {"type": "seq_attentive_lstm", "max_tokens": 512, "chunk": 8,
                 "checkpoint": False, "eval_workers": 0, "lstm_hidden": 16,
                 "lstm_layers": 1, "bidirectional": True, "heads": 4, "dropout": 0.1,
                 "epochs": 2, "warmup": 1, "class_weight": "balanced"}}

tr = np.arange(0, 16); va = np.arange(16, 20); te = np.arange(20, 24)
tm, pred, vf1, state, bep = E.train_probe(cfg, feats, off, y, tr, va, te, "PWSX",
                                          torch.device("cpu"), E.BINARY_CLASSES)
assert state is not None, "no best_state returned"
print(f"train_probe returned state with {len(state)} tensors, best_epoch={bep}")

# emulate run()'s save block
ck = {"probe_states": {"PWSX": state}, "probe_kind": "seq_attentive_lstm",
      "dim": D, "classes": E.BINARY_CLASSES, "heads": 4, "dropout": 0.1,
      "lstm_hidden": 16, "lstm_layers": 1, "bidirectional": True, "chunk": 8}
p = "/tmp/SLURM_10439186/claude-605458/-project2-shrikann-35-hongn-vjepa2-dev-artiJEPA/" \
    "a76f1ec1-fcd6-4801-aa45-6e779071f977/scratchpad/_probe_test.pt"
torch.save(ck, p)

ck2 = E.load_probe(p)
clf = E.build_probe_from_ckpt(ck2, "PWSX", device="cpu")
print("rebuilt probe:", type(clf).__name__, "| kind =", clf.kind)

# predictions from the rebuilt probe must match the saved weights exactly
ds = E._RaggedDS(feats, off, te, y[te])
x, ln, _yy, _pos = next(iter(E._ragged_loader(ds, cfg, shuffle=False, eval_mode=True)))
with torch.no_grad():
    out_new = clf(x.float(), ln).argmax(-1).numpy()

ref = E.DynamicSeqProbe(D, 2, kind="seq_attentive_lstm", heads=4, dropout=0.1,
                        lstm_hidden=16, lstm_layers=1, bidirectional=True, chunk=8)
ref.load_state_dict(state); ref.eval()
with torch.no_grad():
    out_ref = ref(x.float(), ln).argmax(-1).numpy()

print("reloaded preds == original preds:", bool((out_new == out_ref).all()))
assert (out_new == out_ref).all()
print("ALL OK — weights round-trip exactly")
