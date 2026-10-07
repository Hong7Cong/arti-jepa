#!/usr/bin/env python3
"""Reproduce a phoneme-probe result from its SAVED WEIGHTS -- no retraining.

Loads a `.pt` probe checkpoint written by eval_phoneme.run() and re-scores the
frozen feature cache with the exact same eval helpers run() uses, then diffs the
recomputed metrics against the ones stored inside the checkpoint.

Works for BOTH heads in RESULTS_phonepred.md:
  * attentive       (Phase-1, chunk-mode CE)     -> predict() + evaluate()
  * attentive_lstm  (Phase-2, utterance-mode)     -> _UttSpatialDS + _utt_eval()

No encoder is loaded: extract() cache-hits the frozen features on disk (keyed by
the checkpoint's feature_tag), so this is cheap and touches only CPU+1 GPU.

Usage:
  python scratchpad/repro_phoneme_probe.py \
      --config dev_artiJEPA/configs/eval_phoneme_annot16_combined.yaml \
      --probe  /scratch1/hongn/artijepa/eval/phoneme_usc_lss_tssl256comb215sp_43c1fe20dd_attentive_lstm_ce_s0.pt
"""
import argparse, json, torch
import artijepa.eval_phoneme as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--probe", required=True, help="path to a *.pt probe checkpoint")
    args = ap.parse_args()

    clf, ck = E.load_probe(args.probe)                 # reconstructs TokenProbe + .eval()
    ptype, loss_kind = ck["probe_kind"], ck["loss"]
    print(f"[repro] probe={ptype} loss={loss_kind} seed={ck['seed']} "
          f"best_epoch={ck['best_epoch']} feature_tag={ck['feature_tag']}")

    cfg = E.load_config(args.config)
    cfg["probe"]["type"] = ptype
    cfg["probe"]["loss"] = loss_kind
    # spatial heads consume the un-pooled [N,T',S',D] grid cache (same as run()).
    SPATIAL = {"tcn_spatial", "attentive", "attentive_lstm"}
    cfg["data"]["pool_spatial"] = ptype not in SPATIAL
    # CRITICAL: the feature-cache tag hashes on encoder.spec + names by meta.tag.
    # Restore BOTH from the checkpoint, else _tag() cache-hits a DIFFERENT encoder's
    # cache and the probe scores garbage features (degenerate PER=1.0).
    cfg["encoder"]["spec"] = ck["encoder_spec"]
    cfg["meta"]["tag"] = ck["feature_tag"].rsplit("_", 1)[0].removesuffix("sp")
    got_tag = E._tag(cfg, "train")[0]
    assert got_tag == ck["feature_tag"], (
        f"tag mismatch: recomputed {got_tag!r} != stored {ck['feature_tag']!r} "
        f"-- would score the wrong feature cache")
    print(f"[repro] feature cache tag OK: {got_tag}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clf = clf.to(device)
    dtype = torch.float16
    extra_names = cfg["data"].get("extra_test_splits", []) or []

    # cache-hit: encoder=None is never dereferenced when the cache exists on disk.
    fva, lva, mva = E.extract(None, cfg, "val", device, dtype)
    fte, lte, mte = E.extract(None, cfg, "test", device, dtype)
    extra_raw = {nm: E.extract(None, cfg, nm, device, dtype) for nm in extra_names}

    val_ds = E.build_dataset(cfg, "val")[0]
    test_ds = E.build_dataset(cfg, "test")[0]
    ref_va, ref_te = val_ds.reference_sequences(), test_ds.reference_sequences()
    extra_ref = {nm: E.build_dataset(cfg, nm)[0].reference_sequences() for nm in extra_names}
    num_classes = val_ds.num_classes
    drop = tuple(val_ds.collapse_drop)

    got = {}
    if ptype in {"attentive_lstm"}:                    # utterance mode
        mt, workers, amp = 1024, 2, (device.type == "cuda")
        def _ev(f, l, m, r):
            ds = E._UttSpatialDS(f, l, m, r)
            return E._utt_eval(clf, ds, device, loss_kind, num_classes, drop, mt, workers, amp)
        got["val"] = _ev(fva, lva, mva, ref_va)
        got["test"] = _ev(fte, lte, mte, ref_te)
        got["tests"] = {nm: _ev(*extra_raw[nm], extra_ref[nm]) for nm in extra_names}
    else:                                              # chunk mode (attentive / tcn_spatial / ...)
        def _ev(f, l, m, r):
            return E.evaluate(E.predict(clf, f, device), l, m, r, num_classes, drop)
        got["val"] = _ev(fva, lva, mva, ref_va)
        got["test"] = _ev(fte, lte, mte, ref_te)
        got["tests"] = {nm: _ev(*extra_raw[nm][:3], extra_ref[nm]) for nm in extra_names}

    stored = ck["metrics"]
    print("\n===== RECOMPUTED vs STORED =====")
    for split in ["val", "test"]:
        print(f"[{split}]      recomputed: {got[split]}")
        print(f"[{split}]      stored:     {stored.get(split)}")
    for nm in extra_names:
        print(f"[{nm}] recomputed: {got['tests'][nm]}")
        print(f"[{nm}] stored:     {stored.get('tests', {}).get(nm)}")


if __name__ == "__main__":
    main()
