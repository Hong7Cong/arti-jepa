# Stuttering rtMRI Corpus — Annotation Statistics

Documentation of the stuttering disfluency corpus used for Arti-JEPA downstream
**Task 8** (segment-level disfluency-type classification). Statistics below are
computed directly from the raw Praat `.TextGrid` annotations and from the derived
manifest (`disfluency_manifest.csv`).

- **Location:** `/data1/span_data/stuttering/`
- **Loader / parser:** `artijepa/stutter.py`
- **Binary (fluent vs disfluent) dataloader:** `artijepa/stutter_binary.py` (§7)
- **Type eval:** `artijepa/eval_disfluency.py`, config `configs/eval_disfluency.yaml`
- **Binary train+eval:** `artijepa/eval_stutter_binary.py`, config
  `configs/eval_stutter_binary.yaml`, launcher `scripts/20_eval_stutter_binary.sh` (§8)
- **Stats reproduced:** 2026-07-09, from the raw TextGrids (ground truth) and manifest.

> ⚠️ **Decoder bug (important).** Every stuttering `.avi` is `rawvideo` /
> `pix_fmt=pal8` (8-bit palettized). **`decord`'s `VideoReader` decodes these as
> all-zero (black) frames** — verified: every frame `max=0`. `stutter.py` and
> `eval_disfluency.py` load with decord, so any features/eval they produce on this
> corpus are computed on **black clips** and are invalid. **OpenCV decodes the files
> correctly** (`max=255`); `stutter_binary.py` uses OpenCV. Any pipeline touching
> this corpus must not use decord until the codec is transcoded (e.g. to mpeg4/
> h264 yuv420p) or the loader is switched to OpenCV.

---

## 1. Corpus overview

Seven **persons-who-stutter (PWS)** with real-time MRI (rtMRI) of the vocal tract,
paired 1:1 with audio and Praat TextGrid annotations. Same acquisition geometry as
`usc_lss`.

| Property | Value |
|---|---|
| Speakers | PWS3, PWS4, PWS5, PWS6, PWS7, PWS8, PWS10 (7 total) |
| Video | 104 × 104 grayscale rtMRI, **~99.4 fps** (`.avi`) |
| Audio | `.wav`, one per video stem |
| Annotations | Praat long-format `.TextGrid`, one per stem |
| Annotated TextGrids | **476** total |
| Task type | Segment classification (each labeled interval = one example) |

Per-speaker file counts (TextGrid / avi / wav):

| Speaker | TextGrid | avi | wav |
|---|---|---|---|
| PWS3  | 62 | 62 | 62 |
| PWS4  | 77 | 77 | 77 |
| PWS5  | 76 | 86 | 86 |
| PWS6  | 78 | 85 | 85 |
| PWS7  | 78 | 95 | 69 |
| PWS8  | 78 | 86 | 86 |
| PWS10 | 27 | 27 | 27 |

Note: not every `.avi`/`.wav` has a matching TextGrid, and vice-versa. The manifest
builder keeps only stems that have **both** a TextGrid event **and** a video file
(419 distinct stems contribute ≥1 event).

### Tiers present in the TextGrids

Each TextGrid can carry several interval tiers. Files containing each tier:

| Tier | # files | Role |
|---|---|---|
| `words` | 450 | word segmentation |
| `disfluency` | 450 | **primary** disfluency events (`<phoneme>_<type>`) |
| `phones` | 343 | phone segmentation |
| `IPs` | 90 | intonational phrases |
| `disfluency2` | 27 | secondary/overlapping disfluency (PWS10 only) |

The two `disfluency*` tiers hold the labels used for Task 8. Each event is a labeled
interval `[xmin, xmax]` in **seconds**, with text of the form `<phoneme>_<type>`
(e.g. `DH_block`, `W_pro+rep`).

---

## 2. Raw annotated events (ground truth)

Counting only **non-empty** labeled intervals directly from the TextGrids:

| Tier | Events | Speakers with events | Total duration | Median dur | Mean dur | Range |
|---|---|---|---|---|---|---|
| `disfluency`  | **2,108** | all 7 | 5,383 s | 2.12 s | 2.55 s | 0.45–25.1 s |
| `disfluency2` | **126**   | PWS10 only | 733 s | 5.17 s | 5.81 s | 1.20–24.6 s |

### Events per speaker (`disfluency` tier)

| Speaker | Events |
|---|---|
| PWS3  | 579 |
| PWS4  | 246 |
| PWS5  | 267 |
| PWS6  | 507 |
| PWS7  | 107 |
| PWS8  | 248 |
| PWS10 | 154 |

The corpus is **imbalanced across speakers** (PWS3 and PWS6 account for ~51% of all
events; PWS7 only 5%). This motivates the leave-one-speaker-out (LOSO) eval protocol
and inverse-frequency class weighting.

---

## 3. Disfluency-type distribution

Type = the substring after the first `_`. Labels are canonicalized (typo repair,
lowercasing, `?` stripped); compound events use `+` (e.g. `pro+rep`).

### 3.1 Primary type (7-type space, `disfluency` tier)

Taking the **first** canonical component of each event:

| Type | Count | % |
|---|---|---|
| rep (repetition)      | 706 | 33.5% |
| block                 | 703 | 33.4% |
| pro (prolongation)    | 630 | 29.9% |
| osci (oscillation)    | 42  | 2.0% |
| revert                | 18  | 0.9% |
| abandon               | 4   | 0.2% |
| filler                | 4   | 0.2% |

- **1 event** was uncanonicalizable (raw text `error`).
- The three dominant types (**rep / block / pro**) cover **96.8%** of all events.
- The tail (osci / revert / abandon / filler) is folded into `other` for the eval.

### 3.2 5-way bucket (`bucket5`, used by the eval)

| Bucket | Count |
|---|---|
| rep    | 706 |
| block  | 703 |
| pro    | 630 |
| osci   | 42 |
| other  | 26 |

### 3.3 Compound / multi-type events

**368 events (17.5%)** carry more than one type via `+`. The most common raw type
strings (single- and multi-label):

| Raw type | Count |
|---|---|
| `rep`        | 586 |
| `block`      | 531 |
| `pro`        | 463 |
| `pro+rep`    | 116 |
| `rep+block`  | 72 |
| `block?`     | 71 |
| `block+rep`  | 43 |
| `osci`       | 35 |
| `pro?`       | 32 |
| `block+pro`  | 21 |
| `rep+pro`    | 20 |
| `block+osci` | 14 |

For the single-label head, the **primary** (first) component is used; the full
ordered set is retained in the manifest `multi` column for a multi-label variant.

### 3.4 Secondary tier (`disfluency2`, PWS10 only)

Almost entirely repetitions: **rep 125, pro 1** (126 events). This tier is optional
in the eval (`data.tiers`); the canonical setup uses only `disfluency`.

---

## 4. Phoneme carrying the disfluency

The prefix before the first `_` names the phoneme on which the disfluency occurred.
**133 distinct** phoneme prefixes appear. Top 20:

| Phoneme | Count | | Phoneme | Count |
|---|---|---|---|---|
| DH | 199 | | K  | 59 |
| W  | 186 | | D  | 56 |
| B  | 160 | | IH | 51 |
| S  | 134 | | SH | 48 |
| R  | 120 | | H  | 46 |
| P  | 105 | | AY | 42 |
| T  | 91  | | G  | 38 |
| M  | 83  | | Y  | 38 |
| L  | 75  | | AH | 37 |
| F  | 75  | | N  | 62 |

Consonants (esp. `DH`, `W`, `B`, `S`, `R`, `P`) dominate the onset positions where
disfluencies are annotated.

---

## 5. Derived manifest (`disfluency_manifest.csv`)

Built by `stutter.build_manifest`. One row per labeled event that (a) has a video,
(b) canonicalizes, and (c) has duration in **[min_dur, max_dur] = [0.10, 8.0] s**.
The distributed manifest also samples **fluent** (empty-interval) negatives to seed
the binary fluent-vs-disfluent baseline.

- **3,130 rows total** across 419 stems.
- By tier: `disfluency` 3,029, `disfluency2` 101.
- **2,171 disfluent** rows + **959 fluent** negatives.

The manifest counts differ from the raw ground-truth §3 because it (a) adds the
`disfluency2` tier, (b) drops events longer than 8 s or shorter than 0.10 s, and
(c) adds sampled fluent negatives.

### `bucket5` × speaker (non-fluent rows, as used by the eval)

| Speaker | block | rep | pro | osci | other | Total |
|---|---|---|---|---|---|---|
| PWS3  | 18  | 381 | 166 | 11 | 0  | 576 |
| PWS4  | 89  | 28  | 124 | 1  | 3  | 245 |
| PWS5  | 139 | 65  | 56  | 5  | 1  | 266 |
| PWS6  | 292 | 11  | 192 | 10 | 0  | 505 |
| PWS7  | 83  | 5   | 18  | 0  | 1  | 107 |
| PWS8  | 42  | 168 | 14  | 12 | 10 | 246 |
| PWS10 | 30  | 137 | 50  | 3  | 6  | 226 |
| **Total** | **693** | **795** | **620** | **42** | **21** | **2,171** |

Note the **per-speaker type distribution varies sharply** (PWS3 is rep-heavy, PWS6
is block/pro-heavy, PWS7 has almost no rep). Under LOSO this makes the type-priors
differ between train and held-out test — a genuine domain shift that the eval must
handle.

---

## 6. Task setup implications

- **Class imbalance** is severe (block/rep/pro vs. osci vs. tail). Primary metric is
  **macro-F1**; the probe uses inverse-frequency class weighting
  (`probe.class_weight: balanced`).
- **Speaker imbalance + speaker-specific priors** motivate LOSO evaluation
  (`data.split_mode: loso`).
- **Label tasks** (`stutter.label_space`):
  - `type5` — block / rep / pro / osci / other (canonical)
  - `type3` — block / rep / pro (rare types dropped)
  - `binary` — fluent / disfluent (needs the sampled fluent negatives)
- Alignment is in **seconds**, so the ~99 fps video is uniformly resampled to
  `frames_per_clip` positions spanning each (padded) event — no fps special-casing.

---

## 7. Binary fluent-vs-disfluent dataloader (`stutter_binary.py`)

A dataloader that yields **disfluency clips (label 1)** vs **regular fluent-speech
clips (label 0)** for binary classification, with negatives drawn to **match the
positive duration distribution** (so clip length is not a give-away feature).

**Clip definitions**

| Class | Source |
|---|---|
| Positive (disfluent, 1) | every `disfluency`-tier event `[xmin, xmax]` (canonicalizable, dur ∈ [`min_dur`, `max_dur`]) |
| Negative (fluent, 0)    | a window carved from a *fluent-speech region* = (non-empty `words` intervals, merged across pauses ≤ `merge_gap` s) **minus** every `disfluency` event |

Negatives use the `disfluency` tier as truth for "is this stretch disfluent" and the
`words` tier only to stay on actual speech (not silence). This does **not** rely on
the `flue_`/`disf_` word-prefix convention (only ~2,170 words use it; the rest are
plain), so it works corpus-wide.

**Duration matching.** Each negative's target length is drawn from the empirical
positive-duration pool and placed (non-overlapping) inside a fluent region that fits;
if none fits, the largest region is used whole. Because disfluencies are frequent in
these PWS recordings, fluent stretches are naturally short, so realized negatives
skew a bit shorter than positives — the best achievable from real fluent speech:

| | n | p50 (s) | mean (s) | p90 (s) | max (s) |
|---|---|---|---|---|---|
| Positives | 2,070 | 2.09 | 2.41 | 4.21 | 7.96 |
| Negatives (matched) | 1,831 | 1.36 | 1.55 | 2.77 | 7.94 |

(defaults: `neg_per_pos=1`, `seed=0`, `tiers=[disfluency]`, `min_dur=0.20`,
`max_dur=8.0`, `merge_gap=0.25`; a larger `merge_gap` closes the gap slightly.)

Balanced ~1:1 per file → **2,070 pos / 1,831 neg** (3,901 clips) across the same 7
speakers, so **leave-one-speaker-out** splits are clean (`filter_speakers`).

**Frames.** Each clip is **200 frames sampled uniformly across its window at the
video's native ~99 fps** (linear interp between bracketing native frames), then the
standard Arti-JEPA preprocessing (percentile-clip → z-score/minmax → bicubic resize
→ grayscale ×3). Output tensor: `[3, 200, S, S]` (default `S=256`).

**Usage**

```python
from artijepa import stutter_binary as SB
rows, stats = SB.build_rows(seed=0, neg_per_pos=1)         # pos + duration-matched neg
loader, ds = SB.make_loader(rows, num_frames=200, batch_size=8, shuffle=True)
for clips, labels, meta in loader:      # clips [B,3,200,256,256], labels ∈ {0,1}
    ...

# leave-one-speaker-out:
train = SB.filter_speakers(rows, keep={"PWS3","PWS4","PWS5","PWS6","PWS7","PWS8"})
test  = SB.filter_speakers(rows, keep={"PWS10"})
```

CLI (build a manifest and/or sanity-check one batch; needs the `artijepa` env):

```bash
python -m artijepa.stutter_binary --check --num-frames 200 --neg-per-pos 1
python -m artijepa.stutter_binary --out /data1/span_data/stuttering/binary_manifest.csv
```

---

## 8. Binary fluent-vs-disfluent eval (`eval_stutter_binary.py`)

Frozen-encoder **train + eval** for the binary task, built on the §7 dataloader.
Script `artijepa/eval_stutter_binary.py`, config `configs/eval_stutter_binary.yaml`,
launcher `scripts/20_eval_stutter_binary.sh`.

> ⚠️ Use **this** script, not `eval_disfluency.py --task binary`, on this corpus:
> `eval_disfluency` decodes with **decord** → all-black clips (§ decoder bug). This
> eval uses the OpenCV `stutter_binary` loader.

**Pipeline.** (1) build binary rows (§7); (2) run the frozen encoder once, pool each
clip to one feature, **cache it**; (3) train an attentive/mean/mlp probe (inverse-freq
CE), model-select on a stratified val split by macro-F1, report the held-out speaker.
LOSO folds reuse the single cache.

**Encoder / geometry.** The 256px combined T-SSL V-JEPA2 checkpoint
(`/data1/hongn/arti-jepa/tssl_vitl_256_combined/ckpt_100.pt`), loaded exactly as in
`examples/demo.ipynb`. Clips are fed at the checkpoint's **256px / 32-frame** geometry
(→ 4096 tokens/clip), **not** the loader's 200-frame default — 200f ≈ 25k tokens is
far OOD for a 32f-pretrained ViT-L and ~memory-prohibitive.

**Resource caps.** 1 GPU (peak ~2–3 GiB VRAM at batch 8; `CUDA_VISIBLE_DEVICES=0`)
and ≤ 8 CPU cores (`num_workers=6` decode workers pinned to 1 thread each +
`cpu_threads=2`; tune via `--num-workers` / `--cpu-threads`). Extraction of all 3,901
clips takes ~9 min (video-decode bound); the 7 probe folds are seconds each on cache.

**Artifacts (all on `/data1`).** Feature cache
`/data1/hongn/arti-jepa/feat_cache/stutter_binary/<tag>/` (~31 GB: the `[N,4096,1024]`
fp16 attentive token grid); results + log
`/data1/hongn/arti-jepa/eval/stutter_binary/`.

> **grayscale-stats caveat.** The combined checkpoint trained with
> `grayscale_stats_combined.json`, which was **not** preserved when the artifact tree
> moved from `/data2/hongn/artijepa` → `/data1/hongn/arti-jepa` (2026-07-11). Absent
> the file, the global channel-norm falls back to **mean=0/std=1** — which
> `compute_stats.py` documents these values land near anyway (they are computed on
> already per-clip z-scored clips), so the effect is a negligible global affine
> offset. Restore the json at `/data1/hongn/arti-jepa/grayscale_stats_combined.json`
> to override. The script prints which path it took.

### Results — frozen tssl-256, attentive probe, LOSO (seed 0, 2026-07-11)

Pooled over all 3,901 held-out clips (`neg_per_pos=1`, 2,070 disfluent / 1,831 fluent):

| Metric | Value |
|---|---|
| **macro-F1 (pooled)** | **0.828** |
| balanced acc | 0.827 |
| accuracy | 0.829 |
| fluent  P / R / F1 | 0.83 / 0.80 / 0.81 |
| disfluent P / R / F1 | 0.83 / 0.85 / 0.84 |
| mean-over-folds macro-F1 | 0.816 |

Per held-out speaker (macro-F1): PWS7 0.935 · PWS6 0.895 · PWS3 0.868 · PWS5 0.855 ·
PWS10 0.802 · PWS4 0.700 · PWS8 0.656. The spread (0.66–0.94) reflects the
per-speaker domain shift documented in §5 — val macro-F1 sits ~0.90 on every fold,
so the drop on PWS4/PWS8 is held-out-speaker generalization, not underfitting.

Reproduce:

```bash
bash scripts/20_eval_stutter_binary.sh                 # frozen tssl256, LOSO, attentive
bash scripts/20_eval_stutter_binary.sh --probe mean    # cheaper linear-on-mean probe
bash scripts/20_eval_stutter_binary.sh --split fixed --test-speaker PWS10 --val-speaker PWS7
```

### Probe types — `attentive` · `pooled_attentive` · `attentive_lstm`

The head pools the frozen encoder's `[T'·S', D]` token grid to one label. The probe
type also chooses **how the grid is reduced at extraction** (`pool_mode`, which sets
the cache shape) — see `pool_mode_for_probe`:

- **`attentive`** (`pool_mode=none`, cache `[N, T'·S', D]`) — one V-JEPA
  `AttentivePooler` (1 query) over **all** T'·S' tokens → linear. Simple, but its cost
  and its 31 GiB (32f) / 190 GiB (200f) cache grow with the *joint* token count.
- **`pooled_attentive`** (`pool_mode=spatial`, cache `[N, T', D]`) — **mean-pool the S'
  spatial tokens of each frame at extraction**, then an `AttentivePooler` over the
  resulting **temporal** sequence → linear. Same head as `attentive`, but over T'
  (16–100) tokens instead of T'·S' (~4k–25k). The cache is **S'× (256×) smaller** —
  32f → **123 MB** (measured), 200f → ~0.8 GiB — so it fits GPU/RAM trivially with no
  page-cache thrash, and probe VRAM is ~0.2 GiB. Cost: within-frame spatial structure
  is averaged away before the head (temporal attention is kept). **This is the
  recommended path for 200f** (§10.6).
- **`attentive_lstm`** (`pool_mode=none`) — *factorized* spatiotemporal pooling that matches the signal's
  structure: an `AttentivePooler` over the **S' spatial tokens of each frame**
  (one shared pooler, applied per temporal step) → a `[T', D]` per-frame sequence →
  **LSTM over time** (bi-dir by default) → linear. Spatial attention is confined to
  S'=256 tokens; temporal modeling is an O(T') recurrence, not an O((T'·S')²)
  attention. The per-frame spatial pool runs in **temporal chunks** (`probe.chunk`)
  with optional **gradient checkpointing** (`probe.checkpoint`), so peak activation
  memory is O(chunk·S') instead of O(T'·S').

`attentive` and `attentive_lstm` share the full-grid cache (no re-extraction between
them); `pooled_attentive` and `mean`/`mlp` each have their own (smaller) cache.

**Measured LOSO (frozen tssl-256, seed 0):**

| probe | frames | cache | probe VRAM | pooled macro-F1 | mean macro-F1 |
|---|---|---|---|---|---|
| attentive        | 32f  | 31 GiB  | ~7 GiB (B64)   | **0.828** | 0.816 |
| pooled_attentive | 32f  | **123 MB** | **~0.2 GiB** | 0.811 | 0.808 |
| attentive_lstm   | 32f  | 31 GiB  | ~7 GiB (B64)   | 0.813 | 0.816 |
| pooled_attentive | 200f | 763 MB  | ~0.2 GiB       | 0.729 | 0.685 |

At 32f, `pooled_attentive` costs ~1.5 macro-F1 points vs the full grid but shrinks the
cache **256×** and probe VRAM ~35× — the practical choice when the full grid won't fit.

> ⚠️ **200f is worse, not better.** `pooled_attentive` @ 200f mechanically *works* — the
> cache is 0.76 GiB, it fits one GPU at 12 GiB, extraction ~52 min — but macro-F1
> **drops to 0.73/0.69**. Cause: feeding 200 raw frames to the **32f-pretrained RoPE
> encoder is far out-of-distribution** (§ geometry note). More frames ≠ better features
> when the encoder never saw that temporal length. To use ~200 frames of context, encode
> in-distribution **32f windows** and concatenate their temporal tokens — do not feed 200
> raw frames to this encoder.

```bash
bash scripts/20_eval_stutter_binary.sh --probe pooled_attentive     # tiny cache, temporal attn
bash scripts/20_eval_stutter_binary.sh --probe attentive_lstm
# high frame counts — bound VRAM with chunking + checkpointing:
bash scripts/20_eval_stutter_binary.sh --frames 200 --probe attentive_lstm \
     --lstm-chunk 20 --lstm-checkpoint --batch 16
```

**Measured VRAM/step (fwd+bwd, fp32, RTX 6000 Ada), head only:**

| grid | probe | B=16 | B=32/64 |
|---|---|---|---|
| 32f (N=4,096)  | attentive          | 1.8 GiB | 7.1 GiB (B64) |
| 32f (N=4,096)  | attentive_lstm     | 1.9 GiB | 7.1 GiB (B64) |
| 200f (N=25,600)| attentive          | 11.0 GiB | 22.0 GiB (B32) |
| 200f (N=25,600)| attentive_lstm     | 11.1 GiB | 22.0 GiB (B32) |
| 200f (N=25,600)| **attentive_lstm + chunk20 + gradckpt** | **5.1 GiB** | **10.1 GiB (B32)** |

Key point (honest): at equal frames, `attentive_lstm` alone is **not** cheaper than
flat attentive — both project every B·T'·S' token. **The VRAM win comes from chunked
+ gradient-checkpointed spatial pooling** (≈½ the peak at 200f, less with smaller
chunks), which is only possible *because* the spatial pool is factorized per frame.
Its other advantage is the inductive bias: an explicit temporal recurrence over the
articulatory trajectory, which suits disfluency dynamics (blocks/prolongations/
repetitions). (For a frozen 32f encoder, prefer chunk-encoding 32f windows over
feeding 200 raw frames to the RoPE ViT — see the geometry note above.)

### Training curves (`plot_stutter_binary.py`)

`eval_stutter_binary` logs only sparse points and keeps just the best-epoch summary.
`artijepa/plot_stutter_binary.py` re-runs the same probe on the **cached** features
(preloaded onto one GPU as fp16 → no per-batch host↔device transfer; the whole thing
in minutes, not the ~½ hr the transfer-bound path takes) and records train/val loss
and macro-F1 at **every** epoch for all 7 folds, writing:

- `eval/stutter_binary/…_curves_s0.png` — 4-panel: mean±std loss & macro-F1, plus
  per-speaker val-F1 (• = selected epoch) and train-loss.
- `…_curves_s0.{json,csv}` — full per-epoch history (long-format CSV for re-plotting).

```bash
python -m artijepa.plot_stutter_binary --config configs/eval_stutter_binary.yaml
```

**What the curves show.** Train loss → 0 / train macro-F1 → 1.0 by ~epoch 25, while
**val loss turns up after ~epoch 8** and val macro-F1 plateaus ~0.90 — a clear
overfit past the elbow. Model selection on val macro-F1 (the • markers, epochs 8–26)
catches the peak; the probe is intentionally tiny + weight-decayed, and the
early-stopping selection is what keeps the held-out numbers honest. (Curves are a
faithful fp16 re-run: mean best test macro-F1 = 0.815, matching the 0.816 above.)

---

## 9. Dynamic-length eval (`eval_stutter_binary_dynamic.py`)

Everything in §8 resamples each event to a **fixed** frame budget. This variant
accepts **variable-length** inputs so a clip's temporal extent tracks the event's
real duration/rate. Module `artijepa/stutter_dynamic.py` (dataloader) +
`artijepa/eval_stutter_binary_dynamic.py` (extract/probe/eval), config
`configs/eval_stutter_binary_dynamic.yaml`, launcher
`scripts/21_eval_stutter_binary_dynamic.sh`. **The fixed-length §8 path is untouched.**

**Pipeline.** Sample each event at a target FPS (`--sample-fps 25`, or `native` ≈99)
→ frame budget `round(dur·fps)` → tile into **K in-distribution `window`(=32)-frame
clips**, K ∝ duration. Encode each window, **mean-pool its S' spatial tokens** → one
vector/frame, concatenate the K windows → a variable-length sequence `[L=K·T', D]`.
A **masked sequence probe** classifies it:

- **`seq_attentive`** — a learned query attends over the L frame-vectors with a
  key-padding mask (length-agnostic; default).
- **`seq_lstm`** — packed bi-LSTM → masked mean of outputs.

The cache is **ragged** (`[ΣL, D]` fp16 + per-clip `offsets`), kept tiny by the
spatial pooling. Batches pad to the batch-max length + a lengths vector; the mask
makes padding a no-op.

**Why windows, not raw frames.** Every window is 32f, so the frozen 32f-pretrained
encoder stays in-distribution — this is the correct way to get long temporal context
(contrast §10.8, where raw 200f collapsed to 0.73 from OOD).

**Measured (frozen tssl-256, LOSO, seed 0, sample_fps=25, window=32).**

| item | value |
|---|---|
| windows/clip | min 1 · median 2 · max 7 (∝ duration) |
| sequence length | median 32 · max 112 tokens |
| ragged cache | **250 MB** (`[127840, 1024]` fp16); extract 476 s |
| `seq_attentive` | macro-F1 **0.767** pooled / 0.755 mean |
| `seq_lstm` | macro-F1 0.744 pooled / 0.730 mean (cache hit — no re-extract) |

Sits above raw-200f (0.73) but below fixed-32f pooled_attentive (0.81): fixed-32f-
spanning-the-event is a strong *normalized* view, while dynamic preserves real
temporal scale/rate — on this corpus that trade didn't beat the fixed view at 25 fps
(a `sample_fps`/`window` sweep is the obvious next knob). Both probe kinds share one
extraction (switching probe is a cache hit).

```bash
bash scripts/21_eval_stutter_binary_dynamic.sh                    # seq_attentive, 25 fps
bash scripts/21_eval_stutter_binary_dynamic.sh --probe seq_lstm   # reuses the cache
bash scripts/21_eval_stutter_binary_dynamic.sh --sample-fps native
```

### 9.1 Dynamic × full spatial grid (`seq_attentive_lstm`) — and its OOM budget

Everything above **mean-pools S′ at extraction**, which is exactly the reduction §12
phase 2c showed destroys the rt-MRI fine-tune's gain (fixed-32f: tssl256 grid 0.817 vs
pooled 0.785, **+0.032**; vjepa_pt −0.023). `--probe seq_attentive_lstm`
(`--pool-mode none`) keeps the grid, so the probe gets **real temporal extent AND
spatial detail** — the one cell the two axes never crossed. Probe: a chunked
AttentivePooler over S′ per frame → `[B,L,D]` → packed bi-LSTM → masked mean → linear
(the dynamic-length counterpart of `eval_disfluency`'s `attentive_lstm`).

The cost is the ragged `[ΣL, S′=256, D=1024]` fp16 cache, which scales with `sample_fps`:

| `sample_fps` | K/clip (p50/max) | L tokens (p50/max) | ΣL | pooled cache | **full-grid cache** |
|---|---|---|---|---|---|
| 25 | 2 / 7 | 32 / 112 | 127,824 | 0.24 GiB | **62.4 GiB** |
| 50 | 3 / 13 | 48 / 208 | 225,680 | 0.43 GiB | **110.2 GiB** |
| **native (~99)** | 6 / 25 | 96 / 400 | 417,584 | 0.80 GiB | **203.9 GiB** |

At that size the cache fits neither host RAM nor page cache — the **same failure the
phoneme Phase-2 utterance probe hit** (149 GiB spatial cache OOM'ing a 64 GB cgroup; see
`RESULTS_phonepred.md` *Loss landscape (Phase 2)*). The full-grid path therefore reuses
its four fixes verbatim; **all four are load-bearing, and only #2 is new work** — the
other three are ports:

1. **`madvise(MADV_DONTNEED)` on the write-side memmap.** Written pages stay resident and
   are charged to the job's cgroup, so extraction OOMs long before 204 GiB lands.
   Periodic `flush()` + advise-away keeps RSS flat (extraction never reads back).
   `fadvise()` on a separate fd does **not** work here — the pages are still mapped.
2. **Padded-token batching** (`_TokenBudget`, `probe.max_tokens` = 1024). Budget
   `len(batch) × max(L)`, **not** `sum(L)`: `_collate_ragged` pads to the batch max, and
   budgeting `sum()` is precisely what packed short clips beside a 400-token one and blew
   up the phoneme run. A fixed 64-clip batch would be ~25× more memory at L=400 than at
   L=16 — the budget keeps peak flat across that spread.
3. **Eval loaders at `probe.eval_workers` (2) × `prefetch_factor=1`** — host RAM ≈
   workers × prefetch × padded-batch bytes; the Jul-15 phoneme crash was 4×2 stacking.
4. **fp16 dataset items** — the old `_RaggedDS` upcast to f32 in the worker, doubling
   every collate and pinned transfer. Irrelevant at 200 KB/clip, 2×200 MiB/clip here.
   The probe upcasts on the GPU instead.

Plus, VRAM-side, `probe.chunk` (32) + `probe.checkpoint` gradient-checkpoint the spatial
pool so peak activations are `O(B·chunk·S′)` not `O(B·Lmax·S′)`.

**Cache-hash safety.** `_tag` hashes `mode: dyn_fullgrid` vs `dyn_spatialpool`, so the
full-grid run gets its own directory and **every existing pooled cache/result is
untouched** — the same discipline as the phoneme `tssl256comb215` tag split (the hash
keys on geometry/manifest, *not* the ckpt epoch, so a shared tag would silently cache-hit
stale features).

```bash
FPS=25 SEEDS=0 sbatch dev_artiJEPA/scripts/27_stutter_binary_dyn_fullgrid.sbatch      # 62 GiB cache
FPS=50 SEEDS=0 sbatch dev_artiJEPA/scripts/27_stutter_binary_dyn_fullgrid.sbatch      # 110 GiB
# per-fold split (needed for native — a single all-folds job cannot finish 7 folds in one wall):
FPS=native SEEDS=0 FOLDS=PWS8 sbatch dev_artiJEPA/scripts/27_stutter_binary_dyn_fullgrid.sbatch
```
Config `configs/eval_stutter_binary_dyn_fullgrid.yaml` (encoder = combined T-SSL
**ckpt_215**, tag `tssl256comb215`, matching the phoneme Phase-2 arm) on 1× V100-32GB.

**Results (seed 0):** 25 fps **0.791**, 50 fps **0.766** — full table + reading in
`RESULTS_stutter.md` *(Dynamic-length × full spatial grid)*. **native dropped** (see §12
phase 2d). **RAM/wall gotcha:** `--mem` is set by loader buffers, NOT the cache — but a job
whose working set (≈cache size) exceeds RAM thrashes anyway (cyclic random scan, LRU evicts
each page just before reuse), so epoch cost is set by cache *residency*: 25 fps (62 GiB,
~fits) ran 1.9 min/epoch, native (204 GiB) 14 min/epoch, and raising `--mem` 64→180 G did
not move it. The fix for a big cache is **per-fold parallelism** (`FOLDS=<spk>`, 7 jobs
~5 h each), not more RAM. Per-fold splits are bit-identical to the all-folds run (the LOSO
loop consumes `rng` for every speaker; only training is skipped) and each writes its own
`…_loso_<spk>_s0.json` — merge the 7 for the pooled metric.

---

## 10. Critical changes & decision log (binary task, 2026-07-11 → 07-12)

The non-obvious decisions and gotchas behind the binary pipeline — read before
changing it.

1. **Separate eval, not `eval_disfluency --task binary`.** `eval_disfluency` decodes
   with **decord**, which reads these `pal8`/`rawvideo` `.avi` as all-black frames
   (§ decoder-bug warning at top). Any feature it makes on this corpus is invalid, so
   the binary path uses `eval_stutter_binary.py` on the **OpenCV** loader
   (`stutter_binary.py`, §7) with duration-matched fluent negatives.

2. **Encoder loaded at the checkpoint's geometry (256px / 32f / tubelet 2), like
   `examples/demo.ipynb` — not the loader's 200-frame default.** 32f → 4,096 tokens;
   200f → 25,600 tokens is both far OOD for the 32f-pretrained RoPE ViT and
   ~memory-prohibitive. `frames_per_clip` is a config/`--frames` knob if you re-run.

3. **Artifact tree moved `/data2/hongn/artijepa` → `/data1/hongn/arti-jepa`
   (2026-07-11).** All cache/outputs now write to `/data1`. `grayscale_stats_combined
   .json` (and `manifest_combined.csv`) were **not** preserved, so global channel-norm
   falls back to **mean=0/std=1** — where `compute_stats.py` says these values land
   anyway (they are computed on already per-clip z-scored clips), a negligible affine
   offset. The eval prints which path it took; restore the json to override.

4. **Resource caps: 1 GPU + ≤8 CPU cores.** Launcher exports `CUDA_VISIBLE_DEVICES=0`
   and BLAS thread caps; the loader pins each of `num_workers` decode workers to 1
   thread (+`cpu_threads` main) so total ≈ `num_workers + cpu_threads` (default 6+2).

5. **New `attentive_lstm` probe (§8).** Attn-pool the S′ spatial tokens **per frame**
   (one shared pooler) → `[T′,D]` sequence → **bi-LSTM over time** → linear. Reuses
   the same feature cache as `attentive` (`pool_spatial` kept False for both). Added
   to the shared `SegmentProbe`, so the disfluency-type eval gets it too.
   - **Honest VRAM finding:** at equal frames it is **not** cheaper than flat
     attentive (both project all B·T′·S′ tokens). The win is **chunked +
     gradient-checkpointed spatial pooling** (`--lstm-chunk`, `--lstm-checkpoint`):
     ≈½ peak VRAM at 200f (11→5 GiB @ B16), less with smaller chunks. Its other value
     is the temporal inductive bias.

6. **200f feasibility (measured).** GPU is **not** the limit (extraction ~3 GiB;
   probe 5–22 GiB by batch/chunk — fits one card). The limit is the **feature cache =
   ~190 GiB** at 200f (full-grid probes): extraction RSS climbs toward that (reclaimable
   mmap-write pages), and probe training wants ~190 GiB in **page cache** to avoid
   random-read thrash. Fits this 251 GB box only barely (and it's shared). **The fix is
   `pooled_attentive`** (`pool_mode=spatial`): mean-pool spatial at extraction →
   `[N,T′,D]` cache = 123 MB (32f, measured) / ~0.8 GiB (200f), which sidesteps the
   whole RAM problem; alternatively chunk-encode 32f windows.

7. **Results (frozen tssl-256, LOSO, seed 0, 32f).** attentive **0.828** pooled /
   0.816 mean macro-F1; attentive_lstm 0.813 / 0.816 — a **tie** (attentive_lstm is
   about VRAM scaling at high frames, not 32f accuracy); **pooled_attentive 0.811 /
   0.808** — ~1.5 pts below the full grid but at a **256× smaller cache** (123 MB) and
   ~0.2 GiB probe VRAM, so it's the go-to when the grid won't fit. Curves + history via
   `plot_stutter_binary.py` (GPU-preloads the cache for speed).

8. **200f is out-of-distribution — negative result (2026-07-12).** Ran
   `pooled_attentive --frames 200` to test the high-frame path end-to-end. The
   *engineering* worked perfectly: cache 0.76 GiB, one GPU at 12 GiB, extraction 52 min
   — the 190 GiB-cache problem is gone. But macro-F1 **fell to 0.729 pooled / 0.685
   mean** (from 0.811 / 0.808 at 32f). The 32f-pretrained RoPE encoder never saw
   200-frame sequences, so its features degrade — **more frames ≠ better here**. The
   only correct way to add temporal context on a frozen 32f encoder is to encode
   in-distribution 32f windows and concatenate their tokens; do not feed 200 raw frames.

---

## 11. Binary results at a glance (frozen tssl-256, LOSO, seed 0)

| setup | probe | frames/fps | pooled macro-F1 | cache | note |
|---|---|---|---|---|---|
| fixed   | attentive        | 32f    | **0.828** | 31 GiB | best; full T'·S' grid |
| fixed   | attentive_lstm   | 32f    | 0.813 | 31 GiB | temporal LSTM; VRAM-bounded via chunk+ckpt |
| fixed   | pooled_attentive | 32f    | 0.811 | **123 MB** | mean-spatial → attend time; tiny cache |
| fixed   | pooled_attentive | 200f   | 0.729 | 763 MB | OOD (encoder pretrained at 32f) |
| dynamic | seq_attentive    | 25 fps | 0.767 | 250 MB | variable-length, in-distribution windows |
| dynamic | seq_lstm         | 25 fps | 0.744 | 250 MB | variable-length |

Takeaways: full-grid `attentive` leads; `pooled_attentive` nearly matches it at 256×
less cache; raw-200f is OOD; the dynamic path is in-distribution but 25 fps trails
fixed-32f. All single-encoder, single-fps, single-seed — the scale-up below (§12) turns
this into a proper **cross-encoder benchmark → `RESULTS_stutter.md`** (6 encoders, done).
The full-grid `attentive` vs `pooled_attentive` gap here (0.828 vs 0.811) is exactly the
spatial-detail question §12 phase 2c now tests across encoders + seeds.

> **Read these numbers against §14, not against chance.** A no-video silence-fraction
> baseline reaches ~0.74 macro-F1 on this task; the binary score is largely a
> silence/duration cue (disfluent windows are 52% silent — mostly blocks), so ~0.83 does
> not by itself demonstrate learned articulatory dynamics. See §14 for the confound
> analysis and the honest baseline.

---

## 12. Scale-up plan — cross-model × fps × probe benchmark

Turn the single-encoder probe into a systematic, matched-protocol benchmark.

> **Status (2026-07-13).** **Phase 1 (infra) + Phase 2 (encoder sweep) DONE.** The
> multi-encoder loader is ported into `eval_stutter_binary.py` (encoder-type switch:
> `vjepa` T-SSL + FAIR-pretrained, `videomae` HF + local rt-MRI repo ckpt,
> `image_baseline` timm), driven by `scripts/22_stutter_binary_sweep.sh` and aggregated
> by `artijepa/agg_stutter_binary.py` → **`RESULTS_stutter.md`** (the planned deliverable;
> named `RESULTS_stutter.md`, not `_binary`). Six-encoder `pooled_attentive`/LOSO/seed-0
> result (pooled macro-F1): **vjepa_pt 0.822 > tssl256 0.808 ≈ videomae_tssl 0.806 ≈
> videomae_pt 0.801 ≫ vitl 0.533 / dinov2 0.528**. Headline findings: (i) **video
> encoders dominate per-frame 2-D image baselines** (~0.80 vs ~0.53, κ≈0.07) — disfluency
> is a temporal signal; (ii) **FAIR-pretrained V-JEPA2 beats the rt-MRI fine-tune**.
> tssl256 0.808 reproduces §11's published 0.811.
>
> **3-seed CIs DONE (2026-07-13, `scripts/run_3seed_pooled.sh`).** vjepa_pt
> **0.812 ± 0.007** vs tssl256 **0.785 ± 0.017** (pooled macro-F1, seeds 0/1/2) — **Δ
> +0.027, vjepa_pt wins all three seeds and is ~2.4× more stable**, so the
> pretrained>fine-tune result is *robust and larger than seed-0 implied*. Per-speaker,
> the seed-0 "tssl wins PWS3/6/7 (via fluent recall)" story **does not survive**: PWS3
> (±0.050) and PWS6 flip to vjepa on the 3-seed mean (tssl won only at seed 0); only the
> tiny/noisy PWS7 fold stays a marginal tssl win (+0.019, not unanimous). The *robust*
> signal is vjepa_pt winning the **hard** speakers PWS4 (Δ−0.072) and PWS8 (Δ−0.076)
> **every seed** with tight CIs. The rt-MRI fine-tune also raised probe-seed variance
> (±0.017 vs ±0.007) — a less stable frozen representation. Full CI tables in
> `RESULTS_stutter.md`.
>
> **DONE 2026-07-15 (SLURM).** **Full-grid `attentive` @ 32f** (no spatial pooling), both
> encoders, 3 seeds (`scripts/23_stutter_binary_attentive.sbatch`) — **hypothesis
> CONFIRMED, with a twist.** tssl256 **0.817 ± 0.009** vs its pooled 0.785 (**+0.032**);
> vjepa_pt **0.789 ± 0.010** vs its pooled 0.812 (**−0.023**). A clean **dissociation**:
> keeping the spatial grid helps the rt-MRI fine-tuned encoder but hurts the pretrained
> one, and the **ranking flips** (pooled: vjepa > tssl; full-grid: tssl > vjepa). rt-MRI
> fine-tuning restructures the per-frame spatial tokens to carry articulatory-dynamics
> signal that mean-pooling destroys — but only a spatially-aware probe reads it out. Full
> tables in `RESULTS_stutter.md`. (Ran one job per seed from the isolated worktree
> `/scratch1/hongn/artijepa/wt_stutter_sweep`, branch `stutter-binary-sweep`, after the
> 3-seeds-serial packing overran the 8h wall — single-seed cache-hit jobs run ~25 min.)

**Held constant.** LOSO over 7 PWS · duration-matched fluent negatives (`seed 0`) ·
balanced CE · macro-F1 primary (+bal-acc, acc) · pooled-spatial cache to keep RAM
small · 3 seeds for the final headline cells.

**Axis A — Encoders (each at its native geometry/norm).**

| key | type | geom / norm | pooled_attn macro-F1 (s0) | status |
|---|---|---|---|---|
| `tssl256` (combined ckpt_100) | vjepa | 256/32f · zscore | 0.808 | ✓ done |
| `vjepa_pt` (FAIR pretrained) | vjepa | 256/32f · zscore | **0.822** | ✓ done |
| `videomae_pt` (Kinetics SSL) | videomae | 224/16f · minmax | 0.801 | ✓ done |
| `videomae_tssl` (rtMRI ckpt-214) | videomae | 224/16f · minmax | 0.806 | ✓ done |
| `vitl` (supervised, per-frame) | image_baseline `vitl` | per-frame · minmax | 0.533 | ✓ done |
| `dinov2` (per-frame) | image_baseline `dinov2` | per-frame · minmax | 0.528 | ✓ done |

Encoder id → checkpoint: `vjepa_pt` = `/scratch1/hongn/artijepa/checkpoints/vitl.pt`
(key `target_encoder`, `module.backbone.*` → `clean_backbone_key`); `videomae_tssl` =
`/scratch1/hongn/videomae_ct/runs/vitl_rtmri_combined_ct/checkpoint-214.pth` (repo→HF).
timm image baselines are cached under `$HF_HOME` (offline-ready).

**Axis B — Temporal sampling.** fixed {32f, 100f}; dynamic {native, 25, 50, 100 fps}.
(Image baselines are per-frame → the fps sets the frame rate feeding a temporal head;
VideoMAE is fixed 16f internally.)

**Axis C — Probes.** `attentive`, `pooled_attentive`, `attentive_lstm` (fixed);
`seq_attentive`, `seq_lstm` (dynamic); `mean`, `mlp` (cheap baselines).

**Axis D — Compute report (per run).** extraction wall-time · cache size · probe peak
VRAM · #probe params · tokens/clip — all already logged; aggregate into one table.

**Phasing.**
1. ✓ **Infra DONE** — multi-encoder loader ported into `eval_stutter_binary.py`
   (encoder-type switch + per-encoder geometry/norm), `scripts/22_stutter_binary_sweep.sh`
   runner, `artijepa/agg_stutter_binary.py` aggregator. (`_dynamic` not yet ported.)
2. ✓ **Encoder sweep DONE** at `pooled_attentive`/32f-equiv/seed-0 → cross-model table
   in `RESULTS_stutter.md` (see Status box above).
2b. ✓ **3-seed CIs DONE** (tssl256, vjepa_pt @ pooled_attentive) — `scripts/run_3seed_pooled.sh`;
   CIs in `RESULTS_stutter.md`. vjepa_pt 0.812±0.007 > tssl256 0.785±0.017 (robust, all seeds).
2c. ✓ **Full-grid `attentive` @ 32f DONE** (tssl256, vjepa_pt, 3 seeds) — the
   **spatial-detail test**: keep the [T′·S′,D] grid (no S′ pooling) so an
   AttentivePooler attends over space+time jointly. **Result: dissociation.** tssl256
   0.817±0.009 (**+0.032** over pooled) but vjepa_pt 0.789±0.010 (**−0.023**); the
   encoder ranking flips (pooled: vjepa>tssl; grid: tssl>vjepa). rt-MRI fine-tuning's
   gain lives in the spatial tokens that mean-pooling destroys. CIs + per-speaker in
   `RESULTS_stutter.md`. `scripts/23_stutter_binary_attentive.sbatch` (~31 GiB
   cache/encoder → SLURM; `SEEDS` env runs one seed/job to fit the 8h wall).
2d. ✓ **Dynamic × full-grid `seq_attentive_lstm` DONE** (tssl256comb215, seed 0, 25 & 50 fps;
   native **dropped**) — full table in `RESULTS_stutter.md` *(Dynamic-length × full spatial
   grid)*. Crossed the two axes that never met (2c's grid win was fixed-32f only; §9's dynamic
   path was spatially-pooled only). **Answers: (i) the spatial-grid win replicates on the
   dynamic path** — @25fps full-grid 0.791 vs dynamic-pooled `seq_attentive` 0.767 (**+0.024**,
   echoing 2c's +0.032); **(ii) higher FPS hurts** — 25→50 fps drops −0.025 (pooled 0.791→0.766),
   and full-grid dynamic still trails the normalized **fixed-32f grid 0.817**. Sampling at real
   rate doesn't beat spanning the event with a fixed 32f budget on this corpus. PWS8 (hardest
   fold) drives the 50fps drop (0.653→0.574). **native (~99 fps) dropped:** with 25 > 50
   monotone-down it would extend a declining trend, at ~14 min/epoch = ~49 h serial (past the
   24 h wall — a single job structurally cannot finish, since results write only after all 7
   folds); the 204 GiB cache `…_dyn_eeef2e500c/` stays on scratch and per-fold launch is wired
   (`FOLDS=<spk>` in `scripts/27_*.sbatch`, splits verified bit-identical) if the 3rd point is
   ever wanted. **Wall-time lesson:** epoch cost is I/O-bound on cache residency — 25fps (62 GiB,
   ~fits RAM) ran 1.9 min/epoch, native (204 GiB) 14 min/epoch; more `--mem` did **not** help
   (a cyclic random scan over a working set > cache thrashes under LRU regardless), so the fix
   is per-fold parallelism, not bigger RAM.
3. **FPS sweep** (dynamic, best encoder) {native, 25, 50, 100} → macro-F1-vs-rate curve.
4. **Probe sweep** (best encoder × best temporal) → probe table.
5. **Compute report** (Axis D) + an accuracy-vs-compute Pareto plot.
6. **3-seed** re-run of the remaining headline cells for CIs.

**Cost.** Extraction dominates; the seed-0 sweep was 6 encoders × ~3–66 min each
(DINOv2 @518px is the outlier). Pooled-spatial keeps every cache 61–122 MB; the
full-grid `attentive` path is ~31 GiB/encoder (hence SLURM for 2c).

**Deliverables.** the auto-aggregated **`RESULTS_stutter.md`** (done); still to add:
per-fold CIs, the full-grid-vs-pooled spatial-detail comparison, and plots (macro-F1 vs
fps per encoder; accuracy-vs-compute Pareto). **Infra gap CLOSED** (phase 1).

---

## 13. Disfluency-TYPE eval (`eval_stutter_type.py`) — block / rep / pro

Task 8c: not *is it disfluent* (§8, solved at ~0.81 macro-F1) but **which kind**.
Block (silent articulatory hold), repetition (repeated gesture cycles), and
prolongation (a held, sustained gesture) are all defined by articulator *dynamics*,
so this is the eval that actually stresses what a video encoder buys over a
per-frame image encoder.

**Label spaces** (`data.task` / `--task`, via `stutter.label_space`):

| task | classes | clips | note |
|---|---|---|---|
| `type3` (default) | block / rep / pro | **2007** | 96.9% of positives; globally near-balanced |
| `type4` | + `fluent` | 3838 | detection **and** typing in one head |
| `type5` | + `osci` / `other` | 2070 | tail classes n=42 / n=21 — macro-F1 gets noisy |

**Class counts under the §7 row builder** (`tiers=[disfluency]`, 0.20–8.0 s):

| Speaker | block | rep | pro | (fluent) |
|---|---|---|---|---|
| PWS3  | 18  | 381 | 166 | 382 |
| PWS4  | 89  | 28  | 124 | 240 |
| PWS5  | 139 | 65  | 56  | 266 |
| PWS6  | 292 | 11  | 192 | 465 |
| PWS7  | 83  | 5   | 18  | 107 |
| PWS8  | 42  | 168 | 14  | 246 |
| PWS10 | 30  | 36  | 50  | 125 |
| **Total** | **693** | **694** | **620** | **1831** |

Globally the three types are almost perfectly balanced (693/694/620) — but the
**per-speaker priors are extreme**: PWS7 has 5 reps, PWS6 has 11, PWS3 has 18 blocks.
Under LOSO that means some folds score a class on a handful of clips, so per-fold
macro-F1 is high-variance by construction and the **pooled** number (all held-out
clips concatenated) is the one to read. Folds missing a class entirely are logged;
`classification_metrics` averages only the classes present.

**Shared feature cache.** Type rows are a *subset* of the binary rows and their clip
windows are identical, so `eval_stutter_type` builds rows and extracts through the
binary eval's code with the binary label space — the cache tag is byte-identical.
`tag=tssl256_215` @ 256px/32f/full-grid already exists
(`feat_cache/stutter_binary/tssl256_215_b5da470386`, 3901×4096×1024 fp16, 31 GiB), so a
default run costs **zero extraction**. A startup guard re-derives the binary labels
from the rebuilt rows and aborts if they disagree with the cached ones (mis-alignment
would silently scramble the type labels). Changing any of `neg_per_pos`, `build_seed`,
`min_dur`, `max_dur`, `merge_gap`, `frames_per_clip`, `spatial_size`, `event_pad_s`
changes the hash and forces a re-extract.

**Probe checkpoints (new).** Unlike the binary eval, this one **saves the trained
probe**: one `fold_<speaker>.pt` per fold under
`meta.out/probes/<tag>_<probe>_<task>_<split>_s<seed>/`, holding the best-val
`state_dict` plus everything needed to rebuild it (`probe_kwargs`, `dim`, `t_steps`,
`classes`, val macro-F1, epoch, feature tag). Rebuild with
`SegmentProbe(dim, num_classes, **probe_kwargs).load_state_dict(state_dict)`.

Also added over the binary trainer: **per-epoch val logging** (macro-F1, kappa,
per-class recall) and optional **`probe.patience`** early stopping — the §9 dynamic
runs showed folds that peak by epoch ~3 and then sit at chance for 25 more epochs.

```
sbatch dev_artiJEPA/scripts/29_stutter_type3.sbatch          # type3, seeds 0 1 2
sbatch dev_artiJEPA/scripts/29_stutter_type3.sbatch type4    # + fluent class
```

### 13.1 Results — `type3`, frozen T-SSL ckpt_215, attentive_lstm, LOSO

Run 2026-07-23 (jobs 10509945 / 10516533 / 10516534, ~2 h per seed, cache hit as
designed — zero extraction). 2007 clips, 7 folds, seeds 0/1/2.

| | macro-F1 | balanced acc | accuracy | κ |
|---|---|---|---|---|
| **fold-mean** (mean ± sd over seeds) | **0.317 ± 0.013** | **0.423 ± 0.005** | 0.418 ± 0.026 | — |
| **pooled** (all held-out clips) | **0.381 ± 0.025** | 0.408 ± 0.024 | 0.411 ± 0.026 | **0.110 ± 0.038** |
| pooled, per seed (s0 / s1 / s2) | 0.400 / 0.353 / 0.389 | 0.430 / 0.382 / 0.411 | 0.435 / 0.384 / 0.414 | 0.145 / 0.069 / 0.116 |

**This is barely above chance.** Balanced accuracy 0.423 against a 3-class chance of
0.333; pooled κ = 0.11. The matched binary run on the **byte-identical cache**
(`stutter_binary_tssl256_215_b5da470386_attentive_loso_s{0,1}`, 2 seeds) scores
**0.830 ± 0.019** macro-F1 / bal-acc 0.831 / **κ = 0.660 ± 0.037** — same encoder,
geometry, rows and LOSO folds, differing only in head (`attentive` vs `attentive_lstm`)
and label space. So *whether* a segment is disfluent is decodable from these features
and *which kind* it is essentially is not — at least not by this probe.

**Pooled per-class** (mean ± sd over seeds; support is the fixed LOSO total):

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| block | 693 | 0.388 ± 0.013 | **0.722 ± 0.056** | 0.505 ± 0.024 |
| rep   | 694 | 0.407 ± 0.133 | **0.183 ± 0.054** | 0.252 ± 0.077 |
| pro   | 620 | 0.488 ± 0.019 | 0.318 ± 0.031 | 0.385 ± 0.024 |

**The head collapses onto `block`.** Row-normalised confusion, summed over the 3 seeds
(rows = true, columns = predicted):

| true \ pred | block | rep | pro |
|---|---|---|---|
| **block** | 72.2% | 14.6% | 13.2% |
| **rep**   | **65.0%** | 18.3% | 16.7% |
| **pro**   | **54.1%** | 14.0% | 31.8% |

Two thirds of repetitions and over half of prolongations are called blocks, despite
`class_weight: balanced`. Note precision is 0.39–0.49 on all three classes — well above
the 0.33 a uniform guesser would get — so there *is* signal; it is recall that the bias
destroys. The balancing weights are computed on the training pool, which does not fix a
held-out speaker whose own prior is inverted.

**Per fold** (val = best in-speaker val macro-F1, mean over seeds; test = held-out):

| fold | n | support (b/r/p) | val F1 | test macro-F1 | bal acc | κ |
|---|---|---|---|---|---|---|
| PWS10 | 116 | 30 / 36 / 50 | 0.773 | 0.414 ± 0.049 | 0.462 | 0.181 |
| PWS7  | 106 | 83 / 5 / 18  | 0.746 | 0.405 ± 0.037 | 0.564 | 0.163 |
| PWS4  | 241 | 89 / 28 / 124| 0.768 | 0.342 ± 0.017 | 0.373 | 0.096 |
| PWS5  | 260 | 139 / 65 / 56| 0.803 | 0.326 ± 0.020 | 0.377 | 0.089 |
| PWS6  | 495 | 292 / 11 / 192| 0.724 | 0.312 ± 0.038 | 0.436 | 0.067 |
| PWS3  | 565 | 18 / 381 / 166| 0.725 | 0.310 ± 0.029 | 0.414 | 0.114 |
| PWS8  | 224 | 42 / 168 / 14| 0.780 | **0.112 ± 0.001** | 0.335 | **0.006** |

Two things to read off this table:

1. **Val 0.72–0.80 vs. test 0.11–0.41 is a speaker-generalisation failure, not an
   optimisation failure.** The probe fits held-out *clips* of a training speaker well
   and transfers almost nothing to a held-out *speaker*. Whatever it keys on is
   speaker-specific articulator appearance rather than the type dynamics.
2. **PWS8 collapses completely** — 222 of 224 clips predicted `block` (confusion
   `[[42,0,0],[166,1,1],[14,0,0]]`), identical to ±0.001 across all three seeds, κ ≈ 0.
   PWS8 is 75% repetitions, the exact inversion of the training prior; the failure is
   deterministic, so it is the class prior and not seed noise.

The two best folds (PWS10, PWS7) are also the two smallest, consistent with §13's
warning that per-fold macro-F1 is high-variance by construction — read the pooled row.

**Not yet run:** `type4` (+ fluent) has never been launched; there is no `attentive`
(joint pooler) or image-encoder baseline for `type3`, so this does not yet say whether
the video encoder buys anything over a per-frame encoder *on typing*. Before spending
more compute on encoders, the class collapse is the thing to fix — per-fold prior
correction / logit adjustment at test time, and a per-speaker rather than pooled
class weighting.

---

## 14. Discussion — what the binary 0.83 actually measures

The binary probe scores 0.83 macro-F1 (§8, §11) while the same features type disfluencies
at chance (§13.1). That gap prompted the question: **is 0.83 a phonetic confound —
disfluent windows biased toward stop consonants (the classic stutter class), so the
encoder reads articulatory posture rather than disfluency dynamics?** We tested it on
the exact probe rows (`build_rows` seed 0, cache tag `b5da470386`) by aligning the
`phones` tier (ARPABET forced-alignment; 343/476 files carry it → 2523/3901 rows, uneven
per speaker) and measuring what each window is acoustically made of. Scripts:
`artijepa/analyze_binary_phoneme_confound.py`, `analyze_disfluency_composition.py`;
figures `eval/stutter_binary/phoneme_confound.png`,
`eval/stutter_type/disfluency_composition.png`.

**The confound is real, but it is silence, not consonant identity.** Phone-class
occupancy (fraction of window-time) of disfluent vs fluent windows:

| | vowel | stop | consonant (all) | **silence** |
|---|---|---|---|---|
| disfluent | 21% | 5% | 27% | **52%** |
| fluent | 43% | 11% | 41% | **16%** |

Disfluent windows are *less* consonant-heavy, not more — the specific hypothesis is not
supported. A no-video LOSO classifier attributes the leakage cleanly:

| no-video feature (LOSO macro-F1) | score |
|---|---|
| phone identity, silence removed | **0.515** (chance) |
| silence fraction alone (1 feature) | **0.743** |
| full phone+silence histogram | 0.767 |
| _video encoder, same rows/protocol_ | _0.830_ |

Once the silence axis is removed, phoneme identity carries nothing; a **one-feature
silence-fraction baseline reproduces 0.74 of the 0.83**, and the full ViT stack adds only
~0.09 on top.

**Why the silence, and why it is not an alignment artifact.** A disfluency event is not a
phoneme but a ~2.3 s stretch (vs 1.6 s fluent) of the attempted word's phones *plus* the
dysfluent element. Composition by type shows silence tracking the behavior, not the
aligner:

| type | n | avg dur | silence | consonant | vowel |
|---|---|---|---|---|---|
| block (silent hold) | 572 | 2.3 s | **54%** | 21% | 21% |
| rep (repeated attempts) | 307 | 2.2 s | 42% | 26% | 28% |
| pro (sustained sound) | 365 | 2.1 s | **25%** | 44% | 26% |
| fluent (negative) | 1245 | 1.6 s | 15% | 41% | 43% |

Blocks are silent articulatory holds (postured for the target, no airflow, 0.9–1.6 s);
repetitions add inter-attempt pauses. If the `<sil>` were the forced aligner failing on
non-canonical stuttered speech, prolongations — the most non-canonical, sustained speech —
would be the *most* silent; they are the **least** (25%), because a prolongation is
sustained sound. So the silence is genuine held/paused articulation.

**Reconciling with §4 (consonant-dominated onset prefixes).** No contradiction: §4 counts
the *annotator's target phoneme* — the sound the speaker is stuck on — which is
consonant-heavy, matching the stuttering literature. §14 measures the *acoustic
realization* of blocking on that target, which is silence (a held posture makes no
sound). Target = consonant; realization = silence. On rt-MRI the encoder most plausibly
reads the **frozen consonant-shaped posture during the silent hold** — which is a real
articulatory correlate of disfluency, not a nuisance to be removed.

**Implications.**
- **0.83 is a genuine correlate but a shallow one.** It is largely a silence/duration cue;
  it does *not* by itself demonstrate learned articulatory *dynamics*. The §8 headline
  ("video encoders 0.78–0.81, per-frame image encoders collapse to ~0.53") still holds —
  a per-frame encoder cannot see silence duration or a sustained hold either — but the
  achievable ceiling on this task is set by a very simple feature, and future claims
  should be made against a **silence-fraction / VAD baseline (~0.74)**, not against 0.50.
- **Prediction: prolongations are the hardest disfluency for the binary probe.** They sit
  at 25% silence, barely above fluent's 15%, so the silence shortcut mostly misses them.
  Testable directly by scoring binary recall per disfluency type.
- **This is why typing collapses (§13.1).** block/rep/pro are *all* disfluent, so the
  silence cue that carries binary detection cannot separate them; the probe falls back on
  the majority (`block`) and lands near chance.
- **The consonant intuition survives only inside prolongations** — the one type more
  consonant-heavy than fluent (44% vs 41%): what you sustain is usually a fricative/nasal.

**Caveats.** Phone coverage is 65% of rows and uneven per speaker (PWS3 26%, PWS5/PWS7
100%); the interview recordings lack a `phones` tier. The no-video classifier is a linear
model on coarse 8-class histograms — a *lower bound* on phonetic leakage, but the
silence-only vs identity-only split is the clean attribution.

**Two concrete follow-ups.** (1) Add a silence-fraction / VAD baseline row to
`RESULTS_stutter.md` as the honest floor the encoder must beat. (2) Rebuild negatives to
be **duration-and-silence-matched** (draw fluent windows with matched pause content) to
isolate whether the encoder sees anything beyond silence.

---

_Regenerate these statistics with the analysis over `artijepa/stutter.py`'s
`parse_textgrid` / `canonicalize` on `/data1/span_data/stuttering/`._
