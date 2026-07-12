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

_Regenerate these statistics with the analysis over `artijepa/stutter.py`'s
`parse_textgrid` / `canonicalize` on `/data1/span_data/stuttering/`._
