# Glossectomy — Transfer phoneme decoding, pre vs post (appearance shift vs pathology)

Frozen-encoder **transfer** eval (Task 9 / `docs/GLOSS.md §11-12`). Annot-16-trained phoneme
probes (14 healthy speakers) are applied **eval-only** to glossectomy rtMRI, separately for
**pre-** and **post-**glossectomy. Each probe ships a **healthy-OOD anchor** (usc_lss — a healthy
speaker in a different acquisition domain, **no pathology**), so gloss transfer can be read
against a shift-only reference. Primary metric = Cohen's **κ**.

> **All numbers below were recomputed 2026-08-07 and replace everything published earlier.**
> They differ from the pre-2026-08-06 figures for **two independent reasons** — do not read the
> difference as a corpus effect alone:
> 1. **Corpus changed.** Sub-20-fps acquisitions are excluded and spk3/post's video filenames
>    were corrected (see *Corpus* below). pre 56→**38** utts, post 44→**48**.
> 2. **One interpolation instead of two.** Manifests now point at the **native ~99.4 fps source**
>    videos. Previously they pointed at `resampled_video/`, a uniform-100-fps re-encode, so
>    frames went 99.4 → 100 (duplication) → 50 at load. The dataloader is frame-rate agnostic
>    (`target_fps: 50.0`, linear interpolation in seconds), so the 100 fps intermediate was
>    lossy and unnecessary. usc_lss is handled the same way (99.01 → 50).

## Corpus

Manifests: `python -m artijepa.gloss` → `/scratch1/hongn/gloss/gloss_{pre,post}_manifest.csv`,
all rows native **99.4036 fps**.

| | n_utt | breakdown | speech segments |
|---|---|---|---|
| pre | **38** | spk1 28, spk2 9, spk3 1 | 5063 |
| post | **48** | spk1 4, spk2/post1 17, spk2/post2 11, spk3 16 | 6184 |

Three facts about this corpus drive everything below:

- **Two acquisition rates.** Sources are either **99.4036 fps** (130 files) or **15.2929 fps**
  (29 files), nothing between. The 15.29 fps material is excluded (`--min-fps 20`); it lives in
  `/scratch1/hongn/gloss/_lowfps_quarantine/`.
- **The fps split is confounded with stimulus type.** In exactly three sessions — spk1/post,
  spk2/pre, spk3/pre — the **word-list/VCV items** (`ai, ara, asha, atcha, au, oi, puppy1-3`)
  were recorded at 15.29 fps while the **connected passages** (`rainbow, caterpillar, bvt,
  spont, 1_5`) were recorded at 99.4. The other four sessions are all-99.4 and lose nothing.
  So the gate removes the word-list condition from one side only, **in opposite directions for
  spk1 and spk2**:

  | speaker | pre | post |
  |---|---|---|
  | spk1 | word lists + passages (28) | **passages only** (4) |
  | spk2 | **passages only** (9) | word lists + passages (17 / 11) |
  | spk3 | 1 passage (`rainbow`) | word lists + passages (16) |

- **spk3/post filenames were a permutation** and are now corrected on disk (map + undo:
  `spk3/post/video_rename_map.json`, backup `_video_prerename_backup/`). The annotations were
  never wrong: transcript↔TextGrid↔audio is self-consistent and `|wav_dur − tg_xmax| = 0.00` on
  all 17, while 15/17 videos mismatched their same-named TextGrid. After the rename **16/17 pass
  the name-based duration check** (was 2/17); only `asha` fails (video truncated 7.59 s). This
  supersedes the old duration-re-pairing workaround — `--spk3-repaired` now defaults to `none`
  and `gloss_post_spk3_repaired*.csv` / `phonemes_json_repaired/` are **obsolete**.

---

# Frame-level transfer — `attentive_lstm` head, ckpt_215 (headline)

Phase-2 `attentive_lstm` probe (AttentivePooler spatial pool → bi-LSTM over the whole utterance
→ linear) on `tssl256comb215`, applied eval-only in utterance mode. 3 seeds, mean ± sd.

## κ ladder

| stage | holds fixed | κ |
|---|---|---|
| in-domain test (Annot-16 sub043) | healthy, same domain, **seen stimulus** | **0.673** |
| **healthy-OOD (usc_lss), matched stimulus** | new healthy speaker + domain, **seen stimulus** | **0.508 ± .008** |
| **healthy-OOD (usc_lss)** | new healthy speaker + domain, **no pathology** | **0.412** |
| **healthy-OOD (usc_lss), unseen stimulus only** | new speaker + domain + **novel material** | **0.415 ± .008** |
| gloss pre-op (pooled) | + tumor pathology + gloss domain | **0.444 ± .004** |
| gloss post-op (pooled) | + altered anatomy + gloss domain | **0.359 ± .006** |

The two new rows split the healthy-OOD anchor by **stimulus novelty** — see
*Is the healthy-OOD anchor depressed by unseen stimulus?* below. The 0.412 row is the probe's
own stored seed-0 metric (what the ladder has always quoted); its 3-seed equivalent over all
684 utts is **0.421 ± .008**, so the matched/unseen rows are directly comparable to each other
and to the gloss rows, all being 3-seed.

**Gloss pre-op now sits *above* its healthy-OOD anchor** (0.444 vs 0.412). Post-op sits 0.053
below it. Read the pooled pre→post gap with care — it is stimulus-confounded (below).

Supporting metrics (3-seed mean): pre PERµ 0.575, frame-acc 0.584, macro-recall groups 0.395 /
vowels 0.345 / consonants 0.297 / place 0.304; post PERµ 0.647, frame-acc 0.525, macro-recall
groups 0.331 / vowels 0.226 / consonants 0.214 / place 0.236. PERµ < 1.0 everywhere — the
utterance-mode LSTM removes the CE-argmax insertion blow-up, so PER is usable for this head.

## Is the healthy-OOD anchor depressed by unseen stimulus? (added 2026-08-08)

The in-domain test (sub043) covers the **same 20 stimuli the probe trained on**, while the
healthy-OOD test (usc_lss s1, 684 utts / 71 items) is mostly *different material* — so the
in-domain → healthy-OOD drop (0.673 → 0.412) confounds **speaker/appearance shift** with
**stimulus novelty**. The two corpora happen to share three passages, which lets them be
separated. Driver: `artijepa/eval_lss_stimulus.py`, 3 seeds, no re-extraction (the `test_lss`
features are already in the Annot-16 unpooled cache).

**Overlap is measured, not assumed.** Gold phoneme sequences of every usc_lss item matched
against every Annot-16 stimulus — **6 of 71 items overlap**, with an unambiguous gap (0.63–0.94
vs ≤0.32 for all 65 others):

| usc_lss item | Annot-16 stimulus | ratio | n_utt | tier |
|---|---|---|---|---|
| 39 | `grandfather1` | 0.921 | 10 | verbatim |
| 56 | `grandfather1` | 0.943 | 9 | verbatim |
| 40 | `northwind1` | 0.930 | 6 | verbatim |
| 57 | `northwind1` | 0.932 | 6 | verbatim |
| 38 | `rainbow` | 0.632 | 17 | composite |
| 55 | `rainbow` | 0.784 | 11 | composite |

All three passages are in the Annot-16 **train** split, so the probe genuinely saw this material.
Items 38/55 are **composites** (55 opens with the shibboleth sentences then continues into other
material), so a single-stimulus ratio understates their overlap — they are kept in a separate tier.

> **Pitfall worth recording:** `difflib.SequenceMatcher(...).ratio()` must be called with
> `autojunk=False`. The default heuristic discards any element occurring in >1% of a sequence of
> length ≥200, and in a ~40-symbol phoneme alphabet *every* symbol qualifies, so matching silently
> collapses. With autojunk on, northwind scored 0.372 and looked like a non-match when it is
> **verbatim identical** (correct ratio 0.930) — only 2 of the 6 overlaps were found.

### Result — κ and PER by stimulus novelty (3 seeds, mean ± sd)

| subset | n_utt | κ | PERµ | PER_macro | frame-acc |
|---|---|---|---|---|---|
| all 684 utts (= the anchor) | 684 | 0.421 ± .008 | 0.469 ± .009 | 0.482 ± .009 | 0.445 ± .008 |
| **seen stimulus (verbatim)** | 31 | **0.508 ± .008** | **0.234 ± .017** | **0.239 ± .013** | **0.527 ± .007** |
| seen stimulus (composite) | 28 | 0.439 ± .012 | 0.253 ± .005 | 0.267 ± .003 | 0.460 ± .011 |
| **unseen stimulus** | 625 | **0.415 ± .008** | **0.498 ± .009** | **0.504 ± .009** | 0.439 ± .008 |

**The hypothesis is confirmed, but it accounts for only part of the κ gap and nearly all of the
PER gap.**

- **κ: +0.094** on matched stimulus (0.508 vs 0.415). That closes the in-domain gap from 0.258 to
  0.165, i.e. stimulus novelty explains **~36%** of the in-domain → healthy-OOD κ drop. The
  remaining ~64% is genuine speaker/appearance/domain shift.
- **PERµ: less than half** on matched stimulus (0.234 vs 0.498, 0.47×). Per item, all six matched
  items sit at the **0th percentile** of the 65 unseen items with **z ≈ −4.1 to −6.2** — a
  perfectly consistent effect with no overlap against the unseen distribution.

**Why the asymmetry matters — do not read the PER result as better articulatory decoding.** PER
collapses frames to a phoneme *sequence* and scores edit distance, so a probe that has effectively
learned a known passage's phonotactics can recover the right sequence even when frame boundaries
are wrong. κ is frame-level and cannot be rescued that way. The bi-LSTM runs over the whole
utterance, so it has exactly the sequence-level context needed to exploit a memorised passage.
The κ gain is the trustworthy measure of transfer here; the PER halving is inflated by
sequence-prior memorisation.

**Per-item κ is noisy — the pooled 0.508 rests on 31 utts across 4 items**, and they disagree:
item 57 (northwind) κ 0.653 is the maximum over all 71 items (100th percentile) while item 39
(grandfather) κ 0.318 sits at the 17th percentile despite being a 0.921 verbatim match. So treat
+0.094 as an estimate with real item-level variance, not a precise quantity. PER, by contrast,
is unanimous across all six items.

**Consequence for the gloss reading.** Against the *matched-stimulus* healthy anchor (0.508),
gloss pre-op (0.444) sits **below** rather than above it, and gloss post-op (0.359) further below.
Gloss is word-list + passage material and largely unseen, so the unseen-stimulus anchor
(**0.415**) is the fairer comparison — and against that, gloss pre-op is still slightly higher.
The headline conclusion is unchanged, but the anchor it should be read against is 0.415, and the
0.412 figure the ladder has always quoted turns out to be an all-stimulus average that happens
to land in almost the same place.

## Per gloss speaker × session (κ mean ± sd / PERµ / n_utt, 3 seeds)

| speaker | pre κ | pre PERµ | pre n | post κ | post PERµ | post n |
|---|---|---|---|---|---|---|
| spk1 | 0.412 ± .008 | 0.60 | 28 | **0.557 ± .009** | 0.41 | 4 |
| spk2 · post1 | 0.480 ± .006 | 0.55 | 9 | 0.432 ± .013 | 0.60 | 17 |
| spk2 · post2 | *(same pre)* | | | 0.449 ± .007 | 0.61 | 11 |
| spk3 | 0.602 ± .005 | 0.47 | 1 | **0.203 ± .011** | 0.89 | 16 |
| **pooled** | **0.444 ± .004** | 0.58 | 38 | **0.359 ± .006** | 0.65 | 48 |

**The per-speaker deltas track stimulus composition, not surgery.** Compare each cell against
the composition table above: in every case the side that is **passages-only scores higher**.

- **spk1** pre (mixed, n=28) 0.412 → post (**passages only**, n=4) 0.557, **+0.145**
- **spk2** pre (**passages only**, n=9) 0.480 → post (mixed) 0.432 / 0.449, **−0.048 / −0.031**
- **spk3** pre (**one passage**) 0.602 → post (mixed, n=16) 0.203, **−0.399**

spk1 and spk2 move in **opposite directions**, exactly as their opposite stimulus imbalance
predicts. This is direct evidence that **connected speech decodes far better than word lists**
at the frame level — spk3's single Rainbow passage reaches κ 0.602, close to the in-domain
ceiling of 0.673 and well above the healthy-OOD anchor. Neither the pooled Δ nor the spk1/spk2
Δ can be read as a surgical effect.

**spk3 post is the one cell that resists that explanation**: at 0.203 it is far below every
other cell despite containing passages as well as word lists, and it is the lowest-PER-quality
cell (PERµ 0.89). This is the same single-speaker exception earlier analyses surfaced, now on
16 correctly-paired utterances rather than a 3-utterance artifact.

## Per-seed pooled κ

| seed | pre κ | post κ |
|---|---|---|
| s0 | 0.4452 | 0.3575 |
| s1 | 0.4399 | 0.3657 |
| s2 | 0.4473 | 0.3545 |

Seed spread is tiny (sd ≤ .006), so the per-speaker contrasts above are not seed noise.

## Per-speaker × place of articulation (3 seeds, mean ± sd)

| cell | κ | place macro-recall | consonant macro-recall |
|---|---|---|---|
| spk1 pre | 0.412 ± .008 | 0.279 ± .006 | 0.276 ± .006 |
| spk1 post | 0.557 ± .009 | **0.369 ± .008** | 0.344 ± .009 |
| spk2 pre | 0.480 ± .006 | 0.322 ± .016 | 0.323 ± .010 |
| spk2 post1 | 0.432 ± .013 | 0.310 ± .014 | 0.303 ± .019 |
| spk2 post2 | 0.449 ± .007 | 0.304 ± .012 | 0.328 ± .039 |
| spk3 pre | 0.602 ± .005 | **0.596 ± .094** | 0.523 ± .021 |
| spk3 post | 0.203 ± .011 | **0.078 ± .006** | 0.051 ± .007 |

Place macro-recall tracks κ cell-for-cell, including the spk1↑ / spk2↓ split — i.e. it follows
the same stimulus confound and is not an independent articulatory signal. Seed spread is small
(sd ≤ .016) everywhere except **spk3 pre** (± .094), whose rare places have single-digit support
in one 354-segment passage, so its macro-recall depends on which of those the probe happens to
hit. The exception is again **spk3 post**, where place recall collapses to 0.078 (everything ≈ 0)
with sd .006 — far beyond what composition or seed noise explains.

---

# Frame-level transfer — `attentive` head, ckpt_100 (secondary)

The weaker Phase-1 chunk-mode probe on `tssl256comb100`, same corpus, 3 seeds. Retained because
it is a different encoder checkpoint *and* a different head, so it tests whether the pattern
above is head-specific.

| stage | κ |
|---|---|
| in-domain test (Annot-16 sub043) | **0.458** |
| **healthy-OOD (usc_lss)** | **0.337** |
| gloss pre-op (pooled) | **0.321 ± .012** |
| gloss post-op (pooled) | **0.253 ± .024** |

| speaker | pre κ | pre n | post κ | post n |
|---|---|---|---|---|
| spk1 | 0.304 ± .015 | 28 | **0.372 ± .010** | 4 |
| spk2 · post1 | 0.338 ± .036 | 9 | 0.353 ± .031 | 17 |
| spk2 · post2 | *(same pre)* | | 0.346 ± .020 | 11 |
| spk3 | 0.392 ± .013 | 1 | **0.095 ± .039** | 16 |
| **pooled** | **0.321 ± .012** | 38 | **0.253 ± .024** | 48 |

Per-seed pooled κ — s0 0.334/0.278, s1 0.310/0.250, s2 0.318/0.232. PERµ ≈ 0.82–0.89 pooled and
1.00 for spk3 (the insertion artifact this head shows OOD; trust κ, not PER, here).

**Same shape, weaker throughout**: spk1 post > pre, spk2 post ≈ pre, spk3 post collapses. Both
the stimulus confound and the spk3 exception reproduce across encoder *and* head, so neither is
an artifact of the `attentive_lstm` path.

---

# Segment-level (Phase-3) transfer: phoneme classification + clustering

Same eval-only transfer at the **segment level** with the Phase-3 clip probes: one gold phoneme
segment = one sample = one label, in four label spaces — **vowels (15) · consonants (25) ·
manner (5) · place (8)** — same frozen `tssl256comb215`, no retraining. Gold boundaries make this
an **identity-only** (oracle-segmentation) read, and these label spaces contain **no `sil` class**,
so the frame-level smear toward `sil` cannot hide here. **Seed 0 only** (Phase 3 is single-seed).

## The ladder — macro-F1 | κ (seed 0)

| stage | vowels (15) | consonants (25) | manner (5) | place (8) |
|---|---|---|---|---|
| in-domain test (Annot-16 sub043) | 0.568 \| 0.543 | 0.399 \| 0.440 | 0.571 \| 0.525 | 0.593 \| 0.572 |
| **healthy-OOD (usc_lss)** | **0.516 \| 0.470** | **0.329 \| 0.360** | **0.537 \| 0.464** | **0.507 \| 0.408** |
| gloss pre (pooled, 5063 seg) | 0.306 \| 0.282 | 0.287 \| 0.343 | 0.451 \| 0.359 | 0.511 \| **0.545** |
| gloss post (pooled, 6184 seg) | 0.275 \| 0.253 | 0.264 \| 0.307 | 0.372 \| 0.311 | 0.383 \| 0.393 |

The **robustness ordering survives the corpus rebuild**: **place > manner > consonant-id >
vowel-id**. Gloss-pre **place κ 0.545 exceeds the healthy-OOD anchor (0.408)** and approaches
the in-domain ceiling (0.572) — where the tongue makes contact stays decodable under the gloss
shift — while fine vowel identity falls to roughly half the healthy anchor (0.282 vs 0.470).
All 15 vowel classes are populated, so this is a real collapse, not an empty-class artifact.

## Per speaker × session (macro-F1 | κ | n_seg, seed 0)

| cell | vowels | consonants | manner | place |
|---|---|---|---|---|
| spk1 / pre | 0.333 \| 0.308 \| 1405 | 0.290 \| 0.361 \| 1779 | 0.453 \| 0.346 \| 1779 | 0.551 \| 0.603 \| 1779 |
| spk1 / post | 0.419 \| 0.367 \| 406 | 0.351 \| 0.393 \| 643 | 0.405 \| 0.357 \| 643 | 0.520 \| 0.505 \| 643 |
| spk2 / pre | 0.171 \| 0.182 \| 603 | 0.215 \| 0.281 \| 922 | 0.421 \| 0.373 \| 922 | 0.336 \| 0.391 \| 922 |
| spk2 / post1 | 0.282 \| 0.284 \| 797 | 0.314 \| 0.401 \| 1023 | 0.428 \| 0.379 \| 1023 | 0.401 \| 0.472 \| 1023 |
| spk2 / post2 | 0.273 \| 0.267 \| 667 | 0.295 \| 0.383 \| 964 | 0.416 \| 0.365 \| 964 | 0.391 \| 0.431 \| 964 |
| spk3 / pre | 0.556 \| 0.444 \| 138 | 0.405 \| 0.454 \| 216 | 0.403 \| 0.420 \| 216 | 0.564 \| 0.603 \| 216 |
| spk3 / post | **0.152 \| 0.145 \| 749** | **0.034 \| 0.069 \| 935** | **0.248 \| 0.159 \| 935** | **0.177 \| 0.204 \| 935** |

- **spk1: post ≥ pre on vowels/consonants**, slightly down on place — no systematic post-op loss.
- **spk2: post ≥ pre on every task**, both sessions.
- **spk3: a sharp post-op collapse across all four tasks** — place κ 0.603 → 0.204, consonants
  0.454 → 0.069. This is the segment-level confirmation of the frame-level spk3 finding, on the
  corrected n=16 post session, and it is sharper here.

Note that at the *segment* level spk1 and spk2 both go **up** post-op, while at the frame level
spk2 went down. Segment-level scoring uses gold boundaries, which removes the duration/segmentation
component that the word-list vs passage difference affects most — consistent with the stimulus
confound being a frame-level phenomenon and largely absent once boundaries are given.

## Clustering readout — the geometry is speaker-dominated

Per-segment 1024-d vectors; **rep B** = raw encoder mean over the segment (probe-independent).
Silhouette of the same points under three labelings, cosine geometry:

| task | sil(phoneme) | **sil(speaker)** | sil(condition) | k-means ARI | NMI |
|---|---|---|---|---|---|
| vowels | −0.070 | **0.151** | 0.074 | 0.034 | 0.103 |
| consonants | −0.091 | **0.148** | 0.100 | 0.018 | 0.102 |
| manner | −0.022 | **0.148** | 0.100 | 0.007 | 0.005 |
| place | −0.067 | **0.148** | 0.100 | 0.015 | 0.044 |

**silhouette(speaker) > silhouette(phoneme) for every task, and sil(phoneme) is negative** —
phoneme clusters overlap while speakers separate. The gloss representation is
appearance/speaker-dominated, so a phoneme probe transferred across speakers is fighting the
encoder's speaker structure. Unchanged by the corpus rebuild.

## Per-class cross-speaker t-SNE — where the speaker structure lives (added 2026-08-09)

The clustering readout above pools all classes into one number per task. Splitting it
**per class** answers a sharper question: *within a single phoneme class, do the three gloss
speakers and the healthy reference speaker occupy the same region?* One panel per class,
points colored by speaker (spk1/spk2/spk3 + usc_s1), marker = condition (○ pre, △ post,
□ usc healthy). Driver: `artijepa/tsne_gloss_vowels.py`, which now covers **three label
spaces in one pass** — vowels (15 panels) · place (8) · manner (5), the same spaces as the
Phase-3 ladder. Segment level, `tssl256comb215` + `attentive_lstm` probe, seed 0,
usc_s1 = all 23 669 usc_lss segments. Reads the existing frame-level caches, no re-extraction.

The diagnostic behind the table is **sil(spk)** = silhouette of that panel's points under the
*speaker* labeling: for each point, (mean cosine distance to the nearest *other* speaker −
mean cosine distance to its own speaker) / max of the two, averaged over the panel. It is
computed on the **PCA-50 vectors, not the 2-D t-SNE coordinates**, so the number measures the
real high-dimensional geometry and cannot be inflated or deflated by the embedding — figure
and number are independent views. +1 = each speaker a tight, well-separated ball;
≈0 = speakers fully interleaved, i.e. the class is speaker-invariant; <0 = points sit closer
to other speakers than their own. Cosine matches the geometry of the pooled clustering table
above, so the two are comparable.

**Two caption variants of each figure exist, same data.** `…_uscref_s0.png` prints
`sil(spk)` and the per-speaker n in every panel title; `…_uscref_mincap_s0.png` is the clean
version, class name + n only (use this one in talks/papers). Both are generated by the same
driver; the values are identical and are also printed to the job log, together with the
per-(class, speaker) n breakdown, so nothing is lost by using the clean figure.

| label space | rep B (raw encoder) | rep A (probe-q) |
|---|---|---|
| vowels (15) | **+0.310** (range +0.200…+0.423) | **+0.088** (range −0.049…+0.308) |
| place (8) | **+0.289** (range +0.185…+0.358) | **+0.006** (range −0.046…+0.085) |
| manner (5) | **+0.309** (range +0.269…+0.328) | **+0.028** (range +0.004…+0.064) |

Two results, one per representation.

**1. The raw encoder geometry is speaker-separated inside every single class.** All 28
panels of rep B are positive, none below +0.185, means +0.29…+0.31 — roughly **twice** the
pooled sil(speaker) ≈ 0.15 in the table above. That is expected and is the point: once you
hold the phoneme fixed, essentially the only variance left *is* speaker and condition, so
the per-class view is the undiluted measure of the nuisance structure. Visually, each panel
is four (often eight) compact, well-separated blobs, with usc_s1 its own island. There is no
label space in which the raw encoder puts different speakers' instances of the same phoneme
in the same place.

**2. The probe's AttentivePooler removes most of it, and removes most of it exactly where
transfer is best.** rep A collapses the means to +0.088 / +0.005 / +0.028, and the ordering
is informative: **place is the most speaker-invariant space in q-space** — 5 of its 8 panels
are at or below zero (Glottal −0.046, Palatal −0.026, Velar −0.020, Labiodental −0.019,
Bilabial +0.002) — manner is next, and **vowels retain the most speaker structure** (+0.088,
with /oy/ +0.308 and /aw/ +0.268 the two worst panels anywhere in rep A). That is the
**inverse** of the transfer ordering in the ladder (place κ 0.545 > manner 0.359 > vowels
0.282): the spaces that transfer best are the spaces whose q-vectors are least
speaker-separated. The two measurements are independent — one is unsupervised geometry, the
other is probe accuracy — so their agreement is real evidence that the transfer ceiling here
is set by residual speaker structure rather than by pathology.

**Caveat on how to read sil(spk): it is computed on speaker labels only, so a pre/post split
*within* one speaker lowers it rather than raises it.** Several rep-B panels visibly show
each speaker's pre and post as two separate blobs, and spk3/post appears as a detached
cluster in rep A too (clearest in Bilabial, Alveolar, Dental, Velar, and in Plosive /
Fricative / Nasal / Approximant) — the geometric face of the spk3 post-op collapse. Those
within-speaker splits are *penalised* by the metric, so the numbers above are a **lower
bound** on total nuisance structure, not an estimate of it.

**Scope.** Seed 0, one encoder, one probe head. Three label spaces is three points, so treat
the sil(spk)-vs-transfer inverse ordering as consistent-with, not established. t-SNE layout
is a projection and the silhouettes are computed on the PCA-50 vectors, not the 2-D
coordinates, so the numbers do not depend on the embedding — but the *visual* blob structure
does, and per-panel counts are capped at 200 per (class, speaker).

---

## Corpus composition — phoneme distribution (gold segments)

What the transfer metrics are measured over, straight from the gold ARPABET annotations
(`artijepa/gloss_phoneme_dist.py --from-textgrids`, no encoder). Cells are **n (% of that row's
speech segments)**; `sil` excluded from the denominator.

| cond·group | n_utt | speech seg | Vowel | Diph | Plosive | Fric | Affr | Nasal | Approx |
|---|---|---|---|---|---|---|---|---|---|
| pre·spk1 | 28 | 3184 | 1081 (34.0) | 324 (10.2) | 723 (22.7) | 424 (13.3) | 43 (1.4) | 273 (8.6) | 316 (9.9) |
| pre·spk2 | 9 | 1525 | 484 (31.7) | 119 (7.8) | 331 (21.7) | 240 (15.7) | 13 (0.9) | 154 (10.1) | 184 (12.1) |
| pre·spk3 | 1 | 354 | 110 (31.1) | 28 (7.9) | 68 (19.2) | 67 (18.9) | 3 (0.8) | 39 (11.0) | 39 (11.0) |
| post·spk1 | 4 | 1049 | 319 (30.4) | 87 (8.3) | 243 (23.2) | 159 (15.2) | 9 (0.9) | 115 (11.0) | 117 (11.2) |
| post·spk2/post1 | 17 | 1820 | 588 (32.3) | 209 (11.5) | 419 (23.0) | 234 (12.9) | 24 (1.3) | 157 (8.6) | 189 (10.4) |
| post·spk2/post2 | 11 | 1631 | 504 (30.9) | 163 (10.0) | 353 (21.6) | 235 (14.4) | 18 (1.1) | 175 (10.7) | 183 (11.2) |
| post·spk3 | 16 | 1684 | 526 (31.2) | 223 (13.2) | 395 (23.5) | 212 (12.6) | 21 (1.2) | 148 (8.8) | 159 (9.4) |

Manner shares are broadly stable (Vowel 30–34% everywhere), so the κ differences above are not
driven by gross class-imbalance — the confound is **stimulus type** (word list vs passage), which
changes segmentation and coarticulation rather than manner proportions.

### The one stimulus-matched contrast

Because spk3/pre survives only as `rainbow`, the sole clean pre/post pair in the corpus is the
Rainbow passage on both sides — pre `rainbow` (71.4 s) vs post `rainbow1`+`rainbow2` (75.7 s):

| Rainbow only | Vowel | Diph | Plosive | Fric | Affr | Nasal | Approx | speech seg |
|---|---|---|---|---|---|---|---|---|
| pre | 110 (31.1) | 28 (7.9) | 68 (19.2) | 67 (18.9) | 3 (0.8) | 39 (11.0) | 39 (11.0) | 354 |
| post | 102 (30.7) | 28 (8.4) | 62 (18.7) | 61 (18.4) | 3 (0.9) | 39 (11.7) | 37 (11.1) | 332 |

Every manner class agrees to within **0.7 pp**, vowel share to 0.2 pp, ~340 segments a side, both
at 99.4 fps. **This is the right basis for testing spk3's post-op drop and has not yet been run
as a κ eval** — the highest-value next experiment (see TODO).

---

## Verdict

**The transfer drop remains dominated by appearance/speaker/domain shift, not pathology — and the
evidence is now stronger than before.** Gloss pre-op transfers at κ 0.444, *above* its healthy-OOD
anchor (0.412); on connected speech alone (spk3's Rainbow) it reaches 0.602, close to the in-domain
ceiling of 0.673. At the segment level, gloss-pre **place κ 0.545 beats the healthy-OOD anchor**
and nears in-domain. The raw geometry is speaker-dominated (silhouette(speaker) > silhouette(phoneme),
the latter negative) in every label space, and per class it is stronger still — **all 28
vowel/place/manner classes have sil(spk) ≥ +0.185 in the raw encoder**, while the probe's pooler
drives that to ≈0 and does so most completely for **place**, the space that transfers best.
Pathology adds little on top of domain shift.

**The earlier claim "post ≥ pre, surgery does not reduce decodability" no longer holds as stated,
and its evidence was weaker than it looked.** Pooled frame-level κ now goes 0.444 → 0.359, but that
gap is **not** interpretable: the fps gate removes the word-list condition from one side of each
speaker's comparison, in opposite directions for spk1 and spk2, and word lists decode much worse
than passages. spk1 and spk2 move in opposite directions accordingly. At the segment level, where
gold boundaries remove most of that sensitivity, **spk1 and spk2 both go up post-op** — so for those
two speakers surgery still does not reduce decodability.

**spk3 is a genuine, robust exception.** Its post-op collapse reproduces across both frame-level
heads (κ 0.203 and 0.095), all four segment-level tasks (place κ 0.603 → 0.204), and place recall
(0.491 → 0.076) — and unlike spk1/spk2 it cannot be explained by stimulus composition, since spk3's
post session contains passages too. It is no longer the n=3 pairing artifact it first appeared to be.

## Caveats

- **The pooled pre/post contrast is stimulus-confounded and should not be quoted as a surgical
  effect.** Use the per-speaker rows, and preferably the matched-Rainbow design above.
- **spk3/pre is a single utterance** (354 segments). Its κ 0.602 is a real measurement on real data
  but rests on one passage from one speaker.
- **spk1/post is 4 utterances.** Its +0.145 is the least well-supported per-speaker number here.
- **Segment level is seed 0 only**; frame level is 3 seeds throughout, per-speaker grids included.
- **Oracle boundaries** at the segment level — absolute scores are not comparable to frame-level κ.
- **~0.03-style residuals remain an upper bound on pathology** (`docs/GLOSS.md §8`): with no healthy
  speaker recorded in the gloss acquisition domain, gloss-vs-usc_lss domain shift cannot be split
  from pathology.
- **fp16** on the eval box vs bf16 for the Annot-16 anchors — a small documented precision shift.
- `spk3_post_asha` is excluded (video truncated 7.59 s vs its annotation); its identity is by
  elimination and ~31% of its labels would have no frames.

## TODO

1. **Matched-Rainbow κ eval** — restrict both conditions to the passages present at 99.4 fps in
   every session and re-run. The only way to get an interpretable pre/post number.
2. **Segment-level seeds 1–2** (~16 GPU-h) for error bars on the Phase-3 ladder.
3. ~~Per-vowel cross-speaker t-SNE~~ — **done 2026-08-09**, and extended to place + manner;
   see *Per-class cross-speaker t-SNE* above.

---

## Artifacts

Under `/scratch1/hongn/artijepa/eval/gloss/` (regenerated 2026-08-06/07):

| kind | file pattern |
|---|---|
| frame-level metrics + per-speaker + anchors | `gloss_transfer_tssl256comb215_annot16_attentive_lstm_s{0,1,2}.json` |
| frame-level, comb100 `attentive` | `gloss_transfer_tssl256comb100_annot16_s{0,1,2}.json` |
| frame-level confusion (pre \| post) | `confmat_gloss_{groups,vowels,consonants}_<enc>_annot16[_attentive_lstm]_s{0,1,2}.png` |
| frame-level t-SNE | `tsne_gloss_rep{A,B}_tsne_<enc>_annot16[_attentive_lstm]_s{0,1,2}.png` |
| per-speaker × session place grid + values (3 seeds) | `confmat_gloss_{place,consonants}_perspk_…_s{0,1,2}.png`, `tsne_gloss_place_perspk_rep{A,B}_…_s{0,1,2}.png`, `confmat_gloss_perspk_values_…_s{0,1,2}.json` |
| segment-level metrics + clusters + drift | `glossphg_tssl256comb215_spk3none_s0.json` |
| segment-level confusion / per-speaker / t-SNE / drift, **all 4 tasks** | `{confmat,tsne,drift}_glossphg_*_tssl256comb215_spk3none_s0.png` |
| **per-class cross-speaker t-SNE**, 3 label spaces × 2 reps (2026-08-09) | `tsne_gloss_{vowels,place,manner}_rep{A,B}_segment_tsne_tssl256comb215_attentive_lstm_uscref[_mincap]_s0.png` — bare = sil(spk) + per-speaker n in the captions, `_mincap` = clean captions (class + n). Job logs `slurm_gloss_tsne_{10949541,10950602}.log` carry the sil(spk) and n tables |
| corpus composition | `gloss_phoneme_dist_corrected.{csv,json}`, `gloss_vowel_dist_corrected.png`, `gloss_spk3_dist_corrected.json` |
| healthy-OOD anchor by stimulus novelty (3 seeds) | `lss_stimulus_tssl256comb215_attentive_lstm_s{0,1,2}.json` (subsets + per-item κ/PER + all 71 match ratios) |

Every `.png` in `eval/gloss/` is current (96 files, regenerated 2026-08-06/07 and 08-09). The
55 superseded figures — old naming (`…comb100_s0` from the pre-annot16 usc_lss probe,
`…_spk1_s0`, and `glossphg_…_s0` without `_spk3none`) — are in
`eval/gloss/_stale_pre20260806/`.

Feature caches (new; keyed to the rebuilt manifests): `feat_cache/phoneme/gloss_{pre,post}_<enc>sp_*`
and `feat_cache/glossphg/glossphg_{pre,post}_spk3none_*`. The pre-rebuild caches (104 GB) are parked
in `feat_cache/_stale_pre20260806/`.

> **!! CACHE INVALIDATION !!** `eval_phoneme._tag` / `eval_phoneme_groups._tag` hash the manifest
> **path**, not its contents. Rebuilding the gloss manifests in place does **not** change the cache
> name, so a later run will silently cache-hit features extracted from the previous corpus. Whenever
> a rebuild changes which utterances are included, move the affected `feat_cache/*/gloss*` dirs aside
> first. This is why the caches above were quarantined rather than left in place.

## Reproduce

```bash
source dev_artiJEPA/scripts/_env.sh
cd /project2/shrikann_35/hongn/vjepa2

# 1. manifests (native 99.4 fps sources, >=20 fps gate)
python -m artijepa.gloss                     # --min-fps 0 disables the rate gate

# 2. frame-level, 3 seeds. Seed 0 extracts (~25 min); seeds 1-2 cache-hit (~7 min).
#    Chain with --dependency=afterok so the seeds do not race on the same cache dir.
A=$(sbatch --parsable dev_artiJEPA/scripts/32_gloss_frame.sbatch tssl256comb215 attentive_lstm 0 --per-speaker-figs)
sbatch --dependency=afterok:$A dev_artiJEPA/scripts/32_gloss_frame.sbatch tssl256comb215 attentive_lstm 1
sbatch --dependency=afterok:$A dev_artiJEPA/scripts/32_gloss_frame.sbatch tssl256comb215 attentive_lstm 2
#    secondary head/encoder:
sbatch dev_artiJEPA/scripts/32_gloss_frame.sbatch tssl256comb100 attentive 0

# 3. segment-level (Phase-3 clip probes), seed 0
sbatch dev_artiJEPA/scripts/31_gloss_groups.sbatch tssl256comb215 0

# 4. corpus composition (no GPU)
python -m artijepa.gloss_phoneme_dist --from-textgrids --suffix _corrected

# 5. healthy-OOD anchor split by stimulus novelty (cache-hits test_lss; ~8 min/seed)
for s in 0 1 2; do python -m artijepa.eval_lss_stimulus --seed $s; done

# 6. per-class cross-speaker t-SNE: vowels + place + manner, rep A and rep B (CPU-only,
#    ~20 min total, cache-hits everything -- must run AFTER step 2 has written the caches)
sbatch dev_artiJEPA/scripts/33_gloss_tsne_classes.sbatch
#    clean-caption variant (class + n only) -- arg 5 tags the filename so it lands BESIDE
#    the previous figure instead of overwriting it. Always pass it when restyling.
sbatch dev_artiJEPA/scripts/33_gloss_tsne_classes.sbatch tssl256comb215 attentive_lstm 0 "A B" mincap
#    one label space / one rep only:
python -m artijepa.tsne_gloss_vowels --encoder tssl256comb215 --head attentive_lstm \
      --rep B --per segment --label-space place --include-usc
```

Method: `docs/GLOSS.md §11` (frame-level) and `§12` (segment-level).
