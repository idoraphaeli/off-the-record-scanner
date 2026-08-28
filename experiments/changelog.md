# Experiment changelog — scratch-detector optimization

## Round 1 — flash dataset (13 photos, direct flash)

### Stage 0 (frozen)
- Paired 13 clean/marked photos. One pair off by 1px width (WhatsApp re-encode) —
  aligned by exact crop search, no resampling.
- Ground truth = clean-vs-marked pixel difference (not colour thresholding, which
  would also catch blue in backgrounds/labels): **7 marked images, 21 zones**;
  6 images confirmed by the annotator as "nothing visible to the naked eye".
- Frozen evaluator: per-zone recall (>=20 det px inside a zone), FP suspects
  (>=50 px components outside every zone). Frozen 10/3 split.

### Result: abandoned — no signal in the data
Baseline scored recall 0.0%. A probe of **7 candidate feature maps** (bright
top-hat, dark black-hat, both polarities, desaturated-brightness, wide top-hat,
sparkle-loss, line-boosted variants) measured the density of above-noise pixels
inside marked zones vs background: **every feature scored 0.0–2.4 (≈1 = chance)**.
Full-resolution crops of the marked zones confirmed it visually — at best a
barely-visible hairline, mostly nothing.

Root cause: (1) direct flash produces iridescent diffraction sparkle across the
whole groove field and flattens scratch relief — the opposite of the raking light
the literature specifies; (2) WhatsApp compression to 1600x1200 puts hairlines at
or below one pixel.

---

## Round 2 — dark-room dataset (15 photos, dim side lighting)

Same frozen Stage-0 rig, re-run: **14 marked images, 63 zones**, split 11 cal / 3
locked test. One pair unusable (marked copy re-encoded at a different size).

Feature probe now scores the primary feature at **density ratio 33.9** (vs 0.0–0.3
on the flash set) — the capture change, not any algorithm change, is what put
signal in the data.

| # | change | recall (cal) | FP suspects/img |
|---|---|---|---|
| n0 | baseline (flash-era params, auto disc detection) | 15.7% | 5.18 |
| n1 | Hough disc detection (blob method was dragged off-centre by the disc's own shadow) + unlit-area masking | 5.9% | 3.64 |
| n2 | adaptive percentile thresholds (measured noise floor moves 18→29 between images; a fixed threshold cannot serve both) | 9.8% | 1.82 |
| n3 | shape limits recalibrated from measured true-positive stats (old MIN_LEN=40 cut 18 of 24 real detections; median true length is 39) | 33.3% | 4.45 |
| n4 | added groove-parallel ("tramline") channel, merged into one map | 27.5% | 5.45 |
| n5 | split the two channels — a merged map lets the louder channel raise the shared threshold and starve the quieter one | 33.3% | 7.00 |
| n6 | **local noise normalisation** — divide by local sigma so one threshold means the same in a bright busy sector and a dim quiet one | **68.6%** | 18.73 |
| n7 | operating point chosen from the recall/FP sweep | 54.9% | 5.00 |

### The decisive measurement (before n6)
Per-zone diagnosis of all 48 calibration zones:

| outcome | count | meaning |
|---|---|---|
| above threshold | 16 | detected |
| **below threshold** | **16** | real signal present, global threshold too high |
| masked out | 8 | fell in unlit / glare areas |
| no signal | 8 | genuine capture limit |

This is what identified local normalisation as the fix rather than more knob
turning, and it sets the honest recall ceiling for this dataset at ~83%.

### Recall / false-positive trade-off (calibration set)
| PCT_STRONG | PCT_WEAK | MIN_LEN | recall | FP/img |
|---|---|---|---|---|
| 99.90 | 99.0 | 22 | 66.7% | 17.6 |
| 99.97 | 99.0 | 22 | 60.8% | 7.6 |
| **99.97** | **99.7** | **22** | **54.9%** | **5.0** |
| 99.97 | 99.7 | 30 | 49.0% | 2.6 |
| 99.99 | 99.7 | 22 | 41.2% | 1.8 |

Note "FP suspects" are unjudged: the annotator marked only what was visible to
the naked eye, so a suspect in a well-lit sector is often a real scratch that was
missed by the human. Visual review of the n7 overlays supports this for several
of them; they were not individually adjudicated.

### Final — locked test set (single run, 3 images, 12 zones)
**recall 25.0%, 3.0 FP suspects/image.**

The gap vs calibration (54.9%) is real and expected: 11 images is a small
calibration set and the operating point was chosen on it. The test number is the
honest one to quote.

## Round 3 — fixing fragmented long scratches

Observation from the annotator: long, obvious scratches were being marked only in
part. Three causes were identified; the first was measured directly.

**1. Self-suppression (measured).** The local normaliser divided each pixel by the
local sigma — but a long bright scratch is *part of* its own neighbourhood, so it
inflated its own denominator and erased everything except its brightest fragment
(the classic CFAR self-masking failure). Replacing it with a robust estimator
(deviations clipped at twice the local average before averaging, so an outlier
cannot set its own noise floor):

| noise estimator | coverage of a marked zone |
|---|---|
| plain local sigma | 2.35% |
| **robust** | **7.68%** (3.3x) |

**2. Angle quantisation.** Line-boost kernels ran at 20° steps; a long scratch
curves along its length, and segments landing between two kernel angles were
smeared rather than boosted. Now 10° steps over ±70°.

**3. Masks cutting scratches in half.** A long scratch usually crosses from the
lit sector into the dark one, or passes a glare spot; both masks zero it
mid-length and each half then fails the length test on its own. Added
`_link_collinear`: rejoins two fragments only when they are close, both elongated
along the connecting direction, and mutually aligned — unlike a morphological
close, which also merges unrelated neighbours.

| # | change | recall (cal) | FP suspects/img |
|---|---|---|---|
| n8 | all three fixes, thresholds unchanged | 41.2% | 1.91 |
| n9 | thresholds re-swept for the (less trigger-happy) robust estimator | **62.7%** | 5.09 |

Post-fix trade-off curve (calibration): 74.5% @ 18.8 FP · 68.6% @ 13.6 ·
**62.7% @ 5.1** · 58.8% @ 5.3 · 41.2% @ 1.9.

Note the shape filter no longer buys much: raising MIN_LEN from 30 to 90 dropped
recall 70.6% → 21.6% while FP only fell 12.5 → 5.4, i.e. the surviving false
positives are themselves long and linear, not short blobs. Length is exhausted as
a discriminator; separating these would need a different cue (contrast profile
along the line, or multi-exposure agreement).

### Final — locked test set (single run)
**recall 25.0%, 3.67 FP suspects/image** (unchanged recall vs round 2; the round-3
gains showed on calibration but did not transfer, which with 3 test images and 12
zones is within noise).

## Round 4 — groove-direction rejection (annotator's proposal)

Observation from the annotator: many false positives are the light reflecting off
the grooves themselves, and those highlights run *along* the disc's geometry.
Proposal: measure each detection's direction and drop it when it matches the
groove direction within a small tolerance — mostly short marks.

Implemented as `_axis_angle_deg`: PCA on each surviving component in the
unwrapped image, where grooves are horizontal, giving the angle away from the
groove direction (0 = along, 90 = across). A component under `GROOVE_TOL_DEG` is
dropped unless it exceeds `GROOVE_KEEP_LEN`, which preserves genuine tramlines —
no single groove highlight runs that long.

| # | change | recall (cal) | FP suspects/img |
|---|---|---|---|
| n9 | before | 62.7% | 5.09 |
| **n10** | **groove-direction rejection (12°, keep >250px)** | **62.7%** | **2.27** |

**False positives cut by 55% with zero loss of recall** — the cleanest single
change in the whole project, and it came from a domain observation rather than
from parameter search. Sweeping the tolerance confirmed 12° is optimal: widening
it to 20/28/35° started removing real scratches (62.7% → 58.8% → 52.9%) for no
further FP gain.

Trying to spend the recovered headroom back on sensitivity did not pay: recall
sits on a plateau at 62.7% across the whole PCT_STRONG/PCT_WEAK grid, only FP
moves. The remaining misses are not threshold-limited.

### Final — locked test set (single run)
**recall 25.0%, 2.33 FP suspects/image** (FP improved from 3.67; recall unchanged).

## Stopping point
Neither target was met (recall >= 75%, FP <= 1/image). Stopped at 7 iterations
with the blockers identified rather than continuing to tune:

1. **Only part of the disc is lit.** A single side lamp leaves most of the record
   unjudgeable — 8 of 48 zones were inside masked areas. Multiple exposures with
   the lamp moved between shots (camera and record fixed) would cover the whole
   surface; this is the single highest-value next change.
2. **8 of 48 zones carry no measurable signal at all** — the marks are visible to
   a human tilting the physical disc, but not present in a fixed-viewpoint photo.
3. **Small dataset.** 11 calibration images cannot support reliable threshold
   selection; the cal/test gap is the evidence.

## Reusable assets
`stage0_extract_gt.py` (blue-marker → masks), `evaluate.py` (frozen metrics),
`make_split.py`, `detector.py`, `run_detector.py`, `sweep.py`, and the diagnostic
tools that did the real work: `feature_probe.py` (is there signal at all?),
`filter_audit.py` (which filter kills true positives?), `zone_response.py`
(per-zone signal vs threshold), `review_overlay.py` (detections + zones for
human adjudication).

---

# Promotion to the server — 2026-08-27

`server/scanner/detector.py` had been running the old parameters since it was
first written. The changes measured on the new dataset were ported into it by
hand, not copied over the file: the two keep different function signatures
(`find_disc` also returns how it found the disc, `scratch_map` also returns the
unjudgeable mask), so a straight copy would have broken `analyze.py`.

Verified after porting by running both copies over the same photographs and
comparing the masks pixel for pixel — identical on every one, and all 34 shared
parameters hold the same value.

## What moved

| | was | now | why |
|---|---|---|---|
| `LABEL_R` / `OUTER_R` | 0.40 / 0.93 | 0.36 / 0.95 | 15% of all misses lay outside the analysed band |
| `PCT_STRONG` / `PCT_WEAK` | 99.8 / 99.5 | 99.3 / 98.7 | 80% of missed scratches produced a response that was then discarded |
| `THR_FLOOR` | 35 | 25 | the heavier of the two gates: worth 5.0 points on its own |
| `MIN_LEN` | 30 | 15 + graded rule | 15.6% of misses were killed by this bar alone |
| `MAX_THICK` | 12 | 16 | |
| `RADIAL_TOL_DEG` / `RADIAL_MIN_LEN` | — | 83 / 45 | new: rejects the lamp's beam off the grooves |

## Where it stands

Measured over 53 records split BY RECORD. The test set was opened once, at the
end, after every parameter was fixed.

| | recall | precision |
|---|---|---|
| calibration (30 records) | 60.7% | 75.9% |
| validation (11 records) | 65.2% | 78.1% |
| test (12 records) | 56.9% | 77.2% |

Precision counts dirt as a correct call and was measured by hand-labelling all
2757 detections across the three sets.

## The one thing this breaks

The new build reports about three times as many marks per photo, and
`analyze._grade` sums length × thickness over them. `GRADE_BANDS` was never
fitted to human-graded records, and it now sits against a much larger damage
index. Measured over ten photographs, grades fall one to two bands — one record
moved from Near Mint to Very Good on the same photograph.

**The bands need refitting against records graded by a person before the grade
is shown to anyone as a number.** The detections themselves are better; the
grade derived from them is now further off than it was.

---

# The outer-ring brightness rule — 2026-08-29

## What made it findable

The feature table had been built only from detections the labelling tool asked
about, and that tool never shows a detection that landed on a pen mark. Several
hundred of the most reliable scratches in the set — the ones sitting on Ido's own
strokes — had therefore never been measured. Fixing that took the scratch sample
from 62 to 588.

It changed conclusions immediately. Local contrast, which looked like a
separator on 62 examples, turned out to be nothing (0.44). Surrounding
brightness, which had been dismissed as nothing on the same small sample, turned
out to be the strongest single signal in the table. Two of the eleven rejected
hypotheses were rejected on evidence too thin to carry the claim.

A first attempt at including those detections used the evaluator's 30px
tolerance, which is right for asking whether a scratch was noticed and far too
loose for asking what a detection is: checked against a rendered gallery, the
loosest pairings were specks of dust beside a marked scratch. Tightened to
requiring the mark to touch the ink, which dropped 87 rows and took the worst
pairing from 17px to 5px.

## The rule

A scratch is a thin bright line on dark vinyl, so the surface around it stays
dark. A lamp reflection is a broad wash of light, and the lines found inside one
belong to the wash. Applied only outside 0.65 of the radius, where the lamp
problem lives — the inner half already runs at 85% precision against the outer
half's 70%.

    reject if  radius >= 0.65  AND  the patch around it is brighter than 70

Both numbers chosen on a grid over the CALIBRATION records, holding at least 90%
of the scratches, and measured in the unwrapped ring rather than the photograph,
because that is where it runs.

## What it did

|            | recall        | precision     |
|------------|---------------|---------------|
| calibration| 60.7 -> 57.2  | 83.3 -> 88.4  |
| validation | 65.2 -> 58.9  | 84.3 -> 89.8  |
| test       | 56.9 -> 52.3  | 77.9 -> 85.2  |

On the test set it removed 41% of the false detections for 7% of the scratches,
and detections per photo fell from 17.0 to 14.8. Ido chose precision over recall
explicitly, knowing this puts recall below the 55% floor discussed earlier.

The precision figures come from the labelled detections that can be joined back
to a component, so they sit a little above the official 77.2%; the DIFFERENCE is
the measured part. Recall is exact — it is scored against the pen marks and needs
no labelling.

## Promoted the same day

Ported by hand into server/scanner/detector.py, and analyze.py now hands the
ring down so the rule can run. Verified after porting by running both copies
over the same photographs and comparing the masks pixel for pixel -- identical
on every one, with all 37 shared parameters holding the same value.

Worth recording how that check nearly passed for the wrong reason: the first run
agreed perfectly while calling extract WITHOUT the ring, so the new rule was
skipped in both copies and the comparison proved only that the old behaviour
matched. It was the detection counts being unchanged from before the rule that
gave it away. A test that cannot fail is not a test.
