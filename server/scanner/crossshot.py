# -*- coding: utf-8 -*-
"""
Confirming a mark against a second photograph of the same side.

A scratch is cut into the vinyl: it sits at a fixed radius and a fixed angle on
the disc and is there however the record is held. A reflection belongs to the
light, not the record — move the disc and it lands somewhere else. So a mark
found in the same place on the disc in two separate shots is far more likely to
be real, and that was measured: 89% of confirmed marks were real against 71% of
marks seen only once.

The two photographs are never registered to each other, and that is the point.
A record is very nearly rotationally symmetric — thousands of near-identical
circular grooves — so an image aligner has almost nothing to lock onto, and an
earlier attempt at this returned the same wrong answer for every pair. Radius is
invariant between shots and is already measured, so the whole difference between
two shots is a SINGLE unknown angle.

That angle is read off the centre label: printed, asymmetric, high contrast, and
turning rigidly with the disc. Unrolling the label turns a rotation into a
sideways slide, and a circular cross-correlation finds the slide directly. The
lighting has to be removed first — a lamp to one side makes that side of the
label brighter, that gradient is the strongest thing in the strip, and it belongs
to the ROOM rather than to the record. Measured against hand-drawn marks, this
agreed within 8 degrees on every side that could be checked, median 4, and found
a rotation for 71 of 74 sides.

Promoted 2026-08-27 from experiments/cross_shot.py. The detection-vote fallback
in that file is deliberately NOT carried over: it was never checked against
ground truth, it aligned one side in 74 on its own, and a wrong rotation turns
"confirmed" into coincidence.
"""

import cv2
import numpy as np

# Label band, as fractions of the disc radius. The detector crops this away
# before it looks for damage, which is exactly why it is free to use here.
LABEL_LO, LABEL_HI = 0.12, 0.33
POLAR_STEPS = 1440              # 0.25 degrees per column
LABEL_MIN_RATIO = 4.0           # correlation peak over its own background
# Blur wide enough to erase printed letters but not the lighting gradient, which
# spans the whole strip. 60 columns of 1440 is 15 degrees of arc.
BLUR_W = 60

RAD_TOL = 0.025                 # radius agreement, as a fraction of disc radius
# Angular agreement once the rotation is known. Was 6 degrees, which at a typical
# radius is well over a hundred pixels against a scratch about eight wide -- a
# window that loose confirms marks that are merely near each other. It could not
# be tightened before, because the label's angle was itself a couple of degrees
# out and the window was silently paying for that. With the correction below in
# place, 2 degrees confirms MORE marks than 6 did without it (314 against 148,
# measured over calibration and validation) at 98% precision against 95%.
ANG_TOL = 2.0
# ...but only where the angle could actually be corrected. Correcting it needs a
# handful of paired marks to fit on, and a side with too few gets the label's raw
# angle back -- still a couple of degrees out, and a 2 degree window around a
# wrong angle confirms nothing at all. Measured on one such side: 2 confirmed at
# 6 degrees, 0 at 2. So the old window is kept exactly where the correction could
# not run, which makes this change no worse than before on any side and better on
# the ones that can be corrected.
ANG_TOL_UNCORRECTED = 6.0

# Correcting the label's angle from the marks themselves.
SEARCH_W = 8.0        # how far out a pair may sit while the angle is still wrong
MIN_PAIRS = 4         # below this the median is one outlier from meaningless
REFINE_ROUNDS = 4


def label_profile(gray, center, radius):
    """The centre label unrolled into a strip indexed by angle, with the
    lighting removed so only the PRINT is left."""
    if radius < 40:
        return None
    polar = cv2.warpPolar(gray, (radius, POLAR_STEPS), center, radius,
                          cv2.WARP_POLAR_LINEAR)
    strip = cv2.transpose(polar)[int(LABEL_LO * radius):int(LABEL_HI * radius)]
    if strip.shape[0] < 4 or strip.size < POLAR_STEPS:
        return None
    s = strip.astype(np.float32)
    # wrap the ends round before blurring: the strip is a circle cut open, and a
    # blur that treats its edges as edges invents a seam that never existed
    wide = np.hstack([s[:, -BLUR_W:], s, s[:, :BLUR_W]])
    lighting = cv2.GaussianBlur(wide, (BLUR_W * 2 + 1, 1), 0)[:, BLUR_W:BLUR_W + s.shape[1]]
    s = s - lighting
    s -= s.mean(axis=1, keepdims=True)
    return s / np.maximum(s.std(axis=1, keepdims=True), 1e-3)


def rotation_from_label(pa, pb):
    """Degrees that map disc A onto disc B, or None if the label cannot say.

    Returns (degrees, confidence). A label with no print — a plain single-colour
    one — has no direction to give and is refused here rather than answering at
    random.
    """
    if pa is None or pb is None:
        return None, 0.0
    # The strips are as tall as the disc's radius IN PIXELS, which changes with
    # how close the phone was held. Refusing to compare on a two-row difference
    # threw out 54 of 74 sides; stretching to a common height costs nothing,
    # because the radius axis is only there to give the match more evidence.
    if pa.shape[0] != pb.shape[0]:
        h = min(pa.shape[0], pb.shape[0])
        pa = cv2.resize(pa, (pa.shape[1], h), interpolation=cv2.INTER_AREA)
        pb = cv2.resize(pb, (pb.shape[1], h), interpolation=cv2.INTER_AREA)
    corr = np.fft.irfft(np.fft.rfft(pb, axis=1) *
                        np.conj(np.fft.rfft(pa, axis=1)), axis=1).sum(axis=0)
    k = int(corr.argmax())
    peak = float(corr[k])
    # judge the peak against the rest of the curve, with its own neighbourhood
    # excluded so a broad peak does not dilute its own background
    mask = np.ones(corr.size, bool)
    w = max(corr.size // 60, 3)
    mask[(np.arange(corr.size) - k) % corr.size <= w] = False
    mask[(k - np.arange(corr.size)) % corr.size <= w] = False
    bg, sd = corr[mask].mean(), corr[mask].std()
    if sd <= 0:
        return None, 0.0
    ratio = float((peak - bg) / sd)
    if ratio < LABEL_MIN_RATIO:
        return None, ratio
    return (k * 360.0 / corr.size) % 360.0, ratio


MAP_MIN_RATIO = 4.0     # how far the alignment peak must stand above the rest


def _normalised(m, rows):
    """A response map made comparable to another: stretched to a common height,
    then each row centred and scaled, so a brighter photograph cannot outvote a
    dimmer one when the two are correlated."""
    if m.shape[0] != rows:
        m = cv2.resize(m, (m.shape[1], rows), interpolation=cv2.INTER_LINEAR)
    s = m.astype(np.float32)
    s = s - s.mean(axis=1, keepdims=True)
    return s / np.maximum(s.std(axis=1, keepdims=True), 1e-3)


def rotation_from_maps(map_a, map_b):
    """How far the disc turned, read off the response maps themselves.

    Same trick as the label, applied to the thing that actually has to line up.
    A rotation of the disc is a sideways slide of the unwrapped map, so the
    circular cross-correlation of the two peaks at exactly that slide — and a
    map carries a whole disc of marks rather than one small printed circle, so
    the peak is far sharper. Measured against an exhaustive search it lands
    within a degree, where the label refused outright on one side of ten and was
    5.5 degrees out on another; across 55 calibration sides it answered 52,
    against the label's 53 with three of those wrong enough to matter.

    Returns (degrees, how far the peak stands above the rest), or (None, ratio)
    when nothing stands out enough to trust.
    """
    rows = min(map_a.shape[0], map_b.shape[0])
    pa, pb = _normalised(map_a, rows), _normalised(map_b, rows)
    corr = np.fft.irfft(np.fft.rfft(pb, axis=1) *
                        np.conj(np.fft.rfft(pa, axis=1)), axis=1).sum(axis=0)
    k = int(corr.argmax())
    mask = np.ones(corr.size, bool)
    w = max(corr.size // 60, 3)
    mask[(np.arange(corr.size) - k) % corr.size <= w] = False
    mask[(k - np.arange(corr.size)) % corr.size <= w] = False
    bg, sd = corr[mask].mean(), corr[mask].std()
    if sd <= 0:
        return None, 0.0
    ratio = float((corr[k] - bg) / sd)
    if ratio < MAP_MIN_RATIO:
        return None, ratio
    return (k * 360.0 / corr.size) % 360.0, ratio


def align(map_a, map_b, profile_a, profile_b):
    """The rotation between two shots: from the maps, or the label, or nothing.

    Both are tried because they fail for unrelated reasons — the maps need marks
    to lock onto, the label needs print — so a side one cannot answer is often
    one the other can. Returns (degrees, confidence, which method), with degrees
    None when neither could say.
    """
    delta, ratio = rotation_from_maps(map_a, map_b)
    if delta is not None:
        return delta, ratio, "maps"
    delta, ratio = rotation_from_label(profile_a, profile_b)
    if delta is not None:
        return delta, ratio, "label"
    return None, ratio, "none"


# How much slack the combination allows for what is left of the misalignment.
# The angle is good to about a degree and a scratch is about eight pixels wide,
# so a strict pixel-to-pixel minimum would pair a scratch with the vinyl beside
# it and score it zero.
TOL_ROWS, TOL_COLS = 9, 15


def combine(map_a, map_b, delta, polar_steps):
    """The soft AND of two response maps, once the rotation is taken out.

    The pixelwise MINIMUM: a point that both shots saw keeps its value, and a
    point only one of them saw takes the other's, which is nothing. That is what
    removes a lamp's reflection — it moves when the disc is tilted, so the other
    shot has nothing there — and it is why what comes out is only what both
    photographs agreed on.
    """
    if map_b.shape[0] != map_a.shape[0]:
        map_b = cv2.resize(map_b, (map_b.shape[1], map_a.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    grown = cv2.dilate(map_b, np.ones((TOL_ROWS, TOL_COLS), np.uint8))
    return np.minimum(map_a, np.roll(grown, int(round(-delta / 360.0 * polar_steps)),
                                     axis=1))


def _offsets(marks, others, delta, window):
    """For each mark, how far its nearest partner sits from where this rotation
    says it should be. A mark with no partner inside the window says nothing."""
    out = []
    for m in marks:
        want = (m["angle_deg"] + delta) % 360.0
        best = None
        for o in others:
            if abs(m["radius_frac"] - o["radius_frac"]) > RAD_TOL:
                continue
            d = (o["angle_deg"] - want + 180.0) % 360.0 - 180.0
            if abs(d) <= window and (best is None or abs(d) < abs(best)):
                best = d
        out.append(best)
    return out


def refine_rotation(marks, others, delta):
    """The label's angle, corrected by the offset its own pairs still show.

    Unwrapping both shots and drawing them into one strip showed matched pairs
    sitting SIDE BY SIDE rather than on top of each other, by a median of 2.4
    degrees -- and by the same amount for every pair on a side, which is the
    signature of a wrong angle rather than of pairs that do not belong together.
    Measured across 51 sides, the correction moves the angle a median of 2.8
    degrees while the pairs stay within 0.75 degrees of each other, so they are
    plainly moving together.

    A MEDIAN is used rather than a mean because most marks have no true partner,
    and the ones that pair up by accident sit anywhere; a mean would let them
    drag the correction. The starting angle comes from the label rather than from
    the marks, so this cannot manufacture agreement out of nothing -- it can only
    remove a bias that is already common to the pairs.

    Returns (angle, how far it moved, the spread of the pairs around it, how many
    pairs it was fitted on). Spread is None when there were too few pairs to fit,
    in which case the label's own angle is handed back unchanged.
    """
    start, res = delta, []
    for _ in range(REFINE_ROUNDS):
        res = [d for d in _offsets(marks, others, delta, SEARCH_W) if d is not None]
        if len(res) < MIN_PAIRS:
            return start, 0.0, None, 0
        step = float(np.median(res))
        delta = (delta + step) % 360.0
        if abs(step) < 0.05:
            break
    spread = float(np.median(np.abs(np.array(res) - np.median(res))))
    moved = (delta - start + 180.0) % 360.0 - 180.0
    return delta, moved, spread, len(res)


def confirm(marks, others, delta, window=None):
    """Which of `marks` also appear in `others`, once the rotation is undone.

    Both lists carry disc coordinates (`radius_frac`, `angle_deg`), so a match
    means the same place on the RECORD — not the same place in a photograph.

    `window` is how much angular disagreement is still a match; it is passed in
    rather than fixed here because how tight it may be depends on whether the
    angle was corrected for this side.
    """
    window = ANG_TOL if window is None else window
    flags = []
    for m in marks:
        want = (m["angle_deg"] + delta) % 360.0
        hit = False
        for o in others:
            if abs(m["radius_frac"] - o["radius_frac"]) > RAD_TOL:
                continue
            if abs((o["angle_deg"] - want + 180.0) % 360.0 - 180.0) <= window:
                hit = True
                break
        flags.append(hit)
    return flags
