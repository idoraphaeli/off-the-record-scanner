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
ANG_TOL = 6.0                   # angular agreement once the rotation is known


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


def confirm(marks, others, delta):
    """Which of `marks` also appear in `others`, once the rotation is undone.

    Both lists carry disc coordinates (`radius_frac`, `angle_deg`), so a match
    means the same place on the RECORD — not the same place in a photograph.
    """
    flags = []
    for m in marks:
        want = (m["angle_deg"] + delta) % 360.0
        hit = False
        for o in others:
            if abs(m["radius_frac"] - o["radius_frac"]) > RAD_TOL:
                continue
            if abs((o["angle_deg"] - want + 180.0) % 360.0 - 180.0) <= ANG_TOL:
                hit = True
                break
        flags.append(hit)
    return flags
