# -*- coding: utf-8 -*-
"""MODEL v2 -- a detector built on what the last two days measured.

Three findings, in one pipeline.

  the two shots are combined BEFORE the decision, not after it. The detector
  produces a response map -- a number per point saying how scratch-like it is --
  and today each shot is thresholded on its own and the surviving marks matched
  afterwards, so a faint scratch answering just under the bar in both shots is
  thrown away twice. Here the two maps are brought into the same frame and the
  pixelwise MINIMUM is taken, which demands presence in both without demanding
  strength in either. What comes out is only what both shots saw.

  the three anti-reflection rules come off. They reject marks aimed at the
  centre, marks inside a bright patch, and marks lying along a groove -- each of
  them a lamp's signature, and each of them also a real scratch sometimes. They
  were tuned when a single photograph had to be clean on its own; the minimum
  above removes reflections far more directly, since a reflection moves when the
  disc is tilted and cannot survive being intersected with the other shot.

  the thresholds come down. The bar was high to keep reflections out. It no
  longer has to be.

The part that is new here and was not in any of the tests is the ALIGNMENT.
Everything above depends on knowing how far the disc turned between the two
shots, and the centre label -- which is where that came from until now -- was
measured refusing to answer on one side in ten and being 5.5 degrees out on
another, against a slack of less than one. So the rotation is read off the
response maps themselves instead: the maps are what has to line up, they carry
far more signal than a label's print, and they exist for every record.

The label is kept as a cross-check and as the fallback for a side the maps
cannot align, and a side that neither can align is reported as such rather than
being silently graded on one photograph.

Usage:  python model_v2.py <photo A> <photo B>       one side, printed
        python model_v2.py --evaluate [n records]    scored against the pen marks
"""

import os
import sys

import cv2
import numpy as np

import detector
from cross_shot import label_profile, rotation_from_label
from detector import P

# --- what this model changes about the detector -----------------------------
RULES_OFF = dict(RADIAL_TOL_DEG=91.0,    # an axis angle never exceeds 90
                 GLARE_PATCH_MAX=256,    # a mean of 8-bit pixels never does
                 GROOVE_TOL_DEG=-1.0)    # nor is an angle ever below zero
THRESHOLDS = dict(PCT_STRONG=98.6, PCT_WEAK=97.6, THR_FLOOR=17)

# --- combining ---------------------------------------------------------------
TOL_ROWS = 9        # px of slack across the grooves, for residual misalignment
TOL_COLS = 15       # columns of slack around the disc, about 1.5 degrees
MAP_MIN_RATIO = 4.0  # how far the alignment peak must stand above the rest


def maps_of(path_or_image):
    """The ring and its two response maps, before any threshold is applied."""
    img = (detector.load_image(path_or_image) if isinstance(path_or_image, str)
           else path_or_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner, outer = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer]
    radial, tram = detector.scratch_map(ring)[:2]
    return {"img": img, "shape": gray.shape, "ring": ring, "radial": radial,
            "tram": tram, "inner": inner, "radius": radius, "center": center,
            # kept for the label fallback, which reads the file itself
            "path": path_or_image if isinstance(path_or_image, str) else None}


def _normalised(m, rows):
    """A map made comparable to another: stretched to a common height and with
    each row centred and scaled, so a brighter photograph cannot outvote a
    dimmer one when the two are correlated."""
    if m.shape[0] != rows:
        m = cv2.resize(m, (m.shape[1], rows), interpolation=cv2.INTER_LINEAR)
    s = m.astype(np.float32)
    s = s - s.mean(axis=1, keepdims=True)
    return s / np.maximum(s.std(axis=1, keepdims=True), 1e-3)


def rotation_from_maps(a, b):
    """How far the disc turned, read off the response maps themselves.

    A rotation of the disc is a sideways slide of the unwrapped map, so the
    circular cross-correlation of the two maps peaks at exactly that slide. This
    is the same trick used on the label's print, applied to the thing that
    actually has to line up -- and the maps carry a whole disc of marks rather
    than one small printed circle, so the peak is far sharper: measured 6 to 27
    standard deviations above its own background, against a label that failed
    outright on one side and was 5.5 degrees out on another.

    Returns (degrees, how far the peak stands above the rest), or (None, ratio)
    when nothing stands out enough to trust.
    """
    rows = min(a["radial"].shape[0], b["radial"].shape[0])
    pa, pb = _normalised(a["radial"], rows), _normalised(b["radial"], rows)
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


def align(a, b):
    """The rotation between two shots: from the maps, or the label, or nothing.

    Both are tried because they fail for unrelated reasons -- the maps need
    marks to lock onto, the label needs print -- so a side one cannot answer is
    often one the other can.
    """
    delta, ratio = rotation_from_maps(a, b)
    if delta is not None:
        return delta, ratio, "maps"
    if a["path"] and b["path"]:
        delta, ratio = rotation_from_label(label_profile(a["path"]),
                                           label_profile(b["path"]))
        if delta is not None:
            return delta, ratio, "label"
    return None, ratio, "none"


def combined(ma, mb, delta):
    """The soft AND of two response maps, once the rotation is taken out.

    The second map is grown by a few pixels first: the alignment is good to
    about a degree, a scratch is about eight pixels wide, and a strict
    pixel-to-pixel minimum would pair a scratch with the vinyl beside it.
    """
    if mb.shape[0] != ma.shape[0]:
        mb = cv2.resize(mb, (mb.shape[1], ma.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    mb = cv2.dilate(mb, np.ones((TOL_ROWS, TOL_COLS), np.uint8))
    cols = int(round(-delta / 360.0 * P["POLAR_STEPS"]))
    return np.minimum(ma, np.roll(mb, cols, axis=1))


def analyse_side(photo_a, photo_b):
    """One side of a record, from two photographs of it.

    Returns the detection mask in the FIRST photograph's frame, plus what the
    alignment did, so a side that could not be aligned can be told apart from
    one that is clean.
    """
    keep = {k: P[k] for k in list(RULES_OFF) + list(THRESHOLDS)}
    P.update(RULES_OFF)
    P.update(THRESHOLDS)
    try:
        a, b = maps_of(photo_a), maps_of(photo_b)
        delta, ratio, how = align(a, b)
        if delta is None:
            # Nothing to intersect with. Falling back to one photograph would
            # report marks this model has no second opinion on, so the side is
            # returned unjudged and the caller decides what to say about it.
            return {"aligned": False, "how": how, "confidence": ratio,
                    "mask": np.zeros(a["shape"], np.uint8), "marks": 0,
                    "img": a["img"], "center": a["center"], "radius": a["radius"]}

        rad = combined(a["radial"], b["radial"], delta)
        tra = combined(a["tram"], b["tram"], delta)
        m1, _ = detector.extract(rad, None, a["ring"], a["inner"], a["radius"])
        m2, _ = detector.extract(tra, P["TRAM_MIN_LEN"], a["ring"],
                                 a["inner"], a["radius"])
        det = detector.rewrap(cv2.bitwise_or(m1, m2), a["inner"], a["center"],
                              a["radius"], a["shape"])
        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            (det > 127).astype(np.uint8), connectivity=8)
        big = [i for i in range(1, n) if stats[i][4] >= 40]
        det = (np.isin(lab, big).astype(np.uint8) * 255) if big else det * 0
        return {"aligned": True, "how": how, "confidence": ratio,
                "rotation": delta, "mask": det, "marks": len(big),
                "img": a["img"], "center": a["center"], "radius": a["radius"]}
    finally:
        P.update(keep)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] != "--evaluate":
        r = analyse_side(sys.argv[1], sys.argv[2])
        print(f"aligned : {r['aligned']}  by the {r['how']}"
              f"  (peak {r['confidence']:.1f} sd)")
        if r["aligned"]:
            print(f"rotation: {r['rotation']:.1f} deg")
            print(f"marks   : {r['marks']}")
        return
    print(__doc__)


if __name__ == "__main__":
    main()
