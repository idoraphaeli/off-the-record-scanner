# -*- coding: utf-8 -*-
"""
Multi-shot analysis: several photos of the same record side, combined.

Each photo is normalised independently (disc found, unwrapped to polar), which
removes camera translation, distance and viewing angle. What can remain is an
angular offset, when the RECORD itself was turned between shots -- estimated
here from the record's own fine texture, never from the lighting or the
detections.

Fusion is deliberately hybrid, because the two shots serve two different jobs:
  * where only one shot lit an area  -> use it alone (fixes coverage: a sector
    that was too dark to judge in shot A may be lit in shot B)
  * where both shots lit an area     -> require agreement (fixes false alarms:
    a scratch is fixed to the record and appears in both, while a lamp
    reflection sits wherever the lamp is and appears in only one)

Each detection therefore carries a confidence, and the report states how much of
the playing surface was judgeable at all.

Usage: python multishot.py <folder> [--grade]
"""

import os
import sys

import cv2
import numpy as np

import detector
from detector import P

LABEL_LO, LABEL_HI = 0.12, 0.36   # radius band holding the printed label
ALIGN_COARSE = 4       # coarse angular search step, in polar columns
ALIGN_STEP = 1         # refinement step
MIN_SHARPNESS = 3.0    # below this the alignment is a guess, not a measurement
MIN_COVERAGE = 55.0    # % of the ring below which the result is not trustworthy
# Agreement tolerance. The alignment is good to about a degree, not to a pixel,
# and a scratch is only 2-3 px wide -- so two views of the SAME scratch rarely
# overlap exactly. Without slack the agreement rule rejects everything, including
# the real marks (observed: 5 and 6 marks per shot, 0 after a strict AND).
AGREE_TOL = 21


def analyse_photo(path):
    """One photo -> ring, its two scratch maps, and its judgeable mask."""
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner_px, outer_px = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner_px:outer_px]
    radial, tram = detector.scratch_map(ring)
    judgeable = ~((detector.glare_mask(ring) > 0) | (detector.unlit_mask(ring) > 0))
    return dict(path=path, img=img, center=center, radius=radius,
                inner_px=inner_px, ring=ring, radial=radial, tram=tram,
                judgeable=judgeable, label=label_strip(gray, center, radius))


def label_strip(gray, center, radius):
    """The printed label, unwrapped. This is the alignment anchor: it rotates
    rigidly with the record and carries high-contrast text, so unlike groove
    texture it cannot be confused with a lighting pattern that stays put while
    the record turns. (Groove-texture alignment was tried first and locked onto
    the lighting, reporting 1 deg for a record that had actually turned 90.)
    """
    polar = detector.unwrap(gray, center, radius)
    lo, hi = int(LABEL_LO * radius), int(LABEL_HI * radius)
    strip = polar[lo:hi].astype(np.float32)
    return strip - cv2.blur(strip, (61, 1))       # drop the lighting gradient


def estimate_offset(strip_ref, strip_other):
    """Angular offset (in polar columns) that best aligns the two labels."""
    width = strip_ref.shape[1]
    scores = np.array([float((strip_ref * np.roll(strip_other, s, axis=1)).mean())
                       for s in range(0, width, ALIGN_COARSE)])
    coarse = int(np.argmax(scores)) * ALIGN_COARSE
    sharpness = (scores.max() - scores.mean()) / (scores.std() + 1e-9)

    # refine around the coarse winner at full resolution
    fine = [(float((strip_ref * np.roll(strip_other, s % width, axis=1)).mean()), s)
            for s in range(coarse - ALIGN_COARSE, coarse + ALIGN_COARSE + 1, ALIGN_STEP)]
    return max(fine)[1] % width, sharpness


def fuse(shots):
    """Combine aligned shots into one evidence map plus a confidence map."""
    ref = shots[0]
    h, w = ref["radial"].shape
    votes = np.zeros((h, w), np.int32)      # how many shots judged this pixel
    hits = np.zeros((h, w), np.int32)       # how many shots flagged it
    best = np.zeros((h, w), np.float32)

    kernel = np.ones((AGREE_TOL, AGREE_TOL), np.uint8)
    for s in shots:
        shift = s["shift"]
        judge = np.roll(s["judgeable"], shift, axis=1)
        smap = np.maximum(np.roll(s["radial"], shift, axis=1),
                          np.roll(s["tram"], shift, axis=1)).astype(np.float32)
        valid = smap[judge]
        if valid.size < 1000:
            continue
        thr = max(float(np.percentile(valid, P["PCT_WEAK"])), P["THR_FLOOR"] / 2)
        hit = (judge & (smap > thr)).astype(np.uint8)
        votes += judge.astype(np.int32)
        # widen each shot's hits so "the other shot saw it nearby" counts as
        # agreement; the evidence map itself keeps the original thin shape
        hits += cv2.dilate(hit, kernel).astype(np.int32)
        best = np.maximum(best, smap * judge)

    # confirmed where two or more shots saw it and agreed; provisional where only
    # one shot could see the area at all
    confirmed = (votes >= 2) & (hits >= 2)
    provisional = (votes == 1) & (hits == 1)
    evidence = best.copy()
    evidence[~(confirmed | provisional)] = 0
    coverage = 100.0 * (votes >= 1).mean()
    return evidence.astype(np.uint8), confirmed, provisional, coverage


def grade(scratches, ring_shape, coverage):
    """Goldmine-scale suggestion from the surviving marks.

    Severity weights length and thickness, not pixel count: one long deep gouge
    matters more than many light hairlines covering the same area. Thresholds are
    UNCALIBRATED -- they need a set of records graded by a human to be meaningful.
    """
    if not scratches:
        return "Near Mint (NM)", 0.0
    area = ring_shape[0] * ring_shape[1]
    severity = sum(s["length"] * max(s["thickness"], 1.0) for s in scratches)
    index = 1000.0 * severity / area
    for limit, name in ((0.5, "Near Mint (NM)"), (2.0, "Very Good Plus (VG+)"),
                        (6.0, "Very Good (VG)"), (15.0, "Good Plus (G+)"),
                        (30.0, "Good (G)"), (60.0, "Fair (F)")):
        if index <= limit:
            return name, index
    return "Poor (P)", index


def main():
    folder = sys.argv[1]
    paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"{len(paths)} photos in {os.path.basename(folder)}")

    shots = []
    for p in paths:
        s = analyse_photo(p)
        print(f"  {os.path.basename(p)[-22:]:>24} | disc r={s['radius']:3d} "
              f"| judgeable {100*s['judgeable'].mean():5.1f}%")
        shots.append(s)

    shots[0]["shift"] = 0
    weak_alignment = False
    for s in shots[1:]:
        shift, sharp = estimate_offset(shots[0]["label"], s["label"])
        s["shift"] = shift
        deg = 360.0 * shift / shots[0]["ring"].shape[1]
        weak = sharp < MIN_SHARPNESS
        weak_alignment |= weak
        print(f"  alignment (via label): {deg:6.1f} deg  sharpness {sharp:.1f}"
              f"{'   << WEAK -- agreement between shots is unreliable' if weak else ''}")

    evidence, confirmed, provisional, coverage = fuse(shots)
    mask, scratches = detector.extract(evidence)

    conf_count = sum(1 for lbl in [0] for _ in [0])  # placeholder replaced below
    n, labels, _, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    high = 0
    for i in range(1, n):
        if np.count_nonzero(confirmed[labels == i]) > 0.3 * np.count_nonzero(labels == i):
            high += 1

    name, index = grade(scratches, shots[0]["ring"].shape, coverage)
    print(f"\n  surface judged: {coverage:.1f}%"
          f"{'   << too low to be reliable' if coverage < MIN_COVERAGE else ''}")
    print(f"  marks found: {len(scratches)}  (high confidence: {high})")
    for s in sorted(scratches, key=lambda x: -x["length"]):
        print(f"    length={s['length']:4d}px  thickness={s['thickness']:4.1f}px"
              f"  angle_to_groove={s.get('angle_to_groove', '?')}")
    print(f"  damage index: {index:.2f}")
    print(f"  suggested grade: {name}")

    out = os.path.join(folder, "analysis")
    os.makedirs(out, exist_ok=True)
    ref = shots[0]
    det = detector.rewrap(mask, ref["inner_px"], ref["center"], ref["radius"],
                          ref["img"].shape[:2])
    vis = ref["img"].copy().astype(np.float32)
    band = cv2.dilate((det > 127).astype(np.uint8), np.ones((9, 9), np.uint8))
    band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.0)
    a = np.clip(band, 0, 1)[..., None] * 0.45
    vis = (vis * (1 - a) + np.array([90, 255, 255], np.float32) * a).astype(np.uint8)
    cv2.imencode(".jpg", vis)[1].tofile(os.path.join(out, "detections.jpg"))
    cv2.imencode(".jpg", ref["ring"])[1].tofile(os.path.join(out, "ring_ref.jpg"))
    cv2.imencode(".jpg", (confirmed * 255).astype(np.uint8))[1].tofile(
        os.path.join(out, "confirmed.jpg"))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
