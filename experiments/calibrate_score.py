# -*- coding: utf-8 -*-
"""Re-calibrate the grade curve for the strict model.

The grade comes from a damage index -- every mark's area, weighted by how much
it looks like a cut rather than a blob, by how many tracks it crosses, and by
whether both shots saw it -- put through

    score = 100 / (1 + index / SCORE_HALF_AT)

SCORE_HALF_AT is the index at which a record scores 50, and it is currently 35.
It was chosen for a detector that paints 14.8 marks a photograph. The strict
model paints 9.6, so the same record produces a smaller index and comes out with
a better grade -- not because it is in better condition, but because we changed
what we count. Left alone, every record in the shop quietly moves up a band.

So the constant is refitted to hold the GRADE DISTRIBUTION where it is. That
choice carries an assumption worth stating: it takes today's grades as correct.
They have never been checked against a human grading the same records by ear or
by eye, so this keeps us consistent with ourselves rather than with the Goldmine
standard. Calibrating against real graded records is a separate job and this
does not replace it.

Usage:  python calibrate_score.py [cal|val|both]
"""

import collections
import json
import math
import os
import sys

import cv2
import numpy as np

import detector
from compare_precision import BAR_NOW, detect_with
from detector import P
from evaluate_frozen import MIN_EXTRA_AREA
from model_v2 import RULES_OFF, align, combined, maps_of
from validate_strict import sides_of

# the grade's own constants, copied from server/scanner/analyze.py
DIRT_ELONG, CUT_ELONG = 4.0, 8.0
DIRT_WEIGHT = 0.30
SPAN_GAIN = 2.0
CONF_SEEN_TWICE, CONF_SEEN_ONCE, CONF_NO_SECOND = 1.16, 0.92, 1.00
SCORE_HALF_AT = 35.0
GRADE_BANDS = [(93, "M- (Near Mint)"), (80, "VG+ (Very Good Plus)"),
               (62, "VG (Very Good)"), (40, "G+ (Good Plus)"),
               (20, "G (Good)"), (0, "P (Poor)")]

RAD_TOL, ANG_TOL, ANG_TOL_RAW = 0.025, 2.0, 6.0


def band(score):
    for floor, name in GRADE_BANDS:
        if score >= floor:
            return name
    return GRADE_BANDS[-1][1]


def marks_of_mask(ring_mask, ring_h, inner, radius):
    """Every mark in a ring mask, measured the way the grade needs.

    Read off the components directly rather than from the detector's own list,
    so that the combined map -- which the detector has no notion of -- is
    measured by exactly the same rule as a single shot.
    """
    n, lab, stats, cent = cv2.connectedComponentsWithStats(
        (ring_mask > 127).astype(np.uint8), connectivity=8)
    ring_area = ring_mask.shape[0] * ring_mask.shape[1]
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_EXTRA_AREA:
            continue
        comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cont, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = max((cv2.arcLength(c, True) for c in cont), default=0)
        length = max(perimeter / 2, max(w, h))
        thickness = area / max(length, 1)
        out.append({
            "length_px": length, "thickness_px": thickness,
            "radial_span_frac": h / max(ring_h, 1),
            "area_frac": (length * max(thickness, 1.0)) / max(ring_area, 1),
            "rad": (float(cent[i][1]) + inner) / max(radius, 1),
            "ang": float(cent[i][0]) / P["POLAR_STEPS"] * 360.0,
        })
    return out


def weight_of(m, conf):
    elong = m["length_px"] / max(m["thickness_px"], 1.0)
    t = (elong - DIRT_ELONG) / (CUT_ELONG - DIRT_ELONG)
    cut = DIRT_WEIGHT + (1.0 - DIRT_WEIGHT) * min(max(t, 0.0), 1.0)
    tracks = 1.0 + SPAN_GAIN * min(max(m["radial_span_frac"], 0.0), 1.0)
    return cut * tracks * conf


def index_of(marks_with_conf):
    return 1000.0 * sum(m["area_frac"] * weight_of(m, c) for m, c in marks_with_conf)


def score_of(index, half_at):
    return 100.0 / (1.0 + index / half_at)


def ring_masks(a, b, delta, combine):
    """The ring-space mask this model would produce for shot A (and for shot B
    when the server model needs both)."""
    keep = {k: P[k] for k in list(RULES_OFF) + list(BAR_NOW)}
    P.update(BAR_NOW)
    try:
        def one(m, other=None, d=None):
            rad = m["radial"] if other is None else combined(m["radial"], other["radial"], d)
            tra = m["tram"] if other is None else combined(m["tram"], other["tram"], d)
            m1, _ = detector.extract(rad, None, m["ring"], m["inner"], m["radius"])
            m2, _ = detector.extract(tra, P["TRAM_MIN_LEN"], m["ring"],
                                     m["inner"], m["radius"])
            return cv2.bitwise_or(m1, m2)
        if combine:
            return one(a, b, delta), None
        return one(a), one(b)
    finally:
        P.update(keep)


def confirm_flags(ma, mb, delta, window):
    flags = []
    for m in ma:
        want = (m["ang"] + delta) % 360.0
        flags.append(any(abs(m["rad"] - o["rad"]) <= RAD_TOL
                         and abs((o["ang"] - want + 180.0) % 360.0 - 180.0) <= window
                         for o in mb))
    return flags


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    sets = ("cal", "val") if which == "both" else (which,)
    rows = []

    for s in sets:
        for rec, side, shots in sides_of(s):
            (_, path_a), (_, path_b) = shots
            try:
                a, b = maps_of(path_a), maps_of(path_b)
            except Exception:
                continue
            delta, _, _ = align(a, b)
            if delta is None:
                continue          # the strict model refuses this side anyway

            # --- the server today: each shot on its own, then cross-checked ---
            mask_a, mask_b = ring_masks(a, b, delta, combine=False)
            ma = marks_of_mask(mask_a, a["ring"].shape[0], a["inner"], a["radius"])
            mb = marks_of_mask(mask_b, b["ring"].shape[0], b["inner"], b["radius"])
            in_b = confirm_flags(ma, mb, delta, ANG_TOL)
            in_a = confirm_flags(mb, ma, -delta, ANG_TOL)
            server = [(m, CONF_SEEN_TWICE if f else CONF_SEEN_ONCE)
                      for m, f in zip(ma, in_b)]
            server += [(m, CONF_SEEN_ONCE) for m, f in zip(mb, in_a) if not f]

            # --- the strict model: one set, every mark seen in both ---
            mask_c, _ = ring_masks(a, b, delta, combine=True)
            strict = [(m, CONF_SEEN_TWICE)
                      for m in marks_of_mask(mask_c, a["ring"].shape[0],
                                             a["inner"], a["radius"])]

            rows.append((rec, side, index_of(server), index_of(strict)))
            print(f"  {rec[:26]:<28}{side}   server {rows[-1][2]:>7.2f}"
                  f"   strict {rows[-1][3]:>7.2f}")

    # keep the measured indices, so the constant can be refitted later without
    # paying for the detection again
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "score_indices.json"), "w", encoding="utf-8") as fh:
        json.dump([{"record": r, "side": s_, "server": o, "strict": n}
                   for r, s_, o, n in rows], fh, ensure_ascii=False)

    pairs = [(o, n) for _, _, o, n in rows if o > 0.01 and n > 0.01]
    ratios = np.array([n / o for o, n in pairs])
    k = float(np.median(ratios))
    proposed = SCORE_HALF_AT * k

    print(f"\n{len(rows)} sides, {len(pairs)} with damage on both")
    print(f"  the strict model's index is {k:.3f} of the server's (median)")
    print(f"  quartiles {np.percentile(ratios,25):.2f} to {np.percentile(ratios,75):.2f}")
    print(f"\n  SCORE_HALF_AT   now {SCORE_HALF_AT:.0f}"
          f"   ->   proposed {proposed:.1f}")

    # what the grades actually do, which is the thing being held still
    head = f"\n{'grade':<26}{'server':>9}{'strict, 35':>13}{'strict, refitted':>19}"
    print(head)
    print("-" * len(head))
    old = collections.Counter(band(score_of(o, SCORE_HALF_AT)) for _, _, o, _ in rows)
    naive = collections.Counter(band(score_of(n, SCORE_HALF_AT)) for _, _, _, n in rows)
    fixed = collections.Counter(band(score_of(n, proposed)) for _, _, _, n in rows)
    for _, name in GRADE_BANDS:
        if old[name] or naive[name] or fixed[name]:
            print(f"{name:<26}{old[name]:>9}{naive[name]:>13}{fixed[name]:>19}")
    # the median ratio holds the AVERAGE record still; sweeping for the constant
    # that moves the fewest records holds them still one by one, which is the
    # thing a seller actually notices
    best, best_moved = proposed, None
    for h in np.arange(5.0, 35.01, 0.1):
        m = sum(1 for _, _, o, n in rows
                if band(score_of(o, SCORE_HALF_AT)) != band(score_of(n, h)))
        if best_moved is None or m < best_moved:
            best, best_moved = float(h), m
    print(f"  the constant that moves the fewest sides is {best:.1f}"
          f", moving {best_moved} of {len(rows)}")

    moved = sum(1 for _, _, o, n in rows
                if band(score_of(o, SCORE_HALF_AT)) != band(score_of(n, proposed)))
    print(f"\n  sides that change band after refitting: {moved} of {len(rows)}")
    print(f"  without refitting                     : "
          f"{sum(1 for _, _, o, n in rows if band(score_of(o, SCORE_HALF_AT)) != band(score_of(n, SCORE_HALF_AT)))}")


if __name__ == "__main__":
    main()
