# -*- coding: utf-8 -*-
"""
Should the match window widen towards the rim, and what does the cross-check
look like used as a confidence level rather than a filter?

The physics pulls both ways, which is why this is measured rather than argued.
The rotation is uncertain by about 3 degrees, and that is an ANGULAR error — a
window measured in degrees already covers it identically at every radius. But
the disc centre is uncertain by several pixels, and that produces an angular
error which is LARGER near the middle: ten pixels at a quarter of the radius is
2.3 degrees, the same ten pixels at the rim is 0.9. So centre error asks for a
wider window INWARD, the opposite of the intuition that damage further out is
less certain.

Both directions are tried, along with a constant-arc-length window, which is
what "the same physical distance everywhere" actually means.

Run on a third of the calibration records — enough to rank the schemes, and
quick enough to iterate on.

Usage:  python tune_radial.py [how many records]
"""

import collections
import csv
import json
import math
import os
import sys

import numpy as np

import cross_shot as cs

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")

RAD_TOL = 0.025
INNER, OUTER = 0.36, 0.95      # the band the detector analyses


def ramp(at_inner, at_outer):
    """Angular window that moves linearly with radius across the band."""
    def f(r):
        t = min(max((r - INNER) / (OUTER - INNER), 0.0), 1.0)
        return at_inner + t * (at_outer - at_inner)
    return f


def flat(deg):
    return lambda r: deg


def arc(mm_like):
    """Constant arc LENGTH: the window covers the same distance along the
    groove at every radius, so the angle shrinks as the radius grows."""
    return lambda r: mm_like / max(r, 0.05)


SCHEMES = [
    ("flat 6 deg (current)", flat(6.0)),
    ("flat 4 deg", flat(4.0)),
    ("wider OUT  3 -> 9", ramp(3.0, 9.0)),
    ("wider OUT  2 -> 12", ramp(2.0, 12.0)),
    ("wider OUT  4 -> 8", ramp(4.0, 8.0)),
    ("wider IN   9 -> 3", ramp(9.0, 3.0)),
    ("wider IN   12 -> 2", ramp(12.0, 2.0)),
    ("constant arc length", arc(3.9)),
]


def confirm(p, others, delta, ang_of):
    tol = ang_of(p["rad"])
    want = (p["ang"] + delta) % 360.0
    for q in others:
        if abs(p["rad"] - q["rad"]) > RAD_TOL:
            continue
        if abs((q["ang"] - want + 180.0) % 360.0 - 180.0) <= tol:
            return True
    return False


def main():
    n_rec = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    cal = sorted(split["cal"])
    step = max(len(cal) // n_rec, 1)
    chosen = set(cal[::step][:n_rec])          # spread, not the first ten

    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in chosen]
    labels = cs.load_labels("cal")

    print(f"{len(chosen)} of {len(cal)} calibration records, "
          f"{len(rows)} photos\ndetecting once; the schemes are free\n")

    sides = collections.defaultdict(list)
    for r in rows:
        path = os.path.join(PHOTOS, r["photo_file"])
        if not os.path.exists(path):
            continue
        try:
            sides[(r["record"], r["side"])].append(
                (r["pair"], cs.polar_detections(path), cs.label_profile(path)))
        except Exception:
            continue

    plan = []
    for key, shots in sorted(sides.items()):
        if len(shots) < 2:
            continue
        for i, (pair_i, pts_i, prof_i) in enumerate(shots):
            partners = []
            for j, (pair_j, pts_j, prof_j) in enumerate(shots):
                if i == j:
                    continue
                d, _ = cs.rotation_from_label(prof_i, prof_j)
                if d is not None:
                    partners.append((pts_j, d))
            if partners:
                plan.append((pair_i, pts_i, partners))
    print(f"  {len(plan)} photos with an aligned partner\n")

    head = (f"{'window scheme':<24}{'CONFIRMED':>10}{'prec':>7}"
            f"{'of real':>9}{'NOT CONF':>10}{'prec':>7}{'lift':>7}"
            f"{'scratches kept':>16}")
    print(head)
    print("-" * len(head))

    for name, ang_of in SCHEMES:
        t = collections.Counter()
        for pair, pts, partners in plan:
            rows_l = labels.get(pair)
            if not rows_l or not pts:
                continue
            arr = np.array([[p["vx"], p["vy"]] for p in pts], float)
            ok = [any(confirm(p, o, d, ang_of) for o, d in partners) for p in pts]
            for r in rows_l:
                dd = np.hypot(arr[:, 0] - r["cx"], arr[:, 1] - r["cy"])
                k = int(dd.argmin())
                if dd[k] > 25:
                    continue
                g = "conf" if ok[k] else "once"
                t[(g, "real" if r["label"] in ("scratch", "dirt") else "false")] += 1
                # scratches tracked on their own: they are what recall is
                # measured against, so this is the cost if the cross-check is
                # ever used to DISCARD rather than to rank confidence
                if r["label"] == "scratch":
                    t[(g, "scratch")] += 1

        cr, cf = t[("conf", "real")], t[("conf", "false")]
        orr, of = t[("once", "real")], t[("once", "false")]
        if not (cr + cf) or not (orr + of):
            continue
        base = 100.0 * (cr + orr) / (cr + cf + orr + of)
        pc = 100.0 * cr / (cr + cf)
        po = 100.0 * orr / (orr + of)
        sc = t[("conf", "scratch")]
        sc_all = sc + t[("once", "scratch")]
        print(f"{name:<24}{cr + cf:>10}{pc:>6.0f}%"
              f"{100*cr/max(cr+orr,1):>8.0f}%{orr + of:>10}{po:>6.0f}%"
              f"{pc - base:>6.0f}{f'{sc} of {sc_all}':>16}")

    print(f"\n  'of real' = the share of all REAL detections that end up "
          f"confirmed.\n  'lift'    = precision of the confirmed group over the "
          f"set as a whole.")


if __name__ == "__main__":
    main()
