# -*- coding: utf-8 -*-
"""
How tight should the cross-shot match be?

Confirmation currently accepts a partner within 6 degrees and 2.5% of the disc
radius. That window was picked before the alignment was measured, and it is far
looser than the alignment deserves: matching the label's print puts the rotation
within 3 degrees of what the hand marks say, median, on every side tested.

A loose window costs precision directly. With ~40 detections in the other shot,
a 6-degree window gives any detection about an 11% chance of finding a partner
by pure coincidence — and a coincidence confirms a reflection just as happily as
a scratch. Tightening should raise the confirmed group's precision and lower how
much of it survives; this measures where that trade turns.

The detections and rotations are computed once and reused, so the sweep itself
is instant — the expensive pass is the detector, and it does not depend on the
window at all.

Usage:  python tune_crosscheck.py [cal|val|both]
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

WINDOWS = [
    (0.025, 6.0), (0.025, 4.0), (0.020, 4.0), (0.015, 3.0),
    (0.012, 2.5), (0.010, 2.0), (0.008, 1.5),
]


def confirm(p, others, delta, rad_tol, ang_tol):
    want = (p["ang"] + delta) % 360.0
    for q in others:
        if abs(p["rad"] - q["rad"]) > rad_tol:
            continue
        if abs((q["ang"] - want + 180.0) % 360.0 - 180.0) <= ang_tol:
            return True
    return False


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    sets = ("cal", "val") if which == "both" else (which,)

    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    records = set()
    for s in sets:
        records |= set(split[s])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in records]

    labels = {}
    for s in sets:
        labels.update(cs.load_labels(s))

    print("detecting once; the sweep afterwards is free\n")
    sides = collections.defaultdict(list)
    for r in rows:
        path = os.path.join(PHOTOS, r["photo_file"])
        if not os.path.exists(path):
            continue
        try:
            pts = cs.polar_detections(path)
            prof = cs.label_profile(path)
        except Exception:
            continue
        sides[(r["record"], r["side"])].append((r["pair"], pts, prof))
    print(f"  {sum(len(v) for v in sides.values())} photos, "
          f"{sum(1 for v in sides.values() if len(v) > 1)} sides with a partner")

    # rotations, computed once — they do not depend on the match window
    plan = []
    for key, shots in sorted(sides.items()):
        if len(shots) < 2:
            continue
        for i, (pair_i, pts_i, prof_i) in enumerate(shots):
            partners = []
            for j, (pair_j, pts_j, prof_j) in enumerate(shots):
                if i == j:
                    continue
                delta, _ = cs.rotation_from_label(prof_i, prof_j)
                if delta is not None:
                    partners.append((pts_j, delta))
            if partners:
                plan.append((pair_i, pts_i, partners))

    head = (f"\n{'window':<16}{'confirmed':>11}{'kept':>7}{'PRECISION':>11}"
            f"{'once':>7}{'prec':>7}{'scratches':>11}")
    print(head)
    print("-" * len(head))

    for rad_tol, ang_tol in WINDOWS:
        t = collections.Counter()
        for pair, pts, partners in plan:
            rows_l = labels.get(pair)
            if not rows_l or not pts:
                continue
            arr = np.array([[p["vx"], p["vy"]] for p in pts], float)
            ok = [any(confirm(p, o, d, rad_tol, ang_tol) for o, d in partners)
                  for p in pts]
            for r in rows_l:
                dd = np.hypot(arr[:, 0] - r["cx"], arr[:, 1] - r["cy"])
                k = int(dd.argmin())
                if dd[k] > 25:
                    continue
                g = "conf" if ok[k] else "once"
                t[(g, "real" if r["label"] in ("scratch", "dirt") else "false")] += 1
                if r["label"] == "scratch":
                    t[(g, "scratch")] += 1

        cr, cf = t[("conf", "real")], t[("conf", "false")]
        orr, of = t[("once", "real")], t[("once", "false")]
        tot = cr + cf + orr + of
        if not tot or not (cr + cf):
            continue
        sc, sc_all = t[("conf", "scratch")], t[("conf", "scratch")] + t[("once", "scratch")]
        print(f"{rad_tol:.3f}R / {ang_tol:>4.1f}deg{cr + cf:>11}"
              f"{100*(cr+cf)/tot:>6.0f}%{100*cr/(cr+cf):>10.0f}%"
              f"{orr + of:>7}{100*orr/max(orr+of,1):>6.0f}%"
              f"{sc:>6} of {sc_all:<4}")

    print(f"\n  'kept' is the share of all detections that survive as confirmed.")
    print(f"  A window of {WINDOWS[0][1]:.0f} deg gives any detection roughly an")
    print(f"  11% chance of a partner by coincidence; 2 deg gives about 3%.")


if __name__ == "__main__":
    main()
