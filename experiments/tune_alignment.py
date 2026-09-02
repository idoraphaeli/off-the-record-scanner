# -*- coding: utf-8 -*-
"""Is the cross-shot match window wide because it has to be, or because we are?

Drawing both shots into one unwrapped strip showed matched pairs sitting SIDE BY
SIDE rather than on top of each other, by a couple of degrees -- and the offset
was the same for every pair on a side, which is the signature of a wrong angle
rather than of pairs that do not belong together.

That matters because the window we accept a match in is +/-6 degrees. At a
typical radius six degrees is well over a hundred pixels, against a scratch
about eight pixels wide. A window that loose confirms pairs that are merely
near each other, so "seen in both shots" is worth less than it should be.

So: start from the angle read off the label, then correct it by the MEDIAN
offset of the pairs it produced, and repeat. A median is used rather than a mean
because most detections have no true partner and the ones that pair up by
accident sit anywhere -- they must not be allowed to drag the correction.

The correction is one number per side fitted from many pairs, on top of an
estimate that came from the label and not from the detections, so it cannot
invent agreement out of nothing. Two things are reported to show it did not:
the SPREAD of the offsets around their own median, which stays small only if the
pairs really do move together, and precision against the hand labels at each
window, which is the thing we actually want to improve.

Usage:  python tune_alignment.py [cal|val|both]
"""

import collections
import csv
import json
import os
import sys

import numpy as np

from cross_shot import (RAD_TOL, label_profile, load_labels, polar_detections,
                        rotation_from_label)

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")

SEARCH_W = 8.0       # how far out a pair may be while the angle is still wrong
MIN_PAIRS = 4        # fewer than this and the median is one outlier away from
                     # meaningless -- keep the label's answer instead
ROUNDS = 4
WINDOWS = (6.0, 4.0, 3.0, 2.0, 1.0)


def offsets(a_pts, b_pts, delta, window):
    """For each detection in A, how far its nearest partner in B sits from where
    this rotation says it should be. Detections with no partner say nothing."""
    out = []
    for p in a_pts:
        want = (p["ang"] + delta) % 360.0
        best = None
        for q in b_pts:
            if abs(p["rad"] - q["rad"]) > RAD_TOL:
                continue
            d = (q["ang"] - want + 180.0) % 360.0 - 180.0
            if abs(d) <= window and (best is None or abs(d) < abs(best)):
                best = d
        out.append(best)
    return out


def refine(a_pts, b_pts, delta):
    """The label's angle, corrected by the offset its own pairs still show.

    Returns the corrected angle, how much it moved, and the spread of the
    offsets around their median at the end -- the number that says whether the
    pairs moved together or were never a set.
    """
    start, res = delta, []
    for _ in range(ROUNDS):
        res = [d for d in offsets(a_pts, b_pts, delta, SEARCH_W) if d is not None]
        if len(res) < MIN_PAIRS:
            return start, 0.0, None, 0
        step = float(np.median(res))
        delta = (delta + step) % 360.0
        if abs(step) < 0.05:
            break
    spread = float(np.median(np.abs(np.array(res) - np.median(res))))
    moved = (delta - start + 180.0) % 360.0 - 180.0
    return delta, moved, spread, len(res)


def flags(a_pts, b_pts, delta, window):
    return [d is not None for d in offsets(a_pts, b_pts, delta, window)]


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    sets = ("cal", "val") if which == "both" else (which,)

    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        index = list(csv.DictReader(fh))
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    wanted = set()
    for s in sets:
        wanted |= set(split[s])
    rows = [r for r in index if r["record"] in wanted]

    labels = {}
    for s in sets:
        labels.update(load_labels(s))

    sides = collections.defaultdict(list)
    for r in rows:
        path = os.path.join(PHOTOS, r["photo_file"])
        if not os.path.exists(path):
            continue
        try:
            sides[(r["record"], r["side"])].append(
                (r["pair"], polar_detections(path), label_profile(path)))
        except Exception as exc:
            print(f"  skip {r['pair'][:40]}: {type(exc).__name__}")

    # per photo, per method, per window: which detections were confirmed
    conf = collections.defaultdict(dict)
    moves, spreads = [], []
    aligned = 0

    for key, shots in sorted(sides.items()):
        if len(shots) < 2:
            continue
        for i, (pair_i, pts_i, prof_i) in enumerate(shots):
            got = {(m, w): [False] * len(pts_i)
                   for m in ("label", "refined") for w in WINDOWS}
            ok_any = False
            for j, (pair_j, pts_j, prof_j) in enumerate(shots):
                if i == j:
                    continue
                delta, _ = rotation_from_label(prof_i, prof_j)
                if delta is None:
                    continue
                ok_any = True
                fixed, moved, spread, n = refine(pts_i, pts_j, delta)
                if spread is not None and i < j:
                    moves.append(moved)
                    spreads.append(spread)
                for w in WINDOWS:
                    for m, d in (("label", delta), ("refined", fixed)):
                        for k, f in enumerate(flags(pts_i, pts_j, d, w)):
                            got[(m, w)][k] = got[(m, w)][k] or f
            if ok_any:
                aligned += 1
                conf[pair_i] = (pts_i, got)

    if moves:
        mv = np.abs(np.array(moves))
        sp = np.array(spreads)
        print(f"\nangle correction on {len(mv)} sides")
        print(f"  how far the label's angle moved : median {np.median(mv):.2f} deg"
              f"   90th pct {np.percentile(mv, 90):.2f} deg")
        print(f"  spread of the pairs around it   : median {np.median(sp):.2f} deg")
        print("  (a small spread beside a large correction means the pairs moved")
        print("   together -- one wrong angle, not a bag of unrelated matches)")

    # join the hand-labelled verdicts on by position, exactly as cross_shot does
    tally = collections.Counter()
    for pair, rows_l in labels.items():
        if pair not in conf:
            continue
        pts, got = conf[pair]
        if not pts:
            continue
        arr = np.array([[p["vx"], p["vy"]] for p in pts], float)
        for r in rows_l:
            d = np.hypot(arr[:, 0] - r["cx"], arr[:, 1] - r["cy"])
            k = int(d.argmin())
            if d[k] > 25:
                continue
            real = r["label"] in ("scratch", "dirt")
            for m in ("label", "refined"):
                for w in WINDOWS:
                    g = "confirmed" if got[(m, w)][k] else "once"
                    tally[(m, w, g, "real" if real else "false")] += 1
                    if r["label"] == "scratch":
                        tally[(m, w, g, "scratch")] += 1

    head = (f"\n{'angle':<10}{'window':>8}{'confirmed':>11}{'precision':>11}"
            f"{'seen once':>12}{'precision':>11}{'separation':>12}"
            f"{'scratches kept':>16}")
    print(head)
    print("-" * len(head))
    for m in ("label", "refined"):
        for w in WINDOWS:
            cr = tally[(m, w, "confirmed", "real")]
            cf = tally[(m, w, "confirmed", "false")]
            orr = tally[(m, w, "once", "real")]
            of = tally[(m, w, "once", "false")]
            if not (cr + cf):
                continue
            pc = 100.0 * cr / (cr + cf)
            po = 100.0 * orr / max(orr + of, 1)
            sc = tally[(m, w, "confirmed", "scratch")]
            st = sc + tally[(m, w, "once", "scratch")]
            print(f"{m:<10}{w:>7.0f}d{cr+cf:>11}{pc:>10.0f}%{orr+of:>12}"
                  f"{po:>10.0f}%{pc-po:>11.0f}pt{f'{sc} of {st}':>16}")
    print("\nseparation is what the cross-check is worth: how many points cleaner")
    print("a confirmed detection is than one seen in a single shot.")
    print(f"\nphotos on a side that could be aligned: {aligned}")


if __name__ == "__main__":
    main()
