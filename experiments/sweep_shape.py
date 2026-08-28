# -*- coding: utf-8 -*-
"""
Sweep the SHAPE filters, at the thresholds just chosen.

The shape limits were last tuned before the ink leak was found, so they were
fitted to a detector that could partly see the answer sheet. They have never
been re-examined on clean data. They must also be swept AFTER the thresholds
rather than beside them: the two interact hard — in the first round, easing the
thresholds gained almost nothing because the shape filter destroyed whatever
came through, so a shape answer measured at the old thresholds would not hold at
the new ones.

The steps are aimed by the miss diagnosis, which attributed the surviving
shape-caused misses on calibration as: groove-aligned 8.0%, not elongated 4.9%,
outside the analysed band 4.9%, too short 1.7%. MIN_LEN is therefore left alone —
there is almost nothing left behind it.

Usage:  python sweep_shape.py [cal|val]
"""

import copy
import csv
import json
import os
import sys

import cv2
import numpy as np

import detector
from detector import P
from evaluate_frozen import score
from sweep_recall2 import detect_with

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")

NOW = copy.deepcopy(P)

VARIANTS = [
    ("current", {}),
    # groove rule: the largest remaining shape category. A mark running ALONG
    # the grooves is thrown out as a groove highlight unless it is very long.
    ("groove tol 12 -> 8", {"GROOVE_TOL_DEG": 8}),
    ("groove tol 12 -> 6", {"GROOVE_TOL_DEG": 6}),
    ("groove keep 250 -> 150", {"GROOVE_KEEP_LEN": 150}),
    ("groove tol 8 + keep 150", {"GROOVE_TOL_DEG": 8,
                                 "GROOVE_KEEP_LEN": 150}),
    # elongation: how much longer than wide a component must be
    ("short elong 6.0 -> 4.5", {"SHORT_ELONG": 4.5}),
    ("min elong 2.5 -> 2.0", {"MIN_ELONG": 2.0}),
    ("short 4.5 + min 2.0", {"SHORT_ELONG": 4.5, "MIN_ELONG": 2.0}),
    # the analysed band, widened a little at both ends
    ("band .36/.95 -> .34/.96", {"LABEL_R": 0.34, "OUTER_R": 0.96}),
    ("max thick 16 -> 22", {"MAX_THICK": 22}),
]


def run(rows):
    tot = dict(zones=0, found=0, shown=0, fp=0, clean=0)
    for r in rows:
        gt_path = os.path.join(GT, r["pair"] + ".png")
        photo = os.path.join(PHOTOS, r["photo_file"])
        if not (os.path.exists(gt_path) and os.path.exists(photo)):
            continue
        img, det = detect_with(photo, detector.extract)
        gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        found, zones, extra, shown = score(det, gt)
        tot["zones"] += zones
        tot["found"] += found
        tot["shown"] += shown
        if zones == 0:
            tot["fp"] += shown
            tot["clean"] += 1
    return tot


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "cal"
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    records = set(split[which])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in records]

    print(f"SET = {which}   {len(records)} records, {len(rows)} photos")
    print(f"thresholds fixed at {NOW['PCT_STRONG']} / {NOW['PCT_WEAK']}"
          f"  floor {NOW['THR_FLOOR']}")
    print("cost = extra detections per extra scratch recovered, vs current\n")
    print(f"{'setting':<30}{'recall':>8}{'found':>7}{'shown':>7}"
          f"{'per clean':>11}{'cost':>7}")
    print("-" * 70)

    base = None
    for name, changes in VARIANTS:
        P.clear()
        P.update(copy.deepcopy(NOW))
        P.update(changes)
        t = run(rows)
        recall = 100.0 * t["found"] / max(t["zones"], 1)
        per_clean = t["fp"] / max(t["clean"], 1)
        if base is None:
            base, cost = t, "-"
        else:
            gain = t["found"] - base["found"]
            cost = f"{(t['shown'] - base['shown']) / gain:.0f}" if gain > 0 else "-"
        print(f"{name:<30}{recall:>7.1f}%{t['found']:>7}{t['shown']:>7}"
              f"{per_clean:>11.1f}{cost:>7}")

    P.clear()
    P.update(copy.deepcopy(NOW))


if __name__ == "__main__":
    main()
