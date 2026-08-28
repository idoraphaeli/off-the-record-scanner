# -*- coding: utf-8 -*-
"""
Sweep the thresholds, now that the miss diagnosis says they are the wall.

Across both sets, 80% of missed scratches produced a response the detector
measured and then discarded as not standing out enough, and only 2 of 527 left
no trace at all. The shape filter, which blocked this same move last time, now
accounts for 7%.

Two levers are swept. The PERCENTILES adapt to each image's own noise, so
lowering them admits more of whatever that image contains. The FLOOR is an
absolute cut underneath them, and it is what keeps a genuinely clean record
clean — on a quiet image the percentiles alone would always "find" the top 0.1%
of pure noise, so if the floor is what binds, moving the percentiles does
nothing.

The groove rule is swept alongside because it is the second-largest fixable
category: a mark running ALONG the grooves is rejected as a groove highlight
unless it is very long, and 37 real scratches across both sets were lost that
way.

Detections on photos with no marks at all are reported separately: those records
were judged undamaged by eye, so anything found there is the cost.

Usage:  python sweep_thresholds.py [cal|val]
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

# The first sweep found the two thresholds are gates in SERIES: opening either
# one alone gains 3-10 points, opening both gains 21. So the interesting curve
# runs through the middle of the square, not along its edges, and these steps
# walk it looking for where the price per recovered scratch turns.
VARIANTS = [
    ("current  99.5 / 99.0  floor 35", {}),
    ("99.45 / 98.9   floor 32", {"PCT_STRONG": 99.45, "PCT_WEAK": 98.9,
                                 "THR_FLOOR": 32}),
    ("99.4 / 98.8    floor 30", {"PCT_STRONG": 99.4, "PCT_WEAK": 98.8,
                                 "THR_FLOOR": 30}),
    ("99.3 / 98.7    floor 30", {"PCT_STRONG": 99.3, "PCT_WEAK": 98.7,
                                 "THR_FLOOR": 30}),
    ("99.3 / 98.7    floor 25", {"PCT_STRONG": 99.3, "PCT_WEAK": 98.7,
                                 "THR_FLOOR": 25}),
    ("99.2 / 98.4    floor 28", {"PCT_STRONG": 99.2, "PCT_WEAK": 98.4,
                                 "THR_FLOOR": 28}),
    ("99.2 / 98.4    floor 25", {"PCT_STRONG": 99.2, "PCT_WEAK": 98.4,
                                 "THR_FLOOR": 25}),
    ("99.0 / 98.0    floor 30", {"PCT_STRONG": 99.0, "PCT_WEAK": 98.0,
                                 "THR_FLOOR": 30}),
    ("99.0 / 98.0    floor 25", {"PCT_STRONG": 99.0, "PCT_WEAK": 98.0,
                                 "THR_FLOOR": 25}),
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
    which = sys.argv[1] if len(sys.argv) > 1 else "val"
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    records = set(split[which])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in records]

    print(f"SET = {which}   {len(records)} records, {len(rows)} photos")
    print("cost = extra detections per extra scratch recovered, vs current\n")
    print(f"{'setting':<38}{'recall':>8}{'found':>7}{'shown':>7}"
          f"{'per clean':>11}{'cost':>7}")
    print("-" * 78)

    base = None
    for name, changes in VARIANTS:
        P.clear()
        P.update(copy.deepcopy(NOW))
        P.update(changes)
        t = run(rows)
        recall = 100.0 * t["found"] / max(t["zones"], 1)
        per_clean = t["fp"] / max(t["clean"], 1)
        if base is None:
            base = t
            cost = "-"
        else:
            gain = t["found"] - base["found"]
            cost = f"{(t['shown'] - base['shown']) / gain:.0f}" if gain > 0 else "-"
        print(f"{name:<38}{recall:>7.1f}%{t['found']:>7}{t['shown']:>7}"
              f"{per_clean:>11.1f}{cost:>7}")

    P.clear()
    P.update(copy.deepcopy(NOW))


if __name__ == "__main__":
    main()
