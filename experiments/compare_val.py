# -*- coding: utf-8 -*-
"""
Old settings versus new, on the validation records only.

The new numbers were chosen on the calibration set, so quoting them alone proves
nothing: any set of parameters can be made to look good on the data used to pick
them. What matters is the same comparison on records that took no part in the
choice. The old values are restored explicitly here rather than read from the
detector, which now holds the new ones.

Usage:  python compare_val.py [cal|val|test]
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

NEW = copy.deepcopy(P)
# SHORT_LEN 0 makes the graded rule inert, restoring the single flat elongation
OLD = {**NEW, "LABEL_R": 0.40, "OUTER_R": 0.93, "MIN_LEN": 30, "MAX_THICK": 12,
       "SHORT_LEN": 0, "PCT_STRONG": 99.8, "PCT_WEAK": 99.5}


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

    print(f"SET = {which}   {len(records)} records, {len(rows)} photos\n")
    print(f"{'settings':<12}{'recall':>9}{'found':>8}{'shown':>8}"
          f"{'per clean photo':>18}")
    print("-" * 56)
    for name, params in (("old", OLD), ("new", NEW)):
        P.clear()
        P.update(copy.deepcopy(params))
        t = run(rows)
        recall = 100.0 * t["found"] / max(t["zones"], 1)
        print(f"{name:<12}{recall:>8.1f}%{t['found']:>8}{t['shown']:>8}"
              f"{t['fp']/max(t['clean'],1):>17.1f}")
    P.clear()
    P.update(copy.deepcopy(NEW))


if __name__ == "__main__":
    main()
