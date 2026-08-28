# -*- coding: utf-8 -*-
"""
Third and final sweep. Two questions are left open by sweep 2.

1. The outward half of the widening is suspect. Towards the centre there is no
   sharp edge to trip over, but 0.97R sits close to the rim, and if find_disc
   overestimates the radius by a few percent that line lands on the background —
   a bright hard edge, exactly what the detector hunts. So the inward and
   outward halves are measured separately instead of as one move.

2. Where to sit on the threshold axis. Sweep 2 showed the graded shape rule
   dominates the blunt one (same recall, fewer detections), so every variant
   here builds on it.

Usage:  python sweep_recall3.py [cal|val]
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
from sweep_recall2 import extract_graded, detect_with, run

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "gt_new")

BASE = copy.deepcopy(P)
G6 = {"MIN_LEN": 15, "MAX_THICK": 16, "SHORT_LEN": 30, "SHORT_ELONG": 6.0}
G5 = {**G6, "SHORT_ELONG": 5.0}
MILD = {"PCT_STRONG": 99.7, "PCT_WEAK": 99.2}
FULL = {"PCT_STRONG": 99.5, "PCT_WEAK": 99.0}

VARIANTS = [
    ("baseline",                    {}),
    ("g6, band unchanged",          {**G6}),
    ("g6, band in only  .36/.93",   {**G6, "LABEL_R": 0.36}),
    ("g6, band .36/.95",            {**G6, "LABEL_R": 0.36, "OUTER_R": 0.95}),
    ("g6, band .36/.97",            {**G6, "LABEL_R": 0.36, "OUTER_R": 0.97}),
    ("g6 .36/.95 + mild thr",       {**G6, "LABEL_R": 0.36, "OUTER_R": 0.95, **MILD}),
    ("g5 .36/.95 + mild thr",       {**G5, "LABEL_R": 0.36, "OUTER_R": 0.95, **MILD}),
    ("g6 .36/.95 + full thr",       {**G6, "LABEL_R": 0.36, "OUTER_R": 0.95, **FULL}),
]

BASE_FOUND, BASE_SHOWN = 254, 545


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "cal"
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    records = set(split[which])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in records]

    print(f"SET = {which}   {len(rows)} photos\n")
    print(f"{'variant':<28}{'recall':>8}{'found':>7}{'shown':>7}"
          f"{'clean FP':>10}{'cost':>8}")
    print("-" * 68)

    results = []
    for name, changes in VARIANTS:
        P.clear()
        P.update(copy.deepcopy(BASE))
        P.update(changes)
        ex = extract_graded if "SHORT_ELONG" in changes else detector.extract
        t = run(rows, ex)
        recall = 100.0 * t["found"] / max(t["zones"], 1)
        fp = t["fp_clean"] / max(t["clean"], 1)
        gain = t["found"] - BASE_FOUND
        cost = (t["shown"] - BASE_SHOWN) / gain if gain > 0 else float("nan")
        print(f"{name:<28}{recall:>7.1f}%{t['found']:>7}{t['shown']:>7}"
              f"{fp:>9.1f}{cost:>8.1f}")
        results.append({"name": name, "recall": round(recall, 1),
                        "found": t["found"], "shown": t["shown"],
                        "clean_fp": round(fp, 2), "zones": t["zones"],
                        "params": changes})

    with open(os.path.join(HERE, f"frontier_{which}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    P.clear()
    P.update(copy.deepcopy(BASE))
    print(f"\nwrote frontier_{which}.json")


if __name__ == "__main__":
    main()
