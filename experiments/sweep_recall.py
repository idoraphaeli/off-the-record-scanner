# -*- coding: utf-8 -*-
"""
Test the three fixes the miss-attribution points at, individually and together.

why_missed.py found that of 347 missed scratches only ONE produced no response
at all — the feature sees almost every scratch, and they are lost afterwards:

    57%  thresholding (below the flood level, or never reaching a seed)
    24%  the shape filter (too short, too thick)
    15%  cropped out of the analysed band before anything ran

So the candidates are: widen the band, relax the shape limits, lower the
thresholds. Each is scored with the frozen evaluator, and false positives on
CLEAN photos are reported alongside recall, since any relaxation buys recall
with them.

Usage:  python sweep_recall.py [cal|val]
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

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")

BASE = copy.deepcopy(P)

VARIANTS = [
    ("baseline",                     {}),
    ("A widen band",                 {"LABEL_R": 0.36, "OUTER_R": 0.97}),
    ("B relax shape",                {"MIN_LEN": 15, "MAX_THICK": 16}),
    ("C lower thresholds",           {"PCT_STRONG": 99.5, "PCT_WEAK": 99.0}),
    ("A+B",                          {"LABEL_R": 0.36, "OUTER_R": 0.97,
                                      "MIN_LEN": 15, "MAX_THICK": 16}),
    ("A+B+C",                        {"LABEL_R": 0.36, "OUTER_R": 0.97,
                                      "MIN_LEN": 15, "MAX_THICK": 16,
                                      "PCT_STRONG": 99.5, "PCT_WEAK": 99.0}),
    ("A+B+C stronger",               {"LABEL_R": 0.36, "OUTER_R": 0.97,
                                      "MIN_LEN": 12, "MAX_THICK": 18,
                                      "PCT_STRONG": 99.0, "PCT_WEAK": 98.0}),
]


def detect_with(photo):
    img = detector.load_image(photo)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner, outer = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer]
    radial, tram = detector.scratch_map(ring)[:2]
    m1, _ = detector.extract(radial)
    m2, _ = detector.extract(tram, min_len=P["TRAM_MIN_LEN"])
    det = detector.rewrap(cv2.bitwise_or(m1, m2), inner, center, radius, gray.shape)
    return img, det


def run(rows):
    tot = dict(zones=0, found=0, shown=0, fp_clean=0, clean_photos=0)
    for r in rows:
        gt_path = os.path.join(GT, r["pair"] + ".png")
        photo = os.path.join(PHOTOS, r["photo_file"])
        if not (os.path.exists(gt_path) and os.path.exists(photo)):
            continue
        img, det = detect_with(photo)
        gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        found, zones, extra, shown = score(det, gt)
        tot["zones"] += zones
        tot["found"] += found
        tot["shown"] += shown
        if zones == 0:
            tot["fp_clean"] += shown
            tot["clean_photos"] += 1
    return tot


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "cal"
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    records = set(split[which])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in records]

    print(f"SET = {which}   {len(rows)} photos\n")
    print(f"{'variant':<22}{'recall':>9}{'found':>8}{'shown':>8}"
          f"{'clean FP/photo':>16}")
    print("-" * 64)

    for name, changes in VARIANTS:
        P.clear()
        P.update(copy.deepcopy(BASE))
        P.update(changes)
        t = run(rows)
        recall = 100.0 * t["found"] / max(t["zones"], 1)
        fp = t["fp_clean"] / max(t["clean_photos"], 1)
        print(f"{name:<22}{recall:>8.1f}%{t['found']:>8}{t['shown']:>8}{fp:>16.1f}")

    P.clear()
    P.update(copy.deepcopy(BASE))


if __name__ == "__main__":
    main()
