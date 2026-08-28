# -*- coding: utf-8 -*-
"""
Second sweep, driven by what the first one measured rather than by the
attribution table.

The attribution said 57% of missed scratches die at a threshold, so lowering the
thresholds looked like the big win. It is not: on its own it recovered 6 of 347
misses while adding 331 detections. The attribution names the FIRST stage that
could drop a scratch, not the only one — admit those 121 weak regions and the
shape filter kills them at the next step. That is why the same threshold change
is worth +61 once MIN_LEN is relaxed. Order matters: shape first, thresholds
after.

Relaxing MIN_LEN to 15 outright also admits every short stubby blob. A real
short scratch is a thin sharp line; a speck of dirt is stubby. So the graded
rule below asks a SHORT component to be more elongated than a long one, instead
of asking every component for the same elongation and then banning short ones
wholesale.

extract() is monkey-patched here rather than edited in detector.py, so nothing
is promoted into the detector until a variant has earned it.

Usage:  python sweep_recall2.py [cal|val]
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
BASE_RECALL, BASE_FOUND, BASE_SHOWN = 40.3, 254, 545

A = {"LABEL_R": 0.36, "OUTER_R": 0.97}
B = {"MIN_LEN": 15, "MAX_THICK": 16}
GRADED = {"SHORT_LEN": 30}            # below this length the strict rule applies

VARIANTS = [
    ("baseline",                 {}),
    ("A+B  (best of sweep 1)",   {**A, **B}),
    ("A + short len only",       {**A, "MIN_LEN": 15}),
    ("A + thick only",           {**A, "MAX_THICK": 16}),
    ("A+B graded elong 4",       {**A, **B, **GRADED, "SHORT_ELONG": 4.0}),
    ("A+B graded elong 6",       {**A, **B, **GRADED, "SHORT_ELONG": 6.0}),
    ("A+B graded 4 + mild thr",  {**A, **B, **GRADED, "SHORT_ELONG": 4.0,
                                  "PCT_STRONG": 99.7, "PCT_WEAK": 99.2}),
    ("A+B graded 4 + full thr",  {**A, **B, **GRADED, "SHORT_ELONG": 4.0,
                                  "PCT_STRONG": 99.5, "PCT_WEAK": 99.0}),
    ("A+B graded 6 + full thr",  {**A, **B, **GRADED, "SHORT_ELONG": 6.0,
                                  "PCT_STRONG": 99.5, "PCT_WEAK": 99.0}),
]


def extract_graded(smap, min_len=None):
    """extract() with one change: the elongation demanded of a component depends
    on its length. Everything else is copied so the comparison stays honest."""
    min_len = P["MIN_LEN"] if min_len is None else min_len
    judgeable = smap[smap > 0]
    if judgeable.size < 1000:
        return np.zeros_like(smap), []
    thr_strong = max(float(np.percentile(judgeable, P["PCT_STRONG"])), P["THR_FLOOR"])
    thr_weak = max(float(np.percentile(judgeable, P["PCT_WEAK"])), P["THR_FLOOR"] / 2)

    weak = (smap > thr_weak).astype(np.uint8)
    strong = smap >= thr_strong
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    seeds = set(np.unique(labels[strong])) - {0}
    binary = np.isin(labels, list(seeds)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones(P["CLOSE"], np.uint8))
    binary = detector._link_collinear(binary)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros_like(binary)
    scratches = []
    short_len = P.get("SHORT_LEN", 0)
    short_elong = P.get("SHORT_ELONG", P["MIN_ELONG"])
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = max((cv2.arcLength(c, True) for c in contours), default=0)
        length = max(perimeter / 2, max(w, h))
        thickness = area / max(length, 1)
        if length < min_len or thickness > P["MAX_THICK"]:
            continue
        need = short_elong if length < short_len else P["MIN_ELONG"]
        if length / max(thickness, 1) < need:
            continue
        angle = detector._axis_angle_deg(comp)
        if angle < P["GROOVE_TOL_DEG"] and length < P["GROOVE_KEEP_LEN"]:
            continue
        mask[labels == i] = 255
        scratches.append({"length": int(length)})
    return mask, scratches


def detect_with(photo, ex):
    img = detector.load_image(photo)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner, outer = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer]
    radial, tram = detector.scratch_map(ring)[:2]
    m1, _ = ex(radial)
    m2, _ = ex(tram, min_len=P["TRAM_MIN_LEN"])
    det = detector.rewrap(cv2.bitwise_or(m1, m2), inner, center, radius, gray.shape)
    return img, det


def run(rows, ex):
    tot = dict(zones=0, found=0, shown=0, extra=0, marked=0,
               fp_clean=0, clean=0)
    for r in rows:
        gt_path = os.path.join(GT, r["pair"] + ".png")
        photo = os.path.join(PHOTOS, r["photo_file"])
        if not (os.path.exists(gt_path) and os.path.exists(photo)):
            continue
        img, det = detect_with(photo, ex)
        gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        found, zones, extra, shown = score(det, gt)
        tot["zones"] += zones
        tot["found"] += found
        tot["shown"] += shown
        if zones:
            tot["extra"] += extra
            tot["marked"] += 1
        else:
            tot["fp_clean"] += shown
            tot["clean"] += 1
    return tot


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "cal"
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    records = set(split[which])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in records]

    print(f"SET = {which}   {len(rows)} photos")
    print("cost = extra detections added per extra scratch recovered, "
          "vs the baseline\n")
    print(f"{'variant':<26}{'recall':>8}{'found':>7}{'shown':>7}"
          f"{'clean FP':>10}{'cost':>8}")
    print("-" * 66)

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
        print(f"{name:<26}{recall:>7.1f}%{t['found']:>7}{t['shown']:>7}"
              f"{fp:>9.1f}{cost:>8.1f}")

    P.clear()
    P.update(copy.deepcopy(BASE))


if __name__ == "__main__":
    main()
