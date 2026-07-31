# -*- coding: utf-8 -*-
"""Per-zone diagnosis: does each human-marked zone carry any detector response,
and how does it compare to that image's threshold? Separates "zone has no
signal" (capture limit) from "signal present but threshold too high" (tuning)."""

import json
import os

import cv2
import numpy as np

import detector
from detector import P

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\User\Desktop\Ido private\Computer science\off-the-record\record_pics"
CLEAN_DIR = next(os.path.join(BASE, d) for d in os.listdir(BASE)
                 if os.path.isdir(os.path.join(BASE, d))
                 and "ללא סימונים" in d and "חדש" in d)
GT_DIR = os.path.join(HERE, "gt")

split = json.load(open(os.path.join(HERE, "split.json"), encoding="utf-8"))
buckets = {"no_signal": 0, "below_thr": 0, "above_thr": 0, "masked_out": 0}

for name in split["cal"]:
    stem = os.path.splitext(name)[0]
    gt_path = os.path.join(GT_DIR, stem + "_mask.png")
    if not os.path.exists(gt_path):
        continue
    img = detector.load_image(os.path.join(CLEAN_DIR, name))
    center, radius = detector.find_disc(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inner_px, outer_px = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner_px:outer_px]
    radial, tram = detector.scratch_map(ring)
    smap = np.maximum(radial, tram)

    judge = smap[smap > 0]
    thr = max(float(np.percentile(judge, P["PCT_STRONG"])), P["THR_FLOOR"]) if judge.size else 99

    gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_ring = cv2.transpose(cv2.warpPolar(
        gt, (radius, P["POLAR_STEPS"]), center, radius,
        cv2.WARP_POLAR_LINEAR | cv2.INTER_NEAREST))[inner_px:outer_px] > 127

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        gt_ring.astype(np.uint8), connectivity=8)
    rows = []
    for i in range(1, n):
        if stats[i][4] < 50:
            continue
        z = smap[labels == i]
        if z.size == 0:
            continue
        dead_frac = float(np.count_nonzero(z == 0)) / z.size
        peak = float(z.max())
        if dead_frac > 0.9:
            buckets["masked_out"] += 1
            verdict = "MASKED"
        elif peak >= thr:
            buckets["above_thr"] += 1
            verdict = "ok"
        elif peak >= thr * 0.5:
            buckets["below_thr"] += 1
            verdict = "BELOW-THR"
        else:
            buckets["no_signal"] += 1
            verdict = "NO-SIGNAL"
        rows.append(f"peak={peak:5.1f}/thr={thr:4.1f} {verdict}")
    print(f"{stem[-22:]:>24} | " + " | ".join(rows))

print("\nzone outcome tally:", buckets)
