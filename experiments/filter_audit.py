# -*- coding: utf-8 -*-
"""Why do candidates die? Count components surviving each filter stage, and
report the measurements of components that overlap a GT zone (those are the
ones we WANT to keep -- their stats tell us where to set the limits)."""

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


def ring_of(mask2d, center, radius, inner_px, outer_px, nearest=True):
    flags = cv2.WARP_POLAR_LINEAR | (cv2.INTER_NEAREST if nearest else 0)
    polar = cv2.warpPolar(mask2d, (radius, P["POLAR_STEPS"]), center, radius, flags)
    return cv2.transpose(polar)[inner_px:outer_px]


split = json.load(open(os.path.join(HERE, "split.json"), encoding="utf-8"))
hit_stats, all_counts = [], {"raw": 0, "len": 0, "thick": 0, "elong": 0}

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
    smap = detector.scratch_map(ring)

    gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_ring = ring_of(gt, center, radius, inner_px, outer_px) > 127

    judgeable = smap[smap > 0]
    if judgeable.size < 1000:
        continue
    thr_s = max(float(np.percentile(judgeable, P["PCT_STRONG"])), P["THR_FLOOR"])
    thr_w = max(float(np.percentile(judgeable, P["PCT_WEAK"])), P["THR_FLOOR"] / 2)
    weak = (smap > thr_w).astype(np.uint8)
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    seeds = set(np.unique(labels[smap >= thr_s])) - {0}
    binary = np.isin(labels, list(seeds)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones(P["CLOSE"], np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        per = max((cv2.arcLength(c, True) for c in cs), default=0)
        length = max(per / 2, max(w, h))
        thick = area / max(length, 1)
        elong = length / max(thick, 1)
        all_counts["raw"] += 1
        if length >= P["MIN_LEN"]:
            all_counts["len"] += 1
        if thick <= P["MAX_THICK"]:
            all_counts["thick"] += 1
        if elong >= P["MIN_ELONG"]:
            all_counts["elong"] += 1
        overlaps_gt = np.count_nonzero(gt_ring[labels == i]) > 10
        if overlaps_gt:
            hit_stats.append((length, thick, elong, area))

print("component counts across cal set:", all_counts)
if hit_stats:
    arr = np.array(hit_stats)
    print(f"\ncomponents overlapping a GT zone: {len(arr)}")
    for idx, label in enumerate(["length", "thickness", "elongation", "area"]):
        v = arr[:, idx]
        print(f"  {label:>11}: min={v.min():7.1f} p25={np.percentile(v,25):7.1f}"
              f" median={np.median(v):7.1f} p75={np.percentile(v,75):7.1f} max={v.max():7.1f}")
    passing = ((arr[:, 0] >= P["MIN_LEN"]) & (arr[:, 1] <= P["MAX_THICK"])
               & (arr[:, 2] >= P["MIN_ELONG"])).sum()
    print(f"  of these, passing current shape filter: {passing}/{len(arr)}")
