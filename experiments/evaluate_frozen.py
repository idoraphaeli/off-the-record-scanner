# -*- coding: utf-8 -*-
"""
FROZEN evaluator for the new dataset. Do not change after the baseline is taken —
a scoring rule that moves with the thing it scores cannot show whether anything
improved.

The annotator drew ALONG each scratch, so a ground-truth stroke marks the
scratch's position directly. Hand-drawn lines wander a little, so a scratch
counts as FOUND when a detection lands within TOLERANCE pixels of the stroke,
not on top of it exactly.

Three numbers are always reported together, because any one of them alone hides
the trade-off:
    recall     of the scratches marked by hand, how many were found
    extra      detections that sit near no marked scratch
    clean FP   detections on photos with no marks at all — the cleanest signal
               there is, since those records were judged undamaged by eye

Usage:  python evaluate_frozen.py [cal|val|test|all]
"""

import collections
import csv
import json
import os
import sys

import cv2
import numpy as np

import detector
from detector import P

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")

TOLERANCE = 30       # px, at the detector's working scale
MIN_HIT_PX = 8       # detection pixels near a stroke for it to count as found
MIN_EXTRA_AREA = 40  # smaller stray detections are ignored as speck-level


def load_index():
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def detect(path):
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner, outer = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer]
    radial, tram = detector.scratch_map(ring)
    m1, _ = detector.extract(radial, None, ring, inner, radius)
    m2, _ = detector.extract(tram, P["TRAM_MIN_LEN"], ring, inner, radius)
    det = detector.rewrap(cv2.bitwise_or(m1, m2), inner, center, radius, gray.shape)
    return img, det


def score(det, gt):
    det_b = (det > 127).astype(np.uint8)
    gt_b = (gt > 127).astype(np.uint8)

    near_det = cv2.dilate(det_b, np.ones((TOLERANCE, TOLERANCE), np.uint8))
    n, labels, _, _ = cv2.connectedComponentsWithStats(gt_b, connectivity=8)
    found = sum(1 for i in range(1, n)
                if np.count_nonzero(near_det[labels == i]) >= MIN_HIT_PX)

    near_gt = cv2.dilate(gt_b, np.ones((TOLERANCE, TOLERANCE), np.uint8))
    outside = det_b & (near_gt == 0)
    nd, _, ds, _ = cv2.connectedComponentsWithStats(outside, connectivity=8)
    extra = sum(1 for i in range(1, nd) if ds[i][4] >= MIN_EXTRA_AREA)

    nt, _, dts, _ = cv2.connectedComponentsWithStats(det_b, connectivity=8)
    shown = sum(1 for i in range(1, nt) if dts[i][4] >= MIN_EXTRA_AREA)
    return found, n - 1, extra, shown


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "cal"
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    records = set(sum(split.values(), [])) if which == "all" else set(split[which])
    rows = [r for r in load_index() if r["record"] in records]

    tot = collections.Counter()
    per_record = collections.defaultdict(lambda: collections.Counter())

    for r in rows:
        gt_path = os.path.join(GT, r["pair"] + ".png")
        photo = os.path.join(PHOTOS, r["photo_file"])
        if not (os.path.exists(gt_path) and os.path.exists(photo)):
            continue
        img, det = detect(photo)
        gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST)

        found, zones, extra, shown = score(det, gt)
        tot["zones"] += zones
        tot["found"] += found
        tot["shown"] += shown
        tot["photos"] += 1
        if zones:
            tot["extra_marked"] += extra
            tot["photos_marked"] += 1
        else:
            tot["fp_clean"] += shown
            tot["photos_clean"] += 1
        pr = per_record[r["record"]]
        pr["zones"] += zones
        pr["found"] += found
        pr["shown"] += shown

    print(f"SET = {which}   {len(per_record)} records, {tot['photos']} photos\n")
    recall = 100.0 * tot["found"] / max(tot["zones"], 1)
    print(f"  marked scratches        : {tot['zones']}")
    print(f"  found by the model      : {tot['found']}   ->  recall {recall:.1f}%")
    print(f"  detections shown total  : {tot['shown']}")
    if tot["photos_marked"]:
        print(f"  extra on marked photos  : {tot['extra_marked']}"
              f"   ({tot['extra_marked']/tot['photos_marked']:.1f} per photo)")
    if tot["photos_clean"]:
        print(f"  on CLEAN photos ({tot['photos_clean']:>3})    : {tot['fp_clean']}"
              f"   ({tot['fp_clean']/tot['photos_clean']:.1f} per clean photo)"
              f"   <- all false")
    if tot["shown"]:
        hit_rate = 100.0 * (tot["shown"] - tot["extra_marked"] - tot["fp_clean"]) \
            / tot["shown"]
        print(f"  of what is shown, near a marked scratch: {hit_rate:.0f}%")

    print(f"\n{'record':<34}{'marked':>8}{'found':>7}{'shown':>7}")
    for rec in sorted(per_record):
        d = per_record[rec]
        print(f"  {rec[:32]:<34}{d['zones']:>8}{d['found']:>7}{d['shown']:>7}")


if __name__ == "__main__":
    main()
