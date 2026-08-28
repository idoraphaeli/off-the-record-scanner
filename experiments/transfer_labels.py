# -*- coding: utf-8 -*-
"""
How much of the hand-labelling survives the ground-truth rebuild?

The labels themselves are still valid — each one is a judgement made by eye
about a spot on a record, and closing the ink leak did not change what is on
those records. What changed is the SET of detections: the detector now reads the
clean half of each pair, so it fires in slightly different places, and a label
is attached to a detection by id.

A label transfers when a new detection lands on the same spot as the labelled
one. Positions are compared in the tool's own view coordinates (1100 px wide),
which are directly comparable across the two halves of a pair because they frame
the same disc — only the resolution differs, and that is normalised away.

Reported both ways: how many old labels find a home, and how many new detections
have nobody to inherit from and would still need judging.

Usage:  python transfer_labels.py [cal|val] [mode]
"""

import collections
import csv
import json
import os
import sys

import cv2
import numpy as np

import detector
from evaluate_frozen import TOLERANCE, MIN_EXTRA_AREA, detect

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
VIEW_W = 1100

RADII = (10, 20, 35)          # view px; 1100 px spans the whole photo


def new_detections(photo, gt_path):
    """Detections at the current settings, in view coordinates, split by
    whether they sit on a pen mark — the tool separates those two modes."""
    img, det = detect(photo)
    H, W = img.shape[:2]
    scale = VIEW_W / float(W)

    gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
    near_gt = cv2.dilate((gt > 127).astype(np.uint8),
                         np.ones((TOLERANCE, TOLERANCE), np.uint8))

    n, labels, stats, cent = cv2.connectedComponentsWithStats(
        (det > 127).astype(np.uint8), connectivity=8)
    extra, matched = [], []
    for i in range(1, n):
        if stats[i][4] < MIN_EXTRA_AREA:
            continue
        cx, cy = int(cent[i][0]), int(cent[i][1])
        pt = (cx * scale, cy * scale)
        if near_gt[min(cy, H - 1), min(cx, W - 1)]:
            matched.append(pt)
        else:
            extra.append(pt)
    return extra, matched


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "cal"
    mode = sys.argv[2] if len(sys.argv) > 2 else "extra"
    suffix = "" if mode == "extra" else "_matched"
    tool = os.path.join(HERE, f"label_tool_{which}{suffix}")

    index = json.load(open(os.path.join(tool, "index.json"), encoding="utf-8"))
    labels_path = os.path.join(tool, f"labels_{which}.json")
    if not os.path.exists(labels_path):
        sys.exit(f"no labels at {labels_path}")
    labelled = [r for r in json.load(open(labels_path, encoding="utf-8"))["rows"]
                if r["label"]]
    by_id = {e["id"]: e for e in index["items"]}

    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        gt_rows = {r["pair"]: r for r in csv.DictReader(fh)}

    old_by_pair = collections.defaultdict(list)
    for r in labelled:
        e = by_id.get(r["id"])
        if e:
            old_by_pair[r["pair"]].append((e["cx"], e["cy"], r["label"]))

    hit = {r: 0 for r in RADII}
    total_old = orphan_new = total_new = gone_pairs = 0
    kept_by_label = collections.Counter()

    for pair, olds in sorted(old_by_pair.items()):
        meta = gt_rows.get(pair)
        if meta is None:
            gone_pairs += 1
            total_old += len(olds)
            continue
        photo = os.path.join(PHOTOS, meta["photo_file"])
        gt_path = os.path.join(GT, pair + ".png")
        if not (os.path.exists(photo) and os.path.exists(gt_path)):
            gone_pairs += 1
            total_old += len(olds)
            continue

        extra, matched = new_detections(photo, gt_path)
        news = extra if mode == "extra" else matched
        total_new += len(news)
        arr = np.array(news, float) if news else np.zeros((0, 2))

        used = set()
        for cx, cy, lab in olds:
            total_old += 1
            if not len(arr):
                continue
            d = np.hypot(arr[:, 0] - cx, arr[:, 1] - cy)
            j = int(d.argmin())
            for r in RADII:
                if d[j] <= r:
                    hit[r] += 1
            if d[j] <= RADII[-1]:
                used.add(j)
                kept_by_label[lab] += 1
        orphan_new += len(news) - len(used)
        print(f"  {pair[:44]:<46}{len(olds):>4} old{len(news):>5} new")

    print(f"\nSET = {which}   MODE = {mode}")
    print(f"  labels made by hand                    : {total_old}")
    print(f"  detections the model reports now       : {total_new}")
    if gone_pairs:
        print(f"  pairs that no longer exist             : {gone_pairs}")
    print()
    for r in RADII:
        print(f"  transfer within {r:>2} px : {hit[r]:>5}"
              f"   ({100*hit[r]/max(total_old,1):.0f}% of your work)")
    print(f"\n  new detections with nobody to inherit from : {orphan_new}"
          f"   ({100*orphan_new/max(total_new,1):.0f}% of what is shown now)")
    if kept_by_label:
        print("\n  what transfers, by verdict:")
        for k, v in kept_by_label.most_common():
            print(f"    {k:<14}{v:>5}")


if __name__ == "__main__":
    main()
