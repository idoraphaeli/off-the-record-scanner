# -*- coding: utf-8 -*-
"""
Stage 0b — FROZEN evaluation script. Do not modify after Stage 0 approval.

Usage:
    python evaluate.py <detections_dir> [cal|test|all]

<detections_dir> must contain one binary mask per image, named <stem>_det.png,
in ANY resolution (it is nearest-resized to the ground-truth resolution).

Metrics per image:
  - per-scratch recall: a GT zone counts as FOUND if at least MIN_HIT_PX
    detection pixels fall inside it
  - false-positive suspects: detection components (>= MIN_FP_AREA px) that do
    not touch any GT zone; reported with centroid + area for human review
Images with zero GT zones contribute only to the false-positive count.
"""

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(HERE, "gt")
SPLIT_FILE = os.path.join(HERE, "split.json")

MIN_HIT_PX = 20     # detection pixels inside a GT zone to count it as found
MIN_FP_AREA = 50    # smaller stray detections are ignored as dust-level noise


def load_mask(path, like=None):
    m = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if like is not None and m.shape != like:
        m = cv2.resize(m, (like[1], like[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def evaluate_image(det_mask, gt_mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gt_mask, connectivity=8)
    zones_found, zones_total = 0, n - 1
    for i in range(1, n):
        if int(np.count_nonzero(det_mask[labels == i])) >= MIN_HIT_PX:
            zones_found += 1

    outside = det_mask & (gt_mask == 0)
    nd, dlabels, dstats, dcent = cv2.connectedComponentsWithStats(
        outside.astype(np.uint8), connectivity=8)
    fps = []
    for i in range(1, nd):
        x, y, w, h, area = dstats[i]
        if area < MIN_FP_AREA:
            continue
        # a component that mostly overlaps GT but spills out is not an FP
        full_comp = det_mask[y:y + h, x:x + w]
        gt_comp = gt_mask[y:y + h, x:x + w]
        if np.count_nonzero(full_comp & gt_comp) >= MIN_HIT_PX:
            continue
        fps.append({"cx": int(dcent[i][0]), "cy": int(dcent[i][1]), "area": int(area)})
    return zones_found, zones_total, fps


def main():
    det_dir = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "cal"
    split = json.load(open(SPLIT_FILE, encoding="utf-8"))
    names = split["cal"] + split["test"] if which == "all" else split[which]

    total_found, total_zones, total_fps, per_image = 0, 0, 0, {}
    for name in names:
        stem = os.path.splitext(name)[0]
        gt_path = os.path.join(GT_DIR, stem + "_mask.png")
        det_path = os.path.join(det_dir, stem + "_det.png")
        if not os.path.exists(det_path):
            print(f"MISSING detection: {stem}")
            continue
        gt = load_mask(gt_path) if os.path.exists(gt_path) else None
        det_ref = cv2.imdecode(np.fromfile(det_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gt is None:   # unmarked image: all-zero GT at detection resolution
            gt = np.zeros(det_ref.shape[:2], np.uint8)
        det = load_mask(det_path, like=gt.shape)

        found, zones, fps = evaluate_image(det, gt)
        total_found += found
        total_zones += zones
        total_fps += len(fps)
        per_image[stem] = {"found": found, "zones": zones, "false_positives": fps}
        print(f"{stem[-22:]:>24} | {found}/{zones} zones | {len(fps)} FP suspects")

    recall = 100.0 * total_found / total_zones if total_zones else float("nan")
    avg_fp = total_fps / max(len(per_image), 1)
    print("=" * 60)
    print(f"SET={which}  recall={recall:.1f}%  ({total_found}/{total_zones} zones)"
          f"  avg FP suspects/image={avg_fp:.2f}")
    out = {"set": which, "recall_pct": round(recall, 1), "zones_found": total_found,
           "zones_total": total_zones, "avg_fp_per_image": round(avg_fp, 2),
           "per_image": per_image}
    with open(os.path.join(det_dir, f"eval_{which}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
