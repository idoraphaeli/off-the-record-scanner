# -*- coding: utf-8 -*-
"""
Build one browsable results folder covering every photo in the dataset.

Per record, writes into results/:
  <id>_result.jpg   detections in yellow + human-marked zones outlined in green
  <id>_detect.jpg   detections only (no green), i.e. what the model would output
Plus results/summary.csv and results/README.txt with the per-record table.
"""

import csv
import json
import os
import sys

import cv2
import numpy as np

import detector

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\User\Desktop\Ido private\Computer science\off-the-record\record_pics"
CLEAN_DIR = next(os.path.join(BASE, d) for d in os.listdir(BASE)
                 if os.path.isdir(os.path.join(BASE, d))
                 and "ללא סימונים" in d and "חדש" in d)
GT_DIR = os.path.join(HERE, "gt")
# optional CLI arg names the output folder, so earlier runs stay for comparison
OUT = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "results")

MIN_HIT_PX = 20     # same rule as the frozen evaluator
MIN_FP_AREA = 50


MARK_ALPHA = 0.45      # translucent enough to still read the scratch underneath
MARK_HALO = 9          # px: the highlight is widened so it is findable at page scale


def draw(img, det, gt=None):
    """Translucent yellow highlight: a wide soft band marks WHERE the scratch is
    while leaving the scratch itself visible through it (a solid fill hides the
    very thing the user needs to judge)."""
    vis = img.copy().astype(np.float32)
    band = cv2.dilate((det > 127).astype(np.uint8),
                      np.ones((MARK_HALO, MARK_HALO), np.uint8))
    band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.0)  # soft edges
    a = np.clip(band, 0, 1)[..., None] * MARK_ALPHA
    yellow = np.array([90, 255, 255], np.float32)                  # BGR, light yellow
    vis = vis * (1 - a) + yellow * a

    vis = vis.astype(np.uint8)
    if gt is not None:
        cs, _ = cv2.findContours((gt > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (0, 255, 0), 3)
    return vis


def main():
    os.makedirs(OUT, exist_ok=True)
    split = json.load(open(os.path.join(HERE, "split.json"), encoding="utf-8"))
    role = {n: "test" for n in split["test"]}
    role.update({n: "calibration" for n in split["cal"]})

    rows = []
    names = sorted(f for f in os.listdir(CLEAN_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    for idx, name in enumerate(sorted(names), 1):
        stem = os.path.splitext(name)[0]
        rec_id = f"record_{idx:02d}"
        img = detector.load_image(os.path.join(CLEAN_DIR, name))
        det, _, info = detector.detect(os.path.join(CLEAN_DIR, name))

        gt_path = os.path.join(GT_DIR, stem + "_mask.png")
        gt = None
        found = zones = 0
        extra = 0
        if os.path.exists(gt_path):
            gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
            gt_b = (gt > 127).astype(np.uint8)
            det_b = (det > 127).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(gt_b, 8)
            zones = n - 1
            for i in range(1, n):
                if int(np.count_nonzero(det_b[labels == i])) >= MIN_HIT_PX:
                    found += 1
            outside = det_b & (gt_b == 0)
            nd, _, dstats, _ = cv2.connectedComponentsWithStats(outside, 8)
            extra = sum(1 for i in range(1, nd) if dstats[i][4] >= MIN_FP_AREA)

        cv2.imencode(".jpg", draw(img, det, gt))[1].tofile(
            os.path.join(OUT, f"{rec_id}_result.jpg"))
        cv2.imencode(".jpg", draw(img, det))[1].tofile(
            os.path.join(OUT, f"{rec_id}_detect.jpg"))

        rows.append({
            "id": rec_id,
            "set": role.get(name, "unused"),
            "marks_detected": info["n_scratches"],
            "human_zones": zones,
            "zones_found": found,
            "detections_outside_zones": extra,
            "source_file": name,
        })
        print(f"{rec_id} | {role.get(name,'unused'):>11} | detected={info['n_scratches']:2d}"
              f" | zones {found}/{zones} | outside={extra}")

    with open(os.path.join(OUT, "summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    tot_found = sum(r["zones_found"] for r in rows)
    tot_zones = sum(r["human_zones"] for r in rows)
    with open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Off The Record - scratch detection results\n"
            "==========================================\n\n"
            "<id>_detect.jpg  what the model found (yellow) - the model's own output\n"
            "<id>_result.jpg  same, plus your hand-marked zones outlined in green\n"
            "summary.csv      per-record table\n\n"
            "Column meaning:\n"
            "  marks_detected           how many separate marks the model reported\n"
            "  human_zones / zones_found  your marked zones, and how many the model hit\n"
            "  detections_outside_zones   found by the model, not marked by you --\n"
            "                             may be a real scratch you missed, or a false alarm\n\n"
            f"Overall: {tot_found}/{tot_zones} marked zones hit "
            f"({100.0 * tot_found / max(tot_zones, 1):.1f}%)\n"
        )
    print(f"\nwrote {len(rows)} records to {OUT}")


if __name__ == "__main__":
    main()
