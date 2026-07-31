# -*- coding: utf-8 -*-
"""Parameter sweep scored with the FROZEN evaluation logic (evaluate.evaluate_image).
Reports recall and FP/image per combo on the calibration set."""

import itertools
import json
import os

import cv2
import numpy as np

import detector
from detector import P
from evaluate import evaluate_image

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\User\Desktop\Ido private\Computer science\off-the-record\record_pics"
CLEAN_DIR = next(os.path.join(BASE, d) for d in os.listdir(BASE)
                 if os.path.isdir(os.path.join(BASE, d))
                 and "ללא סימונים" in d and "חדש" in d)
GT_DIR = os.path.join(HERE, "gt")

split = json.load(open(os.path.join(HERE, "split.json"), encoding="utf-8"))
cases = []
for name in split["cal"]:
    stem = os.path.splitext(name)[0]
    gt_path = os.path.join(GT_DIR, stem + "_mask.png")
    if not os.path.exists(gt_path):
        continue
    cases.append((name, stem, gt_path))
print(f"sweeping on {len(cases)} calibration images")

GRID = dict(
    PCT_STRONG=[99.5, 99.65, 99.8],
    PCT_WEAK=[99.0, 99.3, 99.5],
)

keys = list(GRID)
results = []
for combo in itertools.product(*(GRID[k] for k in keys)):
    for k, v in zip(keys, combo):
        P[k] = v
    found_tot, zones_tot, fp_tot = 0, 0, 0
    for name, stem, gt_path in cases:
        det, _, _ = detector.detect(os.path.join(CLEAN_DIR, name))
        gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (det.shape[1], det.shape[0]), interpolation=cv2.INTER_NEAREST)
        found, zones, fps = evaluate_image((det > 127).astype(np.uint8),
                                           (gt > 127).astype(np.uint8))
        found_tot += found
        zones_tot += zones
        fp_tot += len(fps)
    recall = 100.0 * found_tot / max(zones_tot, 1)
    fp_img = fp_tot / len(cases)
    results.append((recall, fp_img, dict(zip(keys, combo))))
    print(f"{dict(zip(keys, combo))}  ->  recall={recall:5.1f}%  FP/img={fp_img:5.2f}")

print("\n-- ranked by recall among combos with FP/img <= 2 --")
ok = [r for r in results if r[1] <= 2.0]
for recall, fp, params in sorted(ok, key=lambda r: -r[0])[:5]:
    print(f"  recall={recall:5.1f}%  FP/img={fp:5.2f}  {params}")
if not ok:
    print("  (none met the FP bar; best by recall overall:)")
    for recall, fp, params in sorted(results, key=lambda r: -r[0])[:5]:
        print(f"  recall={recall:5.1f}%  FP/img={fp:5.2f}  {params}")
