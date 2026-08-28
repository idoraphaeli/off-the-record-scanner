# -*- coding: utf-8 -*-
"""
Is the detector reading the PEN-MARKED photo instead of the clean one?

If it is, the model can be firing on the annotator's own ink. Blue ink on black
vinyl is a bright elongated line, exactly the shape the detector rewards, so
every such photo hands the model the answer sheet and inflates recall directly.

Earlier this was tested by comparing how much blue each half of a pair contains.
That was useless: black vinyl has a bluish sheen worth tens of thousands of
pixels and a pen stroke is worth a few thousand, so the sheen decided the answer.

This asks the question directly instead. The ground-truth mask already says
exactly where the pen is. So look at the file the detector actually reads, at
precisely those pixels, and see whether they are pen-blue there. A clean file is
plain vinyl under the mask; a marked one is blue.

Usage:  python check_pen_leak.py [cal|val|test|all]
"""

import csv
import json
import os
import sys

import cv2
import numpy as np

import detector
from evaluate_frozen import detect

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")

PEN_LO, PEN_HI = (95, 120, 90), (125, 255, 255)
INKED = 0.35          # fraction of mask pixels that must be blue to call it inked


def blue_mask(img):
    return cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), PEN_LO, PEN_HI)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    records = (set(sum(split.values(), [])) if which == "all"
               else set(split[which]))
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["record"] in records and int(r["marks"]) > 0]

    leaked, checked = [], 0
    det_on_ink = det_total = 0

    for r in rows:
        photo = os.path.join(PHOTOS, r["photo_file"])
        gt_path = os.path.join(GT, r["pair"] + ".png")
        if not (os.path.exists(photo) and os.path.exists(gt_path)):
            continue
        img = detector.load_image(photo)
        gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        sel = gt > 127
        if not sel.any():
            continue
        checked += 1

        frac = float(np.count_nonzero(blue_mask(img)[sel])) / np.count_nonzero(sel)
        if frac < INKED:
            continue
        leaked.append((frac, r))

    print(f"SET = {which}   {checked} marked photos checked\n")
    print(f"  photos the detector reads WITH the pen on them : {len(leaked)}"
          f"   ({100*len(leaked)/max(checked,1):.0f}%)")

    if not leaked:
        print("\n  the detector is reading clean files throughout.")
        return

    # how much of the damage is real: detections sitting on actual ink
    for frac, r in leaked:
        img, det = detect(os.path.join(PHOTOS, r["photo_file"]))
        ink = cv2.dilate(blue_mask(img), np.ones((7, 7), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (det > 127).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats[i][4] < 40:
                continue
            det_total += 1
            if np.count_nonzero(ink[labels == i]) > 0.25 * stats[i][4]:
                det_on_ink += 1

    print(f"  detections on those photos                     : {det_total}")
    print(f"  detections sitting on actual ink               : {det_on_ink}"
          f"   ({100*det_on_ink/max(det_total,1):.0f}%)")

    leaked.sort(reverse=True, key=lambda t: t[0])
    print(f"\n{'photo file':<54}{'mask pixels that are blue':>26}")
    print("-" * 82)
    for frac, r in leaked[:20]:
        print(f"{r['photo_file'][:52]:<54}{100*frac:>25.0f}%")
    if len(leaked) > 20:
        print(f"  ... and {len(leaked) - 20} more")


if __name__ == "__main__":
    main()
