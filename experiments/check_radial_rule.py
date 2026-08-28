# -*- coding: utf-8 -*-
"""
Confirm the radial rule against the real detector, not the simulation.

The rule was chosen on the hand-labelled table, and that table only holds
detections which did NOT land on a pen mark — the ones that did were counted
correct by construction and never listed. So the simulation could not see
whether the rule also deletes detections sitting ON a marked scratch, and those
are exactly what recall is made of.

This runs the whole detector twice, with the rule and with it switched off, and
reports recall from the frozen evaluator both ways.

Usage:  python check_radial_rule.py [cal|val]
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
from sweep_recall2 import detect_with

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
NOW = copy.deepcopy(P)


def run(rows):
    tot = dict(zones=0, found=0, shown=0)
    for r in rows:
        gt_path = os.path.join(GT, r["pair"] + ".png")
        photo = os.path.join(PHOTOS, r["photo_file"])
        if not (os.path.exists(gt_path) and os.path.exists(photo)):
            continue
        img, det = detect_with(photo, detector.extract)
        gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        found, zones, _extra, shown = score(det, gt)
        tot["zones"] += zones
        tot["found"] += found
        tot["shown"] += shown
    return tot


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "val"
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    records = set(split[which])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in records]

    print(f"SET = {which}   {len(rows)} photos\n")
    print(f"{'':<22}{'recall':>9}{'found':>8}{'shown':>8}")
    print("-" * 47)

    results = {}
    # 999 degrees can never be exceeded, so the rule is off without touching
    # any other part of the pipeline
    for name, tol in (("rule OFF", 999.0), ("rule ON", NOW["RADIAL_TOL_DEG"])):
        P.clear()
        P.update(copy.deepcopy(NOW))
        P["RADIAL_TOL_DEG"] = tol
        t = run(rows)
        results[name] = t
        print(f"{name:<22}{100.0 * t['found'] / max(t['zones'], 1):>8.1f}%"
              f"{t['found']:>8}{t['shown']:>8}")

    P.clear()
    P.update(copy.deepcopy(NOW))

    off, on = results["rule OFF"], results["rule ON"]
    print(f"\n  scratches lost to the rule : {off['found'] - on['found']}"
          f" of {off['zones']}")
    print(f"  detections removed         : {off['shown'] - on['shown']}"
          f" of {off['shown']}")


if __name__ == "__main__":
    main()
