# -*- coding: utf-8 -*-
"""
Read back the verification of the MATCHED detections and say what it does to
recall.

Recall has always been computed as "hand-marked zones that a detection landed
within TOLERANCE of". That credits the model twice over on trust: that the zone
really is a pen stroke, and that the detection really sits on it. Ground truth
came from differencing the two copies of a pair and keeping what is pen-blue,
and blue-lit vinyl and cyan labels both pass a colour test — so a zone can exist
where nothing was ever drawn, and every such zone inflates recall.

Detections are counted per ZONE, not per detection, because recall counts zones:
several detections can land on one stroke, and scaling a zone-level score by a
detection-level rate would be wrong.

Usage:  python analyse_matched.py [cal|val]
"""

import collections
import csv
import json
import os
import sys

import cv2
import numpy as np

import detector
from evaluate_frozen import TOLERANCE, MIN_HIT_PX, detect

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "gt_new")

# The verdicts split along two questions: is the stroke real, and did the model
# earn the credit?
#
#   hit            on the stroke — earned outright
#   dust_on_mark   touching the stroke, but dirt is what lit the detector up.
#                  The damaged spot did get flagged, so credit is defensible,
#                  but it is not evidence the model can see that scratch.
#   other_scratch  on real damage that was never marked — the model found
#                  something true, just not this stroke
#   false_near     on nothing; the stroke merely happens to be within tolerance.
#                  This is the one that inflates recall outright.
#   nomark         no stroke there at all — ground truth is wrong
#
# "nearby" is a legacy value from before this split and is reported separately
# rather than folded into a guess.
GOOD = ("hit",)
EARNED = ("hit", "dust_on_mark")
STROKE_REAL = ("hit", "dust_on_mark", "other_scratch", "false_near", "nearby")


def zone_verdicts(which, rows, by_id):
    """Map each hand-marked zone to the verdicts of the detections credited to
    it, so a zone can be judged once rather than once per detection."""
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        gt_index = {r["pair"]: r for r in csv.DictReader(fh)}
    photos = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")

    by_pair = collections.defaultdict(list)
    for r in rows:
        by_pair[r["pair"]].append(r)

    zones = collections.defaultdict(list)
    for pair, group in by_pair.items():
        meta = gt_index.get(pair)
        if meta is None:
            continue
        gt_path = os.path.join(GT, pair + ".png")
        photo = os.path.join(photos, meta["photo_file"])
        if not (os.path.exists(gt_path) and os.path.exists(photo)):
            continue

        img = detector.load_image(photo)
        gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        n, labels, _, _ = cv2.connectedComponentsWithStats(
            (gt > 127).astype(np.uint8), connectivity=8)
        near = cv2.dilate(labels.astype(np.uint16), np.ones((TOLERANCE, TOLERANCE),
                                                            np.uint8))
        H, W = labels.shape
        for r in group:
            e = by_id.get(r["id"])
            if e is None:
                continue
            x = int(round(r["cx"] / (e["vw"] / float(W))))
            y = int(round(r["cy"] / (e["vw"] / float(W))))
            z = int(near[min(max(y, 0), H - 1), min(max(x, 0), W - 1)])
            if z:
                zones[(pair, z)].append(r["label"])
        print(f"  {pair[:46]:<48}{len(group):>4}")
    return zones


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "val"
    tool = os.path.join(HERE, f"label_tool_{which}_matched")
    labels_path = os.path.join(tool, f"labels_{which}.json")
    if not os.path.exists(labels_path):
        sys.exit(f"no labels at {labels_path} — export them from the tool first")

    index = json.load(open(os.path.join(tool, "index.json"), encoding="utf-8"))
    by_id = {e["id"]: e for e in index["items"]}
    rows = [r for r in json.load(open(labels_path, encoding="utf-8"))["rows"]
            if r["label"]]

    print(f"SET = {which}   {len(rows)} of {index['total']} verified\n")
    tally = collections.Counter(r["label"] for r in rows)
    for k in ("hit", "dust_on_mark", "other_scratch", "false_near",
              "nomark", "unsure", "nearby"):
        if k == "nearby" and not tally[k]:
            continue
        n = tally[k]
        bar = "#" * int(round(40 * n / max(tally.values(), default=1)))
        print(f"  {k:<9}{n:>5}{100*n/max(len(rows),1):>7.1f}%  {bar}")

    print()
    zones = zone_verdicts(which, rows, by_id)
    if not zones:
        return

    strict = sum(1 for v in zones.values() if any(x in GOOD for x in v))
    earned = sum(1 for v in zones.values() if any(x in EARNED for x in v))
    loose = sum(1 for v in zones.values() if any(x in STROKE_REAL for x in v))
    phantom = sum(1 for v in zones.values()
                  if v and all(x == "nomark" for x in v))

    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        total_zones = sum(int(r["marks"]) for r in csv.DictReader(fh)
                          if r["record"] in set(split[which]))

    print(f"\nzones the model was credited with finding   : {len(zones)}")
    print(f"  a detection really on the stroke           : {strict}")
    print(f"  reached only via dirt sitting on the mark  : {earned - strict}")
    print(f"  credited to a detection on OTHER real damage: "
          f"{sum(1 for v in zones.values() if 'other_scratch' in v and not any(x in EARNED for x in v))}")
    print(f"  credited to a detection on NOTHING         : "
          f"{sum(1 for v in zones.values() if 'false_near' in v and not any(x in EARNED for x in v))}"
          f"   <- inflated")
    print(f"  NO stroke there at all (ground truth wrong) : {phantom}"
          f"   <- corrupt")

    real_zones = max(total_zones - phantom, 1)
    print(f"\nrecall as reported                          : "
          f"{100*len(zones)/max(total_zones,1):.1f}%  ({len(zones)}/{total_zones})")
    print(f"recall over real strokes only               : "
          f"{100*loose/real_zones:.1f}%")
    print(f"recall the model actually earned            : "
          f"{100*earned/real_zones:.1f}%")
    print(f"recall counting direct hits alone           : "
          f"{100*strict/real_zones:.1f}%")
    if phantom:
        print(f"\nNOTE: the {phantom} phantom zones were removed from the "
              f"DENOMINATOR too — they are not\n      real scratches either, so "
              f"dropping them from the numerator alone would\n      have "
              f"understated the model rather than corrected it.")


if __name__ == "__main__":
    main()
