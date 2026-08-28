# -*- coding: utf-8 -*-
"""
Everything known about the alignment of one side, for when a pair behaves
differently from the rest.

Three independent answers to the same question — how far did the disc turn
between the two shots — are printed side by side:

  label     the printed pattern on the centre label, which is what the pipeline
            uses and what was checked against the hand marks
  marks     the hand-drawn scratches, which are the same physical damage in both
            shots and therefore the closest thing to truth available
  best      the rotation that would make the MOST detections line up, found by
            trying all of them

If all three agree, the alignment is right and a low overlap is telling us
something about the photographs — different light, or dirt that moved. If
"best" sits somewhere else entirely, the alignment is simply wrong and the
overlap curve will show where the right answer was.

Usage:  python diag_side.py <record> <side>
"""

import collections
import csv
import math
import os
import sys

import cv2
import numpy as np

import detector
import cross_shot as cs
from show_alignment import gt_points, true_rotation

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")

RAD_TOL, ANG_TOL = 0.025, 6.0


def overlap(a, b, delta):
    n = 0
    for p in a:
        want = (p["ang"] + delta) % 360.0
        for q in b:
            if abs(p["rad"] - q["rad"]) > RAD_TOL:
                continue
            if abs((q["ang"] - want + 180.0) % 360.0 - 180.0) <= ANG_TOL:
                n += 1
                break
    return n


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: python diag_side.py <record> <side>")
    record, side = sys.argv[1], sys.argv[2].lower()

    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        shots = [r for r in csv.DictReader(fh)
                 if r["record"] == record and r["side"].lower() == side]
    if len(shots) < 2:
        sys.exit(f"{record} side {side}: found {len(shots)} shots, need 2")

    info = []
    for r in shots[:2]:
        path = os.path.join(PHOTOS, r["photo_file"])
        img = detector.load_image(path)
        center, radius = detector.find_disc(img)
        info.append({
            "row": r, "path": path,
            "det": cs.polar_detections(path),
            "prof": cs.label_profile(path),
            "marks": gt_points(r["pair"], center, radius, img.shape[:2]),
            "center": center, "radius": radius, "size": img.shape[:2],
        })

    print(f"{record}  side {side}\n")
    for i, s in enumerate(info):
        print(f"  shot {s['row']['shot']}  {s['row']['photo_file'][:44]}")
        print(f"      image {s['size'][1]}x{s['size'][0]}   disc radius "
              f"{s['radius']}px   centre ({s['center'][0]}, {s['center'][1]})")
        print(f"      detections {len(s['det'])}   hand marks {len(s['marks'])}"
              f"   marks in index {s['row']['marks']}")

    a, b = info[0], info[1]

    d_label, ratio = cs.rotation_from_label(a["prof"], b["prof"])
    d_marks, votes = true_rotation(a["marks"], b["marks"])

    curve = np.array([overlap(a["det"], b["det"], d) for d in range(0, 360, 2)])
    order = curve.argsort()[::-1]
    best = int(order[0]) * 2

    print(f"\n  rotation from the LABEL  : "
          f"{'%.0f deg' % d_label if d_label is not None else 'none'}"
          f"   (peak strength {ratio:.1f})")
    print(f"  rotation from HAND MARKS : "
          f"{'%.0f deg' % d_marks if d_marks is not None else 'none'}"
          f"   ({votes} votes)")
    print(f"  rotation that best fits the detections : {best} deg"
          f"   ({int(curve.max())} overlap)")

    if d_label is not None:
        print(f"\n  overlap at the label's answer : "
              f"{overlap(a['det'], b['det'], d_label)} of {len(a['det'])}")
    if d_marks is not None and d_label is not None:
        gap = abs((d_label - d_marks + 180.0) % 360.0 - 180.0)
        print(f"  label vs hand marks           : {gap:.0f} deg apart")

    print(f"\n  overlap as the rotation is swept — the five best:")
    shown = []
    for k in order:
        d = int(k) * 2
        if any(abs((d - s + 180) % 360 - 180) < 12 for s in shown):
            continue
        shown.append(d)
        print(f"      {d:>4} deg : {int(curve[k]):>3} overlap")
        if len(shown) == 5:
            break
    print(f"  median across all rotations : {np.median(curve):.0f}"
          f"   (what pure coincidence gives)")


if __name__ == "__main__":
    main()
