# -*- coding: utf-8 -*-
"""What is left if the cross-shot check is used as a FILTER, not as a weight.

Today a detection confirmed in both shots of a side is only weighted more
heavily in the grade; nothing is ever thrown away. Measured against the hand
labels, a confirmed detection is right 98% of the time against 78% for one seen
in a single shot -- so dropping the unconfirmed ones would buy a large amount of
precision. It would also throw away most of the real damage: only about a
quarter of the hand-marked scratches show up in both shots at all.

That trade cannot be settled by the two numbers alone, because they count marks
and a buyer cares about records. Losing four faint marks on a record that keeps
two obvious ones costs nothing -- the record is still called scratched. Losing
the only mark on an otherwise clean-looking record turns a bad record into a
good one, which is the failure that matters.

So this lays it out to be judged by eye: the record, everything the model finds,
and what survives the filter. Five records, one side each.

The angle between the two shots is corrected before matching, and the match
window is the tightened +/-2 degrees -- the filter is only worth considering at
all once the alignment is right.

Usage:  python make_filter_sheets.py [how many records] [seed]
"""

import collections
import csv
import json
import math
import os
import random
import shutil
import sys

import cv2
import numpy as np

import detector
from cross_shot import label_profile, rotation_from_label
from evaluate_frozen import MIN_EXTRA_AREA, detect
from tune_alignment import offsets, refine

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
OUT = os.path.join(os.path.dirname(HERE), "Model_FilterCompare")

PANE = 820
WINDOW = 2.0          # the tightened match window the refined angle allows
RING_R = 26           # radius of the circle drawn round each mark, in pane px


def marks_of(path):
    """Detections on one photo: the photo, a label image, and disc coordinates."""
    img, det = detect(path)
    center, radius = detector.find_disc(img)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(
        (det > 127).astype(np.uint8), connectivity=8)
    marks, keep = [], np.zeros(n, bool)
    for i in range(1, n):
        if stats[i][4] < MIN_EXTRA_AREA:
            continue
        keep[i] = True
        dx, dy = cent[i][0] - center[0], cent[i][1] - center[1]
        marks.append({"id": i,
                      "rad": math.hypot(dx, dy) / max(radius, 1),
                      "ang": math.degrees(math.atan2(dy, dx)) % 360.0,
                      "cx": float(cent[i][0]), "cy": float(cent[i][1])})
    return {"img": img, "labels": lab, "marks": marks, "keep": keep,
            "center": center, "radius": radius}


def painted(shot, which, colour, crop, scale):
    """The photo with a chosen subset of the marks painted on and circled.

    A mark is a handful of pixels wide; on a pane this size that is a speck, so
    the pixels are painted for the truth of the shape and a circle is drawn
    round them so the eye can find it at all.
    """
    x, y, s = crop
    view = shot["img"][y:y+s, x:x+s].copy()
    ids = [shot["marks"][k]["id"] for k in which]
    if ids:
        m = np.isin(shot["labels"][y:y+s, x:x+s], ids)
        m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        view[m] = colour
    view = cv2.resize(view, (PANE, PANE), interpolation=cv2.INTER_AREA)
    for k in which:
        p = shot["marks"][k]
        cx = int((p["cx"] - x) * scale)
        cy = int((p["cy"] - y) * scale)
        cv2.circle(view, (cx, cy), RING_R, colour, 1, cv2.LINE_AA)
    return view


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)

    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    cal = set(split["cal"])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in cal]

    sides = collections.defaultdict(list)
    for r in rows:
        p = os.path.join(PHOTOS, r["photo_file"])
        if os.path.exists(p):
            sides[(r["record"], r["side"])].append(p)

    # one side per record, and a different record every time -- the point is to
    # see five records, not five views of the same pressing
    by_record = collections.defaultdict(list)
    for (rec, side), paths in sorted(sides.items()):
        if len(paths) >= 2:
            by_record[rec].append((side, paths))
    picks = sorted(by_record)
    random.Random(seed).shuffle(picks)

    made, totals = 0, [0, 0]
    for rec in picks:
        if made >= want:
            break
        side, paths = by_record[rec][0]
        try:
            a, b = marks_of(paths[0]), marks_of(paths[1])
            delta, _ = rotation_from_label(label_profile(paths[0]),
                                           label_profile(paths[1]))
        except Exception:
            continue
        if delta is None or not a["marks"]:
            continue

        fixed, moved, spread, _ = refine(a["marks"], b["marks"], delta)
        hit = [d is not None for d in offsets(a["marks"], b["marks"], fixed, WINDOW)]
        kept = [k for k, f in enumerate(hit) if f]
        lost = [k for k, f in enumerate(hit) if not f]
        totals[0] += len(kept)
        totals[1] += len(a["marks"])

        c, rr = a["center"], a["radius"]
        s = int(rr * 2.16)
        x, y = max(c[0] - s // 2, 0), max(c[1] - s // 2, 0)
        s = min(s, a["img"].shape[1] - x, a["img"].shape[0] - y)
        scale = PANE / float(s)
        crop = (x, y, s)

        plain = cv2.resize(a["img"][y:y+s, x:x+s], (PANE, PANE),
                           interpolation=cv2.INTER_AREA)
        every = painted(a, range(len(a["marks"])), (60, 200, 255), crop, scale)
        only = painted(a, kept, (80, 230, 80), crop, scale)

        gap = 12
        sheet = np.full((PANE + 92, PANE * 3 + gap * 2, 3), 18, np.uint8)
        for i, pane in enumerate((plain, every, only)):
            sheet[0:PANE, i*(PANE+gap):i*(PANE+gap)+PANE] = pane
        f = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(sheet, f"{rec}   side {side}", (8, PANE + 32), f, 0.72,
                    (225, 220, 210), 1, cv2.LINE_AA)
        heads = ("the record, as photographed",
                 f"everything the model finds: {len(a['marks'])}",
                 f"kept by the cross-check: {len(kept)}"
                 f"   ({len(lost)} dropped)")
        for i, t in enumerate(heads):
            cv2.putText(sheet, t, (i*(PANE+gap) + 8, PANE + 62), f, 0.56,
                        (150, 145, 138), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"angle from the label {delta:.0f} deg, corrected by "
                           f"{moved:+.1f} deg   match window +/-{WINDOW:.0f} deg",
                    (8, PANE + 84), f, 0.48, (120, 116, 110), 1, cv2.LINE_AA)

        fn = f"{made+1:02d}_{rec[:28]}_{side}.jpg"
        cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(
            os.path.join(OUT, fn))
        print(f"  {fn:<44} found {len(a['marks']):>3}   kept {len(kept):>3}"
              f"   dropped {len(lost):>3}   correction {moved:>+5.1f} deg")
        made += 1

    if totals[1]:
        print(f"\nacross the {made} records: {totals[0]} of {totals[1]} detections "
              f"survive the filter ({100*totals[0]/totals[1]:.0f}%)")
    print(f"\nsheets written to {OUT}")


if __name__ == "__main__":
    main()
