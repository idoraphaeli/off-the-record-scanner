# -*- coding: utf-8 -*-
"""Ten sides, each shown as the photograph beside the two shots overlaid.

The cross-check is the strongest discriminator measured -- marks confirmed in
both shots of a side are right 89% of the time against 71% for marks seen once --
but it has only ever been looked at as a number. This lays it out to be read by
eye: what the record looks like, and where the two shots agree and disagree once
the rotation between them is taken out.

Green is the first shot, red the second, and one layer is nudged a few pixels so
a pair that landed on the same spot stays visible as two marks side by side
rather than one covering the other. A green blob with a red blob beside it is a
mark that survived both angles. A lone blob is a mark that appeared once.

Only sides whose rotation could be recovered from the label are included -- on
the rest there is nothing to overlay.

Usage:  python make_crossshot_sheets.py [how many sides]
"""

import collections
import csv
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np

import detector
from detector import P
import cross_shot as cs
from evaluate_frozen import MIN_EXTRA_AREA

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
OUT = os.path.join(os.path.dirname(HERE), "Model_CrossShot_Sheets")

NUDGE = 7            # px the second layer is shifted, so pairs stay readable
RAD_TOL = 0.025
ANG_TOL = 6.0
PANE = 900           # each half of the sheet


def marks_of(path):
    """Detections on one photo, as a photo-space mask plus disc coordinates."""
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner = int(P["LABEL_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:int(P["OUTER_R"] * radius)]
    rad_map, tram = detector.scratch_map(ring)[:2]
    m1, _ = detector.extract(rad_map, None, ring, inner, radius)
    m2, _ = detector.extract(tram, P["TRAM_MIN_LEN"], ring, inner, radius)
    det = detector.rewrap(cv2.bitwise_or(m1, m2), inner, center, radius, gray.shape)

    n, lb, stats, cent = cv2.connectedComponentsWithStats(
        (det > 127).astype(np.uint8), connectivity=8)
    marks = []
    for i in range(1, n):
        if stats[i][4] < MIN_EXTRA_AREA:
            continue
        dx, dy = cent[i][0] - center[0], cent[i][1] - center[1]
        marks.append({"id": i,
                      "rad": math.hypot(dx, dy) / max(radius, 1),
                      "ang": math.degrees(math.atan2(dy, dx)) % 360.0})
    return {"img": img, "labels": lb, "marks": marks,
            "center": center, "radius": radius}


def confirmed(a_marks, b_marks, delta):
    flags = []
    for m in a_marks:
        want = (m["ang"] + delta) % 360.0
        hit = any(abs(m["rad"] - o["rad"]) <= RAD_TOL
                  and abs((o["ang"] - want + 180.0) % 360.0 - 180.0) <= ANG_TOL
                  for o in b_marks)
        flags.append(hit)
    return flags


def layer(shot, colour, dx, dy):
    """One shot's detections as a coloured layer, shifted by (dx, dy)."""
    h, w = shot["labels"].shape
    m = (shot["labels"] > 0).astype(np.uint8)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    m = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST)
    out = np.zeros((h, w, 3), np.uint8)
    out[m > 0] = colour
    return out


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 10
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

    made = 0
    for (rec, side), paths in sorted(sides.items()):
        if made >= want or len(paths) < 2:
            continue
        try:
            a, b = marks_of(paths[0]), marks_of(paths[1])
            delta, _ = cs.rotation_from_label(cs.label_profile(paths[0]),
                                              cs.label_profile(paths[1]))
        except Exception:
            continue
        if delta is None:
            continue

        flags = confirmed(a["marks"], b["marks"], delta)
        both = int(sum(flags))

        # the two layers, the second nudged so a pair reads as a pair
        comp = np.zeros_like(a["img"])
        comp = cv2.add(comp, layer(a, (60, 240, 60), 0, 0))
        comp = cv2.add(comp, layer(b, (60, 60, 240), NUDGE, NUDGE))

        c, rr = a["center"], a["radius"]
        s = int(rr * 2.16)
        x, y = max(c[0] - s // 2, 0), max(c[1] - s // 2, 0)
        s = min(s, a["img"].shape[1] - x, a["img"].shape[0] - y)
        left = cv2.resize(a["img"][y:y+s, x:x+s], (PANE, PANE), interpolation=cv2.INTER_AREA)
        right = cv2.resize(comp[y:y+s, x:x+s], (PANE, PANE), interpolation=cv2.INTER_AREA)

        gap = 14
        sheet = np.full((PANE + 74, PANE * 2 + gap, 3), 18, np.uint8)
        sheet[0:PANE, 0:PANE] = left
        sheet[0:PANE, PANE+gap:] = right
        f = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(sheet, f"{rec}  side {side}", (8, PANE + 30), f, 0.7,
                    (225, 220, 210), 1, cv2.LINE_AA)
        cv2.putText(sheet, "the record", (8, PANE + 58), f, 0.55,
                    (150, 145, 138), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"green = shot 1 ({len(a['marks'])})   "
                           f"red = shot 2 ({len(b['marks'])})   "
                           f"agreeing on {both}   rotation {delta:.0f} deg",
                    (PANE + gap + 8, PANE + 58), f, 0.55, (150, 145, 138), 1, cv2.LINE_AA)

        fn = f"{made+1:02d}_{rec[:26]}_{side}.jpg"
        cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(
            os.path.join(OUT, fn))
        print(f"  {fn:<44} shot1 {len(a['marks']):>3}   shot2 {len(b['marks']):>3}"
              f"   agree {both:>3}   rot {delta:>5.0f}")
        made += 1

    print(f"\n{made} sheets written to {OUT}")


if __name__ == "__main__":
    main()
