# -*- coding: utf-8 -*-
"""The two shots of a side, unwrapped and laid one on top of the other.

In the disc view a pair of matching marks is two small blobs somewhere on a
circle, and whether they line up has to be taken on trust. Unwrapped, the
rotation between the shots becomes a horizontal shift and nothing else, so the
second strip can be rolled back by the angle recovered from the label and the
two drawn into the SAME strip. A mark that survived both angles then lands on
itself: green sitting inside red, in one place.

Nothing is nudged apart. The only shift applied is the rotation between the
photographs, so any offset left between a green blob and its red one is real
alignment error and can be read off directly -- which is the point of drawing it
this way rather than side by side.

The red layer is drawn thicker than the green so a coincident pair still reads
as two marks: red shows as a halo around the green rather than being hidden
underneath it.

Height in the strip is distance from the centre; the top is the label end of the
playing surface and the bottom is the rim.

Usage:  python make_crossshot_strips.py [how many sides]
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
OUT = os.path.join(os.path.dirname(HERE), "Model_CrossShot_Strips")

WIDTH = 1800          # how wide the unwrapped strip is drawn
RED_FAT = 5           # px the red layer is grown, so green cannot bury it
RAD_TOL = 0.025
ANG_TOL = 6.0


def unwrapped(path):
    """The ring, its detection mask, and the marks in disc coordinates."""
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner, outer = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer]
    rad_map, tram = detector.scratch_map(ring)[:2]
    m1, _ = detector.extract(rad_map, None, ring, inner, radius)
    m2, _ = detector.extract(tram, P["TRAM_MIN_LEN"], ring, inner, radius)
    mask = cv2.bitwise_or(m1, m2)

    n, lb, stats, cent = cv2.connectedComponentsWithStats(
        (mask > 127).astype(np.uint8), connectivity=8)
    marks = []
    for i in range(1, n):
        if stats[i][4] < MIN_EXTRA_AREA:
            continue
        marks.append({"rad": (float(cent[i][1]) + inner) / max(radius, 1),
                      "ang": float(cent[i][0]) / P["POLAR_STEPS"] * 360.0})
    return {"ring": ring, "mask": mask, "marks": marks,
            "inner": inner, "radius": radius, "photo": img}


def confirmed(a_marks, b_marks, delta):
    """Which of shot 1's marks have a partner in shot 2, and how far off it sat.

    The residual is what is left over after the rotation has been taken out, so
    it is alignment error and nothing else. If it is a few tenths of a degree the
    label is doing its job; if it is systematic and large, the tolerance is being
    spent covering for the alignment instead of on real damage.
    """
    out, resid = [], []
    for m in a_marks:
        want = (m["ang"] + delta) % 360.0
        best = None
        for o in b_marks:
            if abs(m["rad"] - o["rad"]) > RAD_TOL:
                continue
            d = (o["ang"] - want + 180.0) % 360.0 - 180.0
            if abs(d) <= ANG_TOL and (best is None or abs(d) < abs(best)):
                best = d
        out.append(best is not None)
        if best is not None:
            resid.append(best)
    return out, resid


def sized(a, height):
    """One array at the drawing size.

    Both shots are brought to the same height even though the two photographs
    give slightly different disc radii -- a couple of rows apart, from how close
    the phone was held. Without that the same distance from the centre would land
    on a different row in each layer, and a matching pair would look like a miss.
    """
    return cv2.resize(a, (WIDTH, height), interpolation=cv2.INTER_NEAREST)


def draw_overlay(a, b, height, roll_cols):
    """Both shots' detections painted into one strip, drawn over shot 1's ring."""
    base = cv2.cvtColor(sized(a["ring"], height), cv2.COLOR_GRAY2BGR)
    base = (base.astype(np.float32) * 0.55).astype(np.uint8)   # dim, so marks pop

    ma = sized(a["mask"], height) > 127
    mb = sized(b["mask"], height) > 127
    if roll_cols:
        mb = np.roll(mb, roll_cols, axis=1)   # take the rotation out, nothing else

    # red first and fattened, green second and untouched: a mark found in both
    # shots reads as green with a red rim, which no single-shot mark can look like
    mb = cv2.dilate(mb.astype(np.uint8), np.ones((RED_FAT, RED_FAT), np.uint8)) > 0
    base[mb] = (60, 60, 240)
    base[ma] = (60, 240, 60)
    return base


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

    made, all_resid = 0, []
    for (rec, side), paths in sorted(sides.items()):
        if made >= want or len(paths) < 2:
            continue
        try:
            a, b = unwrapped(paths[0]), unwrapped(paths[1])
            delta, _ = cs.rotation_from_label(cs.label_profile(paths[0]),
                                              cs.label_profile(paths[1]))
        except Exception:
            continue
        if delta is None:
            continue

        flags, resid = confirmed(a["marks"], b["marks"], delta)
        both = int(sum(flags))
        all_resid += resid
        off = float(np.median(resid)) if resid else 0.0

        # roll the second layer back by the rotation, so the two share a frame.
        # WIDTH is not POLAR_STEPS, so the roll has to be scaled to the drawing.
        roll = int(round(-delta / 360.0 * WIDTH))
        h = min(a["ring"].shape[0], b["ring"].shape[0])
        strip = draw_overlay(a, b, h, roll)

        pad = 96
        sheet = np.full((h + pad, WIDTH, 3), 18, np.uint8)
        sheet[0:h] = strip
        f = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(sheet, f"{rec}   side {side}", (8, h + 30), f, 0.68,
                    (225, 220, 210), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"green = shot 1, {len(a['marks'])} marks      "
                           f"red = shot 2, {len(b['marks'])} marks, "
                           f"rolled back {delta:.0f} deg      agreeing on {both}",
                    (8, h + 58), f, 0.54, (150, 145, 138), 1, cv2.LINE_AA)
        cv2.putText(sheet, "left to right = around the disc    "
                           "top = near the label, bottom = the rim    "
                           "green inside red = a mark that survived both angles; "
                           "a lone blob appeared in one shot only",
                    (8, h + 82), f, 0.48, (120, 116, 110), 1, cv2.LINE_AA)

        fn = f"{made+1:02d}_{rec[:26]}_{side}.jpg"
        cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(
            os.path.join(OUT, fn))
        print(f"  {fn:<44} shot1 {len(a['marks']):>3}  shot2 {len(b['marks']):>3}"
              f"  agree {both:>3}  rot {delta:>5.0f}  median offset {off:>+5.2f} deg")
        made += 1

    if all_resid:
        v = np.abs(np.array(all_resid, float))
        print(f"\nalignment residual over {len(v)} matched pairs, after the rotation")
        print(f"  median {np.median(v):.2f} deg   90th pct {np.percentile(v, 90):.2f} deg"
              f"   (the match window is +/-{ANG_TOL:.0f} deg)")
    print(f"\n{made} strips written to {OUT}")


if __name__ == "__main__":
    main()
