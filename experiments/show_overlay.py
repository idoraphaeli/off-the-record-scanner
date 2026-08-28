# -*- coding: utf-8 -*-
"""
Both shots of a side, drawn on ONE photo, so the alignment can be judged by eye.

The second shot's detections are carried onto the first shot's photo using the
rotation measured from the label print. If that rotation is right, a real defect
appears twice in the same spot — a red ring with a green box around it. A
reflection appears once, because it belongs to the light rather than the record,
and the disc was moved between shots.

Both shots are marked with rings; only the colour separates them.

Usage:  python show_overlay.py [how many] [min marks]
"""

import collections
import csv
import json
import math
import os
import sys

import cv2
import numpy as np

import detector
import cross_shot as cs

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
OUT = os.path.join(os.path.dirname(HERE), "Model_Overlay")

VIEW_W = 1150
RED = (60, 60, 235)         # shot A — circles
GREEN = (90, 210, 90)       # shot B — squares
BAR = 128
RAD_TOL, ANG_TOL = 0.025, 6.0


def caption(img, lines):
    strip = np.full((BAR, img.shape[1], 3), 22, np.uint8)
    y = 34
    for text, colour, sz in lines:
        cv2.putText(strip, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, sz, colour,
                    2, cv2.LINE_AA)
        y += 32
    return np.vstack([strip, img])


def ring(vis, x, y, r):
    cv2.circle(vis, (x, y), r, (15, 15, 15), 8, cv2.LINE_AA)
    cv2.circle(vis, (x, y), r, RED, 3, cv2.LINE_AA)


def box(vis, x, y, r):
    cv2.circle(vis, (x, y), r, (15, 15, 15), 8, cv2.LINE_AA)
    cv2.circle(vis, (x, y), r, GREEN, 3, cv2.LINE_AA)


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    min_marks = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    os.makedirs(OUT, exist_ok=True)

    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    sides = collections.defaultdict(list)
    for r in rows:
        sides[(r["record"], r["side"])].append(r)

    # heavily marked sides first: that is where two shots have something to
    # agree or disagree about
    ranked = []
    for key, shots in sides.items():
        if len(shots) < 2:
            continue
        marks = max(int(s["marks"]) for s in shots)
        if marks >= min_marks:
            ranked.append((marks, key, shots))
    ranked.sort(reverse=True, key=lambda t: t[0])
    if not ranked:
        sys.exit("no side has two shots and that many marks")

    # the most damaged one, then a spread across the rest
    picks = [ranked[0]]
    rest = ranked[1:]
    if rest and want > 1:
        step = max(len(rest) // (want - 1), 1)
        picks += rest[::step][:want - 1]

    print(f"{'record':<34}{'side':>5}{'marks':>7}{'rot':>7}"
          f"{'A':>5}{'B':>5}{'both':>6}")
    for marks, key, shots in picks:
        a, b = shots[0], shots[1]
        pa = os.path.join(PHOTOS, a["photo_file"])
        pb = os.path.join(PHOTOS, b["photo_file"])
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        try:
            det_a = cs.polar_detections(pa)
            det_b = cs.polar_detections(pb)
            delta, _ = cs.rotation_from_label(cs.label_profile(pa),
                                              cs.label_profile(pb))
        except Exception as exc:
            print(f"  skip {key[0][:30]}: {type(exc).__name__}")
            continue
        if delta is None:
            print(f"  skip {key[0][:30]}: no rotation")
            continue

        img = detector.load_image(pa)
        center, radius = detector.find_disc(img)
        k = VIEW_W / img.shape[1]
        vis = cv2.resize(img, (VIEW_W, int(round(img.shape[0] * k))),
                         interpolation=cv2.INTER_AREA)
        cxv, cyv, rv = center[0] * k, center[1] * k, radius * k
        for f in (detector.P["LABEL_R"], detector.P["OUTER_R"]):
            cv2.circle(vis, (int(cxv), int(cyv)), int(rv * f), (80, 80, 80), 1,
                       cv2.LINE_AA)

        both = 0
        for p in det_a:
            x = int(cxv + p["rad"] * rv * math.cos(math.radians(p["ang"])))
            y = int(cyv + p["rad"] * rv * math.sin(math.radians(p["ang"])))
            ring(vis, x, y, 22)
            for q in det_b:
                if abs(p["rad"] - q["rad"]) > RAD_TOL:
                    continue
                want_ang = (p["ang"] + delta) % 360.0
                if abs((q["ang"] - want_ang + 180.0) % 360.0 - 180.0) <= ANG_TOL:
                    both += 1
                    break

        # shot B's detections rotated back onto shot A's photo
        for q in det_b:
            ang = (q["ang"] - delta) % 360.0
            x = int(cxv + q["rad"] * rv * math.cos(math.radians(ang)))
            y = int(cyv + q["rad"] * rv * math.sin(math.radians(ang)))
            box(vis, x, y, 22)

        vis = caption(vis, [
            (f"{key[0][:40]}   side {key[1]}   {marks} hand-marked scratches",
             (245, 245, 245), 0.72),
            (f"RED = shot {a['shot']}  ({len(det_a)})", RED, 0.66),
            (f"GREEN = shot {b['shot']}, rotated {delta:.0f} deg onto this"
             f" photo  ({len(det_b)})", GREEN, 0.66),
            (f"{both} of {len(det_a)} land on each other", (245, 245, 245), 0.66),
        ])
        name = f"ov_{marks:02d}marks_{key[0][:26]}_{key[1]}.jpg"
        cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(
            os.path.join(OUT, name))
        print(f"{key[0][:32]:<34}{key[1]:>5}{marks:>7}{delta:>7.0f}"
              f"{len(det_a):>5}{len(det_b):>5}{both:>6}")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
