# -*- coding: utf-8 -*-
"""
Draw the analysed band on real photos, so "widening the band" can be seen
instead of argued about.

The band is INSIDE the disc, not around it. The detector finds the disc's own
circle first, then analyses only the ring between LABEL_R and OUTER_R of that
radius: the label carries printed text and the outermost rim is a smooth
run-out, and both produce lines that are not damage. Everything outside the ring
is never looked at, so a marked scratch that lands there cannot be found at any
threshold.

Each picture shows the current ring, the proposed one, and every hand-marked
scratch that sits in the strip between them.

Usage:  python show_band.py [output folder]
"""

import csv
import json
import os
import re
import sys

import cv2
import numpy as np

import detector
from detector import P

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(HERE), "Model_Examples_Band")

NOW_IN, NOW_OUT = 0.40, 0.93
NEW_IN, NEW_OUT = 0.36, 0.97

WHITE = (255, 255, 255)
ORANGE = (0, 150, 255)
BAR = 150


def shade(img, center, radius, r_lo, r_hi, colour, alpha):
    """Tint one annulus, so the added strip reads as an area and not as a line."""
    ring = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(ring, center, int(r_hi * radius), 255, -1)
    cv2.circle(ring, center, int(r_lo * radius), 0, -1)
    tint = np.zeros_like(img)
    tint[:] = colour
    out = img.copy()
    out[ring > 0] = cv2.addWeighted(img, 1 - alpha, tint, alpha, 0)[ring > 0]
    return out


def caption(img, lines):
    w = img.shape[1]
    strip = np.full((BAR, w, 3), 22, np.uint8)
    scale = max(w / 1400.0, 0.55)
    y = int(40 * scale) + 10
    for text, colour in lines:
        cv2.putText(strip, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85 * scale, colour, max(1, int(2 * scale)), cv2.LINE_AA)
        y += int(38 * scale)
    return np.vstack([strip, img])


def marks_by_zone(gt, center, radius):
    """Split the hand-marked scratches by where they fall relative to the rings."""
    n, labels, stats, cent = cv2.connectedComponentsWithStats(
        (gt > 127).astype(np.uint8), connectivity=8)
    inside, added, lost = [], [], []
    for i in range(1, n):
        if stats[i][4] < 40:
            continue
        cx, cy = float(cent[i][0]), float(cent[i][1])
        frac = np.hypot(cx - center[0], cy - center[1]) / max(radius, 1)
        pt = (int(cx), int(cy))
        if NOW_IN <= frac <= NOW_OUT:
            inside.append(pt)
        elif NEW_IN <= frac <= NEW_OUT:
            added.append((pt, frac))
        else:
            lost.append((pt, frac))
    return inside, added, lost


def main():
    os.makedirs(OUT, exist_ok=True)
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    cal = set(split["cal"])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["record"] in cal and int(r["marks"]) > 0]

    # rank photos by how much the widening would actually buy them
    scored = []
    for r in rows:
        gt_path = os.path.join(GT, r["pair"] + ".png")
        photo = os.path.join(PHOTOS, r["photo_file"])
        if not (os.path.exists(gt_path) and os.path.exists(photo)):
            continue
        img = detector.load_image(photo)
        center, radius = detector.find_disc(img)
        gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        inside, added, lost = marks_by_zone(gt, center, radius)
        scored.append((len(added), r, img, center, radius, inside, added, lost))

    scored.sort(key=lambda s: -s[0])
    picked = scored[:4]

    total_added = sum(s[0] for s in scored)
    total_lost = sum(len(s[7]) for s in scored)
    total_in = sum(len(s[5]) for s in scored)
    print(f"across {len(scored)} marked calibration photos:")
    print(f"  marks already inside the current ring : {total_in}")
    print(f"  marks the widening would let us see   : {total_added}")
    print(f"  marks still outside even after it     : {total_lost}\n")

    for i, (n_add, r, img, center, radius, inside, added, lost) in enumerate(picked, 1):
        vis = img.copy()
        # the two strips the widening adds, tinted so they read as areas
        vis = shade(vis, center, radius, NEW_IN, NOW_IN, ORANGE, 0.30)
        vis = shade(vis, center, radius, NOW_OUT, NEW_OUT, ORANGE, 0.30)

        for rr, colour, thick in ((NOW_IN, WHITE, 3), (NOW_OUT, WHITE, 3),
                                  (NEW_IN, ORANGE, 3), (NEW_OUT, ORANGE, 3)):
            cv2.circle(vis, center, int(rr * radius), (15, 15, 15), thick + 4,
                       cv2.LINE_AA)
            cv2.circle(vis, center, int(rr * radius), colour, thick, cv2.LINE_AA)
        cv2.circle(vis, center, radius, (15, 15, 15), 6, cv2.LINE_AA)
        cv2.circle(vis, center, radius, (120, 255, 120), 2, cv2.LINE_AA)

        for k, ((cx, cy), frac) in enumerate(added, 1):
            cv2.circle(vis, (cx, cy), 46, (15, 15, 15), 9, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 46, ORANGE, 5, cv2.LINE_AA)
            cv2.putText(vis, f"{frac:.2f}R", (cx + 52, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (15, 15, 15), 6, cv2.LINE_AA)
            cv2.putText(vis, f"{frac:.2f}R", (cx + 52, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, ORANGE, 2, cv2.LINE_AA)

        vis = caption(vis, [
            (f"{r['record'][:38]}  side {r['side']}  shot {r['shot']}",
             (235, 235, 235)),
            ("GREEN = edge of the disc.  WHITE = ring analysed today "
             f"({NOW_IN:.2f}R - {NOW_OUT:.2f}R)", (200, 255, 200)),
            (f"ORANGE = proposed ring ({NEW_IN:.2f}R - {NEW_OUT:.2f}R). "
             "The tinted strips are what it adds - both still inside the disc.",
             ORANGE),
            (f"marks inside today: {len(inside)}    marks the widening "
             f"reaches: {len(added)}    still out: {len(lost)}", (235, 235, 235)),
        ])

        name = f"band_{i:02d}_{r['record'][:26]}_{r['side']}{r['shot']}.jpg"
        cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(
            os.path.join(OUT, name))
        print(f"  {name}")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
