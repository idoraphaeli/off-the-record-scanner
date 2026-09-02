# -*- coding: utf-8 -*-
"""Ten records, each as the plain photograph and the same photograph marked.

The marking is the server's own: the same soft yellow, the same translucency,
the same halo drawn wider than the mark so a hairline shows at all, and the same
clip at the edge of the playing surface. Nothing here is drawn for illustration
-- what these images show is what the app would show.

The one difference is WHICH marks get painted. The app paints everything the
model found; here only the marks that appeared in BOTH shots of the side are
painted, so the question can be looked at directly: if the cross-check were used
to throw detections away rather than to weigh them, is what remains a fair
picture of the record.

The angle between the two shots is corrected before matching and the window is
the tightened +/-2 degrees, so the filter is being judged at its best.

Usage:  python make_confirmed_pairs.py [how many records] [seed]
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
from detector import P
from evaluate_frozen import MIN_EXTRA_AREA, detect
from tune_alignment import offsets, refine

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
OUT = os.path.join(os.path.dirname(HERE), "Model_ConfirmedOnly")

WINDOW = 2.0
# the server's own drawing constants, copied rather than imported: the server
# package will not load outside its own container
MARK_ALPHA = 0.45
MARK_HALO = 9
YELLOW = (90, 255, 255)


def marks_of(path):
    img, det = detect(path)
    center, radius = detector.find_disc(img)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(
        (det > 127).astype(np.uint8), connectivity=8)
    marks = []
    for i in range(1, n):
        if stats[i][4] < MIN_EXTRA_AREA:
            continue
        dx, dy = cent[i][0] - center[0], cent[i][1] - center[1]
        marks.append({"id": i,
                      "rad": math.hypot(dx, dy) / max(radius, 1),
                      "ang": math.degrees(math.atan2(dy, dx)) % 360.0})
    return {"img": img, "labels": lab, "marks": marks,
            "center": center, "radius": radius}


def inside_disc(center, radius, shape):
    """The server's clamp: nothing may be painted past the analysed band.

    Drawn arithmetically rather than with a filled circle -- an antialiased
    circle leaves a fringe of part-value pixels at its boundary, and that fringe
    was once painting a dashed ring of its own around the rim.
    """
    h, w = shape[:2]
    limit = int(P["OUTER_R"] * radius)
    yy, xx = np.ogrid[:h, :w]
    return ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= limit ** 2)


def paint(img, det_mask, keep_inside):
    vis = img.copy().astype(np.float32)
    band = cv2.dilate((det_mask > 127).astype(np.uint8),
                      np.ones((MARK_HALO, MARK_HALO), np.uint8))
    band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.0)
    band = band * keep_inside
    a = np.clip(band, 0, 1)[..., None] * MARK_ALPHA
    return (vis * (1 - a) + np.array(YELLOW, np.float32) * a).astype(np.uint8)


def save(img, path):
    cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(path)


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 10
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

    by_record = collections.defaultdict(list)
    for (rec, side), paths in sorted(sides.items()):
        if len(paths) >= 2:
            by_record[rec].append((side, paths))
    picks = sorted(by_record)
    random.Random(seed).shuffle(picks)

    made, kept_all, found_all = 0, 0, 0
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

        fixed, moved, _, _ = refine(a["marks"], b["marks"], delta)
        hit = [d is not None for d in offsets(a["marks"], b["marks"], fixed, WINDOW)]
        ids = [m["id"] for m, f in zip(a["marks"], hit) if f]

        det = np.isin(a["labels"], ids).astype(np.uint8) * 255 if ids else \
            np.zeros(a["labels"].shape, np.uint8)
        marked = paint(a["img"], det,
                       inside_disc(a["center"], a["radius"], a["img"].shape))

        stem = f"{made+1:02d}_{rec[:28]}_{side}"
        save(a["img"], os.path.join(OUT, f"{stem}.jpg"))
        save(marked, os.path.join(OUT, f"{stem}_marked.jpg"))

        kept_all += len(ids)
        found_all += len(a["marks"])
        print(f"  {stem:<40} found {len(a['marks']):>3}   marked {len(ids):>3}"
              f"   correction {moved:>+5.1f} deg")
        made += 1

    if found_all:
        print(f"\n{made} records, {made*2} images. {kept_all} of {found_all} "
              f"detections are painted ({100*kept_all/found_all:.0f}%)")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
