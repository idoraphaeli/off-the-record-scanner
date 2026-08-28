# -*- coding: utf-8 -*-
"""
Did the old ground-truth extraction invent scratches on blue-lit records?

build_gt keeps a pixel when the two copies DIFFER and the drawn-on one is
pen-blue. On a record that is blue all over — blue lighting, a turquoise label —
the colour test passes across the whole disc, and re-encoding makes those same
pixels differ slightly. Verifying the folder rule showed exactly that: pairs
scoring 60,000-90,000 "blue changes" where nobody had drawn anything.

Ink is blue in one copy and NOT blue in the other, so recounting that way
isolates real strokes. Any pair with a high mark count but no ink is carrying
phantom scratches, which corrupt recall from both ends at once: the model is
credited for "finding" them, and they pad the denominator.

Usage:  python check_phantom_gt.py
"""

import csv
import glob
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCANNER = os.path.dirname(HERE)
PHOTOS = os.path.join(SCANNER, "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")

PEN_LO, PEN_HI = (95, 120, 90), (125, 255, 255)
COPY_RE = re.compile(r"\(1\)$")
MIN_INK = 200


def stem(n):
    return re.sub(r"\.jpe?g$", "", n, flags=re.I).strip()


def key(n):
    return COPY_RE.sub("", stem(n)).strip().lower()


def read(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def blue(img):
    return cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), PEN_LO, PEN_HI) > 0


def ink_pixels(a, b):
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        thr = 55
    else:
        thr = 30
    changed = cv2.absdiff(a, b).max(axis=2) > thr
    ba, bb = blue(a), blue(b)
    k = np.ones((3, 3), np.uint8)
    ia = cv2.morphologyEx((changed & ba & ~bb).astype(np.uint8), cv2.MORPH_OPEN, k)
    ib = cv2.morphologyEx((changed & bb & ~ba).astype(np.uint8), cv2.MORPH_OPEN, k)
    return int(np.count_nonzero(ia)), int(np.count_nonzero(ib))


def main():
    files = [f for f in os.listdir(PHOTOS) if f.lower().endswith((".jpg", ".jpeg"))]
    groups = {}
    for f in files:
        groups.setdefault(key(f), []).append(f)

    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    suspect, fine, total_marks, phantom_marks = [], 0, 0, 0
    for r in rows:
        group = groups.get(r["pair"], [])
        if len(group) != 2:
            continue
        a, b = read(os.path.join(PHOTOS, group[0])), read(os.path.join(PHOTOS, group[1]))
        if a is None or b is None:
            continue
        marks = int(r["marks"])
        total_marks += marks
        ia, ib = ink_pixels(a, b)
        if marks > 0 and max(ia, ib) < MIN_INK:
            suspect.append((marks, r["pair"], max(ia, ib)))
            phantom_marks += marks
        else:
            fine += 1

    suspect.sort(reverse=True)
    print(f"{len(rows)} pairs, {total_marks} marks in the current ground truth\n")
    print(f"  pairs whose marks are backed by real ink : {fine}")
    print(f"  pairs with marks but NO ink found        : {len(suspect)}")
    print(f"  marks on those pairs (phantom)           : {phantom_marks}"
          f"   ({100*phantom_marks/max(total_marks,1):.1f}% of all marks)")

    if suspect:
        print(f"\n{'pair':<56}{'marks':>7}{'ink px':>9}")
        print("-" * 74)
        for marks, pair, ink in suspect[:25]:
            print(f"{pair[:54]:<56}{marks:>7}{ink:>9}")
        if len(suspect) > 25:
            print(f"  ... and {len(suspect) - 25} more")


if __name__ == "__main__":
    main()
