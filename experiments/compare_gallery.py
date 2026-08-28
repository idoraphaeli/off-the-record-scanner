# -*- coding: utf-8 -*-
"""
Same photo, current model on the left, candidate on the right — so a change in
recall can be looked at instead of read off a table.

Three glyphs, told apart by SHAPE and not by colour alone:

    circle   a detection that matches one of the hand-drawn marks
    square   a detection with no mark near it
    X        a mark the model did not find

The annotator's own blue pen is left in the photo exactly as drawn, so the
"truth" in the picture is his, not something repainted by this script.

Usage:  python compare_gallery.py [variant name from frontier_cal.json]
"""

import copy
import csv
import json
import os
import re
import sys

import cv2
import numpy as np

import detector
from detector import P
from evaluate_frozen import TOLERANCE, MIN_EXTRA_AREA
from sweep_recall2 import extract_graded, detect_with

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
OUT = os.path.join(os.path.dirname(HERE), "Model_Examples_Compare")

BASE = copy.deepcopy(P)
# the elbow of the measured curve: the last point before the price per
# recovered scratch doubles. Thresholds are deliberately left untouched.
DEFAULT_AFTER = {"MIN_LEN": 15, "MAX_THICK": 16, "SHORT_LEN": 30,
                 "SHORT_ELONG": 6.0, "LABEL_R": 0.36, "OUTER_R": 0.95}

ORANGE = (0, 150, 255)
WHITE = (245, 245, 245)
BAR = 118
N_EXAMPLES = 6


def blobs(mask):
    n, labels, stats, cent = cv2.connectedComponentsWithStats(
        (mask > 127).astype(np.uint8), connectivity=8)
    return [(int(cent[i][0]), int(cent[i][1]), stats[i][2], stats[i][3])
            for i in range(1, n) if stats[i][4] >= MIN_EXTRA_AREA]


def draw_circle(vis, cx, cy, r, colour):
    cv2.circle(vis, (cx, cy), r, (15, 15, 15), 9, cv2.LINE_AA)
    cv2.circle(vis, (cx, cy), r, colour, 4, cv2.LINE_AA)


def draw_square(vis, cx, cy, r, colour):
    p1, p2 = (cx - r, cy - r), (cx + r, cy + r)
    cv2.rectangle(vis, p1, p2, (15, 15, 15), 9, cv2.LINE_AA)
    cv2.rectangle(vis, p1, p2, colour, 4, cv2.LINE_AA)


def draw_x(vis, cx, cy, r, colour):
    for (a, b) in (((cx - r, cy - r), (cx + r, cy + r)),
                   ((cx - r, cy + r), (cx + r, cy - r))):
        cv2.line(vis, a, b, (15, 15, 15), 11, cv2.LINE_AA)
        cv2.line(vis, a, b, colour, 4, cv2.LINE_AA)


def annotate(img, det, gt):
    """Draw the three glyphs and return the counts behind them."""
    vis = img.copy()
    det_b = (det > 127).astype(np.uint8)
    gt_b = (gt > 127).astype(np.uint8)
    near_det = cv2.dilate(det_b, np.ones((TOLERANCE, TOLERANCE), np.uint8))
    near_gt = cv2.dilate(gt_b, np.ones((TOLERANCE, TOLERANCE), np.uint8))

    hit = extra = 0
    for cx, cy, w, h in blobs(det):
        r = max(w, h) // 2 + 30
        if near_gt[min(cy, near_gt.shape[0] - 1), min(cx, near_gt.shape[1] - 1)]:
            draw_circle(vis, cx, cy, r, ORANGE)
            hit += 1
        else:
            draw_square(vis, cx, cy, r, WHITE)
            extra += 1

    missed = 0
    n, labels, stats, cent = cv2.connectedComponentsWithStats(gt_b, connectivity=8)
    for i in range(1, n):
        if np.count_nonzero(near_det[labels == i]) >= 8:
            continue
        cx, cy = int(cent[i][0]), int(cent[i][1])
        draw_x(vis, cx, cy, max(stats[i][2], stats[i][3]) // 2 + 26, WHITE)
        missed += 1
    return vis, hit, extra, missed


def panel(vis, title, lines):
    w = vis.shape[1]
    strip = np.full((BAR, w, 3), 22, np.uint8)
    scale = max(w / 1400.0, 0.55)
    cv2.putText(strip, title, (18, int(42 * scale) + 6), cv2.FONT_HERSHEY_SIMPLEX,
                1.0 * scale, (255, 255, 255), max(2, int(3 * scale)), cv2.LINE_AA)
    y = int(84 * scale) + 6
    for text, colour in lines:
        cv2.putText(strip, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.78 * scale, colour, max(1, int(2 * scale)), cv2.LINE_AA)
        y += int(34 * scale)
    return np.vstack([strip, vis])


def marked_photo(pair):
    """The copy of the pair that carries the pen, so the blue stays as drawn."""
    best, best_blue = None, -1
    for f in os.listdir(PHOTOS):
        if not f.lower().endswith((".jpg", ".jpeg")):
            continue
        if re.sub(r"\(1\)$", "", re.sub(r"\.jpe?g$", "", f, flags=re.I)
                  ).strip().lower() != pair:
            continue
        img = detector.load_image(os.path.join(PHOTOS, f))
        blue = int(np.count_nonzero(cv2.inRange(
            cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (95, 120, 90), (125, 255, 255))))
        if blue > best_blue:
            best, best_blue = img, blue
    return best


def render(changes):
    ex = extract_graded if "SHORT_ELONG" in changes else detector.extract
    P.clear()
    P.update(copy.deepcopy(BASE))
    P.update(changes)
    return ex


def main():
    os.makedirs(OUT, exist_ok=True)
    after = DEFAULT_AFTER
    label = "candidate"
    if len(sys.argv) > 1:
        path = os.path.join(HERE, "frontier_cal.json")
        rows_f = json.load(open(path, encoding="utf-8"))
        match = [r for r in rows_f if r["name"] == sys.argv[1]]
        if match:
            after, label = match[0]["params"], match[0]["name"]

    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    cal = set(split["cal"])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in cal]

    # a spread: some clean, some light, some heavy, one record each
    seen, picked = set(), []
    for r in sorted(rows, key=lambda r: int(r["marks"])):
        if r["record"] in seen:
            continue
        seen.add(r["record"])
        picked.append(r)
    # even spread that INCLUDES the last entry: taking every N-th dropped the
    # heavily marked records, which are the ones a recall change shows up on
    idx = np.linspace(0, len(picked) - 1, N_EXAMPLES).round().astype(int)
    picked = [picked[i] for i in dict.fromkeys(idx.tolist())]

    print(f"candidate = {label}\n")
    print(f"{'file':<34}{'before':>18}{'after':>18}")
    print(f"{'':<34}{'found/extra/miss':>18}{'found/extra/miss':>18}")

    for i, r in enumerate(sorted(picked, key=lambda r: int(r["marks"])), 1):
        photo = os.path.join(PHOTOS, r["photo_file"])
        gt_path = os.path.join(GT, r["pair"] + ".png")
        if not (os.path.exists(photo) and os.path.exists(gt_path)):
            continue

        base_img = marked_photo(r["pair"])
        panels = []
        counts = []
        for name, changes in (("CURRENT MODEL", {}), (f"PROPOSED", after)):
            ex = render(changes)
            img, det = detect_with(photo, ex)
            gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
            canvas = base_img if (base_img is not None and
                                  base_img.shape[:2] == img.shape[:2]) else img
            vis, hit, extra, missed = annotate(canvas, det, gt)
            total = hit + missed
            pct = f"{100*hit/total:.0f}%" if total else "-"
            panels.append(panel(vis, name, [
                (f"O found {hit} of {total}  ({pct})", ORANGE),
                (f"[] reported, not marked: {extra}", WHITE),
                (f"X your marks it missed: {missed}", WHITE),
            ]))
            counts.append((hit, extra, missed))

        h = min(p.shape[0] for p in panels)
        combo = np.hstack([p[:h] for p in panels])
        gap = np.full((combo.shape[0], 14, 3), 22, np.uint8)
        combo = np.hstack([panels[0][:h], gap, panels[1][:h]])

        name = f"cmp_{i:02d}_{r['record'][:24]}_{r['side']}{r['shot']}.jpg"
        cv2.imencode(".jpg", combo, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tofile(
            os.path.join(OUT, name))
        b, a = counts
        print(f"{name[:32]:<34}{f'{b[0]}/{b[1]}/{b[2]}':>18}"
              f"{f'{a[0]}/{a[1]}/{a[2]}':>18}")

    with open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            "Left: the model as it stands. Right: the proposed settings.\n"
            "The blue pen is yours, left exactly as you drew it.\n\n"
            "  O  circle   the model found one of your marks\n"
            "  []  square  the model reported something you did not mark\n"
            "  X   cross   one of your marks the model missed\n\n"
            "Fewer X on the right is the gain. More squares on the right is\n"
            "the price. Both are visible in the same picture on purpose.\n")
    P.clear()
    P.update(copy.deepcopy(BASE))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
