# -*- coding: utf-8 -*-
"""
Build a small, deliberately varied gallery of what the detector currently does.

Ten photos are chosen to span the range actually present in the data — from
records with no marks at all to the most heavily marked — because a gallery of
only easy or only hard cases gives a misleading impression of where the model
stands.

Each image carries a caption strip with its own numbers, so a picture can be
read without cross-referencing a table:

    GREEN  the scratches marked by hand
    YELLOW what the model reports
    a scratch counts as found when a detection lands within TOLERANCE of it

Usage:  python make_examples.py [output folder]
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
from evaluate_frozen import detect, score, TOLERANCE, MIN_EXTRA_AREA

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(HERE), "Model_Examples")

# how many to take from each band, chosen to cover the whole range
WANT = [("clean", 2), ("light", 3), ("medium", 3), ("heavy", 2)]
BAR = 132


def band(n):
    return "clean" if n == 0 else ("light" if n <= 3 else
                                   "medium" if n <= 10 else "heavy")


def caption(img, lines):
    """A strip above the photo, so nothing is drawn over the record itself."""
    w = img.shape[1]
    strip = np.full((BAR, w, 3), 22, np.uint8)
    scale = max(w / 1400.0, 0.55)
    y = int(42 * scale) + 8
    for text, colour in lines:
        cv2.putText(strip, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9 * scale, colour, max(1, int(2 * scale)), cv2.LINE_AA)
        y += int(40 * scale)
    return np.vstack([strip, img])


def ring_detections(img, det, colour=(0, 140, 255), thickness=4):
    """Outline each detection instead of filling it.

    The annotator's own pen is blue and stays in the photo untouched, so the
    model's marker must be told apart from it without relying on colour alone:
    it is an open ring (a different SHAPE) drawn in orange, which is the pairing
    that stays distinguishable under the common forms of colour blindness. An
    outline of fixed width also reads evenly everywhere, unlike a soft fill,
    whose brightness varied with how many pixels happened to overlap.
    """
    vis = img.copy()
    n, labels, stats, cent = cv2.connectedComponentsWithStats(
        (det > 127).astype(np.uint8), connectivity=8)
    drawn = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_EXTRA_AREA:
            continue
        cx, cy = int(cent[i][0]), int(cent[i][1])
        r = int(max(w, h) / 2) + 26
        cv2.circle(vis, (cx, cy), r, (20, 20, 20), thickness + 4, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), r, colour, thickness, cv2.LINE_AA)
        drawn.append((len(drawn) + 1, cx, cy, r))

    for k, cx, cy, r in drawn:
        px, py = cx + int(r * 0.72), cy - int(r * 0.72)
        cv2.circle(vis, (px, py), 20, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(vis, (px, py), 20, colour, 3, cv2.LINE_AA)
        t = str(k)
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(vis, t, (px - tw // 2, py + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return vis, len(drawn)


def marked_photo(pair_row):
    """The file of the pair that actually carries the pen, so the blue marks are
    visible in their original colour rather than being repainted."""
    base = pair_row["pair"]
    candidates = [f for f in os.listdir(PHOTOS)
                  if f.lower().endswith((".jpg", ".jpeg"))
                  and re.sub(r"\(1\)$", "", re.sub(r"\.jpe?g$", "", f, flags=re.I)
                             ).strip().lower() == base]
    best, best_blue = None, -1
    for f in candidates:
        img = detector.load_image(os.path.join(PHOTOS, f))
        blue = int(np.count_nonzero(cv2.inRange(
            cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (95, 120, 90), (125, 255, 255))))
        if blue > best_blue:
            best, best_blue = img, blue
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    cal = set(split["cal"])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in cal]

    # one photo per record at most, so ten examples mean ten different records
    by_band, seen = {b: [] for b, _ in WANT}, set()
    for r in sorted(rows, key=lambda r: int(r["marks"])):
        if r["record"] in seen:
            continue
        b = band(int(r["marks"]))
        if b in by_band:
            by_band[b].append(r)
            seen.add(r["record"])

    picked = []
    for b, n in WANT:
        pool = by_band[b]
        if not pool:
            continue
        step = max(len(pool) // max(n, 1), 1)      # spread across the band
        picked += pool[::step][:n]

    print(f"{len(picked)} examples\n")
    print(f"{'file':<40}{'marked':>8}{'found':>7}{'shown':>7}")

    for i, r in enumerate(sorted(picked, key=lambda r: int(r["marks"])), 1):
        photo = os.path.join(PHOTOS, r["photo_file"])
        gt_path = os.path.join(GT, r["pair"] + ".png")
        img, det = detect(photo)
        gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        found, zones, extra, shown = score(det, gt)

        # draw on the photo that carries the pen, so the blue stays as drawn
        base = marked_photo(r)
        if base is None or base.shape[:2] != img.shape[:2]:
            base = img
        vis, n_rings = ring_detections(base, det)
        pct = f"{100*found/zones:.0f}%" if zones else "-"
        vis = caption(vis, [
            (f"{r['record'][:36]}  side {r['side']}  shot {r['shot']}",
             (235, 235, 235)),
            (f"BLUE lines = your marks ({zones})   "
             f"ORANGE rings = model ({n_rings})", (215, 215, 215)),
            (f"found {found} of {zones}  ({pct})    rings not near a mark: {extra}",
             (0, 170, 255)),
        ])

        name = f"{i:02d}_{band(zones)}_{r['record'][:26]}_{r['side']}{r['shot']}.jpg"
        cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(
            os.path.join(OUT, name))
        print(f"{name[:38]:<40}{zones:>8}{found:>7}{shown:>7}")

    with open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            "What the detector currently reports, on ten photos spanning the\n"
            "whole range in the dataset.\n\n"
            "BLUE lines   your own marks, left exactly as you drew them\n"
            "ORANGE rings what the model reports, numbered\n\n"
            "The two are told apart by shape as well as colour: your marks are\n"
            "lines, the model's are open rings.\n\n"
            "  a blue line with no ring around it  = missed\n"
            "  a ring with no blue line inside it  = reported where you marked nothing\n"
            "  a ring around a blue line           = found\n\n"
            f"A marked scratch counts as found when a detection lands within\n"
            f"{TOLERANCE} px of it, since a hand-drawn line wanders a little.\n"
            f"Detections smaller than {MIN_EXTRA_AREA} px are ignored as speck-level.\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
