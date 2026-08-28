# -*- coding: utf-8 -*-
"""Every detection that was counted as a scratch because it sat on a pen mark.

613 detections were just admitted to the feature table on the strength of landing
within tolerance of a hand-drawn stroke. That is a claim, and it is worth being
able to check by eye rather than trusting the arithmetic: if the tolerance is too
generous, or a mask holds something that was never drawn, then hundreds of rows
labelled "scratch" are nothing of the kind and every conclusion drawn from them
is wrong.

Each image is the same spot twice: the clean photograph the model actually saw,
with the detection ringed, and beside it the copy Ido drew on. The blue line
should be running through the ring. Where it is not, the pairing is wrong.

Files are named so the worst cases sort to the top: the distance from the
detection to the nearest ink, in pixels, leads the filename.

Usage:  python show_onmark.py [cal|val|both] [max per record]
"""

import collections
import csv
import json
import math
import os
import re
import shutil
import sys

import cv2
import numpy as np

import detector
from detector import P
from evaluate_frozen import MIN_EXTRA_AREA
from build_features import ON_MARK_SLACK

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PHOTOS = os.path.join(ROOT, "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
OUT = os.path.join(ROOT, "Model_OnMark_Check")

CROP = 300          # px around the detection, at working scale
TILE = 440          # rendered size of each half
COPY_RE = re.compile(r"\(1\)$")
PEN_LO, PEN_HI = (95, 120, 90), (125, 255, 255)


def pair_key(name):
    return COPY_RE.sub("", re.sub(r"\.jpe?g$", "", name, flags=re.I).strip()).lower()


def penned_copy(by_pair, pair, plain_name, shape):
    """The half of the pair that carries the ink, resized onto the clean one."""
    others = [f for f in by_pair.get(pair, []) if f != plain_name]
    if not others:
        return None
    other = detector.load_image(os.path.join(PHOTOS, others[0]))
    if other is None:
        return None
    if other.shape[:2] != shape:
        other = cv2.resize(other, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return other


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    per_record = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    sets = ("cal", "val") if which == "both" else (which,)

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)

    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    wanted = set()
    for s in sets:
        wanted |= set(split[s])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in wanted]

    by_pair = {}
    for f in os.listdir(PHOTOS):
        if f.lower().endswith((".jpg", ".jpeg")):
            by_pair.setdefault(pair_key(f), []).append(f)

    made = collections.Counter()
    dists = []
    for r in rows:
        photo = os.path.join(PHOTOS, r["photo_file"])
        gtp = os.path.join(GT, r["pair"] + ".png")
        if not (os.path.exists(photo) and os.path.exists(gtp)):
            continue
        if made[r["record"]] >= per_record:
            continue

        img = detector.load_image(photo)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        center, radius = detector.find_disc(img)
        inner = int(P["LABEL_R"] * radius)
        ring = detector.unwrap(gray, center, radius)[inner:int(P["OUTER_R"] * radius)]
        rad_map, tram = detector.scratch_map(ring)[:2]
        m1, _ = detector.extract(rad_map)
        m2, _ = detector.extract(tram, min_len=P["TRAM_MIN_LEN"])
        det = detector.rewrap(cv2.bitwise_or(m1, m2), inner, center, radius, gray.shape)

        gt = cv2.imdecode(np.fromfile(gtp, np.uint8), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (gray.shape[1], gray.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        ink = (gt > 127).astype(np.uint8)
        if not ink.any():
            continue
        near = cv2.dilate(ink, np.ones((ON_MARK_SLACK, ON_MARK_SLACK), np.uint8))
        # distance to the nearest stroke, so each crop can say how close it was
        dist = cv2.distanceTransform(1 - ink, cv2.DIST_L2, 3)

        pen = penned_copy(by_pair, r["pair"], r["photo_file"], gray.shape)

        n, lb, stats, cent = cv2.connectedComponentsWithStats(
            (det > 127).astype(np.uint8), connectivity=8)
        H, W = gray.shape
        for i in range(1, n):
            if stats[i][4] < MIN_EXTRA_AREA:
                continue
            cx, cy = int(cent[i][0]), int(cent[i][1])
            if not near[min(cy, H - 1), min(cx, W - 1)]:
                continue                       # not an on-mark detection
            d = float(dist[min(cy, H - 1), min(cx, W - 1)])
            dists.append(d)

            half = CROP // 2
            x0, y0 = max(cx - half, 0), max(cy - half, 0)
            x1, y1 = min(cx + half, W), min(cy + half, H)
            clean = cv2.resize(img[y0:y1, x0:x1], (TILE, TILE),
                               interpolation=cv2.INTER_CUBIC)
            k = TILE / max(x1 - x0, 1)
            pt = (int((cx - x0) * k), int((cy - y0) * TILE / max(y1 - y0, 1)))
            rr = int(max(stats[i][2], stats[i][3]) * 0.5 * k) + 16
            cv2.circle(clean, pt, rr, (15, 15, 15), 6, cv2.LINE_AA)
            cv2.circle(clean, pt, rr, (0, 150, 255), 3, cv2.LINE_AA)

            if pen is not None:
                marked = cv2.resize(pen[y0:y1, x0:x1], (TILE, TILE),
                                    interpolation=cv2.INTER_CUBIC)
                cv2.circle(marked, pt, rr, (15, 15, 15), 5, cv2.LINE_AA)
                cv2.circle(marked, pt, rr, (0, 150, 255), 2, cv2.LINE_AA)
            else:
                marked = np.full((TILE, TILE, 3), 30, np.uint8)
                cv2.putText(marked, "no pen copy", (30, TILE // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            gap = 12
            panel = np.full((TILE + 46, TILE * 2 + gap, 3), 22, np.uint8)
            panel[0:TILE, 0:TILE] = clean
            panel[0:TILE, TILE + gap:] = marked
            f = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(panel, "what the model saw", (8, TILE + 30), f, 0.6,
                        (170, 165, 155), 1, cv2.LINE_AA)
            cv2.putText(panel, f"your pen  -  {d:.0f} px from the ink",
                        (TILE + gap + 8, TILE + 30), f, 0.6,
                        (90, 200, 255) if d <= ON_MARK_SLACK else (90, 90, 255),
                        1, cv2.LINE_AA)

            fn = f"{int(round(d)):03d}px_{r['record'][:22]}_{r['side']}{r['shot']}_{i}.jpg"
            cv2.imencode(".jpg", panel, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1] \
                .tofile(os.path.join(OUT, fn))
            made[r["record"]] += 1
            if made[r["record"]] >= per_record:
                break

    d = np.array(dists) if dists else np.array([0.0])
    print(f"\nwrote {sum(made.values())} comparisons to")
    print(f"  {OUT}")
    print(f"\nhow far each detection sat from the nearest ink:")
    for q in (50, 75, 90, 95, 99):
        print(f"   {q}th percentile  {np.percentile(d, q):>5.0f} px")
    print(f"   furthest          {d.max():>5.0f} px      (slack is {ON_MARK_SLACK})")
    print(f"\nfiles are named by that distance, so the loosest pairings sort first.")


if __name__ == "__main__":
    main()
