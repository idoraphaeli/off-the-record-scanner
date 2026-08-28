# -*- coding: utf-8 -*-
"""
Every detection carrying one verdict, with the exact pixels the model fired on
painted in, not a ring around the area.

A ring says "something here"; it hides the shape, and the shape is what is being
argued about. Each example is shown twice side by side: the crop untouched, and
the same crop with the detected pixels painted bright yellow. The left panel
shows what is actually on the record, the right shows precisely what the model
claimed was a scratch.

Crops are cut at native resolution and enlarged afterwards, so the painted mark
stays as thin as the model drew it rather than being thickened by resampling.

Outputs into Model_<Verdict>_Painted/: one image per detection, plus sheets/

Usage:  python paint_false.py [cal|val|both] [false|scratch|dirt|unsure]
"""

import collections
import csv
import json
import os
import re
import sys

import cv2
import numpy as np

import detector
from evaluate_frozen import MIN_EXTRA_AREA, detect

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
SCANNER = os.path.dirname(HERE)

VIEW_W = 1100
YELLOW = (60, 255, 255)
MIN_CROP = 190          # native px; smaller crops lose all context
ZOOM = 2.6
COLS, ROWS = 4, 3
CAP = 30


def safe(name):
    return re.sub(r'[\\/:*?"<>|]', "_", name)[:30]


def panels(img, mask, cx, cy, w, h):
    H, W = img.shape[:2]
    half = max(MIN_CROP, int(max(w, h) * 2.2)) // 2
    y0, y1 = max(cy - half, 0), min(cy + half, H)
    x0, x1 = max(cx - half, 0), min(cx + half, W)
    raw = img[y0:y1, x0:x1]
    if raw.size == 0:
        return None
    sub = mask[y0:y1, x0:x1] > 127

    out_w = int(raw.shape[1] * ZOOM)
    out_h = int(raw.shape[0] * ZOOM)
    left = cv2.resize(raw, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    right = left.copy()
    # nearest-neighbour on the mask keeps the mark exactly as thin as the model
    # drew it; a smooth resize would fatten it and change what is being judged
    big = cv2.resize(sub.astype(np.uint8), (out_w, out_h),
                     interpolation=cv2.INTER_NEAREST) > 0
    right[big] = YELLOW

    gap = np.full((out_h, 8, 3), 22, np.uint8)
    return np.hstack([left, gap, right])


def label_bar(width, text, sub):
    bar = np.full((CAP + 22, width, 3), 22, np.uint8)
    cv2.putText(bar, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(bar, sub, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                YELLOW, 1, cv2.LINE_AA)
    return bar


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    verdict = sys.argv[2] if len(sys.argv) > 2 else "false"
    sets = ("cal", "val") if which == "both" else (which,)
    out = os.path.join(SCANNER, f"Model_{verdict.capitalize()}_Painted")
    os.makedirs(os.path.join(out, "sheets"), exist_ok=True)

    labels = collections.defaultdict(list)
    for s in sets:
        p = os.path.join(HERE, f"label_tool_{s}", f"labels_{s}.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p, encoding="utf-8"))["rows"]:
            if r["label"] == verdict:
                labels[r["pair"]].append(r)

    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        source = {r["pair"]: r["photo_file"] for r in csv.DictReader(fh)}

    made = []
    for pair, rows_l in sorted(labels.items()):
        name = source.get(pair)
        path = os.path.join(PHOTOS, name) if name else None
        if not path or not os.path.exists(path):
            continue
        try:
            img, det = detect(path)
        except Exception:
            continue
        n, lab, stats, cent = cv2.connectedComponentsWithStats(
            (det > 127).astype(np.uint8), connectivity=8)
        keep = [i for i in range(1, n) if stats[i][4] >= MIN_EXTRA_AREA]
        if not keep:
            continue
        k = VIEW_W / img.shape[1]
        arr = np.array([[cent[i][0] * k, cent[i][1] * k] for i in keep], float)

        for r in rows_l:
            d = np.hypot(arr[:, 0] - r["cx"], arr[:, 1] - r["cy"])
            j = int(d.argmin())
            if d[j] > 30:
                continue
            i = keep[j]
            x, y, w, h, area = stats[i]
            comp = np.zeros(det.shape, np.uint8)
            comp[lab == i] = 255
            pic = panels(img, comp, int(cent[i][0]), int(cent[i][1]), w, h)
            if pic is None:
                continue
            bar = label_bar(pic.shape[1],
                            f"{r['record'][:30]}  side {r['side']} shot {r['shot']}",
                            f"left: as photographed    right: what the model marked"
                            f"   ({int(area)} px)")
            tile = np.vstack([bar, pic])
            fname = (f"{safe(r['record'])}_{r['side']}{r['shot']}"
                     f"_{r['id'][:6]}.jpg")
            cv2.imencode(".jpg", tile, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1] \
                .tofile(os.path.join(out, fname))
            made.append((r["record"], os.path.join(out, fname)))
        print(f"  {pair[:46]:<48}{len(rows_l):>4}")

    made.sort()
    print(f"\n{len(made)} '{verdict}' detections painted")

    per = COLS * ROWS
    for page in range((len(made) + per - 1) // per):
        chunk = made[page * per:(page + 1) * per]
        tiles = []
        for _, f in chunk:
            im = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                tiles.append(cv2.resize(im, (620, 340),
                                        interpolation=cv2.INTER_AREA))
        if not tiles:
            continue
        while len(tiles) % COLS:
            tiles.append(np.full_like(tiles[0], 22))
        grid = np.vstack([np.hstack(tiles[i:i + COLS])
                          for i in range(0, len(tiles), COLS)])
        cv2.imencode(".jpg", grid, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(
            os.path.join(out, "sheets", f"{verdict}_painted_{page + 1:02d}.jpg"))

    with open(os.path.join(out, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            f"Every detection you judged {verdict.upper()}.\n\n"
            "Each image is one detection, twice:\n"
            "  LEFT   the record as photographed, untouched\n"
            "  RIGHT  the same view with the exact pixels the model marked,\n"
            "         painted yellow\n\n"
            "The yellow is drawn at the model's own resolution, so its\n"
            "thickness is the thickness the model actually found.\n"
            "sheets/ holds the same images 12 to a page.\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

