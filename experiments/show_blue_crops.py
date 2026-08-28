# -*- coding: utf-8 -*-
"""
Find the crops in the labelling tool that contain pen-blue pixels, and lay a few
of them out for the annotator to judge.

The doubt is simple: if the detector has been reading the pen-marked copy of a
pair rather than the clean one, then some of what it reports is his own ink, and
recall was measured against an answer sheet the model could see. Rather than
guess from colour statistics — black vinyl has a bluish sheen worth far more
pixels than a pen stroke, so the counting is unreliable — the candidates are
shown to the only person who knows what he drew.

Usage:  python show_blue_crops.py [how many]
"""

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "label_tool")
OUT = os.path.join(os.path.dirname(HERE), "Model_Examples_BlueCheck")

PEN_LO, PEN_HI = (95, 120, 90), (125, 255, 255)
WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 6
COLS = 3
BAR = 46


def pen_pixels(img):
    m = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), PEN_LO, PEN_HI)
    # a real stroke survives an opening; scattered sheen does not
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    index = json.load(open(os.path.join(TOOL, "index.json"), encoding="utf-8"))

    scored = []
    for e in index["items"]:
        path = os.path.join(TOOL, e["crop"].replace("/", os.sep))
        if not os.path.exists(path):
            continue
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        n = int(np.count_nonzero(pen_pixels(img)))
        if n > 0:
            scored.append((n, e, img))

    scored.sort(key=lambda s: -s[0])
    total = len(scored)
    print(f"{total} of {index['total']} crops contain pen-blue pixels "
          f"({100*total/max(index['total'],1):.1f}%)\n")

    tiles = []
    for k, (n, e, img) in enumerate(scored[:WANT], 1):
        strip = np.full((BAR, img.shape[1], 3), 22, np.uint8)
        cv2.putText(strip, f"{k}   {e['record'][:24]}  side {e['side']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (240, 240, 240), 2, cv2.LINE_AA)
        tiles.append(np.vstack([strip, img]))
        print(f"  {k}. {e['record'][:34]:<36} side {e['side']} shot {e['shot']}"
              f"   {n} blue px   file: {e['pair'][:30]}")

    while len(tiles) % COLS:
        tiles.append(np.full_like(tiles[0], 22))
    rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
    sheet = np.vstack(rows)
    path = os.path.join(OUT, "blue_check.jpg")
    cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
