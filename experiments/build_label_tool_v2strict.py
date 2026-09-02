# -*- coding: utf-8 -*-
"""Build the labelling tool for the strict model's marks.

The strict model -- the two shots intersected, the three anti-glare rules left
on, the bar as it ships -- measures 95.2% precision, and 40% of the marks that
figure rests on have never been judged by anyone. The hand labelling done so far
covered what OTHER settings produce; this model paints a different set, and a
precision computed on three fifths of it is not one to change the server over.

So this asks about exactly what is missing:

  - the strict model's marks, and nothing else
  - not the ones sitting on a pen mark: correct by construction, never worth
    your time
  - every verdict already given, from all three earlier rounds, carried over by
    position, so the same speck is never put in front of you twice
  - what is left comes FIRST in the page

Usage:  python build_label_tool_v2strict.py [how many records]
"""

import hashlib
import json
import os
import shutil
import sys

import cv2
import numpy as np

from build_label_tool import (INHERIT_PX, VIEW_W, crop_window, inherit,
                              load_previous, ringed_crop, save_jpg)
from compare_precision import BAR_NOW, detect_with
from evaluate_frozen import MIN_EXTRA_AREA, TOLERANCE
from model_v2 import align, maps_of
from test_01_loosen_then_confirm import gt_for, pick_sides

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "label_page.html")
OUT = os.path.join(HERE, "label_tool_v2strict")
SET = "v2strict"

PRIOR = [(os.path.join(HERE, "label_tool_cal"), "cal"),
         (os.path.join(HERE, "label_tool_test01"), "test01"),
         (os.path.join(HERE, "label_tool_test02"), "test02"),
         (OUT, SET)]


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    previous = {}
    for folder, which in PRIOR:
        for pair, rows in load_previous(folder, which).items():
            previous.setdefault(pair, []).extend(rows)

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "photos"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "crops"), exist_ok=True)

    entries, on_mark, carried_n = [], 0, 0
    for rec, side, shots in pick_sides(want):
        (pair_a, path_a), (_, path_b) = shots
        try:
            a, b = maps_of(path_a), maps_of(path_b)
        except Exception as exc:
            print(f"  {rec} {side}: failed ({type(exc).__name__})")
            continue
        gt = gt_for(pair_a, a["img"].shape)
        delta, _, _ = align(a, b)
        det = detect_with(a, b, delta, True, False, BAR_NOW)

        n, lab, stats, cent = cv2.connectedComponentsWithStats(
            (det > 127).astype(np.uint8), connectivity=8)
        blobs = [i for i in range(1, n) if stats[i][4] >= MIN_EXTRA_AREA]
        if not blobs:
            print(f"  {rec[:26]:<28}{side}   nothing marked")
            continue

        img = a["img"]
        H, W = img.shape[:2]
        near_gt = (cv2.dilate((gt > 127).astype(np.uint8),
                              np.ones((TOLERANCE, TOLERANCE), np.uint8)) > 0
                   if gt is not None else None)
        scale = VIEW_W / float(W)
        view = cv2.resize(img, (VIEW_W, int(round(H * scale))),
                          interpolation=cv2.INTER_AREA)
        view_name = f"photos/{pair_a}.jpg"
        save_jpg(view, os.path.join(OUT, view_name.replace("/", os.sep)), 82)

        asked, fresh = 0, 0
        for i in blobs:
            cx, cy = int(round(cent[i][0])), int(round(cent[i][1]))
            if near_gt is not None and near_gt[min(cy, H-1), min(cx, W-1)]:
                on_mark += 1
                continue
            x, y, bw, bh, _ = stats[i]
            blob = {"cx": cx, "cy": cy, "w": int(bw), "h": int(bh)}

            uid = hashlib.md5(f"{pair_a}|{cx}|{cy}".encode()).hexdigest()[:12]
            crop_name = f"crops/{uid}.jpg"
            save_jpg(ringed_crop(img, blob, crop_window(blob, H, W)),
                     os.path.join(OUT, crop_name.replace("/", os.sep)))

            e = {"id": uid, "tier": 0, "pair": pair_a, "record": rec,
                 "side": side, "shot": "1",
                 "photo": view_name, "crop": crop_name,
                 "cx": round(cx * scale, 1), "cy": round(cy * scale, 1),
                 "r": round((max(bw, bh) * 0.5 + 22) * scale, 1),
                 "vw": view.shape[1], "vh": view.shape[0]}
            got = inherit(previous.get(pair_a, []), e["cx"], e["cy"])
            if got:
                e["prefill"] = got
                carried_n += 1
            else:
                fresh += 1
            entries.append(e)
            asked += 1
        print(f"  {rec[:26]:<28}{side}   marks {len(blobs):>3}"
              f"   shown {asked:>3}   new to you {fresh:>3}")

    # the ones nobody has seen come first, so a session that stops early still
    # closes the part of the figure that is actually open
    entries.sort(key=lambda e: ("prefill" in e, e["pair"], e["cy"], e["cx"]))
    for i, e in enumerate(entries):
        e["n"] = i + 1

    data = {"set": SET, "mode": "extra", "total": len(entries),
            "matched": on_mark, "items": entries}
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    with open(os.path.join(OUT, "index.js"), "w", encoding="utf-8") as fh:
        fh.write("window.LABEL_DATA = ")
        json.dump(data, fh, ensure_ascii=False)
        fh.write(";\n")
    shutil.copyfile(TEMPLATE, os.path.join(OUT, "label.html"))

    print(f"\nthe strict model: both shots intersected, rules on, bar as it ships")
    print(f"  marks shown           : {len(entries)}")
    print(f"  already on a pen mark : {on_mark}   (correct, not shown)")
    print(f"  verdicts carried over : {carried_n}"
          f"   (within {INHERIT_PX} view px of one you already gave)")
    print(f"  LEFT FOR YOU TO JUDGE : {len(entries) - carried_n}"
          f"   (they come FIRST in the page)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
