# -*- coding: utf-8 -*-
"""Build the labelling tool for TEST 1's confirmed detections at level 4.

The sweep left a hole it could not fill by itself: at the loosened thresholds,
121 of the 184 confirmed detections are ones nobody has ever judged, because the
hand labelling was done on what the CURRENT thresholds produce. Precision over
the judged third is 93%, but a number computed on a third of the evidence is not
a number worth deciding on.

So this asks about exactly the detections that are missing and nothing else:

  - level 4 thresholds, the operating point the sweep pointed at
  - only marks CONFIRMED in both shots of the side, since those are the ones the
    filter would keep and therefore the only ones whose verdict changes anything
  - not the ones sitting on a pen mark: those are correct by construction and
    were never worth your time
  - verdicts you already gave are carried over by position, so the same speck is
    never put in front of you twice

What is left is the genuinely new work. Open label.html, judge each crop, and
the page downloads labels_test01.json when you are done.

Usage:  python build_label_tool_test01.py [how many records]
"""

import hashlib
import json
import os
import shutil
import sys

import cv2
import numpy as np

from build_label_tool import (CROP_OUT, INHERIT_PX, VIEW_W, crop_window,
                              inherit, load_previous, ringed_crop, save_jpg)
from cross_shot import label_profile, rotation_from_label
from detector import P
from evaluate_frozen import TOLERANCE
from test_01_loosen_then_confirm import (LEVELS, WINDOW, analysed, gt_for,
                                         pick_sides)
from tune_alignment import offsets, refine

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "label_page.html")
OUT = os.path.join(HERE, "label_tool_test01")
PREV = os.path.join(HERE, "label_tool_cal")
SET = "test01"
LEVEL = "4_loose"


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    params = dict(LEVELS)[LEVEL]
    baseline = {k: P[k] for k in params}
    P.update(params)

    # read the earlier verdicts before anything is written, and from the tool
    # they were given in -- this folder has none of its own on a first build
    previous = load_previous(PREV, "cal")
    previous_here = load_previous(OUT, SET)
    for pair, rows in previous_here.items():
        previous.setdefault(pair, []).extend(rows)

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "photos"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "crops"), exist_ok=True)

    entries, on_mark, carried_n = [], 0, 0
    try:
        for rec, side, shots in pick_sides(want):
            (pair_a, path_a), (_, path_b) = shots
            try:
                a, b = analysed(path_a), analysed(path_b)
                delta, _ = rotation_from_label(label_profile(path_a),
                                               label_profile(path_b))
            except Exception as exc:
                print(f"  {rec} {side}: failed ({type(exc).__name__})")
                continue
            if delta is None:
                print(f"  {rec} {side}: no rotation, nothing confirmed")
                continue

            fixed, _, _, _ = refine(a["marks"], b["marks"], delta)
            kept = [k for k, d in
                    enumerate(offsets(a["marks"], b["marks"], fixed, WINDOW))
                    if d is not None]
            if not kept:
                continue

            img = a["img"]
            H, W = img.shape[:2]
            gt = gt_for(pair_a, img.shape)
            near_gt = (cv2.dilate((gt > 127).astype(np.uint8),
                                  np.ones((TOLERANCE, TOLERANCE), np.uint8)) > 0
                       if gt is not None else None)

            scale = VIEW_W / float(W)
            view = cv2.resize(img, (VIEW_W, int(round(H * scale))),
                              interpolation=cv2.INTER_AREA)
            view_name = f"photos/{pair_a}.jpg"
            save_jpg(view, os.path.join(OUT, view_name.replace("/", os.sep)), 82)

            asked = 0
            for k in kept:
                m = a["marks"][k]
                cx, cy = int(round(m["cx"])), int(round(m["cy"]))
                if near_gt is not None and near_gt[min(cy, H-1), min(cx, W-1)]:
                    on_mark += 1
                    continue
                ys, xs = np.where(a["labels"] == m["id"])
                bw = int(xs.max() - xs.min() + 1) if xs.size else 6
                bh = int(ys.max() - ys.min() + 1) if ys.size else 6
                blob = {"cx": cx, "cy": cy, "w": bw, "h": bh}

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
                entries.append(e)
                asked += 1
            print(f"  {rec[:26]:<28}{side}   confirmed {len(kept):>3}"
                  f"   to show {asked:>3}")
    finally:
        P.update(baseline)

    entries.sort(key=lambda e: (e["pair"], e["cy"], e["cx"]))
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

    print(f"\nlevel {LEVEL}   {params}")
    print(f"  confirmed detections shown : {len(entries)}")
    print(f"  already on a pen mark      : {on_mark}   (correct, not shown)")
    print(f"  verdicts carried over      : {carried_n}"
          f"   (within {INHERIT_PX} view px of one you already gave)")
    print(f"  LEFT FOR YOU TO JUDGE      : {len(entries) - carried_n}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
