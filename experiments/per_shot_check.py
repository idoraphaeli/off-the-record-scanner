# -*- coding: utf-8 -*-
"""Sanity check before trusting a 'no damage' verdict: what does each shot find
on its own, and how strong is the response at all? A clean record and a record
photographed under light that hides everything both produce zero detections --
these numbers tell them apart."""

import os
import sys

import cv2
import numpy as np

import detector
from detector import P

folder = sys.argv[1]
paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
               if f.lower().endswith((".jpg", ".jpeg", ".png")))

for p in paths:
    img = detector.load_image(p)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner_px, outer_px = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner_px:outer_px]
    radial, tram = detector.scratch_map(ring)
    smap = np.maximum(radial, tram)
    judge = smap[smap > 0]
    thr = max(float(np.percentile(judge, P["PCT_STRONG"])), P["THR_FLOOR"])

    mask_a, sa = detector.extract(radial)
    mask_b, sb = detector.extract(tram, min_len=P["TRAM_MIN_LEN"])

    # how many components existed BEFORE the shape filter rejected them
    weak = (smap > max(float(np.percentile(judge, P["PCT_WEAK"])), P["THR_FLOOR"] / 2))
    ncomp, _, stats, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), 8)
    big = sum(1 for i in range(1, ncomp) if stats[i][4] >= 50)

    print(f"{os.path.basename(p)[-24:]:>26}")
    print(f"    ring brightness mean={ring.mean():5.1f}  p95={np.percentile(ring,95):5.0f}")
    print(f"    map: max={smap.max():3d}  p99.9={np.percentile(judge,99.9):5.1f}  strong_thr={thr:.0f}")
    print(f"    candidate blobs (>=50px, before shape filter): {big}")
    print(f"    kept after shape filter: radial={len(sa)} tram={len(sb)}")
    out = os.path.join(folder, "analysis")
    os.makedirs(out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(p))[0][-6:]
    amp = np.clip(smap.astype(np.float32) * 3, 0, 255).astype(np.uint8)
    cv2.imencode(".jpg", amp)[1].tofile(os.path.join(out, f"map_{tag}_x3.jpg"))
