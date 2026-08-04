# -*- coding: utf-8 -*-
"""
Decisive test on what the per-shot detections actually are.

A mark fixed to the RECORD lines up when the two shots are aligned by the label.
A mark created by the LIGHTING stays where the lamp is, so it lines up at zero
offset instead. Measuring the overlap at both offsets says which we are seeing.
"""

import os
import sys

import cv2
import numpy as np

import detector
from detector import P
from multishot import analyse_photo, estimate_offset, AGREE_TOL

folder = sys.argv[1]
paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
               if f.lower().endswith((".jpg", ".jpeg", ".png")))
a, b = analyse_photo(paths[0]), analyse_photo(paths[1])
shift, sharp = estimate_offset(a["label"], b["label"])
width = a["ring"].shape[1]
print(f"label alignment: {360.0*shift/width:.1f} deg (sharpness {sharp:.1f})")


def hits_of(shot):
    smap = np.maximum(shot["radial"], shot["tram"]).astype(np.float32)
    valid = smap[shot["judgeable"]]
    thr = max(float(np.percentile(valid, P["PCT_WEAK"])), P["THR_FLOOR"] / 2)
    return (shot["judgeable"] & (smap > thr)).astype(np.uint8)


ha, hb = hits_of(a), hits_of(b)
kernel = np.ones((AGREE_TOL, AGREE_TOL), np.uint8)
ha_d = cv2.dilate(ha, kernel)
print(f"hit pixels: shot A={int(ha.sum())}  shot B={int(hb.sum())}")

for name, s in (("record-aligned (label)", shift), ("lighting-aligned (0 deg)", 0)):
    hb_s = np.roll(hb, s, axis=1)
    overlap = int(np.count_nonzero(ha_d & hb_s))
    pct = 100.0 * overlap / max(int(hb_s.sum()), 1)
    print(f"  {name:>26}: {overlap:6d} px of shot B agree ({pct:5.1f}%)")

# also sweep every offset, to see whether ANY rotation makes them agree
best = (0, -1)
for s in range(0, width, 20):
    ov = int(np.count_nonzero(ha_d & np.roll(hb, s, axis=1)))
    if ov > best[1]:
        best = (s, ov)
print(f"  best possible over all offsets: {360.0*best[0]/width:5.1f} deg -> {best[1]} px")

out = os.path.join(folder, "analysis")
os.makedirs(out, exist_ok=True)
vis = np.zeros((ha.shape[0], ha.shape[1], 3), np.uint8)
vis[:, :, 2] = cv2.dilate(ha, np.ones((5, 5), np.uint8)) * 255           # red = A
vis[:, :, 1] = cv2.dilate(np.roll(hb, shift, axis=1),
                          np.ones((5, 5), np.uint8)) * 255               # green = B
cv2.imencode(".jpg", vis)[1].tofile(os.path.join(out, "overlap.jpg"))
print("wrote analysis/overlap.jpg (red = shot A, green = shot B aligned, yellow = agree)")
