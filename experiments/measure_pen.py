# -*- coding: utf-8 -*-
"""
Measure the pen the annotator actually used, instead of assuming it.

The editor's pen draws one uniform colour at a fixed minimum width, so both are
usable as filters — but only if the values come from the data. Differencing a
pair also catches resampling noise (183 of 200 copies were re-saved at another
size), and that noise is what inflated the mark count.

Samples the pixels where a pair differs, then reports the colour distribution
and the stroke width, so the filter can be set from measurement.
"""

import collections
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(HERE), "Records_Data_New_jpg")
SAMPLE_PAIRS = 25

COPY_RE = re.compile(r"\(1\)$")


def key(name):
    return COPY_RE.sub("", re.sub(r"\.jpe?g$", "", name, flags=re.I).strip()).lower()


def read(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


files = sorted(f for f in os.listdir(ROOT) if f.lower().endswith((".jpg", ".jpeg")))
plains = [f for f in files if not COPY_RE.search(re.sub(r"\.jpe?g$", "", f, flags=re.I))]
copies = {key(f): f for f in files if COPY_RE.search(re.sub(r"\.jpe?g$", "", f, flags=re.I))}

hues, sats, vals, widths = [], [], [], []
pairs_used = 0

for f in plains:
    if key(f) not in copies or pairs_used >= SAMPLE_PAIRS:
        continue
    a, b = read(os.path.join(ROOT, f)), read(os.path.join(ROOT, copies[key(f)]))
    if a is None or b is None:
        continue
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    diff = cv2.absdiff(a, b).max(axis=2)
    # look only where the two differ STRONGLY: resampling noise is small, a pen
    # stroke laid over black vinyl is a large change
    strong = diff > 90
    if np.count_nonzero(strong) < 200:
        continue

    # the drawn-on image is whichever is brighter-blue at those pixels
    ba = a[:, :, 0].astype(np.int16) - np.maximum(a[:, :, 1], a[:, :, 2])
    bb = b[:, :, 0].astype(np.int16) - np.maximum(b[:, :, 1], b[:, :, 2])
    marked = b if bb[strong].mean() > ba[strong].mean() else a

    hsv = cv2.cvtColor(marked, cv2.COLOR_BGR2HSV)
    hues.extend(hsv[:, :, 0][strong].ravel().tolist())
    sats.extend(hsv[:, :, 1][strong].ravel().tolist())
    vals.extend(hsv[:, :, 2][strong].ravel().tolist())

    # stroke width: twice the peak distance-to-edge inside the changed region
    dist = cv2.distanceTransform(strong.astype(np.uint8), cv2.DIST_L2, 5)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        strong.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if stats[i][4] < 120:
            continue
        widths.append(2 * float(dist[labels == i].max()))
    pairs_used += 1

print(f"sampled {pairs_used} pairs, {len(hues)} changed pixels\n")

h = np.array(hues)
s = np.array(sats)
v = np.array(vals)

print("HUE of the changed pixels (OpenCV scale 0-179; blue sits near 100-120):")
hist = collections.Counter((h // 10 * 10).tolist())
for bucket in sorted(hist):
    bar = "#" * int(round(50 * hist[bucket] / max(hist.values())))
    print(f"    {bucket:>3}-{bucket+9:<4}{hist[bucket]:>9}  {bar}")

blue = (h >= 100) & (h <= 125)
print(f"\n  pixels in the blue band: {100*blue.mean():.1f}%")
if blue.any():
    print(f"  among those  hue  median {np.median(h[blue]):.0f}"
          f"  (p5 {np.percentile(h[blue],5):.0f} - p95 {np.percentile(h[blue],95):.0f})")
    print(f"               sat  median {np.median(s[blue]):.0f}"
          f"  (p5 {np.percentile(s[blue],5):.0f} - p95 {np.percentile(s[blue],95):.0f})")
    print(f"               val  median {np.median(v[blue]):.0f}"
          f"  (p5 {np.percentile(v[blue],5):.0f} - p95 {np.percentile(v[blue],95):.0f})")

w = np.array(widths)
if len(w):
    print(f"\nSTROKE WIDTH of changed regions ({len(w)} regions):")
    print(f"    median {np.median(w):.1f} px"
          f"   p10 {np.percentile(w,10):.1f}   p90 {np.percentile(w,90):.1f}"
          f"   min {w.min():.1f}   max {w.max():.1f}")
    print(f"    thinner than 3 px: {100*(w<3).mean():.0f}%"
          f"   (these are resampling noise, not pen)")
