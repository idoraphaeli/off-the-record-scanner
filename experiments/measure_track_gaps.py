# -*- coding: utf-8 -*-
"""How wide is the band between two tracks, in the pixels the model works in?

Ido's idea: a scratch cannot be wider than the separator between songs, so a
detection wider than that is not damage. The web has no fixed figure for it --
the width is a cutting-room decision and varies from pressing to pressing -- so
it gets measured here on our own records instead.

A separator runs all the way around the disc, which is what makes it easy to
find: unwrapped, it becomes a horizontal band, and averaging every column of the
ring collapses it into a peak in a one-dimensional brightness profile. A scratch
crosses only a few columns and averages away to nothing, so it cannot be mistaken
for one.

Width is measured across the peak at half its height, in ring rows -- and the
radial axis of the ring is 1:1 with photograph pixels, so the number is directly
comparable to the widths measured for scratches and dirt.

Usage:  python measure_track_gaps.py [record ...]
"""

import os
import sys

import cv2
import numpy as np

import detector
from detector import P

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
OUT = os.path.join(os.path.dirname(HERE), "Model_TrackGaps")

SMOOTH = 5          # rows; kills groove-level noise, keeps a band of ~5+ rows
MIN_PROMINENCE = 0.7   # grey levels a peak must stand above its surroundings.
                       # The separators are faint -- 2.0 found two of them on a
                       # side that plainly has eight.
MIN_GAP_ROWS = 8       # two peaks closer than this are the same separator


def radial_profile(path):
    """Mean brightness of the ring at each radius, and the disc radius."""
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner, outer = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer].astype(np.float32)
    # the median across angles, not the mean: a lamp lighting one sector would
    # drag a mean around, while a band that goes all the way round survives
    prof = np.median(ring, axis=1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, SMOOTH), 0).ravel()
    return prof, ring, inner, radius, img, center


def find_bands(prof):
    """Peaks in the profile, with the width of each at half its prominence."""
    # a short baseline: an 81-row window spans several separators at once and
    # lifts the floor to meet them, which is what hid most of them
    base = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 31), 0).ravel()
    lift = prof - base
    out = []
    n = len(lift)
    for i in range(2, n - 2):
        if not (lift[i] >= lift[i-1] and lift[i] >= lift[i+1] and lift[i] > MIN_PROMINENCE):
            continue
        if out and i - out[-1][0] < MIN_GAP_ROWS:
            if lift[i] > out[-1][1]:
                out[-1] = (i, lift[i], out[-1][2])
            continue
        half = lift[i] / 2.0
        a = i
        while a > 0 and lift[a] > half:
            a -= 1
        b = i
        while b < n - 1 and lift[b] > half:
            b += 1
        out.append((i, lift[i], b - a))
    return out


def main():
    names = sys.argv[1:] or ["zipi_shavit", "baldi_olier", "hio_laylot"]
    os.makedirs(OUT, exist_ok=True)
    all_w = []

    files = sorted(f for f in os.listdir(PHOTOS)
                   if f.lower().endswith((".jpg", ".jpeg")) and "(1)" not in f)
    for rec in names:
        pick = next((f for f in files if f.lower().startswith(rec.lower())), None)
        if not pick:
            print(f"  {rec}: no photo found")
            continue
        prof, ring, inner, radius, img, center = radial_profile(
            os.path.join(PHOTOS, pick))
        bands = find_bands(prof)
        widths = [w for _, _, w in bands]
        all_w += widths

        print(f"\n{rec}   (disc radius {radius} px, ring is {len(prof)} rows)")
        print(f"  separators found: {len(bands)}")
        if widths:
            v = np.array(widths, float)
            print(f"  width in pixels   median {np.median(v):.1f}"
                  f"   range {v.min():.0f} to {v.max():.0f}")
            mm = 130.0 / radius          # 12in LP: ~130 mm of playing surface
            print(f"  width in mm       median {np.median(v)*mm:.2f}"
                  f"   ({mm*1000:.0f} microns per pixel here)")

        # a picture of what was measured, so the number can be checked by eye
        vis = cv2.cvtColor(ring.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        vis = cv2.resize(vis, (900, vis.shape[0]), interpolation=cv2.INTER_AREA)
        for r, _, w in bands:
            cv2.line(vis, (0, r), (900, r), (0, 160, 255), 1)
        cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(
            os.path.join(OUT, f"{rec}_ring.jpg"))

    if all_w:
        v = np.array(all_w, float)
        print(f"\n{'='*58}")
        print(f"across {len(names)} records, {len(v)} separators")
        print(f"  median width  {np.median(v):.1f} px")
        print(f"  quartiles     {np.percentile(v,25):.1f} to {np.percentile(v,75):.1f} px")
        print(f"\nfor comparison, measured over 588 hand-verified scratches:")
        print(f"  scratch width  median  7.9 px")
        print(f"  dirt width     median 14.2 px")
    print(f"\nrings drawn to {OUT}")


if __name__ == "__main__":
    main()
