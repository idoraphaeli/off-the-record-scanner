# -*- coding: utf-8 -*-
"""
Two cheap checks on Records_Data_New before building anything on it.

  1. What format are the files really in? 112 of them failed to decode, and an
     iPhone that edits and re-saves a photo often writes HEIC while keeping a
     .jpg name — OpenCV cannot read HEIC.
  2. Does restricting the blue search to the GROOVE BAND recover a sane number
     of marks? Searching the whole frame found 9,638, because the background and
     some centre labels are blue; the pen, by contrast, is drawn on the playing
     surface, which is exactly the band the detector already isolates.

Read-only, no output files.
"""

import collections
import os
import re
import sys

import cv2
import numpy as np

import detector
from detector import P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(HERE), "Records_Data_New")

MAGIC = [(b"\xff\xd8\xff", "JPEG"), (b"\x89PNG", "PNG"), (b"GIF8", "GIF"),
         (b"BM", "BMP"), (b"RIFF", "WEBP")]


def file_format(path):
    with open(path, "rb") as fh:
        head = fh.read(16)
    for magic, name in MAGIC:
        if head.startswith(magic):
            return name
    if head[4:8] == b"ftyp":
        brand = head[8:12].decode("ascii", "replace")
        return f"HEIC/{brand}"          # iPhone's native format
    return "unknown/" + head[:4].hex()


def blue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (95, 90, 60), (140, 255, 255))


def marks_in_band(img):
    """Blue components lying inside the grooved band only."""
    center, radius = detector.find_disc(img)
    inner = int(P["LABEL_R"] * radius)
    outer = int(P["OUTER_R"] * radius)
    band = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(band, center, outer, 255, -1)
    cv2.circle(band, center, inner, 0, -1)

    pen = cv2.bitwise_and(blue(img), band)
    pen = cv2.morphologyEx(pen, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(pen, connectivity=8)
    kept = [i for i in range(1, n) if stats[i][4] >= 60]
    whole = int(np.count_nonzero(blue(img)))
    return len(kept), int(np.count_nonzero(pen)), whole


def main():
    files = sorted(f for f in os.listdir(ROOT)
                   if f.lower().endswith((".jpg", ".jpeg")))
    print(f"{len(files)} files\n")

    fmt = collections.Counter()
    readable = []
    for f in files:
        k = file_format(os.path.join(ROOT, f))
        fmt[k] += 1
        if k == "JPEG":
            readable.append(f)
    print("actual formats by magic bytes:")
    for k, v in fmt.most_common():
        print(f"    {k:<18}{v:>5}")
    print()

    if fmt.get("JPEG", 0) < len(files):
        print("  -> the non-JPEG files are why decoding failed; OpenCV cannot")
        print("     read them. They need converting to JPEG before use.\n")

    # sample a handful of readable files to see whether band-limiting works
    sample = readable[:8]
    print(f"blue found in {len(sample)} readable files"
          f"  (band = grooves only, whole = entire frame):")
    print(f"    {'file':<52}{'band comps':>11}{'band px':>9}{'whole px':>10}")
    for f in sample:
        img = detector.load_image(os.path.join(ROOT, f))
        try:
            comps, band_px, whole_px = marks_in_band(img)
        except Exception as exc:
            print(f"    {f[:50]:<52}  failed: {exc}")
            continue
        print(f"    {f[:50]:<52}{comps:>11}{band_px:>9}{whole_px:>10}")


if __name__ == "__main__":
    main()
