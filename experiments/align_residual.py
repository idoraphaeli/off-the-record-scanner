# -*- coding: utf-8 -*-
"""
Measure what is left over after the label alignment, and test whether a finer
stage closes it.

Three things can leave the two unwrapped rings offset from each other:
  1. angular  -- the rotation estimate is a degree or two out (constant shift)
  2. radial   -- the two shots got slightly different disc radii (constant shift)
  3. centre   -- the two shots got slightly different disc centres, which is NOT
                 a constant shift: it warps as a sine wave around the disc, so
                 some sectors line up and others do not

Distinguishing them matters, because 1 and 2 are fixed by one global correction
while 3 needs the centre itself corrected (or a per-sector alignment).
"""

import os
import sys

import cv2
import numpy as np

import detector
from detector import P
from multishot import analyse_photo, estimate_offset

HERE = os.path.dirname(os.path.abspath(__file__))
SECTORS = 12          # how many angular slices to measure separately


def flatten(ring):
    """Strip the lighting so correlation locks onto the record, not the lamp."""
    f = ring.astype(np.float32)
    f -= cv2.blur(f, (P["ROW_FLATTEN"], 1))      # radial gradient
    f -= cv2.blur(f, (1, 151))                   # angular gradient (the lit sector)
    return f


def global_shift(a, b):
    """Sub-pixel (dx, dy) that best aligns b onto a. dx = angle, dy = radius."""
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a * win, b * win)
    return dx, dy, response


def main():
    folder = sys.argv[1]
    paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    a, b = analyse_photo(paths[0]), analyse_photo(paths[1])
    width = a["ring"].shape[1]
    deg_per_col = 360.0 / P["POLAR_STEPS"]

    print(f"disc A: centre={a['center']} r={a['radius']}")
    print(f"disc B: centre={b['center']} r={b['radius']}")
    dr = b["radius"] - a["radius"]
    dc = np.hypot(b["center"][0] - a["center"][0], b["center"][1] - a["center"][1])
    print(f"  radius differs by {dr} px, centre by {dc:.1f} px")
    if abs(dr) > 4 or dc > 4:
        print("  ^ these alone can offset the rings; see per-sector table below")

    coarse, sharp = estimate_offset(a["label"], b["label"])
    print(f"\nlabel alignment: {coarse * deg_per_col:.1f} deg (sharpness {sharp:.1f})")

    fa = flatten(a["ring"])
    fb = flatten(np.roll(b["ring"], coarse, axis=1))

    dx, dy, resp = global_shift(fa, fb)
    print(f"residual after label stage: dx={dx:+.1f} px ({dx * deg_per_col:+.2f} deg)"
          f"  dy={dy:+.1f} px   confidence={resp:.3f}")

    # per-sector: a constant residual means angle/radius; a varying one means centre
    print(f"\nper-sector residual ({SECTORS} slices around the disc):")
    step = width // SECTORS
    dxs, dys = [], []
    for s in range(SECTORS):
        sl = slice(s * step, (s + 1) * step)
        sa, sb = np.ascontiguousarray(fa[:, sl]), np.ascontiguousarray(fb[:, sl])
        sdx, sdy, sresp = global_shift(sa, sb)
        dxs.append(sdx)
        dys.append(sdy)
        print(f"  {s * 360 // SECTORS:>3}deg-{(s + 1) * 360 // SECTORS:>3}deg : "
              f"dx={sdx:+7.1f}  dy={sdy:+6.1f}  conf={sresp:.3f}")

    dxs, dys = np.array(dxs), np.array(dys)
    print(f"\n  dx spread: {dxs.std():.1f} px   dy spread: {dys.std():.1f} px")
    if dxs.std() < 6 and dys.std() < 6:
        print("  -> residual is CONSTANT: one global correction fixes it")
    else:
        print("  -> residual VARIES around the disc: the two discs were found at")
        print("     slightly different centres; per-sector alignment is needed")


if __name__ == "__main__":
    main()
