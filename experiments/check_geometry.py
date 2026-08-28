# -*- coding: utf-8 -*-
"""Sanity-check the disc detection across the dataset before trusting anything
built on top of it: image size, detected centre and radius, and which method
found it. A radius far from ~45% of the short side means the wrong circle was
found, and every downstream number is then meaningless."""

import os
import sys

import cv2
import numpy as np

import detector
from detector import P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "Records_Data")
EXT = (".jpg", ".jpeg", ".png")

print(f"{'photo':<44}{'image':>12}{'centre':>14}{'r':>6}{'r/short':>9}  method")
print("-" * 96)

for rec in sorted(os.listdir(ROOT)):
    rp = os.path.join(ROOT, rec)
    if not os.path.isdir(rp):
        continue
    for sub in sorted(os.listdir(rp)):
        sp = os.path.join(rp, sub)
        if not os.path.isdir(sp):
            continue
        for f in sorted(os.listdir(sp)):
            if not f.lower().endswith(EXT) or f.startswith(("overlap", "confirmed")):
                continue
            img = detector.load_image(os.path.join(sp, f))
            h, w = img.shape[:2]
            # experiments/detector.find_disc returns (centre, radius); the server
            # copy also returns which method won. Accept either.
            res = detector.find_disc(img)
            (cx, cy), r = res[0], res[1]
            how = res[2] if len(res) > 2 else "-"
            frac = r / min(h, w)
            flag = "" if 0.40 <= frac <= 0.50 else "   <-- SUSPECT"
            print(f"{f[:42]:<44}{w}x{h:<7}{f'({cx},{cy})':>14}{r:>6}{frac:>9.2f}  {how}{flag}")
