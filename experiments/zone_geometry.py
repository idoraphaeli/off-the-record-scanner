# -*- coding: utf-8 -*-
"""Where do the human-marked zones actually sit, in disc-radius terms?
If they fall outside the analysed ring band, no detector setting can find them."""

import json
import os

import cv2
import numpy as np

import detector
from detector import P

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\User\Desktop\Ido private\Computer science\off-the-record\record_pics"
CLEAN_DIR = next(os.path.join(BASE, d) for d in os.listdir(BASE)
                 if os.path.isdir(os.path.join(BASE, d))
                 and "ללא סימונים" in d and "חדש" in d)
GT_DIR = os.path.join(HERE, "gt")

split = json.load(open(os.path.join(HERE, "split.json"), encoding="utf-8"))
inside, outside, total = 0, 0, 0
for name in split["cal"]:
    stem = os.path.splitext(name)[0]
    gt_path = os.path.join(GT_DIR, stem + "_mask.png")
    if not os.path.exists(gt_path):
        continue
    img = detector.load_image(os.path.join(CLEAN_DIR, name))
    center, radius = detector.find_disc(img)
    gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    n, labels, stats, cent = cv2.connectedComponentsWithStats((gt > 127).astype(np.uint8), 8)
    rs = []
    for i in range(1, n):
        cx, cy = cent[i]
        r_frac = np.hypot(cx - center[0], cy - center[1]) / radius
        rs.append(r_frac)
        total += 1
        if P["LABEL_R"] <= r_frac <= P["OUTER_R"]:
            inside += 1
        else:
            outside += 1
    print(f"{stem[-22:]:>24} | disc r={radius} | zone radii: "
          + ", ".join(f"{v:.2f}" for v in sorted(rs)))

print(f"\nzones inside analysed band [{P['LABEL_R']}, {P['OUTER_R']}]: {inside}/{total}"
      f"   outside: {outside}")
