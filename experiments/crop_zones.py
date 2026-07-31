# -*- coding: utf-8 -*-
"""Crop each GT zone from the clean photo at full resolution for visual check."""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\User\Desktop\Ido private\Computer science\off-the-record\record_pics"
CLEAN_DIR = next(os.path.join(BASE, d) for d in os.listdir(BASE)
                 if os.path.isdir(os.path.join(BASE, d))
                 and "ללא סימונים" in d and "חדש" in d)
GT_DIR = os.path.join(HERE, "gt")
OUT = os.path.join(HERE, "debug")

name = sys.argv[1]
stem = os.path.splitext(name)[0]
img = cv2.imdecode(np.fromfile(os.path.join(CLEAN_DIR, name), np.uint8), cv2.IMREAD_COLOR)
gt = cv2.imdecode(np.fromfile(os.path.join(GT_DIR, stem + "_mask.png"), np.uint8),
                  cv2.IMREAD_GRAYSCALE)
os.makedirs(OUT, exist_ok=True)

n, labels, stats, _ = cv2.connectedComponentsWithStats((gt > 127).astype(np.uint8), 8)
for i in range(1, n):
    x, y, w, h, area = stats[i]
    pad = 40
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    crop = img[y0:y1, x0:x1]
    scale = max(1, int(600 / max(crop.shape[:2])))
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imencode(".jpg", crop)[1].tofile(os.path.join(OUT, f"zone_{i}.jpg"))
    print(f"zone {i}: bbox=({x},{y},{w},{h}) area={area} -> zone_{i}.jpg")
