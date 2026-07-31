# -*- coding: utf-8 -*-
"""Agent-B review view: detections (yellow) + human zones (green outline) on one
image, so every detection can be judged as hit / real-miss / false alarm."""

import json
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

run = sys.argv[1]
names = sys.argv[2:]
out_dir = os.path.join(HERE, "review", run)
os.makedirs(out_dir, exist_ok=True)

for name in names:
    stem = os.path.splitext(name)[0]
    img = cv2.imdecode(np.fromfile(os.path.join(CLEAN_DIR, name), np.uint8), cv2.IMREAD_COLOR)
    det = cv2.imdecode(np.fromfile(os.path.join(HERE, "runs", run, stem + "_det.png"),
                                   np.uint8), cv2.IMREAD_GRAYSCALE)
    det = cv2.resize(det, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    vis = img.copy()

    # thicken detections so they are visible at page scale
    thick = cv2.dilate((det > 127).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    vis[thick] = (0, 255, 255)

    gt_path = os.path.join(GT_DIR, stem + "_mask.png")
    if os.path.exists(gt_path):
        gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours((gt > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (0, 255, 0), 3)
    cv2.imencode(".jpg", vis)[1].tofile(os.path.join(out_dir, stem + "_review.jpg"))
    print("wrote", stem[-24:])
