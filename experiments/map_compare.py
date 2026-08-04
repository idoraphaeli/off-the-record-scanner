# -*- coding: utf-8 -*-
"""Put a scratch-response map next to one from a photo that worked, at the same
amplification, so the difference between 'signal' and 'texture' is visible."""

import os

import cv2
import numpy as np

import detector
from detector import P

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\User\Desktop\Ido private\Computer science\off-the-record\record_pics"
GOOD_DIR = next(os.path.join(BASE, d) for d in os.listdir(BASE)
                if os.path.isdir(os.path.join(BASE, d))
                and "ללא סימונים" in d and "חדש" in d)
GOOD = "WhatsApp Image 2026-07-31 at 13.02.07 (1).jpeg"     # scored 5/5 zones
TEST = os.path.join(HERE, "Test1", "WhatsApp Image 2026-07-31 at 15.02.24.jpeg")
OUT = os.path.join(HERE, "Test1", "analysis")


def mapof(path):
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    lo, hi = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[lo:hi]
    radial, tram = detector.scratch_map(ring)
    smap = np.maximum(radial, tram)
    return ring, np.clip(smap.astype(np.float32) * 3, 0, 255).astype(np.uint8)


os.makedirs(OUT, exist_ok=True)
ring_g, map_g = mapof(os.path.join(GOOD_DIR, GOOD))
ring_t, map_t = mapof(TEST)

w = min(map_g.shape[1], map_t.shape[1])
h = 260


def band(m, label):
    m = cv2.resize(m[:, :w], (w, h))
    m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    cv2.putText(m, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    return m


sep = np.full((10, w, 3), 60, np.uint8)
stack = np.vstack([band(map_g, "GOOD - side lamp (found 5/5 marked zones)"), sep,
                   band(map_t, "Test1 - soft light (found 0)")])
cv2.imencode(".jpg", stack)[1].tofile(os.path.join(OUT, "map_comparison.jpg"))

for name, m in (("good", map_g), ("test1", map_t)):
    nz = m[m > 0]
    print(f"{name:>6}: median={np.median(nz):5.1f}  p99={np.percentile(nz,99):6.1f}"
          f"  p99.9={np.percentile(nz,99.9):6.1f}  max={m.max():3d}"
          f"  -> peak/median ratio = {np.percentile(nz,99.9)/max(np.median(nz),1):5.1f}")
print("wrote analysis/map_comparison.jpg")
