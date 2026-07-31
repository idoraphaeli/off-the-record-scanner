# -*- coding: utf-8 -*-
"""Dump per-stage debug images for one photo. Usage: debug_one.py <image name>"""

import os
import sys

import cv2
import numpy as np

import detector
from detector import P

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\User\Desktop\Ido private\Computer science\off-the-record\record_pics"
CLEAN_DIR = next(os.path.join(BASE, d) for d in os.listdir(BASE)
                 if os.path.isdir(os.path.join(BASE, d))
                 and "ללא סימונים" in d and "חדש" in d)
OUT = os.path.join(HERE, "debug")


def main():
    name = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    img = detector.load_image(os.path.join(CLEAN_DIR, name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # disc-detection internals
    blur = cv2.medianBlur(gray, 5)
    thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark = (blur < thr).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cv2.imencode(".jpg", dark)[1].tofile(os.path.join(OUT, "1_darkmask.jpg"))

    center, radius = detector.find_disc(img)
    vis = img.copy()
    cv2.circle(vis, center, radius, (0, 255, 0), 3)
    cv2.drawMarker(vis, center, (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    cv2.imencode(".jpg", vis)[1].tofile(os.path.join(OUT, "2_disc.jpg"))
    print("center:", center, "radius:", radius)

    polar = detector.unwrap(gray, center, radius)
    inner_px, outer_px = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = polar[inner_px:outer_px]
    cv2.imencode(".jpg", ring)[1].tofile(os.path.join(OUT, "3_ring.jpg"))

    radial, tram = detector.scratch_map(ring)
    for tag, m in (("radial", radial), ("tram", tram)):
        cv2.imencode(".jpg", m)[1].tofile(os.path.join(OUT, f"4_{tag}.jpg"))
        amp = np.clip(m.astype(np.float32) * 6, 0, 255).astype(np.uint8)
        cv2.imencode(".jpg", amp)[1].tofile(os.path.join(OUT, f"4_{tag}_x6.jpg"))
        print(f"{tag}: max={m.max()}  p99.9={float(np.percentile(m[m>0], 99.9)):.1f}")

    mask_a, sa = detector.extract(radial)
    mask_b, sb = detector.extract(tram, min_len=P["TRAM_MIN_LEN"])
    cv2.imencode(".jpg", cv2.bitwise_or(mask_a, mask_b))[1].tofile(
        os.path.join(OUT, "5_mask.jpg"))
    print(f"scratches: radial={len(sa)} tram={len(sb)}")


if __name__ == "__main__":
    main()
