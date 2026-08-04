# -*- coding: utf-8 -*-
"""
Diagnose a single photo, zone by zone: for every hand-marked zone report where it
sits, whether the detector had any response there, and which stage discarded it.
DIAGNOSTIC ONLY -- reports, never tunes.

Usage: python diagnose_one.py "<file name>"
"""

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
GT_DIR = os.path.join(HERE, "gt")
OUT = os.path.join(HERE, "diagnose")


def to_ring(m, center, radius, inner_px, outer_px):
    polar = cv2.warpPolar(m, (radius, P["POLAR_STEPS"]), center, radius,
                          cv2.WARP_POLAR_LINEAR | cv2.INTER_NEAREST)
    return cv2.transpose(polar)[inner_px:outer_px]


def main():
    name = sys.argv[1]
    stem = os.path.splitext(name)[0]
    os.makedirs(OUT, exist_ok=True)

    img = detector.load_image(os.path.join(CLEAN_DIR, name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner_px, outer_px = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    print(f"disc: center={center} radius={radius}  image={img.shape[1]}x{img.shape[0]}")

    vis = img.copy()
    cv2.circle(vis, center, radius, (0, 255, 0), 3)
    cv2.circle(vis, center, inner_px, (0, 165, 255), 2)
    cv2.circle(vis, center, outer_px, (255, 0, 255), 2)
    cv2.drawMarker(vis, center, (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    cv2.imencode(".jpg", vis)[1].tofile(os.path.join(OUT, "geometry.jpg"))

    ring = detector.unwrap(gray, center, radius)[inner_px:outer_px]
    cv2.imencode(".jpg", ring)[1].tofile(os.path.join(OUT, "ring.jpg"))

    glare = detector.glare_mask(ring) > 0
    unlit = detector.unlit_mask(ring) > 0
    print(f"ring masked out: glare {100*glare.mean():.1f}%  unlit {100*unlit.mean():.1f}%"
          f"  judgeable {100*(~(glare|unlit)).mean():.1f}%")
    print(f"ring brightness: mean={ring.mean():.1f} p05={np.percentile(ring,5):.0f}"
          f" p95={np.percentile(ring,95):.0f}")

    radial, tram = detector.scratch_map(ring)
    smap = np.maximum(radial, tram)
    for tag, m in (("radial", radial), ("tram", tram)):
        amp = np.clip(m.astype(np.float32) * 3, 0, 255).astype(np.uint8)
        cv2.imencode(".jpg", amp)[1].tofile(os.path.join(OUT, f"map_{tag}_x3.jpg"))

    judge = smap[smap > 0]
    thr_s = max(float(np.percentile(judge, P["PCT_STRONG"])), P["THR_FLOOR"])
    thr_w = max(float(np.percentile(judge, P["PCT_WEAK"])), P["THR_FLOOR"] / 2)
    print(f"thresholds: strong={thr_s:.0f} weak={thr_w:.0f}")

    gt_path = os.path.join(GT_DIR, stem + "_mask.png")
    gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_ring = to_ring(gt, center, radius, inner_px, outer_px) > 127

    mask_a, sa = detector.extract(radial)
    mask_b, sb = detector.extract(tram, min_len=P["TRAM_MIN_LEN"])
    final = cv2.bitwise_or(mask_a, mask_b)

    n, labels, stats, cent = cv2.connectedComponentsWithStats(
        gt_ring.astype(np.uint8), connectivity=8)
    print(f"\n{n-1} marked zones (in unwrapped coordinates):")
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 50:
            continue
        sel = labels == i
        z = smap[sel]
        r_frac = (inner_px + cent[i][1]) / radius
        print(f"  zone {i}: radius={r_frac:.2f}R  size={w}x{h}  "
              f"peak={z.max():3.0f} (strong={thr_s:.0f})  "
              f"masked={100*np.count_nonzero(z == 0)/z.size:5.1f}%  "
              f"px>weak={int(np.count_nonzero(z > thr_w)):4d}  "
              f"final_px={int(np.count_nonzero(final[sel])):4d}")

    print(f"\nkept components: radial={len(sa)} tram={len(sb)}")
    for s in sa + sb:
        print(f"   len={s['length']:4d} thick={s['thickness']:4.1f} "
              f"angle_to_groove={s.get('angle_to_groove', '?')}")


if __name__ == "__main__":
    main()
