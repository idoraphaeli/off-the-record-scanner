# -*- coding: utf-8 -*-
"""
Agent A exploration: which per-pixel feature best separates the human-marked
scratch zones from the ring background on THESE flash photos?
For each candidate feature map (computed in polar ring space), report the mean
inside GT zones vs the 95th percentile of the background -- a usable feature
needs zone-mean comfortably above background-p95.
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


def to_ring(img_2d, center, radius, inner_px, outer_px, nearest=False):
    flags = cv2.WARP_POLAR_LINEAR | (cv2.INTER_NEAREST if nearest else 0)
    polar = cv2.warpPolar(img_2d, (radius, P["POLAR_STEPS"]), center, radius, flags)
    return cv2.transpose(polar)[inner_px:outer_px]


def features(bgr_ring):
    gray = cv2.cvtColor(bgr_ring, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(bgr_ring, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    out = {}
    # f1: current pipeline's scratch map
    out["f1_current_smap"] = detector.scratch_map(
        cv2.cvtColor(bgr_ring, cv2.COLOR_BGR2GRAY)).astype(np.float32)

    # f2: bright AND desaturated (white sheen streaks vs colorful sparkle)
    desat = v * (255.0 - s) / 255.0
    out["f2_desat_bright"] = desat - cv2.blur(desat, (P["ROW_FLATTEN"], 1))

    # f3: wide top-hat (allows broad scuff bands up to ~40 px)
    flat = gray - cv2.blur(gray, (P["ROW_FLATTEN"], 1))
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1))
    out["f3_wide_tophat"] = cv2.morphologyEx(flat, cv2.MORPH_TOPHAT, se)

    # f4: LOSS of sparkle: scratched areas look smoother than glittery grooves
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 3)
    sparkle = cv2.blur(hp * hp, (15, 15))
    spk = np.sqrt(sparkle)
    out["f4_smoothness"] = -(spk - cv2.blur(spk, (P["ROW_FLATTEN"], 1)))

    # f5: desat-bright with line boost along scratch directions
    booster = np.zeros_like(out["f2_desat_bright"])
    for k in detector._line_kernels(31, (-60, -40, -20, 0, 20, 40, 60)):
        booster = np.maximum(booster, cv2.filter2D(out["f2_desat_bright"], -1, k))
    out["f5_desat_lineboost"] = booster

    # f6: DARK thin lines (black-hat): on flash photos a scratch blocks the
    # sparkle and reads darker than its surroundings
    se15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    dark_ridge = cv2.morphologyEx(flat, cv2.MORPH_BLACKHAT, se15)
    boost_d = np.zeros_like(dark_ridge)
    for k in detector._line_kernels(P["BOOST_LEN"], P["ANGLES"]):
        boost_d = np.maximum(boost_d, cv2.filter2D(dark_ridge, -1, k))
    out["f6_dark_lineboost"] = boost_d

    # f7: both polarities combined
    bright_ridge = cv2.morphologyEx(flat, cv2.MORPH_TOPHAT, se15)
    both = np.maximum(dark_ridge, bright_ridge)
    boost_b = np.zeros_like(both)
    for k in detector._line_kernels(P["BOOST_LEN"], P["ANGLES"]):
        boost_b = np.maximum(boost_b, cv2.filter2D(both, -1, k))
    out["f7_both_lineboost"] = boost_b
    return out


def main():
    for name in sys.argv[1:]:
        stem = os.path.splitext(name)[0]
        img = detector.load_image(os.path.join(CLEAN_DIR, name))
        center, radius = detector.find_disc(img)
        inner_px, outer_px = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)

        gt = cv2.imdecode(np.fromfile(os.path.join(GT_DIR, stem + "_mask.png"),
                                      dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_ring = to_ring(gt, center, radius, inner_px, outer_px, nearest=True) > 127

        bgr_ring = to_ring(img, center, radius, inner_px, outer_px)
        print(f"\n=== {stem[-24:]} | zone px in ring: {int(gt_ring.sum())} ===")
        for fname, fmap in features(bgr_ring).items():
            zone = fmap[gt_ring]
            bg = fmap[~gt_ring]
            # detection-like metric: threshold at the background's 99.9th
            # percentile; a useful feature leaves many zone pixels above it
            # (the thin scratch inside the loop) and few background pixels.
            thr = np.percentile(bg, 99.9)
            zone_hits = int(np.count_nonzero(zone > thr))
            bg_hits = int(np.count_nonzero(bg > thr))
            ratio = (zone_hits / max(zone.size, 1)) / max(bg_hits / max(bg.size, 1), 1e-9)
            print(f"  {fname:>20}: thr(bg p99.9)={thr:6.1f}  zone_hits={zone_hits:6d}"
                  f"  bg_hits={bg_hits:6d}  density_ratio={ratio:6.1f}")


if __name__ == "__main__":
    main()
