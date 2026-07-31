# -*- coding: utf-8 -*-
"""
Hypothesis: a long bright scratch inflates the local sigma used to normalise it,
so it suppresses itself and only its strongest fragment survives ("self-masking",
a known CFAR failure). Test = compare the current mean/std normaliser against a
ROBUST one whose noise estimate ignores the brightest pixels, and measure the
length of the detected component that overlaps each marked zone.
"""

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


def norm_current(chan):
    win = (P["NOISE_WIN"], P["NOISE_WIN"])
    mean = cv2.blur(chan, win)
    sq = cv2.blur(chan * chan, win)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    return chan / np.maximum(std, P["NOISE_FLOOR"])


def norm_robust(chan):
    """Noise estimated from the LOW half of the local distribution only, so a
    bright mark cannot raise its own denominator."""
    win = (P["NOISE_WIN"], P["NOISE_WIN"])
    med = cv2.medianBlur(np.clip(chan, 0, 255).astype(np.uint8), 51).astype(np.float32)
    dev = np.abs(chan - med)
    # mean absolute deviation of the quiet part: clip deviations at their own
    # local mean before averaging, so outliers (the scratch) stop dominating
    mad = cv2.blur(dev, win)
    mad_clipped = cv2.blur(np.minimum(dev, 2.0 * mad), win)
    return (chan - med) / np.maximum(1.4826 * mad_clipped, P["NOISE_FLOOR"])


def run(norm_fn, ring):
    flat = ring.astype(np.float32)
    flat -= cv2.blur(flat, (P["ROW_FLATTEN"], 1))
    flat = cv2.GaussianBlur(flat, (3, 3), 0)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (P["TOPHAT_W"], 1))
    ridge = cv2.morphologyEx(flat, cv2.MORPH_TOPHAT, se)
    best = np.zeros_like(ridge)
    for k in detector._line_kernels(P["BOOST_LEN"], P["ANGLES"]):
        best = np.maximum(best, cv2.filter2D(ridge, -1, k))
    dead = (detector.glare_mask(ring) > 0) | (detector.unlit_mask(ring) > 0)
    z = norm_fn(best)
    z[dead] = 0
    smap = np.clip(z * 10, 0, 255).astype(np.uint8)
    mask, _ = detector.extract(smap)
    return mask


split = json.load(open(os.path.join(HERE, "split.json"), encoding="utf-8"))
tot = {"current": [], "robust": []}
for name in split["cal"]:
    stem = os.path.splitext(name)[0]
    gt_path = os.path.join(GT_DIR, stem + "_mask.png")
    if not os.path.exists(gt_path):
        continue
    img = detector.load_image(os.path.join(CLEAN_DIR, name))
    center, radius = detector.find_disc(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inner_px, outer_px = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner_px:outer_px]

    gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_ring = cv2.transpose(cv2.warpPolar(
        gt, (radius, P["POLAR_STEPS"]), center, radius,
        cv2.WARP_POLAR_LINEAR | cv2.INTER_NEAREST))[inner_px:outer_px] > 127

    n, labels, stats, _ = cv2.connectedComponentsWithStats(gt_ring.astype(np.uint8), 8)
    for tag, fn in (("current", norm_current), ("robust", norm_robust)):
        mask = run(fn, ring)
        for i in range(1, n):
            zone_area = stats[i][4]
            if zone_area < 200:
                continue
            covered = int(np.count_nonzero(mask[labels == i]))
            tot[tag].append(covered / zone_area)

for tag in ("current", "robust"):
    v = np.array(tot[tag])
    hit = v > 0.001
    print(f"{tag:>8}: zones touched={hit.sum():3d}/{len(v)}  "
          f"mean coverage of zone area={100*v.mean():5.2f}%  "
          f"median coverage when hit={100*np.median(v[hit]) if hit.any() else 0:5.2f}%")
