# -*- coding: utf-8 -*-
"""
Two questions about the detections on records the annotator calls clean:

  1. Are they clustered by ANGLE around the disc, and if so does the cluster
     coincide with the lit sector? (An earlier claim that they sit in the "upper
     arc" was an impression, never measured — this measures it.)
  2. Do they differ in SATURATION from the scratches we did find? Black vinyl
     throws a coloured diffraction sheen, while a scratch scatters white light,
     so saturation is a discriminator the pipeline currently throws away at the
     first line (it converts to grey immediately).

Usage:  python analyse_fp_angle_saturation.py [path-to-Records_Data]
"""

import os
import sys

import cv2
import numpy as np

import detector
from detector import P

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(os.path.dirname(HERE), "Records_Data")
EXT = (".jpg", ".jpeg", ".png")
GENERATED = ("overlap", "confirmed", "judge_", "eval_")
COPY_HINT = "עותק"
MARKED = {"16_ship1", "16_ship2", "Ariel_zilber1", "Benzin2",
          "Bob_Marley2", "Idan_raichel2", "high_window1", "high_window2"}
SECTORS = 12

DIFF_THRESHOLD, STROKE_CLOSE, MIN_ZONE_AREA = 25, 11, 300


def polar_of(img, center, radius, flags=cv2.WARP_POLAR_LINEAR):
    return cv2.transpose(cv2.warpPolar(img, (radius, P["POLAR_STEPS"]),
                                       center, radius, flags))


def analyse(path, gt_full=None):
    """Detections on one photo, each with angle, brightness and saturation."""
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    center, radius = detector.find_disc(img)
    inner_px, outer_px = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)

    ring = polar_of(gray, center, radius)[inner_px:outer_px]
    sat = polar_of(hsv[:, :, 1], center, radius)[inner_px:outer_px].astype(np.float32)
    val = polar_of(hsv[:, :, 2], center, radius)[inner_px:outer_px].astype(np.float32)

    gt_ring = None
    if gt_full is not None:
        g = cv2.resize(gt_full, (img.shape[1], img.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
        gt_ring = polar_of(g, center, radius,
                           cv2.WARP_POLAR_LINEAR | cv2.INTER_NEAREST)[inner_px:outer_px] > 127

    radial, tram = detector.scratch_map(ring)
    m1, _ = detector.extract(radial)
    m2, _ = detector.extract(tram, min_len=P["TRAM_MIN_LEN"])
    mask = cv2.bitwise_or(m1, m2)

    # brightness of the ring per angular sector, to test "they follow the light"
    step = ring.shape[1] // SECTORS
    sector_bright = [float(ring[:, s * step:(s + 1) * step].mean())
                     for s in range(SECTORS)]

    out = []
    n, labels, stats, cent = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        sel = labels == i
        col = int(cent[i][0])
        inside_gt = bool(gt_ring[sel].any()) if gt_ring is not None else False
        out.append({
            "angle_deg": 360.0 * col / P["POLAR_STEPS"],
            "sector": min(col // step, SECTORS - 1),
            "brightness": float(ring[sel].mean()),
            "saturation": float(sat[sel].mean()),
            "value": float(val[sel].mean()),
            "on_marked_scratch": inside_gt,
        })
    return out, sector_bright


def zones(original, marked):
    if original.shape != marked.shape:
        dh, dw = (original.shape[0] - marked.shape[0],
                  original.shape[1] - marked.shape[1])
        if not (0 <= dh <= 2 and 0 <= dw <= 2):
            return None
        original = original[:marked.shape[0], :marked.shape[1]]
    diff = cv2.absdiff(original, marked).max(axis=2)
    if np.count_nonzero(diff > DIFF_THRESHOLD) < 200:
        return None
    stroke = (diff > DIFF_THRESHOLD).astype(np.uint8) * 255
    stroke = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE,
                              np.ones((STROKE_CLOSE, STROKE_CLOSE), np.uint8))
    h, w = stroke.shape
    ff = stroke.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    regions = cv2.bitwise_or(cv2.bitwise_not(ff) & cv2.bitwise_not(stroke), stroke)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(regions, connectivity=8)
    mask = np.zeros_like(regions)
    for i in range(1, n):
        if stats[i][4] >= MIN_ZONE_AREA:
            mask[labels == i] = 255
    return mask


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    fp, tp, sector_counts, sector_bright_sum, n_clean = [], [], [0] * SECTORS, \
        np.zeros(SECTORS), 0

    for rec in sorted(os.listdir(root)):
        rp = os.path.join(root, rec)
        if not os.path.isdir(rp):
            continue
        for sub in sorted(os.listdir(rp)):
            sp = os.path.join(rp, sub)
            if not os.path.isdir(sp):
                continue
            files = [f for f in sorted(os.listdir(sp))
                     if f.lower().endswith(EXT) and not f.startswith(GENERATED)]
            originals = [f for f in files if COPY_HINT not in f]
            copies = [f for f in files if COPY_HINT in f]

            for orig in originals:
                p = os.path.join(sp, orig)
                gt = None
                if sub in MARKED:
                    stem = os.path.splitext(orig)[0]
                    match = next((c for c in copies if stem in c), None)
                    if match:
                        o = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
                        m = cv2.imdecode(np.fromfile(os.path.join(sp, match), np.uint8),
                                         cv2.IMREAD_COLOR)
                        gt = zones(o, m)

                dets, bright = analyse(p, gt)
                if sub in MARKED:
                    for d in dets:
                        (tp if d["on_marked_scratch"] else fp).append(d)
                else:
                    fp.extend(dets)
                    n_clean += 1
                    sector_bright_sum += np.array(bright)
                    for d in dets:
                        sector_counts[d["sector"]] += 1

    # ---- question 1: angle ----
    print(f"ANGULAR distribution of detections on {n_clean} clean photos")
    print("(sector 0 = 0-30 deg, going round the disc)\n")
    avg_bright = sector_bright_sum / max(n_clean, 1)
    total = max(sum(sector_counts), 1)
    print(f"    {'sector':<12}{'count':>7}{'share':>8}{'ring brightness':>18}")
    for s in range(SECTORS):
        bar = "#" * int(round(30 * sector_counts[s] / max(max(sector_counts), 1)))
        print(f"    {s*30:>3}-{(s+1)*30:<8}{sector_counts[s]:>7}"
              f"{100*sector_counts[s]/total:>7.1f}%{avg_bright[s]:>12.1f}   {bar}")

    c = np.array(sector_counts, float)
    if c.sum() and avg_bright.std() > 0:
        r = float(np.corrcoef(c, avg_bright)[0, 1])
        print(f"\n    correlation between detections and sector brightness: {r:+.2f}")
        print("    (+1 = they follow the light exactly, 0 = unrelated)")
    spread = c.std() / max(c.mean(), 1e-9)
    print(f"    spread across sectors: {spread:.2f}"
          f"   ({'clustered' if spread > 0.6 else 'fairly even'})")

    # ---- question 2: saturation ----
    print(f"\n\nSATURATION — false ({len(fp)}) vs on a marked scratch ({len(tp)})")
    if not tp:
        print("    no detections landed on a marked scratch; cannot compare")
        return
    for name, group in (("false", fp), ("true ", tp)):
        s = np.array([g["saturation"] for g in group])
        v = np.array([g["value"] for g in group])
        b = np.array([g["brightness"] for g in group])
        print(f"    {name}: saturation median {np.median(s):5.1f}"
              f"  (p25 {np.percentile(s,25):.1f}  p75 {np.percentile(s,75):.1f})"
              f"   value {np.median(v):5.1f}   grey {np.median(b):5.1f}")

    sf = np.array([g["saturation"] for g in fp])
    st = np.array([g["saturation"] for g in tp])
    overlap = np.mean([(sf < t).mean() for t in st]) if len(st) else 0
    print(f"\n    a true detection is less saturated than {100*(1-overlap):.0f}%"
          f" of the false ones on average")
    print("    (near 50% means saturation does not separate them)")


if __name__ == "__main__":
    main()
