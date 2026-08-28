# -*- coding: utf-8 -*-
"""
Characterise the detections found on records the annotator says are CLEAN.

Every detection on a new, unscratched record is a false positive by definition,
so this is the cleanest sample of "what the detector gets wrong" we have. The
question is whether they share a signature — a place on the disc, a brightness,
an orientation — because a shared signature can be fixed at the source, whereas
scattered errors can only be traded against recall.

Usage:  python analyse_false_positives.py [path-to-Records_Data]
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

# folders the annotator marked; everything else is a clean record
MARKED = {"16_ship1", "16_ship2", "Ariel_zilber1", "Benzin2",
          "Bob_Marley2", "Idan_raichel2", "high_window1", "high_window2"}


def collect(path):
    """Every detection on one photo, with the context it sits in."""
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner_px, outer_px = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner_px:outer_px]

    radial, tram = detector.scratch_map(ring)
    m1, k1 = detector.extract(radial)
    m2, k2 = detector.extract(tram, min_len=P["TRAM_MIN_LEN"])

    glare = detector.glare_mask(ring) > 0
    unlit = detector.unlit_mask(ring) > 0
    # how close each pixel is to a masked-out region, to test the "edge of glare"
    # and "edge of shadow" hypotheses
    near_glare = cv2.dilate(glare.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    near_unlit = cv2.dilate(unlit.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    band_h = ring.shape[0]

    out = []
    for chan_name, mask, marks in (("radial", m1, k1), ("tram", m2, k2)):
        n, labels, stats, cent = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            sel = labels == i
            row = int(cent[i][1])
            out.append({
                "channel": chan_name,
                "radius_frac": (inner_px + row) / radius,
                "band_pos": row / max(band_h - 1, 1),      # 0 = inner edge, 1 = outer
                "brightness": float(ring[sel].mean()),
                "near_glare": bool(near_glare[sel].any()),
                "near_unlit": bool(near_unlit[sel].any()),
                "area": int(area),
            })
    return out, ring


def hist(values, edges, labels):
    counts = [0] * len(labels)
    for v in values:
        for i, e in enumerate(edges):
            if v < e:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    total = max(sum(counts), 1)
    for lab, c in zip(labels, counts):
        bar = "#" * int(round(40 * c / total))
        print(f"    {lab:<14}{c:>5}  {100*c/total:>5.1f}%  {bar}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    all_det, n_photos, ring_stats = [], 0, []

    for rec in sorted(os.listdir(root)):
        rp = os.path.join(root, rec)
        if not os.path.isdir(rp):
            continue
        for sub in sorted(os.listdir(rp)):
            if sub in MARKED:
                continue
            sp = os.path.join(rp, sub)
            if not os.path.isdir(sp):
                continue
            for f in sorted(os.listdir(sp)):
                if (not f.lower().endswith(EXT) or f.startswith(GENERATED)
                        or COPY_HINT in f):
                    continue
                det, ring = collect(os.path.join(sp, f))
                all_det.extend(det)
                ring_stats.append(ring.mean())
                n_photos += 1

    print(f"{n_photos} photos of records you consider clean")
    print(f"{len(all_det)} detections — all of them false by definition")
    print(f"({len(all_det)/max(n_photos,1):.1f} per photo)\n")

    if not all_det:
        return

    print("WHERE on the disc (0 = label edge, 1 = outer rim):")
    hist([d["band_pos"] for d in all_det],
         [0.2, 0.4, 0.6, 0.8], ["0.0-0.2 inner", "0.2-0.4", "0.4-0.6",
                                "0.6-0.8", "0.8-1.0 outer"])

    print("\nBRIGHTNESS of the detection (ring mean is"
          f" {np.mean(ring_stats):.0f}):")
    hist([d["brightness"] for d in all_det],
         [30, 60, 90, 140], ["very dark <30", "30-60", "60-90",
                             "90-140", "bright >140"])

    ng = sum(d["near_glare"] for d in all_det)
    nu = sum(d["near_unlit"] for d in all_det)
    both = sum(d["near_glare"] and d["near_unlit"] for d in all_det)
    neither = sum(not d["near_glare"] and not d["near_unlit"] for d in all_det)
    t = len(all_det)
    print("\nPROXIMITY to a masked-out region (within 20 px):")
    print(f"    near glare      {ng:>5}  {100*ng/t:>5.1f}%")
    print(f"    near dark area  {nu:>5}  {100*nu/t:>5.1f}%")
    print(f"    near both       {both:>5}  {100*both/t:>5.1f}%")
    print(f"    near NEITHER    {neither:>5}  {100*neither/t:>5.1f}%")

    for ch in ("radial", "tram"):
        c = sum(1 for d in all_det if d["channel"] == ch)
        print(f"\nchannel {ch:<8}{c:>5}  {100*c/t:>5.1f}%")


if __name__ == "__main__":
    main()
