# -*- coding: utf-8 -*-
"""
For every hand-marked scratch the detector failed to report, work out WHICH
STAGE lost it. Tuning without this is guesswork: a threshold change cannot help
a scratch that was cropped out of the analysed band, and widening the shape
filter cannot help one that produced no response at all.

Each marked scratch is followed through the pipeline and attributed to the first
stage that could have dropped it:

  outside band      it lies inside the label radius or beyond the rim cut, so it
                    was never looked at
  masked            it sits under the glare or unlit mask
  no response       the response map is essentially flat there — no signal exists
  below weak        there is a response, but under the flood threshold
  no strong seed    above the flood threshold, but never reaches the seed
                    threshold, so hysteresis discards the whole component
  shape: <test>     it survived thresholding but a shape test rejected it, and
                    the failing test is named

Usage:  python why_missed.py [cal|val|test]
"""

import collections
import csv
import json
import os
import sys

import cv2
import numpy as np

import detector
from detector import P

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")

TOLERANCE = 30          # same slack the frozen evaluator allows
NO_SIGNAL_LEVEL = 8     # map value below which we call it "no response at all"


def ring_coords(mask, center, radius, inner, outer, shape):
    """Same polar transform the detector uses, applied to a full-frame mask."""
    m = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    polar = cv2.warpPolar(m, (radius, P["POLAR_STEPS"]), center, radius,
                          cv2.WARP_POLAR_LINEAR | cv2.INTER_NEAREST)
    return cv2.transpose(polar)[inner:outer], cv2.transpose(polar)


def shape_verdict(comp, w, h, area, min_len):
    """Which shape test rejects this component, if any."""
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = max((cv2.arcLength(c, True) for c in contours), default=0)
    length = max(per / 2, max(w, h))
    thickness = area / max(length, 1)
    if length < min_len:
        return f"shape: too short ({length:.0f}<{min_len})"
    if thickness > P["MAX_THICK"]:
        return f"shape: too thick ({thickness:.1f}>{P['MAX_THICK']})"
    # mirror the detector's graded rule: short components are held to a
    # stricter elongation than long ones
    need = P["SHORT_ELONG"] if length < P["SHORT_LEN"] else P["MIN_ELONG"]
    if length / max(thickness, 1) < need:
        return f"shape: not elongated (needed {need:.0f})"
    angle = detector._axis_angle_deg(comp)
    if angle < P["GROOVE_TOL_DEG"] and length < P["GROOVE_KEEP_LEN"]:
        return f"shape: groove-aligned ({angle:.0f}deg)"
    return None


def analyse_photo(photo, gt_full):
    img = detector.load_image(photo)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner, outer = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer]

    radial, tram, = detector.scratch_map(ring)[:2]
    combined = np.maximum(radial, tram)
    glare = detector.glare_mask(ring) > 0
    unlit = detector.unlit_mask(ring) > 0
    dead = glare | unlit

    # thresholds exactly as extract() computes them, on the radial channel
    judge = radial[radial > 0]
    if judge.size >= 1000:
        thr_s = max(float(np.percentile(judge, P["PCT_STRONG"])), P["THR_FLOOR"])
        thr_w = max(float(np.percentile(judge, P["PCT_WEAK"])), P["THR_FLOOR"] / 2)
    else:
        thr_s = thr_w = P["THR_FLOOR"]

    # what the detector actually keeps, to know which marks were found
    m1, _ = detector.extract(radial)
    m2, _ = detector.extract(tram, min_len=P["TRAM_MIN_LEN"])
    kept = cv2.bitwise_or(m1, m2)
    near_kept = cv2.dilate((kept > 0).astype(np.uint8),
                           np.ones((TOLERANCE, TOLERANCE), np.uint8))

    # the surviving-component map after hysteresis but BEFORE the shape filter,
    # so a mark can be attributed to thresholding versus shape
    weak = (radial > thr_w).astype(np.uint8)
    nlab, labels_w = cv2.connectedComponents(weak, connectivity=8)
    seeds = set(np.unique(labels_w[radial >= thr_s])) - {0}
    survived = np.isin(labels_w, list(seeds)).astype(np.uint8) * 255
    survived = cv2.morphologyEx(survived, cv2.MORPH_CLOSE,
                                np.ones(P["CLOSE"], np.uint8))
    survived = detector._link_collinear(survived)
    n_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(survived, 8)

    gt_ring, gt_polar = ring_coords(gt_full, center, radius, inner, outer,
                                    gray.shape)
    band_rows = outer - inner

    verdicts = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (cv2.transpose(cv2.warpPolar(
            cv2.resize(gt_full, (gray.shape[1], gray.shape[0]),
                       interpolation=cv2.INTER_NEAREST),
            (radius, P["POLAR_STEPS"]), center, radius,
            cv2.WARP_POLAR_LINEAR | cv2.INTER_NEAREST)) > 127).astype(np.uint8), 8)

    for i in range(1, n):
        if stats[i][4] < 40:
            continue
        sel_full = labels == i                      # in full polar coords
        rows = np.nonzero(sel_full.any(axis=1))[0]
        if not len(rows):
            continue
        in_band = ((rows >= inner) & (rows < outer)).mean()
        if in_band < 0.3:
            where = "inner of label" if rows.mean() < inner else "beyond rim cut"
            verdicts.append(f"outside band ({where})")
            continue

        sel = sel_full[inner:outer]
        if not sel.any():
            verdicts.append("outside band (rounding)")
            continue

        if np.count_nonzero(near_kept[sel]) > 0:
            continue                                 # found; not a miss

        if dead[sel].mean() > 0.6:
            verdicts.append("masked: " + ("glare" if glare[sel].mean() >
                                          unlit[sel].mean() else "too dark"))
            continue

        peak = float(combined[sel].max())
        if peak < NO_SIGNAL_LEVEL:
            verdicts.append("no response")
            continue
        if peak < thr_w:
            verdicts.append("below weak threshold")
            continue
        if peak < thr_s:
            verdicts.append("no strong seed")
            continue

        # it reached the component stage: find the component covering it
        overlap = labels_s[sel]
        overlap = overlap[overlap > 0]
        if overlap.size == 0:
            verdicts.append("lost in hysteresis")
            continue
        idx = int(np.bincount(overlap).argmax())
        x, y, w, h, area = stats_s[idx]
        comp = (labels_s[y:y + h, x:x + w] == idx).astype(np.uint8)
        why = shape_verdict(comp, w, h, area, P["MIN_LEN"])
        verdicts.append(why or "reached output but not matched")

    return verdicts


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "cal"
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    records = set(split[which])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["record"] in records and int(r["marks"]) > 0]

    tally = collections.Counter()
    photos = 0
    for r in rows:
        gt_path = os.path.join(GT, r["pair"] + ".png")
        photo = os.path.join(PHOTOS, r["photo_file"])
        if not (os.path.exists(gt_path) and os.path.exists(photo)):
            continue
        gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        try:
            for v in analyse_photo(photo, gt):
                tally[v] += 1
        except Exception as exc:
            tally[f"error: {type(exc).__name__}"] += 1
        photos += 1

    total = sum(tally.values())
    print(f"SET = {which}   {photos} marked photos")
    print(f"{total} marked scratches were MISSED. Where each was lost:\n")

    # group the shape reasons together for the headline view
    grouped = collections.Counter()
    for k, v in tally.items():
        grouped[k.split(" (")[0]] += v

    print(f"    {'stage':<34}{'count':>7}{'share':>8}")
    for k, v in grouped.most_common():
        bar = "#" * int(round(34 * v / max(grouped.values())))
        print(f"    {k:<34}{v:>7}{100*v/total:>7.1f}%  {bar}")

    print("\n  detail:")
    for k, v in tally.most_common(14):
        print(f"    {k:<48}{v:>6}")


if __name__ == "__main__":
    main()
