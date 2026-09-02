# -*- coding: utf-8 -*-
"""TEST 1 -- open the detector up and let the cross-shot check do the filtering.

The thresholds today are set so that a SINGLE photograph is already fairly
clean. That is the wrong place to spend them if a strong filter sits downstream:
a detection confirmed in both shots of a side is right 98% of the time against
78% for one seen once. The two-stage arrangement -- propose generously, verify
strictly -- should beat one careful stage.

There is a specific reason to expect the number of CONFIRMATIONS to rise rather
than fall. A real scratch that is caught in shot 1 and missed in shot 2 is
missed because it fell just under the threshold; lower the threshold and shot 2
catches it too, and the mark turns from "seen once" into "seen twice". A false
detection gains nothing that way: it appears in each shot independently and only
pairs up by landing in the same place by accident.

That accident scales with the size of the match window, which the alignment fix
just cut from +/-6 degrees to +/-2. So real marks pair up systematically while
coincidences got three times rarer -- which is why this is worth trying now and
was not before.

The thing to watch is that false detections per photo grow much faster than real
ones as the gates open, and coincidental pairs grow with the PRODUCT of the two
counts. True confirmations rise linearly at best, false ones quadratically, so
there is a sweet spot and then a collapse. That is why this sweeps a curve
instead of trying one setting.

Precision here is measured against the pen marks, which were drawn on scratches
only -- so DIRT COUNTS AS A MISTAKE in these numbers and they read lower than
the hand-labelled figures quoted elsewhere. It has to be this way: the hand
labels cover the detections the current thresholds produce, and a looser level
invents detections nobody has ever judged. What matters here is that all five
levels are scored by the same rule, so they can be compared to each other.

Usage:  python test_01_loosen_then_confirm.py [how many records]
"""

import collections
import csv
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np

import detector
from cross_shot import label_profile, rotation_from_label
from detector import P
from evaluate_frozen import MIN_EXTRA_AREA, MIN_HIT_PX, TOLERANCE, detect
from tune_alignment import offsets, refine

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
TESTS = os.path.join(os.path.dirname(HERE), "Model_Tests")
OUT = os.path.join(TESTS, "01_loosen_then_confirm")

WINDOW = 2.0
MARK_ALPHA, MARK_HALO = 0.45, 9      # the server's own drawing
YELLOW = (90, 255, 255)

# The two percentile gates and the absolute floor open together: the detector's
# own notes record that moving one alone is worth 3-10 points and moving both is
# worth 21, so sweeping them separately would measure the wrong thing.
LEVELS = [
    ("1_current", dict(PCT_STRONG=99.3, PCT_WEAK=98.7, THR_FLOOR=25)),
    ("2_open",    dict(PCT_STRONG=99.0, PCT_WEAK=98.2, THR_FLOOR=21)),
    ("3_wide",    dict(PCT_STRONG=98.6, PCT_WEAK=97.6, THR_FLOOR=17)),
    ("4_loose",   dict(PCT_STRONG=98.0, PCT_WEAK=96.8, THR_FLOOR=13)),
    ("5_loosest", dict(PCT_STRONG=97.0, PCT_WEAK=95.5, THR_FLOOR=10)),
]


def analysed(path):
    """Detections on one photo, as a label image plus disc coordinates."""
    img, det = detect(path)
    center, radius = detector.find_disc(img)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(
        (det > 127).astype(np.uint8), connectivity=8)
    marks = []
    for i in range(1, n):
        if stats[i][4] < MIN_EXTRA_AREA:
            continue
        dx, dy = cent[i][0] - center[0], cent[i][1] - center[1]
        marks.append({"id": i,
                      "rad": math.hypot(dx, dy) / max(radius, 1),
                      "ang": math.degrees(math.atan2(dy, dx)) % 360.0,
                      "cx": float(cent[i][0]), "cy": float(cent[i][1])})
    return {"img": img, "labels": lab, "marks": marks,
            "center": center, "radius": radius}


def mask_of(shot, which):
    ids = [shot["marks"][k]["id"] for k in which]
    if not ids:
        return np.zeros(shot["labels"].shape, np.uint8)
    return np.isin(shot["labels"], ids).astype(np.uint8) * 255


def measure(det, gt):
    """Zones found, and how many detections sit near no pen mark.

    Same rule as the frozen evaluator, applied to whatever subset of the marks
    is handed in, so before and after the filter are scored identically.
    """
    det_b = (det > 127).astype(np.uint8)
    gt_b = (gt > 127).astype(np.uint8)
    near_det = cv2.dilate(det_b, np.ones((TOLERANCE, TOLERANCE), np.uint8))
    n, labels, _, _ = cv2.connectedComponentsWithStats(gt_b, connectivity=8)
    found = sum(1 for i in range(1, n)
                if np.count_nonzero(near_det[labels == i]) >= MIN_HIT_PX)

    near_gt = cv2.dilate(gt_b, np.ones((TOLERANCE, TOLERANCE), np.uint8))
    nd, _, ds, _ = cv2.connectedComponentsWithStats(det_b & (near_gt == 0),
                                                   connectivity=8)
    miss = sum(1 for i in range(1, nd) if ds[i][4] >= MIN_EXTRA_AREA)
    nt, _, dts, _ = cv2.connectedComponentsWithStats(det_b, connectivity=8)
    shown = sum(1 for i in range(1, nt) if dts[i][4] >= MIN_EXTRA_AREA)
    return found, n - 1, miss, shown


def inside_disc(center, radius, shape):
    h, w = shape[:2]
    limit = int(P["OUTER_R"] * radius)
    yy, xx = np.ogrid[:h, :w]
    return ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= limit ** 2)


def paint(img, det_mask, keep_inside):
    vis = img.copy().astype(np.float32)
    band = cv2.dilate((det_mask > 127).astype(np.uint8),
                      np.ones((MARK_HALO, MARK_HALO), np.uint8))
    band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.0) * keep_inside
    a = np.clip(band, 0, 1)[..., None] * MARK_ALPHA
    return (vis * (1 - a) + np.array(YELLOW, np.float32) * a).astype(np.uint8)


def save(img, path):
    cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(path)


def gt_for(pair, shape):
    path = os.path.join(GT, pair + ".png")
    if not os.path.exists(path):
        return None
    gt = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(gt, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def pick_sides(want):
    """A small sample: every side of the first few calibration records that has
    two shots, so each side can be cross-checked against itself."""
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    cal = set(split["cal"])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in cal]
    sides = collections.defaultdict(list)
    for r in rows:
        p = os.path.join(PHOTOS, r["photo_file"])
        if os.path.exists(p):
            sides[(r["record"], r["side"])].append((r["pair"], p))
    by_record = collections.defaultdict(list)
    for (rec, side), shots in sorted(sides.items()):
        if len(shots) >= 2:
            by_record[rec].append((side, shots[:2]))
    out = []
    for rec in sorted(by_record)[:want]:
        for side, shots in by_record[rec]:
            out.append((rec, side, shots))
    return out


def run_level(name, params, chosen, render, report):
    P.update(params)
    t = collections.Counter()
    lines = []

    for rec, side, shots in chosen:
        (pair_a, path_a), (pair_b, path_b) = shots
        try:
            a, b = analysed(path_a), analysed(path_b)
            delta, _ = rotation_from_label(label_profile(path_a),
                                           label_profile(path_b))
        except Exception as exc:
            lines.append(f"  {rec} {side}: failed ({type(exc).__name__})")
            continue

        gt = gt_for(pair_a, a["img"].shape)
        if gt is None:
            continue

        if delta is None:
            kept = []
            t["align_failed"] += 1
        else:
            fixed, _, _, _ = refine(a["marks"], b["marks"], delta)
            hit = offsets(a["marks"], b["marks"], fixed, WINDOW)
            kept = [k for k, d in enumerate(hit) if d is not None]
            t["align_ok"] += 1

        all_mask = mask_of(a, range(len(a["marks"])))
        keep_mask = mask_of(a, kept)
        f0, zones, m0, s0 = measure(all_mask, gt)
        f1, _, m1, s1 = measure(keep_mask, gt)

        t["zones"] += zones
        t["found_all"] += f0
        t["shown_all"] += s0
        t["miss_all"] += m0
        t["found_kept"] += f1
        t["shown_kept"] += s1
        t["miss_kept"] += m1
        t["sides"] += 1

        if (rec, side) in render:
            stem = f"{render[(rec, side)]:02d}_{rec[:26]}_{side}"
            ins = inside_disc(a["center"], a["radius"], a["img"].shape)
            save(paint(a["img"], all_mask, ins),
                 os.path.join(OUT, f"level_{name}", f"{stem}_all.jpg"))
            save(paint(a["img"], keep_mask, ins),
                 os.path.join(OUT, f"level_{name}", f"{stem}_confirmed.jpg"))
        lines.append(f"  {rec[:26]:<28}{side}  found {s0:>4} -> kept {s1:>3}"
                     f"   scratches {f0:>2}/{zones:<3} -> {f1:>2}")

    rec_all = 100.0 * t["found_all"] / max(t["zones"], 1)
    rec_keep = 100.0 * t["found_kept"] / max(t["zones"], 1)
    pre_all = 100.0 * (t["shown_all"] - t["miss_all"]) / max(t["shown_all"], 1)
    pre_keep = 100.0 * (t["shown_kept"] - t["miss_kept"]) / max(t["shown_kept"], 1)

    report.append(dict(name=name, params=params, rec_all=rec_all,
                       rec_keep=rec_keep, pre_all=pre_all, pre_keep=pre_keep,
                       shown_all=t["shown_all"], shown_kept=t["shown_kept"],
                       found_kept=t["found_kept"], zones=t["zones"],
                       sides=t["sides"], align_ok=t["align_ok"],
                       align_failed=t["align_failed"], lines=lines))
    print(f"\nlevel {name}   {params}")
    print(f"  detections per photo {t['shown_all']/max(t['sides'],1):.1f}"
          f"  ->  confirmed {t['shown_kept']/max(t['sides'],1):.1f}")
    print(f"  recall  {rec_all:.1f}%  ->  {rec_keep:.1f}%"
          f"    ({t['found_kept']} of {t['zones']} scratches survive)")
    print(f"  precision  {pre_all:.0f}%  ->  {pre_keep:.0f}%")
    print(f"  alignment ok on {t['align_ok']} of {t['align_ok']+t['align_failed']}"
          f" sides")


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    baseline = {k: P[k] for k in ("PCT_STRONG", "PCT_WEAK", "THR_FLOOR")}
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "photos"), exist_ok=True)
    for name, _ in LEVELS:
        os.makedirs(os.path.join(OUT, f"level_{name}"), exist_ok=True)

    chosen = pick_sides(want)
    render, seen = {}, set()
    for rec, side, _ in chosen:          # one side per record gets pictures
        if rec not in seen:
            seen.add(rec)
            render[(rec, side)] = len(seen)
    print(f"{len(chosen)} sides from {len(seen)} records\n")

    for rec, side, shots in chosen:      # the plain photographs, once
        if (rec, side) in render:
            img = detector.load_image(shots[0][1])
            save(img, os.path.join(OUT, "photos",
                                   f"{render[(rec, side)]:02d}_{rec[:26]}_{side}.jpg"))

    report = []
    try:
        for name, params in LEVELS:
            run_level(name, params, chosen, render, report)
    finally:
        P.update(baseline)

    head = (f"{'level':<12}{'per photo':>11}{'confirmed':>11}{'recall':>9}"
            f"{'after gate':>12}{'prec':>7}{'after':>7}{'scratches kept':>17}")
    body = ["", head, "-" * len(head)]
    for r in report:
        kept = f"{r['found_kept']} of {r['zones']}"
        body.append(
            f"{r['name']:<12}{r['shown_all']/max(r['sides'],1):>11.1f}"
            f"{r['shown_kept']/max(r['sides'],1):>11.1f}{r['rec_all']:>8.1f}%"
            f"{r['rec_keep']:>11.1f}%{r['pre_all']:>6.0f}%{r['pre_keep']:>6.0f}%"
            f"{kept:>17}")
    print("\n".join(body))

    with open(os.path.join(OUT, "results.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\n")
        fh.write("\n".join(body) + "\n")
        for r in report:
            fh.write(f"\n\nlevel {r['name']}   {r['params']}\n")
            fh.write(f"  alignment ok on {r['align_ok']} of "
                     f"{r['align_ok'] + r['align_failed']} sides\n")
            fh.write("\n".join(r["lines"]) + "\n")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
