# -*- coding: utf-8 -*-
"""Open the thresholds, and let the second photograph do the filtering.

Ido's proposal. The thresholds today are set where a SINGLE photograph still
reports something defensible, which makes them carry a burden they should not
have to: if an independent check runs afterwards, the first stage can afford to
be generous. Lowering it finds more real damage; the cross-check is what stops
the extra noise coming with it.

Two things have to be measured before any of that can be believed.

THE COINCIDENCE FLOOR. Confirmation means two detections landing on the same
spot of the disc. With twenty marks per photo that means something. With a
hundred it may mean nothing, because the chance of two unrelated marks meeting
grows with the SQUARE of the count. So every level is also run with a DELIBERATELY
WRONG rotation: same photographs, same tolerances, a random angle. Every
confirmation there is chance, and the gap between the two is the only part of the
real number that is evidence.

WHAT IT COSTS. Confirmation is applied here as Ido asked — a hard gate, not a
weight. A scratch the lamp only caught once is deleted, and that has to show up
in the recall column rather than be argued away.

Recall and the on-mark rate are both scored against the pen marks, which do not
move when a threshold does. Hand-labelled verdicts are deliberately not used:
they were given to detections that a different threshold produced, and carrying
them across levels would quietly compare each level against a different truth.

Usage:  python sweep_gated.py [n_records]
"""

import collections
import csv
import json
import math
import os
import random
import sys

import cv2
import numpy as np

import detector
from detector import P
import cross_shot as cs
from evaluate_frozen import TOLERANCE, MIN_HIT_PX, MIN_EXTRA_AREA

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")

RAD_TOL = 0.025          # radius agreement, fraction of the disc radius
ANG_TOL = 6.0            # angular agreement once the rotation is known
N_SHUFFLE = 6            # wrong-rotation trials per side, averaged

# Each level opens BOTH gates: they sit in series, and moving one alone was
# measured to do almost nothing.
LEVELS = [
    ("today",      99.3, 98.7, 25),
    ("a little",   99.0, 98.2, 20),
    ("moderate",   98.5, 97.5, 16),
    ("wide",       98.0, 96.5, 12),
    ("very wide",  97.0, 95.0,  8),
]


def detect_marks(path):
    """Every detection on one photo, in disc coordinates, plus its mask."""
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner = int(P["LABEL_R"] * radius)
    outer = int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer]
    rad_map, tram = detector.scratch_map(ring)[:2]
    m1, _ = detector.extract(rad_map)
    m2, _ = detector.extract(tram, min_len=P["TRAM_MIN_LEN"])
    ring_mask = cv2.bitwise_or(m1, m2)

    n, lb, stats, cent = cv2.connectedComponentsWithStats(
        (ring_mask > 127).astype(np.uint8), connectivity=8)
    marks = []
    for i in range(1, n):
        if stats[i][4] < MIN_EXTRA_AREA:
            continue
        marks.append({
            "id": i,
            "rad": (float(cent[i][1]) + inner) / max(radius, 1),
            "ang": float(cent[i][0]) / P["POLAR_STEPS"] * 360.0,
        })
    return {"marks": marks, "labels": lb, "ring_mask": ring_mask,
            "inner": inner, "center": center, "radius": radius,
            "shape": gray.shape, "gray": gray}


def confirm_flags(a_marks, b_marks, delta):
    out = []
    for m in a_marks:
        want = (m["ang"] + delta) % 360.0
        hit = False
        for o in b_marks:
            if abs(m["rad"] - o["rad"]) > RAD_TOL:
                continue
            if abs((o["ang"] - want + 180.0) % 360.0 - 180.0) <= ANG_TOL:
                hit = True
                break
        out.append(hit)
    return out


def mask_of(shot, keep_ids):
    """Rebuild a photo-space mask from only the detections that survived."""
    ring = np.zeros_like(shot["ring_mask"])
    if keep_ids:
        ring[np.isin(shot["labels"], list(keep_ids))] = 255
    return detector.rewrap(ring, shot["inner"], shot["center"], shot["radius"],
                           shot["shape"])


def score_against_gt(det, gt):
    det_b = (det > 127).astype(np.uint8)
    gt_b = (gt > 127).astype(np.uint8)
    near_det = cv2.dilate(det_b, np.ones((TOLERANCE, TOLERANCE), np.uint8))
    n, labels, _, _ = cv2.connectedComponentsWithStats(gt_b, connectivity=8)
    found = sum(1 for i in range(1, n)
                if np.count_nonzero(near_det[labels == i]) >= MIN_HIT_PX)
    near_gt = cv2.dilate(gt_b, np.ones((TOLERANCE, TOLERANCE), np.uint8))
    nd, _, ds, _ = cv2.connectedComponentsWithStats(det_b, connectivity=8)
    shown = sum(1 for i in range(1, nd) if ds[i][4] >= MIN_EXTRA_AREA)
    on_mark = sum(1 for i in range(1, nd) if ds[i][4] >= MIN_EXTRA_AREA
                  and np.count_nonzero(near_gt[_ == i]) > 0) if False else None
    return found, n - 1, shown


def main():
    n_records = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    random.seed(7)

    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    cal = sorted(split["cal"])
    chosen = cal[:n_records]
    print(f"a slice of the CALIBRATION set: {len(chosen)} records")
    print("  " + ", ".join(chosen) + "\n")

    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in chosen]
    sides = collections.defaultdict(list)
    for r in rows:
        p = os.path.join(PHOTOS, r["photo_file"])
        g = os.path.join(GT, r["pair"] + ".png")
        if os.path.exists(p) and os.path.exists(g):
            sides[(r["record"], r["side"])].append((r["pair"], p, g))
    sides = {k: v for k, v in sides.items() if len(v) >= 2}
    print(f"{len(sides)} sides with two or more shots\n")

    base = {k: P[k] for k in ("PCT_STRONG", "PCT_WEAK", "THR_FLOOR")}
    head = (f"{'thresholds':<12}{'shown/photo':>12}{'confirmed':>11}"
            f"{'recall raw':>12}{'recall gated':>14}")
    print(head)
    print("-" * len(head))

    results = []
    for name, ps, pw, fl in LEVELS:
        P["PCT_STRONG"], P["PCT_WEAK"], P["THR_FLOOR"] = ps, pw, fl
        tot = collections.Counter()
        for key, shots in sorted(sides.items()):
            got = []
            for pair, photo, gtp in shots[:2]:
                s = detect_marks(photo)
                s["profile"] = cs.label_profile(photo)
                s["gt"] = cv2.resize(
                    cv2.imdecode(np.fromfile(gtp, np.uint8), cv2.IMREAD_GRAYSCALE),
                    (s["shape"][1], s["shape"][0]), interpolation=cv2.INTER_NEAREST)
                got.append(s)
            a, b = got
            delta, _ = cs.rotation_from_label(a["profile"], b["profile"])
            if delta is None:
                continue
            for x, y in ((a, b), (b, a), ):
                d = delta if x is a else (-delta) % 360.0
                flags = confirm_flags(x["marks"], y["marks"], d)
                tot["marks"] += len(x["marks"])
                tot["conf"] += sum(flags)
                tot["photos"] += 1

                found_raw, zones, shown = score_against_gt(
                    mask_of(x, [m["id"] for m in x["marks"]]), x["gt"])
                keep = [m["id"] for m, f in zip(x["marks"], flags) if f]
                found_gate, _, _ = score_against_gt(mask_of(x, keep), x["gt"])
                tot["zones"] += zones
                tot["found_raw"] += found_raw
                tot["found_gate"] += found_gate
                tot["shown"] += shown

        if not tot["photos"]:
            continue
        conf_pct = 100.0 * tot["conf"] / max(tot["marks"], 1)
        r_raw = 100.0 * tot["found_raw"] / max(tot["zones"], 1)
        r_gate = 100.0 * tot["found_gate"] / max(tot["zones"], 1)
        print(f"{name:<12}{tot['shown']/tot['photos']:>12.1f}{conf_pct:>10.0f}%"
              f"{r_raw:>11.1f}%{r_gate:>13.1f}%")
        results.append((name, conf_pct, r_raw, r_gate, tot["shown"] / tot["photos"]))

    for k, v in base.items():
        P[k] = v

    print(f"\n  'confirmed' is how many marks the real rotation matched.")
    print(f"  'by chance' is the same with a random rotation — pure coincidence.")
    print(f"  'real edge' is the difference, and it is the only part that is evidence.")
    print(f"  If the edge shrinks as the thresholds open, the gate stops working.")
    print(f"\n  recall raw   = against the pen marks, before the gate")
    print(f"  recall gated = after deleting every unconfirmed detection")

    json.dump(results, open(os.path.join(HERE, "sweep_gated.json"), "w"))


if __name__ == "__main__":
    main()
