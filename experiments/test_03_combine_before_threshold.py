# -*- coding: utf-8 -*-
"""TEST 3 -- combine the two shots BEFORE the threshold, not after it.

The detector does not produce marks directly. It produces a response map -- a
number per pixel saying how scratch-like that spot is -- and only then a
threshold turns it into marks. Today each shot is thresholded on its own and the
surviving marks are matched afterwards, which means a faint scratch answering 20
in both shots against a threshold of 25 is thrown away twice, even though two
independent measurements agreed on it exactly.

So the combination is moved to before the decision. The two response maps are
brought into the same frame using the rotation read off the label, and the
pixelwise MINIMUM is taken. Minimum is a soft AND: it demands presence in both
shots without demanding strength in either.

    a scratch      20 and 20  ->  20, and it survives a lower threshold
    a reflection   60 and 3   ->  3, dead, because the lamp moved with the disc

That is the whole argument for lowering the threshold afterwards: the bar was
high to keep reflections out, and the combination has already removed them.

Two details decide whether it works.

ALIGNMENT. The angle is good to about 0.75 degrees, which at a typical radius is
around ten pixels -- the width of a scratch. A strict pixel-to-pixel minimum
would therefore pair a scratch with the vinyl beside it and score it zero. So the
second map is dilated first: the question asked is not "is it at exactly this
pixel" but "is there support within a few pixels of here".

MASKED AREAS. A zero in a map means the area was unjudgeable -- blown out by
glare, or too dark. Under a strict minimum, anything unjudgeable in EITHER shot
becomes unjudgeable in the result, so coverage becomes the intersection of the
two rather than their union. The lenient variant keeps the first shot's own
answer where the second has nothing to say, and both are measured here, along
with how much of the disc each one can still judge.

Usage:  python test_03_combine_before_threshold.py [how many records]
"""

import collections
import os
import shutil
import sys

import cv2
import numpy as np

import detector
from cross_shot import label_profile, rotation_from_label
from detector import P
from evaluate_frozen import MIN_EXTRA_AREA
from test_01_loosen_then_confirm import (LEVELS, TESTS, gt_for, inside_disc,
                                         measure, paint, pick_sides, save)
from test_01_precision_with_dirt import classify, load_labels
from tune_alignment import refine

OUT = os.path.join(TESTS, "03_combine_before_threshold")

TOL_ROWS = 9          # px of slack across the grooves, for residual misalignment
TOL_COLS = 15         # columns of slack around the disc -- about 1.5 degrees

LEVEL = dict(LEVELS)
CONFIGS = [
    # name                    how          thresholds        lenient
    ("1_baseline",            "single",    LEVEL["1_current"], False),
    ("2_combined_same_thr",   "combined",  LEVEL["1_current"], False),
    ("3_combined_open",       "combined",  LEVEL["2_open"],    False),
    ("4_combined_wide",       "combined",  LEVEL["3_wide"],    False),
    ("5_combined_loose",      "combined",  LEVEL["4_loose"],   False),
    ("6_combined_wide_lenient", "combined", LEVEL["3_wide"],   True),
]


def maps_of(path):
    """The ring and its two response maps, before any threshold is applied."""
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner, outer = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer]
    radial, tram = detector.scratch_map(ring)[:2]
    return {"img": img, "shape": gray.shape, "ring": ring, "radial": radial,
            "tram": tram, "inner": inner, "radius": radius, "center": center}


def marks_in_ring(m):
    """Where this shot's marks sit on the disc, at the current thresholds.

    Only needed to correct the rotation: the angle read off the label is a couple
    of degrees out, and at this radius that is more slack than the combination
    can absorb, so combining on the raw angle would pair a scratch with the vinyl
    beside it.
    """
    m1, _ = detector.extract(m["radial"], None, m["ring"], m["inner"], m["radius"])
    m2, _ = detector.extract(m["tram"], P["TRAM_MIN_LEN"], m["ring"],
                             m["inner"], m["radius"])
    n, _, stats, cent = cv2.connectedComponentsWithStats(
        (cv2.bitwise_or(m1, m2) > 127).astype(np.uint8), connectivity=8)
    return [{"rad": (float(cent[i][1]) + m["inner"]) / max(m["radius"], 1),
             "ang": float(cent[i][0]) / P["POLAR_STEPS"] * 360.0}
            for i in range(1, n) if stats[i][4] >= MIN_EXTRA_AREA]


def into_frame_of(a, b, delta):
    """Shot B's maps, stretched and rolled into shot A's frame of reference.

    The stretch is needed because the two photographs give slightly different
    disc radii, so the ring is a few rows taller in one than the other; without
    it the same distance from the centre would land on a different row.
    """
    h = a["ring"].shape[0]
    cols = int(round(-delta / 360.0 * P["POLAR_STEPS"]))
    out = {}
    for key in ("ring", "radial", "tram"):
        m = b[key]
        if m.shape[0] != h:
            m = cv2.resize(m, (m.shape[1], h), interpolation=cv2.INTER_LINEAR)
        out[key] = np.roll(m, cols, axis=1)
    return out


def combined(ma, mb, lenient):
    """The soft AND of two response maps, with slack for residual misalignment."""
    grown = cv2.dilate(mb, np.ones((TOL_ROWS, TOL_COLS), np.uint8))
    out = np.minimum(ma, grown)
    if lenient:
        # where the other shot could not judge at all, keep this shot's answer
        # rather than calling the spot unjudgeable in both
        out = np.where(grown <= 0, ma, out)
    return out


def run(name, how, thresholds, lenient, sides, render, labels):
    P.update(thresholds)
    t = collections.Counter()
    verdicts = collections.Counter()
    lines = []

    for rec, side, a, b, delta, gt, tuned in sides:
        if how == "single" or delta is None:
            rad, tra = a["radial"], a["tram"]
            fell_back = delta is None and how != "single"
        else:
            bb = into_frame_of(a, b, delta)
            rad = combined(a["radial"], bb["radial"], lenient)
            tra = combined(a["tram"], bb["tram"], lenient)
            fell_back = False
        t["fell_back"] += int(fell_back)

        m1, _ = detector.extract(rad, None, a["ring"], a["inner"], a["radius"])
        m2, _ = detector.extract(tra, P["TRAM_MIN_LEN"], a["ring"],
                                 a["inner"], a["radius"])
        ring_mask = cv2.bitwise_or(m1, m2)
        det = detector.rewrap(ring_mask, a["inner"], a["center"], a["radius"],
                              a["shape"])
        # drop speck-level blobs BEFORE scoring, exactly as tests 1 and 2 do --
        # otherwise a pixel too small to be shown still credits a scratch
        nn, ll, ss, _ = cv2.connectedComponentsWithStats(
            (det > 127).astype(np.uint8), connectivity=8)
        big = [i for i in range(1, nn) if ss[i][4] >= MIN_EXTRA_AREA]
        det = (np.isin(ll, big).astype(np.uint8) * 255) if big else np.zeros_like(det)

        found, zones, miss, shown = measure(det, gt)
        grp = "tuned" if tuned else "raw"
        t[f"zones_{grp}"] += zones
        t[f"found_{grp}"] += found
        t["zones"] += zones
        t["found"] += found
        t["shown"] += shown
        t["miss"] += miss
        t["sides"] += 1
        t["judgeable"] += int(np.count_nonzero(rad > 0))
        t["judgeable_a"] += int(np.count_nonzero(a["radial"] > 0))

        # the verdicts already given, joined on by position, so precision can be
        # read the way we grade -- with dirt counted as a correct call
        n, lab, stats, cent = cv2.connectedComponentsWithStats(
            (det > 127).astype(np.uint8), connectivity=8)
        marks = [{"cx": float(cent[i][0]), "cy": float(cent[i][1])}
                 for i in range(1, n) if stats[i][4] >= MIN_EXTRA_AREA]
        verdicts += classify({"img": a["img"], "marks": marks},
                             range(len(marks)), gt,
                             labels.get(_pair_of[(rec, side)], []))

        if (rec, side) in render:
            stem = f"{render[(rec, side)]:02d}_{rec[:26]}_{side}"
            ins = inside_disc(a["center"], a["radius"], a["img"].shape)
            save(paint(a["img"], det, ins),
                 os.path.join(OUT, f"config_{name}", f"{stem}.jpg"))
        lines.append(f"  {rec[:26]:<28}{side}  shown {shown:>4}"
                     f"   scratches {found:>2}/{zones}")

    good = verdicts["on_mark"] + verdicts["scratch"] + verdicts["dirt"]
    bad = verdicts["false"]
    row = dict(name=name, lines=lines, thresholds=thresholds, lenient=lenient,
               per_photo=t["shown"] / max(t["sides"], 1),
               recall=100.0 * t["found"] / max(t["zones"], 1),
               found=t["found"], zones=t["zones"],
               prec_pen=100.0 * (t["shown"] - t["miss"]) / max(t["shown"], 1),
               coverage=100.0 * t["judgeable"] / max(t["judgeable_a"], 1),
               judged=good + bad, unjudged=verdicts["unjudged"],
               rec_tuned=100.0 * t["found_tuned"] / max(t["zones_tuned"], 1),
               rec_raw=100.0 * t["found_raw"] / max(t["zones_raw"], 1),
               n_tuned=t["zones_tuned"], n_raw=t["zones_raw"],
               labelled=100.0 * good / max(good + bad, 1),
               scratch=verdicts["scratch"], dirt=verdicts["dirt"],
               false=bad, on_mark=verdicts["on_mark"], fell_back=t["fell_back"])
    print(f"  {name:<26} recall {row['recall']:>5.1f}%   "
          f"{row['per_photo']:>5.1f} shown/photo   "
          f"judgeable area {row['coverage']:>5.1f}% of one shot")
    return row


_pair_of = {}          # (record, side) -> the ground-truth key for shot 1


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    keys = set().union(*(set(c[2]) for c in CONFIGS))
    baseline = {k: P[k] for k in keys}

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "photos"), exist_ok=True)
    for name, _, _, _ in CONFIGS:
        os.makedirs(os.path.join(OUT, f"config_{name}"), exist_ok=True)

    labels = load_labels()
    chosen = pick_sides(want)
    render, seen = {}, set()
    for rec, side, _ in chosen:
        if rec not in seen:
            seen.add(rec)
            render[(rec, side)] = len(seen)

    # the response maps do not depend on any threshold, so they are computed
    # once and every configuration is run against the same ones
    sides, ok = [], 0
    for rec, side, shots in chosen:
        (pair_a, path_a), (_, path_b) = shots
        try:
            a, b = maps_of(path_a), maps_of(path_b)
            delta, _ = rotation_from_label(label_profile(path_a),
                                           label_profile(path_b))
            tuned = False
            if delta is not None:
                delta, _, spread, _ = refine(marks_in_ring(a), marks_in_ring(b),
                                             delta)
                # refine needs a handful of paired marks to fit the correction;
                # with fewer it hands the label's raw angle back, and that angle
                # is a few degrees out -- far more slack than the combination can
                # absorb. Whether it fired is therefore the thing to split on.
                tuned = spread is not None
        except Exception as exc:
            print(f"  {rec} {side}: failed ({type(exc).__name__})")
            continue
        gt = gt_for(pair_a, a["img"].shape)
        if gt is None:
            continue
        _pair_of[(rec, side)] = pair_a
        sides.append((rec, side, a, b, delta, gt, tuned))
        ok += int(delta is not None)
        if (rec, side) in render:
            save(a["img"], os.path.join(
                OUT, "photos", f"{render[(rec, side)]:02d}_{rec[:26]}_{side}.jpg"))
    print(f"{len(sides)} sides, {ok} of them alignable\n")

    report = []
    try:
        for name, how, thr, lenient in CONFIGS:
            P.update(baseline)
            report.append(run(name, how, thr, lenient, sides, render, labels))
    finally:
        P.update(baseline)

    head = (f"\n{'config':<26}{'per photo':>11}{'recall':>9}{'scratches':>12}"
            f"{'aligned':>13}{'not aligned':>11}{'PRECISION':>12}{'unjudged':>10}")
    body = [head, "-" * len(head)]
    for r in report:
        found = f"{r['found']} of {r['zones']}"
        body.append(f"{r['name']:<26}{r['per_photo']:>11.1f}{r['recall']:>8.1f}%"
                    f"{found:>12}{r['rec_tuned']:>9.1f}%{r['rec_raw']:>12.1f}%"
                    f"{r['labelled']:>11.1f}%{r['unjudged']:>10}")
    print("\n".join(body))
    print("\njudgeable = how much of the ring still carries an answer, against")
    print("what one shot alone can judge. PRECISION counts dirt as correct and")
    print("sees only the detections that have been judged.")

    with open(os.path.join(OUT, "results.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\n")
        fh.write("\n".join(body) + "\n")
        for r in report:
            fh.write(f"\n\nconfig {r['name']}   {r['thresholds']}"
                     f"   lenient={r['lenient']}\n")
            fh.write(f"  {r['on_mark']} on a pen mark, {r['scratch']} scratch, "
                     f"{r['dirt']} dirt, {r['false']} false, "
                     f"{r['unjudged']} unjudged\n")
            fh.write("\n".join(r["lines"]) + "\n")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
