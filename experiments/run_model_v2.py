# -*- coding: utf-8 -*-
"""Run MODEL v2 over the calibration records: numbers, and pictures to look at.

Every side is analysed twice -- once by the model that is on the server today,
once by the new one -- from the same two photographs, so the two can be read
against each other rather than against remembered figures.

The threshold is swept underneath the new model, because the combination it now
runs on changes what a given threshold means: intersecting two maps lowers every
value, so the bar that suited a single shot is no longer the same bar. What is
being looked for is where recall is still climbing.

The pictures are painted exactly as the server paints them -- the same soft
yellow, the same halo, the same clip at the edge of the playing surface -- so
what the gallery shows is what the app would show.

Usage:  python run_model_v2.py [how many records]
"""

import collections
import os
import shutil
import sys

import cv2
import numpy as np

import detector
import model_v2
from detector import P
from evaluate_frozen import MIN_EXTRA_AREA
from model_v2 import RULES_OFF, align, combined, maps_of
from test_01_loosen_then_confirm import (TESTS, gt_for, inside_disc, measure,
                                         paint, pick_sides, save)

OUT = os.path.join(TESTS, "05_model_v2")

# the bar, swept underneath the combination
SETTINGS = [
    ("a_current_bar", dict(PCT_STRONG=99.3, PCT_WEAK=98.7, THR_FLOOR=25)),
    ("b_open",        dict(PCT_STRONG=99.0, PCT_WEAK=98.2, THR_FLOOR=21)),
    ("c_wide",        dict(PCT_STRONG=98.6, PCT_WEAK=97.6, THR_FLOOR=17)),
    ("d_loose",       dict(PCT_STRONG=98.0, PCT_WEAK=96.8, THR_FLOOR=13)),
]
GALLERY = "c_wide"


def marks_from(ring_masks, a):
    det = detector.rewrap(cv2.bitwise_or(*ring_masks), a["inner"], a["center"],
                          a["radius"], a["shape"])
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        (det > 127).astype(np.uint8), connectivity=8)
    big = [i for i in range(1, n) if stats[i][4] >= MIN_EXTRA_AREA]
    return (np.isin(lab, big).astype(np.uint8) * 255) if big else det * 0


def today(a):
    """What the server does now: this photograph on its own, rules and bar as
    they ship. Scored on the same photograph as the new model, so the two
    numbers answer the same question."""
    m1, _ = detector.extract(a["radial"], None, a["ring"], a["inner"], a["radius"])
    m2, _ = detector.extract(a["tram"], P["TRAM_MIN_LEN"], a["ring"],
                             a["inner"], a["radius"])
    return marks_from((m1, m2), a)


def new_model(a, b, delta):
    m1, _ = detector.extract(combined(a["radial"], b["radial"], delta), None,
                             a["ring"], a["inner"], a["radius"])
    m2, _ = detector.extract(combined(a["tram"], b["tram"], delta),
                             P["TRAM_MIN_LEN"], a["ring"], a["inner"], a["radius"])
    return marks_from((m1, m2), a)


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    keys = set(RULES_OFF) | set(SETTINGS[0][1])
    baseline = {k: P[k] for k in keys}

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "photos"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "model_v2"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "on_the_server_today"), exist_ok=True)

    chosen = pick_sides(want)
    render, seen = {}, set()
    for rec, side, _ in chosen:
        if rec not in seen:
            seen.add(rec)
            render[(rec, side)] = len(seen)
    print(f"{len(chosen)} sides from {len(seen)} records\n")

    tally = {name: collections.Counter() for name, _ in SETTINGS}
    old = collections.Counter()
    how_count = collections.Counter()
    lines = []

    try:
        for rec, side, shots in chosen:
            (pair_a, path_a), (_, path_b) = shots
            try:
                a, b = maps_of(path_a), maps_of(path_b)
            except Exception as exc:
                print(f"  {rec} {side}: failed ({type(exc).__name__})")
                continue
            gt = gt_for(pair_a, a["img"].shape)
            if gt is None:
                continue

            P.update(baseline)
            det_old = today(a)
            f, zones, miss, shown = measure(det_old, gt)
            old["zones"] += zones
            old["found"] += f
            old["shown"] += shown
            old["sides"] += 1

            P.update(RULES_OFF)
            delta, ratio, how = align(a, b)
            how_count[how] += 1
            row = f"  {rec[:26]:<28}{side}  server {f:>2}/{zones:<3}"

            for name, thr in SETTINGS:
                P.update(thr)
                if delta is None:
                    # no second opinion: this model has nothing to intersect
                    det = np.zeros_like(det_old)
                else:
                    det = new_model(a, b, delta)
                fn, _, mn, sn = measure(det, gt)
                t = tally[name]
                t["zones"] += zones
                t["found"] += fn
                t["shown"] += sn
                t["sides"] += 1
                t["blank"] += int(sn == 0)
                if name == GALLERY:
                    row += f"   new {fn:>2}/{zones:<3}  marks {sn:>3}  by the {how}"
                    if (rec, side) in render:
                        ins = inside_disc(a["center"], a["radius"], a["img"].shape)
                        stem = f"{render[(rec, side)]:02d}_{rec[:26]}_{side}"
                        save(a["img"], os.path.join(OUT, "photos", f"{stem}.jpg"))
                        save(paint(a["img"], det, ins),
                             os.path.join(OUT, "model_v2", f"{stem}.jpg"))
                        save(paint(a["img"], det_old, ins),
                             os.path.join(OUT, "on_the_server_today", f"{stem}.jpg"))
            print(row)
            lines.append(row)
    finally:
        P.update(baseline)

    head = (f"\n{'model':<24}{'marks/photo':>13}{'recall':>9}"
            f"{'scratches found':>18}{'sides with none':>17}")
    body = [head, "-" * len(head)]
    found_old = f"{old['found']} of {old['zones']}"
    body.append(f"{'on the server today':<24}"
                f"{old['shown']/max(old['sides'],1):>13.1f}"
                f"{100*old['found']/max(old['zones'],1):>8.1f}%"
                f"{found_old:>18}{'-':>17}")
    for name, thr in SETTINGS:
        t = tally[name]
        found_new = f"{t['found']} of {t['zones']}"
        body.append(f"{'model v2, ' + name:<24}{t['shown']/max(t['sides'],1):>13.1f}"
                    f"{100*t['found']/max(t['zones'],1):>8.1f}%"
                    f"{found_new:>18}{t['blank']:>17}")
    print("\n".join(body))
    print(f"\nalignment: {dict(how_count)}")

    with open(os.path.join(OUT, "results.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\n")
        fh.write("\n".join(body) + "\n")
        fh.write(f"\nalignment: {dict(how_count)}\n\n")
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
