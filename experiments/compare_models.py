# -*- coding: utf-8 -*-
"""Every model we have, on the same sides, scored the same way.

The question this exists to answer is Ido's and it is about FALSE MARKS: does
demanding that a mark be present in BOTH shots put fewer wrong things in front
of the user. Every run so far reported recall and how many marks were painted,
which cannot answer that.

Three numbers per model, all from the pen marks, so no new hand labelling is
needed and every model is judged by the same rule:

  recall        of the scratches drawn in pen, how many were found
  precision     of the marks painted, how many sit on a drawn scratch. DIRT
                COUNTS AS WRONG here, so every figure reads far below the ones
                quoted from the hand labels -- but it reads low for all of them
                equally, which is what makes them comparable.
  clean photos  marks painted on photographs with NO pen marks at all. Those
                records were judged undamaged by eye, so everything painted
                there is dirt or nothing. It is the closest thing to a direct
                count of what the user should not be seeing.

Usage:  python compare_models.py [how many records]
"""

import collections
import os
import shutil
import sys

import cv2
import numpy as np

import detector
from detector import P
from model_v2 import RULES_OFF, align, combined, maps_of
from run_model_v2 import marks_from, new_model, today
from test_01_loosen_then_confirm import TESTS, gt_for, measure, pick_sides

OUT = os.path.join(TESTS, "06_compare_models")

BAR_NOW = dict(PCT_STRONG=99.3, PCT_WEAK=98.7, THR_FLOOR=25)
BARS = [("bar as it ships", BAR_NOW),
        ("bar opened", dict(PCT_STRONG=99.0, PCT_WEAK=98.2, THR_FLOOR=21)),
        ("bar wide", dict(PCT_STRONG=98.6, PCT_WEAK=97.6, THR_FLOOR=17)),
        ("bar loose", dict(PCT_STRONG=98.0, PCT_WEAK=96.8, THR_FLOOR=13))]

MODELS = (
    # name                               combine  rules off  bar
    ("1 on the server today",            False,   False,     BAR_NOW),
    ("2 server, three rules off",        False,   True,      BAR_NOW),
    ("3 both shots, bar as it ships",    True,    True,      BARS[0][1]),
    ("4 both shots, bar opened",         True,    True,      BARS[1][1]),
    ("5 both shots, bar wide",           True,    True,      BARS[2][1]),
    ("6 both shots, rules KEPT ON",      True,    False,     BARS[0][1]),
    ("7 both shots, rules KEPT, opened", True,    False,     BARS[1][1]),
)


def detect_with(a, b, delta, combine, rules_off, bar):
    keep = {k: P[k] for k in list(RULES_OFF) + list(BAR_NOW)}
    P.update(BAR_NOW)
    if rules_off:
        P.update(RULES_OFF)
    P.update(bar)
    try:
        if not combine:
            return today(a)
        if delta is None:
            return np.zeros(a["shape"], np.uint8)
        return new_model(a, b, delta)
    finally:
        P.update(keep)


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    os.makedirs(OUT, exist_ok=True)
    chosen = pick_sides(want)
    print(f"{len(chosen)} sides\n")

    t = {name: collections.Counter() for name, _, _, _ in MODELS}
    for i, (rec, side, shots) in enumerate(chosen, 1):
        (pair_a, path_a), (_, path_b) = shots
        try:
            a, b = maps_of(path_a), maps_of(path_b)
        except Exception as exc:
            print(f"  {rec} {side}: failed ({type(exc).__name__})")
            continue
        gt = gt_for(pair_a, a["img"].shape)
        if gt is None:
            continue
        delta, _, _ = align(a, b)

        for name, combine, rules_off, bar in MODELS:
            det = detect_with(a, b, delta, combine, rules_off, bar)
            found, zones, miss, shown = measure(det, gt)
            c = t[name]
            c["zones"] += zones
            c["found"] += found
            c["shown"] += shown
            c["miss"] += miss
            c["photos"] += 1
            if zones == 0:                 # nothing was drawn on this record
                c["clean_photos"] += 1
                c["on_clean"] += shown
        print(f"  {i:>3}/{len(chosen)}  {rec[:26]:<28}{side}")

    head = (f"\n{'model':<34}{'marks/photo':>13}{'recall':>9}{'precision':>12}"
            f"{'marks on a clean record':>26}")
    body = [head, "-" * len(head)]
    for name, _, _, _ in MODELS:
        c = t[name]
        body.append(
            f"{name:<34}{c['shown']/max(c['photos'],1):>13.1f}"
            f"{100*c['found']/max(c['zones'],1):>8.1f}%"
            f"{100*(c['shown']-c['miss'])/max(c['shown'],1):>11.0f}%"
            f"{c['on_clean']/max(c['clean_photos'],1):>26.1f}")
    print("\n".join(body))
    print(f"\nprecision counts dirt as WRONG, so every row reads low -- but it")
    print(f"reads low for all of them equally. The last column counts marks on")
    print(f"records with nothing drawn on them at all.")

    with open(os.path.join(OUT, "results.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\n")
        fh.write("\n".join(body) + "\n")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
