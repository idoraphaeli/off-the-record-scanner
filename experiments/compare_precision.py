# -*- coding: utf-8 -*-
"""Recall and precision for every model, both scored the way we normally quote.

The last comparison scored precision against the pen marks, which were drawn on
scratches only, so dirt came out as a mistake and every model read around 15%.
That is fine for ranking models against each other and useless for saying what
any one of them is worth, because it is not the rule we grade by -- a dirty
record really is in worse condition and the scanner is meant to say so.

So precision here counts dirt as a correct call, exactly as the 75.9% quoted for
the model on the server does. Each mark is placed in one of three boxes:

    on a pen mark   correct by construction, and never shown for labelling
    hand-labelled   matched by position to a verdict already given
    unjudged        this model found it and nobody has ever looked at it

The first two make the figure and the third is reported beside it, because a
model that opens the detector up invents marks nobody has judged, and a
precision computed on half the evidence has to say so.

Usage:  python compare_precision.py [how many records]
"""

import collections
import os
import sys

import cv2
import numpy as np

from detector import P
from evaluate_frozen import MIN_EXTRA_AREA
from model_v2 import RULES_OFF, align, maps_of
from run_model_v2 import new_model, today
from test_01_loosen_then_confirm import TESTS, gt_for, measure, pick_sides
from test_01_precision_with_dirt import classify, load_labels

OUT = os.path.join(TESTS, "07_precision_of_each_model")

BAR_NOW = dict(PCT_STRONG=99.3, PCT_WEAK=98.7, THR_FLOOR=25)
BAR_OPEN = dict(PCT_STRONG=99.0, PCT_WEAK=98.2, THR_FLOOR=21)
BAR_WIDE = dict(PCT_STRONG=98.6, PCT_WEAK=97.6, THR_FLOOR=17)

MODELS = (
    # name                                    combine  rules off  bar
    ("1  on the server today",                False,   False,     BAR_NOW),
    ("2  server, three rules off",            False,   True,      BAR_NOW),
    ("v2 both shots, rules on, bar as ships", True,    False,     BAR_NOW),
    ("v2 both shots, rules on, bar opened",   True,    False,     BAR_OPEN),
    ("v2 both shots, rules off, bar as ships", True,   True,      BAR_NOW),
    ("v2 both shots, rules off, bar wide",    True,    True,      BAR_WIDE),
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


def as_marks(det):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(
        (det > 127).astype(np.uint8), connectivity=8)
    return [{"cx": float(cent[i][0]), "cy": float(cent[i][1])}
            for i in range(1, n) if stats[i][4] >= MIN_EXTRA_AREA]


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    os.makedirs(OUT, exist_ok=True)
    labels = load_labels()
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
        verdicts = labels.get(pair_a, [])

        for name, combine, rules_off, bar in MODELS:
            det = detect_with(a, b, delta, combine, rules_off, bar)
            found, zones, miss, shown = measure(det, gt)
            marks = as_marks(det)
            c = t[name]
            c += classify({"img": a["img"], "marks": marks},
                          range(len(marks)), gt, verdicts)
            c["zones"] += zones
            c["found"] += found
            c["shown"] += shown
            c["photos"] += 1
            if zones == 0:
                c["clean_photos"] += 1
                c["on_clean"] += shown
        print(f"  {i:>3}/{len(chosen)}  {rec[:26]:<28}{side}")

    head = (f"\n{'model':<40}{'marks':>8}{'recall':>9}{'PRECISION':>12}"
            f"{'unjudged':>10}{'on a clean record':>19}")
    body = [head, "-" * len(head)]
    for name, _, _, _ in MODELS:
        c = t[name]
        good = c["on_mark"] + c["scratch"] + c["dirt"]
        bad = c["false"]
        body.append(
            f"{name:<40}{c['shown']/max(c['photos'],1):>8.1f}"
            f"{100*c['found']/max(c['zones'],1):>8.1f}%"
            f"{100*good/max(good+bad,1):>11.1f}%"
            f"{100*c['unjudged']/max(sum((good,bad,c['unjudged'],c['unsure'])),1):>9.0f}%"
            f"{c['on_clean']/max(c['clean_photos'],1):>19.1f}")
    print("\n".join(body))
    print("\nPRECISION counts dirt as a correct call. `unjudged` is the share of")
    print("each model's marks that nobody has ever looked at -- the higher it is,")
    print("the less the precision beside it can be leaned on.")

    with open(os.path.join(OUT, "results.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\n")
        fh.write("\n".join(body) + "\n")
        for name, _, _, _ in MODELS:
            c = t[name]
            fh.write(f"\n{name}\n  on a pen mark {c['on_mark']}, "
                     f"scratch {c['scratch']}, dirt {c['dirt']}, "
                     f"false {c['false']}, unsure {c['unsure']}, "
                     f"unjudged {c['unjudged']}\n")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
