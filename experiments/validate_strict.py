# -*- coding: utf-8 -*-
"""The strict model on the VALIDATION records, which took no part in choosing it.

Everything measured so far comes from the calibration set, and the calibration
set is also where the configuration was picked -- out of seven, by looking at
their numbers. A figure chosen that way is optimistic by construction, and the
only way to find out by how much is to score the choice once on records that
were never consulted.

Eleven records, held out since the split was made. Recall is scored against the
pen marks and needs no fresh judgement. Precision uses the verdicts already
given on this set and reports how much of itself it could not see, since the
strict model paints marks nobody has judged here either.

Nothing is tuned here and nothing may be. If the numbers hold, the choice made
on calibration was sound; if they do not, the difference is what the choice
cost, and it is worth knowing before the server changes.

Usage:  python validate_strict.py [cal|val]
"""

import collections
import csv
import json
import os
import sys

import cv2
import numpy as np

from compare_precision import BAR_NOW, as_marks, detect_with
from cross_shot import load_labels as load_one
from evaluate_frozen import MIN_EXTRA_AREA
from model_v2 import align, maps_of
from test_01_loosen_then_confirm import GT, PHOTOS, gt_for, measure
from test_01_precision_with_dirt import classify


def sides_of(which):
    """Every side of the chosen set that has two shots, in record order."""
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    records = set(split[which])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in records]
    sides = collections.defaultdict(list)
    for r in rows:
        p = os.path.join(PHOTOS, r["photo_file"])
        if os.path.exists(p):
            sides[(r["record"], r["side"])].append((r["pair"], p))
    return [(rec, side, shots[:2])
            for (rec, side), shots in sorted(sides.items()) if len(shots) >= 2]


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "val"
    chosen = sides_of(which)
    labels = collections.defaultdict(list)
    for name in ("cal", "val", "test01", "test02", "v2strict"):
        for pair, rows in load_one(name).items():
            labels[pair].extend(rows)
    print(f"SET = {which}   {len(chosen)} sides from "
          f"{len({r for r, _, _ in chosen})} records\n")

    t = {"server": collections.Counter(), "strict": collections.Counter()}
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
        delta, _, how = align(a, b)

        for name, combine in (("server", False), ("strict", True)):
            det = detect_with(a, b, delta, combine, False, BAR_NOW)
            found, zones, miss, shown = measure(det, gt)
            marks = as_marks(det)
            c = t[name]
            c += classify({"img": a["img"], "marks": marks},
                          range(len(marks)), gt, labels.get(pair_a, []))
            c["zones"] += zones
            c["found"] += found
            c["shown"] += shown
            c["photos"] += 1
            c["blank"] += int(shown == 0)
            if zones == 0:
                c["clean_photos"] += 1
                c["on_clean"] += shown
        if delta is None:
            print(f"  {rec[:26]:<28}{side}   NOT ALIGNED")
        else:
            print(f"  {rec[:26]:<28}{side}   server {t['server']['found']:>3}"
                  f"   strict {t['strict']['found']:>3}   by the {how}")

    head = (f"\n{'model':<22}{'marks/photo':>13}{'recall':>9}{'PRECISION':>12}"
            f"{'unjudged':>10}{'on a clean record':>19}{'blank sides':>13}")
    print(head)
    print("-" * len(head))
    for name in ("server", "strict"):
        c = t[name]
        good = c["on_mark"] + c["scratch"] + c["dirt"]
        bad = c["false"]
        seen = good + bad + c["unjudged"] + c["unsure"]
        print(f"{name:<22}{c['shown']/max(c['photos'],1):>13.1f}"
              f"{100*c['found']/max(c['zones'],1):>8.1f}%"
              f"{100*good/max(good+bad,1):>11.1f}%"
              f"{100*c['unjudged']/max(seen,1):>9.0f}%"
              f"{c['on_clean']/max(c['clean_photos'],1):>19.1f}"
              f"{c['blank']:>13}")
        print(f"{'':<22}on a pen mark {c['on_mark']}, scratch {c['scratch']}, "
              f"dirt {c['dirt']}, false {c['false']}, unjudged {c['unjudged']}")


if __name__ == "__main__":
    main()
