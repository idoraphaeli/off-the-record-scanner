# -*- coding: utf-8 -*-
"""
Why does the label alignment refuse to answer on 56 sides out of 76?

Guessing at this is how the last three thresholds went wrong, so each failure is
attributed to the first thing that could have caused it, exactly as the miss
diagnosis does for scratches:

  no strip        the label region came out empty
  size mismatch   the two strips are different heights, and rotation_from_label
                  bails out on that. The height is derived from the disc radius
                  IN PIXELS, which changes with how close the phone was held, so
                  this would fail on almost every pair and would be a defect in
                  the code rather than a property of the record
  weak peak       the correlation ran but nothing stood out — a label with too
                  little print to lock onto, which is the only honest reason to
                  decline

The ratios of the weak cases are printed as a distribution, so the threshold can
be set from what the data does rather than from a number that seemed sensible.

Usage:  python why_no_align.py [cal|val|both]
"""

import collections
import csv
import json
import os
import sys

import numpy as np

from cross_shot import (label_profile, rotation_from_label, LABEL_MIN_RATIO)

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    sets = ("cal", "val") if which == "both" else (which,)
    records = set()
    for s in sets:
        records |= set(split[s])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in records]

    sides = collections.defaultdict(list)
    for r in rows:
        sides[(r["record"], r["side"])].append(r)

    cache = {}

    def prof(r):
        if r["pair"] not in cache:
            path = os.path.join(PHOTOS, r["photo_file"])
            try:
                cache[r["pair"]] = label_profile(path) if os.path.exists(path) else None
            except Exception:
                cache[r["pair"]] = None
        return cache[r["pair"]]

    why = collections.Counter()
    ratios, gaps = [], []

    for key, shots in sorted(sides.items()):
        if len(shots) < 2:
            continue
        a, b = shots[0], shots[1]
        pa, pb = prof(a), prof(b)
        if pa is None or pb is None:
            why["no strip"] += 1
            continue
        if pa.shape[0] != pb.shape[0]:
            # recorded, but no longer a reason to give up: rotation_from_label
            # now stretches the two strips to a common height
            gaps.append(abs(pa.shape[0] - pb.shape[0]))
        delta, ratio = rotation_from_label(pa, pb)
        if delta is None:
            why["weak peak"] += 1
            ratios.append(ratio)
        else:
            why["ALIGNED"] += 1
            ratios.append(ratio)

    total = sum(why.values())
    print(f"{total} sides with two shots\n")
    for k, v in why.most_common():
        print(f"  {k:<16}{v:>5}   {100*v/max(total,1):>5.0f}%")

    if gaps:
        g = np.array(gaps)
        print(f"\nsize mismatch, rows apart: median {np.median(g):.0f}, "
              f"max {g.max()}   (strips are a few hundred rows tall)")
        print("  -> this is a code defect, not the records: the two strips only")
        print("     need to be stretched to a common height before comparing.")

    if ratios:
        r = np.array(ratios)
        print(f"\ncorrelation peak strength, all sides that got that far:")
        for t in (2, 3, 4, 5, 8, 12):
            print(f"  ratio >= {t:>2} : {int((r >= t).sum()):>4} of {len(r)}"
                  f"   {'<- current threshold' if t == int(LABEL_MIN_RATIO) else ''}")
        print(f"  median {np.median(r):.1f}")


if __name__ == "__main__":
    main()
