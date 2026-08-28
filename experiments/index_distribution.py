# -*- coding: utf-8 -*-
"""
How the weighted damage index actually spreads over the collection.

The Goldmine bands are a published standard and are not ours to invent. What
IS ours is the single curve that turns a damage index into a percentage score,
and choosing it by taste would put the arbitrariness straight back in.

So this measures the index on every photograph we hold, and reports the spread.
A collection of second-hand records bought from flea markets is not a uniform
sample of the grade range — it will be short of Mint and short of unplayable —
but its shape is still the only real evidence available for where the curve
should sit.

Usage:  python index_distribution.py
"""

import collections
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PHOTOS = os.path.join(ROOT, "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
sys.path.insert(0, ROOT)

from server.scanner.analyze import analyze   # noqa: E402


def main():
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    per_side = collections.defaultdict(list)
    vals, done = [], 0
    for r in rows:
        path = os.path.join(PHOTOS, r["photo_file"])
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as fh:
                res = analyze(fh.read(), want_overlay=False)
        except Exception:
            continue
        vals.append(res["damage_index"])
        per_side[(r["record"], r["side"])].append(res["damage_index"])
        done += 1
        if done % 20 == 0:
            print(f"  {done} photos...", flush=True)

    v = np.array(vals, float)
    # a side is judged by its worst shot: a defect the lamp missed in one photo
    # is still on the record
    sides = np.array([max(x) for x in per_side.values()], float)

    print(f"\n{len(v)} photographs, {len(sides)} sides\n")
    print(f"{'percentile':>12}{'per photo':>12}{'per side':>11}")
    print("-" * 35)
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"{q:>11}%{np.percentile(v, q):>12.2f}{np.percentile(sides, q):>11.2f}")
    print(f"\n  lowest {v.min():.2f}   highest {v.max():.2f}   mean {v.mean():.2f}")

    print(f"\nhow many sides fall under each index, if that were the cut:")
    for cut in (1, 2, 3, 5, 8, 12, 20, 30):
        print(f"   index <= {cut:>2} : {100*(sides <= cut).mean():>5.0f}% of sides")

    json.dump({"per_photo": vals, "per_side": list(sides)},
              open(os.path.join(HERE, "index_distribution.json"), "w"))
    print(f"\nwrote index_distribution.json")


if __name__ == "__main__":
    main()
