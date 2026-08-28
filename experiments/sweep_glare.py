# -*- coding: utf-8 -*-
"""
What would happen if the detector ignored bright areas of the photo?

The detector already has a rule for this — it marks areas brighter than
GLARE_BRIGHT as unjudgeable — but that number is 200, and the reflections that
fool it sit around 86. The rule is set so high it almost never fires.

This is a SIMULATION, run on the brightness already measured around each of the
1454 hand-labelled detections, so the answer arrives in seconds instead of half
an hour. It approximates the real rule: the real one masks broad bright regions
before detection, with an opening and a margin around them, so it will remove
somewhat more than this predicts. Treat the numbers as the shape of the trade,
not the final figures — a real run follows only if the shape is worth it.

Usage:  python sweep_glare.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "false_features.json")

CUTS = (None, 130, 120, 110, 100, 95, 90, 85, 80, 75)


def main():
    if not os.path.exists(SRC):
        raise SystemExit("run probe_false.py first")
    rows = json.load(open(SRC, encoding="utf-8"))
    b = np.array([r["bright"] for r in rows], float)
    kind = np.array([r["kind"] for r in rows])

    n_scr = int((kind == "scratch").sum())
    print(f"{len(rows)} labelled detections   "
          f"(scratches {n_scr}, dirt {int((kind=='dirt').sum())}, "
          f"false {int((kind=='false').sum())})")
    print("brightness around a detection, 0 = black, 255 = white:")
    for k in ("scratch", "dirt", "false"):
        v = b[kind == k]
        print(f"   {k:<9}median {np.median(v):>5.0f}   "
              f"a quarter are above {np.percentile(v, 75):>5.0f}")

    head = (f"\n{'ignore above':<14}{'kept':>7}{'real':>7}{'false':>7}"
            f"{'PRECISION':>12}{'scratches':>12}{'false cut':>11}")
    print(head)
    print("-" * len(head))

    base_false = int((kind == "false").sum())
    for cut in CUTS:
        keep = np.ones(len(rows), bool) if cut is None else (b <= cut)
        k = kind[keep]
        real = int(((k == "scratch") | (k == "dirt")).sum())
        fake = int((k == "false").sum())
        scr = int((k == "scratch").sum())
        if real + fake == 0:
            continue
        name = "nothing (today)" if cut is None else f"{cut}"
        print(f"{name:<14}{real + fake:>7}{real:>7}{fake:>7}"
              f"{100*real/(real+fake):>11.0f}%"
              f"{f'{scr} of {n_scr}':>12}"
              f"{100*(base_false-fake)/max(base_false,1):>10.0f}%")

    print(f"\n  'false cut' is the share of the model's mistakes removed.")
    print(f"  'scratches' is what survives of the damage you marked by hand —")
    print(f"  that column is the price.")


if __name__ == "__main__":
    main()
