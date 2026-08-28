# -*- coding: utf-8 -*-
"""
Does adding length to the radial-direction rule help?

Reflections do run longer than scratches — 34 pixels against 24 at the median —
so the two conditions might reinforce each other. But they can be joined two
ways, and the difference matters:

  AND   reject only marks that are BOTH long and pointing out from the centre.
        A short radial mark survives, so it is the cautious version.
  OR    reject a mark for either fault on its own. Catches more, costs more.

Both are swept across the same grid. Calibration and validation are shown
together on every line, because a combination that only works on one of them is
the failure mode this whole search is built to catch — and adding a second
condition doubles the ways to overfit.

Usage:  python sweep_angle_length.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "features.json")
RECALL_NOW = 62.5

ANGLES = (88, 84, 80)
LENGTHS = (None, 120, 90, 70, 55, 45)


def stats(rows, keep):
    kinds = np.array([r["kind"] for r in rows])
    real = int(((kinds == "scratch") | (kinds == "dirt"))[keep].sum())
    fake = int((kinds == "false")[keep].sum())
    scr_all = int((kinds == "scratch").sum())
    scr = int((kinds == "scratch")[keep].sum())
    if real + fake == 0:
        return None
    return (100.0 * real / (real + fake),
            RECALL_NOW * scr / max(scr_all, 1), scr, scr_all)


def main():
    rows = json.load(open(SRC, encoding="utf-8"))
    sets = {s: [r for r in rows if r["set"] == s] for s in ("cal", "val")}

    for name, v in (("scratch", "scratch"), ("dirt", "dirt"),
                    ("reflection", "false")):
        L = np.array([r["length"] for r in rows if r["kind"] == v])
        A = np.array([r["angle"] for r in rows if r["kind"] == v])
        print(f"  {name:<11}length median {np.median(L):>5.0f}   "
              f"angle median {np.median(A):>4.0f}   "
              f"long AND radial: {100*((L > 70) & (A > 84)).mean():>4.0f}%")

    for mode in ("AND", "OR"):
        head = (f"\n[{mode}]  reject when it is radial"
                f"{' and long' if mode == 'AND' else ' or long'}\n"
                f"{'angle':>7}{'length':>8}"
                f"{'CAL prec':>10}{'CAL rec':>9}{'scr':>8}"
                f"{'   |':>4}{'VAL prec':>10}{'VAL rec':>9}{'scr':>8}")
        print(head)
        print("-" * 74)
        for a in ANGLES:
            for L in LENGTHS:
                if mode == "OR" and L is None:
                    continue
                line = f"{a:>6}d{('-' if L is None else L):>8}"
                ok = True
                for i, s in enumerate(("cal", "val")):
                    data = sets[s]
                    ang = np.array([r["angle"] for r in data], float)
                    ln = np.array([r["length"] for r in data], float)
                    radial = ang > a
                    long_ = np.zeros(len(data), bool) if L is None else ln > L
                    drop = (radial & long_) if mode == "AND" else (radial | long_)
                    st = stats(data, ~drop)
                    if st is None:
                        ok = False
                        break
                    p, r, scr, scr_all = st
                    line += f"{p:>9.0f}%{r:>8.1f}%{f'{scr}/{scr_all}':>8}"
                    if i == 0:
                        line += f"{'   |':>4}"
                if ok:
                    print(line)

    print(f"\n  Today, untouched: calibration 69% / 62.5%,"
          f" validation 73% / 62.5%.")


if __name__ == "__main__":
    main()
