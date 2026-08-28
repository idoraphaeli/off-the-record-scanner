# -*- coding: utf-8 -*-
"""
Find the tightest "long AND radial" rule that still costs no recall at all.

The coarse grid put the boundary somewhere between 84 degrees with length 45,
which took nothing, and 80 degrees with the same length, which cost one scratch.
This walks that corner finely and separates the settings into two groups: those
that leave every marked scratch alone, and those that do not.

Free is judged on BOTH sets. A setting that spares every scratch on calibration
but drops one on validation is not free — with only 19 marked scratches on
validation, a single loss is three points of recall.

Usage:  python sweep_corner.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "features.json")
RECALL_NOW = 62.5

ANGLES = (86, 85, 84, 83, 82, 81, 80, 79, 78)
LENGTHS = (55, 50, 45, 40, 36, 32, 28)


def stats(rows, drop):
    kinds = np.array([r["kind"] for r in rows])
    keep = ~drop
    real = int(((kinds == "scratch") | (kinds == "dirt"))[keep].sum())
    fake = int((kinds == "false")[keep].sum())
    scr_all = int((kinds == "scratch").sum())
    scr = int((kinds == "scratch")[keep].sum())
    fake_all = int((kinds == "false").sum())
    if real + fake == 0:
        return None
    return {"prec": 100.0 * real / (real + fake),
            "rec": RECALL_NOW * scr / max(scr_all, 1),
            "scr": scr, "scr_all": scr_all,
            "cut": fake_all - fake, "fake_all": fake_all}


def main():
    rows = json.load(open(SRC, encoding="utf-8"))
    sets = {s: [r for r in rows if r["set"] == s] for s in ("cal", "val")}
    prepared = {}
    for s, data in sets.items():
        prepared[s] = (data,
                       np.array([r["angle"] for r in data], float),
                       np.array([r["length"] for r in data], float))

    free, costly = [], []
    for a in ANGLES:
        for L in LENGTHS:
            res = {}
            for s in ("cal", "val"):
                data, ang, ln = prepared[s]
                res[s] = stats(data, (ang > a) & (ln > L))
            if not res["cal"] or not res["val"]:
                continue
            lost = ((res["cal"]["scr_all"] - res["cal"]["scr"])
                    + (res["val"]["scr_all"] - res["val"]["scr"]))
            row = (a, L, res, lost)
            (free if lost == 0 else costly).append(row)

    def show(title, group, n):
        if not group:
            print(f"\n{title}: none")
            return
        group.sort(key=lambda t: -(t[2]["cal"]["prec"] + t[2]["val"]["prec"]))
        print(f"\n{title}")
        print(f"{'angle':>6}{'length':>8}{'CAL prec':>10}{'CAL rec':>9}"
              f"{'   |':>4}{'VAL prec':>10}{'VAL rec':>9}"
              f"{'reflections cut':>18}{'scratches lost':>16}")
        print("-" * 92)
        for a, L, res, lost in group[:n]:
            c, v = res["cal"], res["val"]
            print(f"{a:>5}d{L:>8}{c['prec']:>9.0f}%{c['rec']:>8.1f}%"
                  f"{'   |':>4}{v['prec']:>9.0f}%{v['rec']:>8.1f}%"
                  f"{f'{c['cut']+v['cut']} of {c['fake_all']+v['fake_all']}':>18}"
                  f"{lost:>16}")

    print("today: calibration 69% / 62.5%   validation 73% / 62.5%")
    show("FREE — every marked scratch survives on both sets", free, 10)
    show("costs recall", costly, 6)


if __name__ == "__main__":
    main()
