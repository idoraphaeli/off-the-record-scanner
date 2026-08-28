# -*- coding: utf-8 -*-
"""
Where exactly does the length condition start costing scratches?

The angle is held at 83 degrees and only the length moves, so the boundary is
visible rather than asserted. Shorter means the rule bites more marks: it
catches more reflections, and eventually starts catching scratches too. The
question is the last value before that happens.

Usage:  python sweep_length_only.py [angle]
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "features.json")
RECALL_NOW = 62.5
LENGTHS = (100, 80, 70, 60, 55, 50, 48, 46, 45, 44, 42, 40, 36, 32, 28, 24)


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
            "lost": scr_all - scr, "cut": fake_all - fake}


def main():
    angle = float(sys.argv[1]) if len(sys.argv) > 1 else 83.0
    rows = json.load(open(SRC, encoding="utf-8"))
    sets = {s: [r for r in rows if r["set"] == s] for s in ("cal", "val")}

    print(f"angle held at {angle:.0f} degrees; only the length moves\n")
    head = (f"{'longer than':<13}{'CAL prec':>10}{'CAL rec':>9}{'lost':>6}"
            f"{'   |':>4}{'VAL prec':>10}{'VAL rec':>9}{'lost':>6}"
            f"{'reflections cut':>18}")
    print(head)
    print("-" * len(head))

    prev_free = None
    for L in LENGTHS:
        line = f"{L:<13}"
        res = {}
        for i, s in enumerate(("cal", "val")):
            data = sets[s]
            ang = np.array([r["angle"] for r in data], float)
            ln = np.array([r["length"] for r in data], float)
            st = stats(data, (ang > angle) & (ln > L))
            if st is None:
                break
            res[s] = st
            line += f"{st['prec']:>9.0f}%{st['rec']:>8.1f}%{st['lost']:>6}"
            if i == 0:
                line += f"{'   |':>4}"
        if len(res) < 2:
            continue
        cut = res["cal"]["cut"] + res["val"]["cut"]
        lost = res["cal"]["lost"] + res["val"]["lost"]
        line += f"{f'{cut} of 448':>18}"
        if lost == 0:
            prev_free = L
            line += "   <- free"
        print(line)

    print(f"\n  'lost' is marked scratches the rule would delete.")
    print(f"  The shortest length costing nothing is {prev_free}.")
    print(f"  Today, untouched: calibration 69% / 62.5%, "
          f"validation 73% / 62.5%.")


if __name__ == "__main__":
    main()
