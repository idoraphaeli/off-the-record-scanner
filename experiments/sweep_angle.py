# -*- coding: utf-8 -*-
"""
The full trade curve for one rule: reject marks that point straight out from the
centre of the disc.

The rule search turned up the same condition inside nearly every winning
combination, and the second condition was adding almost nothing. So the pair is
dropped and the one condition is swept on its own, which also makes it far
easier to trust — a single threshold cannot quietly overfit the way a pair can.

The measurement is the angle between a mark's long axis and the groove running
under it: 0 means the mark follows the groove round, 90 means it runs straight
out from the centre like a spoke. Vinyl's characteristic reflection is exactly
that spoke — the circular grooves throw a lamp back as a radial beam — while a
scratch made by hand has no reason to aim at the centre.

Both sets are printed side by side. The threshold is being chosen here, so
calibration is what it may be chosen ON; validation is only the check.

Usage:  python sweep_angle.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "features.json")
RECALL_NOW = 62.5
CUTS = (None, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70)


def stats(rows, keep):
    kinds = np.array([r["kind"] for r in rows])
    real = int(((kinds == "scratch") | (kinds == "dirt"))[keep].sum())
    fake = int((kinds == "false")[keep].sum())
    scr_all = int((kinds == "scratch").sum())
    scr = int((kinds == "scratch")[keep].sum())
    if real + fake == 0:
        return None
    return (100.0 * real / (real + fake),
            RECALL_NOW * scr / max(scr_all, 1),
            scr, scr_all, fake, int((kinds == "false").sum()))


def main():
    rows = json.load(open(SRC, encoding="utf-8"))
    sets = {s: [r for r in rows if r["set"] == s] for s in ("cal", "val")}

    a_scr = np.array([r["angle"] for r in rows if r["kind"] == "scratch"])
    a_dirt = np.array([r["angle"] for r in rows if r["kind"] == "dirt"])
    a_f = np.array([r["angle"] for r in rows if r["kind"] == "false"])
    print("how square-on to the grooves each kind sits "
          "(0 = along the groove, 90 = straight out from the centre):")
    for name, v in (("scratch", a_scr), ("dirt", a_dirt), ("reflection", a_f)):
        print(f"   {name:<11}median {np.median(v):>5.0f}   "
              f"a quarter are above {np.percentile(v, 75):>5.0f}   "
              f"above 80: {100*(v > 80).mean():>4.0f}%")

    head = (f"\n{'reject above':<14}"
            f"{'CAL prec':>10}{'CAL recall':>12}{'scratches':>11}"
            f"{'   |':>4}{'VAL prec':>10}{'VAL recall':>12}{'scratches':>11}")
    print(head)
    print("-" * len(head))

    for cut in CUTS:
        line = f"{'nothing' if cut is None else f'{cut} deg':<14}"
        for i, s in enumerate(("cal", "val")):
            data = sets[s]
            v = np.array([r["angle"] for r in data], float)
            keep = np.ones(len(data), bool) if cut is None else (v <= cut)
            st = stats(data, keep)
            if st is None:
                continue
            prec, rec, scr, scr_all, fake, fake_all = st
            line += f"{prec:>9.0f}%{rec:>11.1f}%{f'{scr}/{scr_all}':>11}"
            if i == 0:
                line += f"{'   |':>4}"
        print(line)

    print(f"\n  A cut removes marks pointing straight out from the centre.")
    print(f"  Both halves must improve together for the rule to be real.")


if __name__ == "__main__":
    main()
