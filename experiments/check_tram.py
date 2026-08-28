# -*- coding: utf-8 -*-
"""
Is the groove-parallel channel worth keeping?

It kept appearing beside the angle condition in the winning rules, which is odd:
on its own the channel looked like noise. The channel exists for scratches that
run ALONG the grooves, and it pays for them by accepting anything groove-parallel
and long — so it may be bringing in more mistakes than damage.

Three questions, all answerable from the labelled table:
  what the channel contributes on its own,
  what the rest of the detector looks like without it,
  and whether the scratches it finds are found by the other channel anyway.

Usage:  python check_tram.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "features.json")
RECALL_NOW = {"cal": 62.5, "val": 65.2}


def summarise(rows, label):
    kinds = np.array([r["kind"] for r in rows])
    real = int(((kinds == "scratch") | (kinds == "dirt")).sum())
    fake = int((kinds == "false").sum())
    scr = int((kinds == "scratch").sum())
    if real + fake == 0:
        return None
    return {"label": label, "n": len(rows), "prec": 100.0 * real / (real + fake),
            "scr": scr, "false": fake}


def main():
    rows = [r for r in json.load(open(SRC, encoding="utf-8"))
            if not (r["angle"] > 83 and r["length"] > 45)]

    for s in ("cal", "val"):
        data = [r for r in rows if r["set"] == s]
        tram = [r for r in data if r["tram"] > 0.5]
        rad = [r for r in data if r["tram"] <= 0.5]
        whole = summarise(data, "both channels (today)")
        t = summarise(tram, "the groove-parallel channel alone")
        r_ = summarise(rad, "without it")
        scr_all = whole["scr"]

        print(f"\nSET = {s}")
        print(f"{'':<38}{'marks':>7}{'prec':>7}{'scratches':>11}{'recall':>9}")
        print("-" * 72)
        for st in (whole, t, r_):
            if st is None:
                continue
            rec = RECALL_NOW[s] * st["scr"] / max(scr_all, 1)
            shown = f"{rec:>8.1f}%" if st is not whole else f"{RECALL_NOW[s]:>8.1f}%"
            print(f"{st['label']:<38}{st['n']:>7}{st['prec']:>6.0f}%"
                  f"{f'{st[chr(115)+chr(99)+chr(114)]}/{scr_all}':>11}{shown}")

    print(f"\n  The middle row is what the channel brings in by itself: if its")
    print(f"  precision is far below the whole, it is costing more than it adds.")
    print(f"  The recall column counts only scratches the OTHER channel would")
    print(f"  miss — a scratch found by both survives the channel being removed.")


if __name__ == "__main__":
    main()
