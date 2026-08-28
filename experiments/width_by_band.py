# -*- coding: utf-8 -*-
"""
In SHORT marks, does width variability separate scratches from reflections, or
only dirt from reflections?

The band table showed the effect Ido spotted is real but sits in the short
marks, and reverses in the long ones. Short is also where dirt lives, so the
question that decides whether a rule is possible is which of the two real kinds
the signal actually distinguishes. A cut that removes reflections and dirt
together while sparing scratches is usable; one that takes scratches with them
is not.

Runs off width_features.json, so it costs nothing.

Usage:  python width_by_band.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "width_features.json")
BANDS = [(0, 25), (25, 40), (40, 10 ** 9)]
KEYS = ("spread", "ratio", "jitter")


def auc(pos, neg):
    if len(pos) < 5 or len(neg) < 5:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) \
        / (len(pos) * len(neg))


def main():
    rows = json.load(open(SRC, encoding="utf-8"))
    print(f"{len(rows)} labelled marks\n")

    for lo, hi in BANDS:
        band = [r for r in rows if lo <= r["steps"] < hi]
        scr = [r for r in band if r["kind"] == "scratch"]
        dirt = [r for r in band if r["kind"] == "dirt"]
        fake = [r for r in band if r["kind"] == "false"]
        tag = f"{lo}-{hi if hi < 10**9 else 'up'}"
        print(f"length {tag:<8}scratches {len(scr):>4}   dirt {len(dirt):>5}"
              f"   reflections {len(fake):>4}")
        if len(fake) < 5:
            print("    too few reflections here\n")
            continue
        for name, grp in (("scratch", scr), ("dirt", dirt)):
            if len(grp) < 5:
                print(f"    {name:<9}too few to judge")
                continue
            parts = []
            for k in KEYS:
                a = auc(np.array([r[k] for r in grp], float),
                        np.array([r[k] for r in fake], float))
                who = "reflections" if a < 0.5 else name
                parts.append(f"{k} -> {who} more variable"
                             f" ({'strong' if abs(a-0.5) >= 0.15 else 'weak'})")
            print(f"    {name:<9}{len(grp):>4} vs {len(fake)}")
            for p in parts:
                print(f"        {p}")
        # what a cut on the strongest short-mark feature would actually remove
        if lo == 0 and len(scr) >= 5:
            v_s = np.array([r["spread"] for r in scr], float)
            v_d = np.array([r["spread"] for r in dirt], float)
            v_f = np.array([r["spread"] for r in fake], float)
            print(f"\n    if short marks above a width-variability cut are dropped:")
            print(f"      {'cut':>6}{'reflections lost':>19}{'dirt lost':>12}"
                  f"{'SCRATCHES LOST':>17}")
            for c in np.percentile(v_f, [50, 60, 70, 80]):
                print(f"      {c:>6.2f}{100*(v_f>c).mean():>18.0f}%"
                      f"{100*(v_d>c).mean():>11.0f}%"
                      f"{100*(v_s>c).mean():>16.0f}%")
        print()


if __name__ == "__main__":
    main()
