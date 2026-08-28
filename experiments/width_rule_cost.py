# -*- coding: utf-8 -*-
"""
Is there a width-steadiness cut gentle enough to spare the scratches?

The constraint is Ido's: dirt may suffer, recall may not. Recall is scored
against his pen marks, so it is scratches and only scratches that must survive.

The rule is applied ONLY to short marks, since that is the band where the effect
runs in the useful direction, and long marks are left alone. Every cut is
reported as what it costs recall directly, rather than as a share of a subgroup,
because a share of the short band is easy to misread as small.

Runs off width_features.json.

Usage:  python width_rule_cost.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "width_features.json")
SHORT = 25          # the rule touches nothing longer than this
RECALL_NOW = 62.5   # calibration, current settings


def main():
    rows = json.load(open(SRC, encoding="utf-8"))
    short = [r for r in rows if r["steps"] < SHORT]

    all_scr = [r for r in rows if r["kind"] == "scratch"]
    s_scr = [r for r in short if r["kind"] == "scratch"]
    s_dirt = [r for r in short if r["kind"] == "dirt"]
    s_false = [r for r in short if r["kind"] == "false"]
    n_false_all = sum(1 for r in rows if r["kind"] == "false")

    print(f"the rule would look at marks shorter than {SHORT}px only")
    print(f"  scratches there {len(s_scr):>4} of {len(all_scr)} in total")
    print(f"  dirt            {len(s_dirt):>4}")
    print(f"  reflections     {len(s_false):>4} of {n_false_all} in total\n")

    v_s = np.array([r["spread"] for r in s_scr], float)
    v_d = np.array([r["spread"] for r in s_dirt], float)
    v_f = np.array([r["spread"] for r in s_false], float)

    head = (f"{'cut':>6}{'reflections gone':>18}{'dirt gone':>11}"
            f"{'scratches gone':>16}{'RECALL':>9}")
    print(head)
    print("-" * len(head))
    print(f"{'none':>6}{'0':>18}{'0':>11}{'0':>16}{RECALL_NOW:>8.1f}%")

    for cut in (0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.50):
        gone_f = int((v_f > cut).sum())
        gone_d = int((v_d > cut).sum())
        gone_s = int((v_s > cut).sum())
        recall = RECALL_NOW * (1 - gone_s / max(len(all_scr), 1))
        print(f"{cut:>6.2f}{f'{gone_f} of {len(s_false)}':>18}"
              f"{gone_d:>11}{f'{gone_s} of {len(all_scr)}':>16}"
              f"{recall:>8.1f}%")

    print(f"\n  RECALL is what the whole day was spent raising.")
    print(f"  A cut is only worth taking if that column does not move.")


if __name__ == "__main__":
    main()
