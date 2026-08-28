# -*- coding: utf-8 -*-
"""
Are the mistakes spread evenly, or do a few records produce most of them?

This decides where it is worth looking next. If every record contributes its
share of false detections, then the difference must be found in the marks
themselves, one at a time. If instead a handful of records produce the bulk of
them, the problem is not really about individual marks at all — something about
those records or those photographs is generating streaks, and a judgement made
at that level would be worth more than any amount of per-mark cleverness.

It also asks the same question of the two channels and of the label ring, and
reports how much precision would rise if the worst records were simply fixed —
an upper bound on what a record-level idea could ever be worth.

Usage:  python where_false_lives.py
"""

import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "features.json")


def prec(rows):
    good = sum(1 for r in rows if r["kind"] != "false")
    return 100.0 * good / len(rows) if rows else 0.0


def main():
    rows = [r for r in json.load(open(SRC, encoding="utf-8"))
            if not (r["angle"] > 83 and r["length"] > 45)]

    for s in ("cal", "val"):
        data = [r for r in rows if r["set"] == s]
        by_rec = collections.defaultdict(list)
        for r in data:
            by_rec[r["record"]].append(r)

        stats = []
        for rec, rs in by_rec.items():
            fake = sum(1 for r in rs if r["kind"] == "false")
            stats.append((fake, len(rs), prec(rs), rec))
        stats.sort(reverse=True)
        total_false = sum(t[0] for t in stats)

        print(f"\nSET = {s}   {len(data)} marks over {len(stats)} records"
              f"   precision {prec(data):.0f}%   ({total_false} mistakes)")
        print(f"{'record':<34}{'marks':>7}{'wrong':>7}{'prec':>7}"
              f"{'share of all mistakes':>24}")
        print("-" * 80)
        run = 0
        for fake, n, p, rec in stats:
            run += fake
            bar = "#" * int(round(20 * fake / max(stats[0][0], 1)))
            print(f"{rec[:32]:<34}{n:>7}{fake:>7}{p:>6.0f}%"
                  f"   {bar:<20} {100.0*run/max(total_false,1):>3.0f}% cum")

        # the concentration question, answered plainly
        k = max(1, len(stats) // 5)
        top = sum(t[0] for t in stats[:k])
        print(f"\n  the worst {k} records of {len(stats)} hold "
              f"{100.0*top/max(total_false,1):.0f}% of all the mistakes")

        for cut in (1, 2, 3):
            keep = [r for rec, rs in by_rec.items() for r in rs
                    if sum(1 for q in rs if q["kind"] == "false") <= cut * 5]
            if keep and len(keep) < len(data):
                scr_all = sum(1 for r in data if r["kind"] == "scratch")
                scr_kept = sum(1 for r in keep if r["kind"] == "scratch")
                print(f"  if records with more than {cut*5} mistakes were "
                      f"perfect: precision {prec(keep):.0f}% "
                      f"(and they hold {scr_all-scr_kept} of {scr_all} scratches)")

    # is one channel doing the damage
    print(f"\nwhere the mistakes sit, across both sets")
    both = rows
    for name, sel in (
            ("groove-parallel channel", lambda r: r["tram"] > 0.5),
            ("radial channel", lambda r: r["tram"] <= 0.5),
            ("inner half of the ring", lambda r: r["rad"] < 0.65),
            ("outer half of the ring", lambda r: r["rad"] >= 0.65),
            ("short, under 30px", lambda r: r["length"] < 30),
            ("long, 30px and over", lambda r: r["length"] >= 30),
            ("seen in the other photo", lambda r: r["confirmed"] > 0.5),
            ("seen in one photo only", lambda r: r["confirmed"] <= 0.5)):
        grp = [r for r in both if sel(r)]
        if not grp:
            continue
        fake = sum(1 for r in grp if r["kind"] == "false")
        scr = sum(1 for r in grp if r["kind"] == "scratch")
        print(f"  {name:<26}{len(grp):>6} marks{fake:>6} wrong"
              f"{prec(grp):>6.0f}% prec{scr:>5} scratches")


if __name__ == "__main__":
    main()
