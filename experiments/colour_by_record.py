# -*- coding: utf-8 -*-
"""
Does colour separate false from real on the records where it should?

Across the whole set, colourfulness above the local background is a weak signal:
false detections are slightly more chromatic, consistently, but not enough to
build a rule on. That average may be hiding the thing that matters, though. The
records that fail are the ones shot in direct sun, where the reflections are of
a blue sky and a gold something; on a record shot indoors under neutral light
there is no colour for the feature to find, and those records dilute the average.

So this asks the question per record, on the ones that actually produce errors.
If colour separates there, a rule conditioned on the photo — not on the
detection — becomes possible.

Usage:  python colour_by_record.py
"""

import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEATS = ("chroma_rel", "chroma", "hue_shift", "band", "rad")
MIN_EACH = 8


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return 0.5
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
    rows = json.load(open(os.path.join(HERE, "false_features.json"),
                          encoding="utf-8"))
    by_rec = collections.defaultdict(list)
    for r in rows:
        by_rec[r["record"]].append(r)

    ranked = sorted(by_rec.items(),
                    key=lambda kv: -sum(1 for r in kv[1] if r["kind"] == "false"))

    print("AUC per record, real vs false. Below 0.5 = false are MORE colourful.")
    print("Only records with at least "
          f"{MIN_EACH} of each are shown.\n")
    head = f"{'record':<34}{'real':>6}{'false':>7}" + \
        "".join(f"{f:>12}" for f in FEATS)
    print(head)
    print("-" * len(head))

    shown = 0
    for rec, group in ranked:
        real = [r for r in group if r["kind"] in ("dirt", "scratch")]
        fake = [r for r in group if r["kind"] == "false"]
        if len(real) < MIN_EACH or len(fake) < MIN_EACH:
            continue
        cells = ""
        for f in FEATS:
            a = auc(np.array([r[f] for r in real], float),
                    np.array([r[f] for r in fake], float))
            mark = "*" if abs(a - 0.5) >= 0.15 else " "
            cells += f"{a:>11.2f}{mark}"
        print(f"{rec[:32]:<34}{len(real):>6}{len(fake):>7}{cells}")
        shown += 1

    if not shown:
        print("  no record has enough of both to judge")
    print("\n* marks a separation strong enough to build a rule on")


if __name__ == "__main__":
    main()
