# -*- coding: utf-8 -*-
"""
Can anything tell dirt from a scratch?

Every measurement so far asked a different question: real against reflection.
That question is now answered as well as it is going to be. This one has never
been asked, and it is the one the grade needs — dirt is meant to weigh less than
damage, and the code can only act on that if something it can measure separates
the two.

The two are physically different in a way that should show: dirt sits ON the
surface and is a lump, damage is cut INTO it and is a line. Whether that survives
in a phone photograph at the working resolution is the question.

Only hand-labelled detections count here, and reflections are excluded outright —
they are a third thing, already handled by the angle rule.

Usage:  python probe_dirt_vs_scratch.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "features.json")
FEATURES = ("length", "thick", "elong", "area", "steps", "wmean", "spread",
            "ratio", "jitter", "rad", "raw", "score", "bright", "contrast",
            "chroma", "band", "angle", "tram", "confirmed")


def auc(pos, neg):
    if len(pos) < 5 or len(neg) < 5:
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
    rows = [r for r in json.load(open(SRC, encoding="utf-8"))
            if not (r["angle"] > 83 and r["length"] > 45)]
    scr = [r for r in rows if r["kind"] == "scratch"]
    dirt = [r for r in rows if r["kind"] == "dirt"]
    print(f"hand-labelled, reflections excluded: "
          f"{len(scr)} scratches, {len(dirt)} dirt\n")

    print(f"{'':<11}{'scratch med':>13}{'dirt med':>11}{'AUC':>7}   verdict")
    print("-" * 56)
    ranked = []
    for f in FEATURES:
        p = np.array([r[f] for r in scr], float)
        q = np.array([r[f] for r in dirt], float)
        a = auc(p, q)
        ranked.append((abs(a - 0.5), f, a, np.median(p), np.median(q)))
    ranked.sort(reverse=True)
    for sep, f, a, mp, mq in ranked:
        v = "USEFUL" if sep >= 0.15 else ("weak" if sep >= 0.08 else "nothing")
        print(f"{f:<11}{mp:>13.2f}{mq:>11.2f}{a:>7.2f}   {v}")

    best = ranked[0]
    print(f"\n  strongest separator: {best[1]}  (AUC {best[2]:.2f})")
    if best[0] < 0.15:
        print(f"  NOTHING reaches useful. A dirt weight cannot be applied by")
        print(f"  measurement — the detector has no way to know which is which.")
        return

    # what a threshold on the best one would actually cost
    f = best[1]
    p = np.array([r[f] for r in scr], float)
    q = np.array([r[f] for r in dirt], float)
    hi_is_scratch = best[2] > 0.5
    print(f"\n  what a single cut on {f} would separate:")
    print(f"{'cut':>10}{'scratches kept':>17}{'dirt kept':>12}")
    for cut in np.quantile(np.concatenate([p, q]), np.arange(.2, .81, .1)):
        ps = (p >= cut).mean() if hi_is_scratch else (p <= cut).mean()
        qs = (q >= cut).mean() if hi_is_scratch else (q <= cut).mean()
        print(f"{cut:>10.2f}{100*ps:>16.0f}%{100*qs:>11.0f}%")
    print(f"\n  a usable split needs the two columns far apart. If they move")
    print(f"  together, the cut is sorting by size, not by what the thing is.")


if __name__ == "__main__":
    main()
