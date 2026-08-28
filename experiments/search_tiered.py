# -*- coding: utf-8 -*-
"""
A rule in two tiers: let the safe marks through, and be strict only with the
rest.

Every search so far joined conditions with AND, which can only ever remove. That
shape cannot express the thing the table is plainly saying. Marks seen in both
photographs of a side are right 89% of the time; marks seen in only one are right
71%. Demanding a strict condition of everything punishes the trustworthy group
for the sins of the other, and any rule strict enough to clean up the second
throws away most of the first.

So the rule here has a shape instead:

    keep it if it is TRUSTED, or if it passes a strict test

The trusted group is chosen from properties the detector already knows without
being told anything new — whether the same spot appeared in the other photograph,
where it sits in the ring, which channel found it. The strict test is then
searched for over the untrusted group alone, one or two plain comparisons.

The same two guards as before: dirt counts as a correct call, and every rule that
survives on the calibration records is re-scored on the validation records, which
had no part in choosing it.

Usage:  python search_tiered.py [target precision] [floor recall]
"""

import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "features.json")

RECALL_NOW = {"cal": 62.5, "val": 65.2}
FEATURES = ("length", "thick", "elong", "area", "steps", "wmean", "spread",
            "ratio", "jitter", "rad", "raw", "score", "bright", "contrast",
            "chroma", "band", "angle")
QUANTILES = np.arange(0.10, 0.91, 0.10)

# ways of saying "this one has already earned its place"
TIERS = {
    "seen in the other photo": lambda r: r["confirmed"] > 0.5,
    "inner part of the ring": lambda r: r["rad"] < 0.65,
    "seen in the other photo, or inner ring":
        lambda r: r["confirmed"] > 0.5 or r["rad"] < 0.65,
    "nothing trusted (plain search)": lambda r: False,
}


def score(rows, keep, which):
    kinds = np.array([r["kind"] for r in rows])
    real = int(((kinds == "scratch") | (kinds == "dirt"))[keep].sum())
    fake = int((kinds == "false")[keep].sum())
    scr_all = int((kinds == "scratch").sum())
    scr_kept = int((kinds == "scratch")[keep].sum())
    if real + fake == 0 or scr_all == 0:
        return None
    return {"precision": 100.0 * real / (real + fake),
            "recall": RECALL_NOW[which] * scr_kept / scr_all,
            "kept": int(keep.sum()), "scr": f"{scr_kept}/{scr_all}"}


def conditions(rows):
    out = []
    for f in FEATURES:
        v = np.array([r[f] for r in rows], float)
        for c in np.unique(np.round(np.quantile(v, QUANTILES), 4)):
            out.append((f"{f} <= {c:g}", v <= c))
            out.append((f"{f} >= {c:g}", v >= c))
    return out


def apply_rule(name, rows):
    keep = np.ones(len(rows), bool)
    for part in name.split(" AND "):
        f, op, c = part.rsplit(" ", 2)
        v = np.array([r[f] for r in rows], float)
        keep &= (v <= float(c)) if op == "<=" else (v >= float(c))
    return keep


def main():
    want_p = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    min_r = float(sys.argv[2]) if len(sys.argv) > 2 else 55.0

    rows = [r for r in json.load(open(SRC, encoding="utf-8"))
            if not (r["angle"] > 83 and r["length"] > 45)]
    cal = [r for r in rows if r["set"] == "cal"]
    val = [r for r in rows if r["set"] == "val"]
    print(f"calibration {len(cal)}   validation {len(val)}")
    print(f"today: cal precision {score(cal, np.ones(len(cal), bool), 'cal')['precision']:.0f}%"
          f"   val {score(val, np.ones(len(val), bool), 'val')['precision']:.0f}%")
    print(f"wanted: precision >= {want_p:.0f}% with recall >= {min_r:.0f}%\n")

    results = []
    for tier_name, tier_fn in TIERS.items():
        trust_c = np.array([tier_fn(r) for r in cal], bool)
        hard_c = ~trust_c
        if hard_c.sum() < 30:
            continue
        sub = [r for r, h in zip(cal, hard_c) if h]
        conds = conditions(sub)

        best = []
        singles = []
        for name, sub_keep in conds:
            keep = trust_c.copy()
            keep[hard_c] |= sub_keep
            s = score(cal, keep, "cal")
            if s and s["recall"] >= min_r:
                best.append((s["precision"], name, s))
            if s and s["recall"] >= min_r - 8:
                singles.append((name, sub_keep))

        for (n1, k1), (n2, k2) in itertools.combinations(singles, 2):
            if n1.split()[0] == n2.split()[0]:
                continue
            keep = trust_c.copy()
            keep[hard_c] |= (k1 & k2)
            s = score(cal, keep, "cal")
            if s and s["recall"] >= min_r:
                best.append((s["precision"], f"{n1} AND {n2}", s))

        best.sort(reverse=True, key=lambda t: t[0])
        seen = set()
        for prec, name, s in best:
            if name in seen:
                continue
            seen.add(name)
            results.append((prec, tier_name, name, s))
            if len(seen) >= 4:
                break

    results.sort(reverse=True, key=lambda t: t[0])
    if not results:
        sys.exit("nothing reaches the recall floor")

    print(f"{'trusted group':<40}{'strict test on the rest':<44}"
          f"{'prec':>6}{'recall':>8}")
    print("-" * 98)
    shown = []
    for prec, tier, name, s in results[:14]:
        print(f"{tier[:38]:<40}{name[:42]:<44}{s['precision']:>5.0f}%"
              f"{s['recall']:>7.1f}%")
        shown.append((tier, name))

    print(f"\n{'the same rules on VALIDATION':<84}{'prec':>6}{'recall':>8}")
    print("-" * 98)
    vb = score(val, np.ones(len(val), bool), "val")
    print(f"{'today, nothing removed':<84}{vb['precision']:>5.0f}%{vb['recall']:>7.1f}%")
    for tier, name in shown:
        trust_v = np.array([TIERS[tier](r) for r in val], bool)
        hard_v = ~trust_v
        sub = [r for r, h in zip(val, hard_v) if h]
        if not sub:
            continue
        keep = trust_v.copy()
        keep[hard_v] |= apply_rule(name, sub)
        sv = score(val, keep, "val")
        if sv:
            flag = "  <-- holds" if sv["precision"] >= want_p - 3 else ""
            print(f"{(tier + '  |  ' + name)[:82]:<84}"
                  f"{sv['precision']:>5.0f}%{sv['recall']:>7.1f}%{flag}")

    print(f"\n  A rule counts only if both tables agree. The left column is what")
    print(f"  gets a free pass; the right is what everything else must satisfy.")


if __name__ == "__main__":
    main()
