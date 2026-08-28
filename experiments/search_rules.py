# -*- coding: utf-8 -*-
"""
Search for a simple rule that reaches 90% precision while holding recall at 60%.

Rules are kept to one or two conditions joined by AND, each a plain comparison,
so whatever comes out can be read as a sentence: "reject it if it is longer than
45 and sits in a patch brighter than 90". Nothing here is a model that has to be
taken on trust.

Two things this must not do:

  Count dirt as an error. Dirt is a correct call — a dirty record IS in worse
  condition — so precision here is (scratch + dirt) against false.

  Fool itself. Trying thousands of combinations will always turn up something
  that looks good by chance. So the search runs on the CALIBRATION records only
  and every survivor is then re-scored on the validation records, which took no
  part in choosing it. A rule that only works on one of the two was luck.

Recall is reported as what it would become: it is scored against Ido's pen
marks, so a rule that drops a tenth of the scratch detections drops recall by a
tenth.

Usage:  python search_rules.py [target precision] [floor recall]
"""

import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "features.json")

RECALL_NOW = 62.5          # calibration, with the radial rule in place
FEATURES = ("length", "thick", "elong", "area", "steps", "wmean", "spread",
            "ratio", "jitter", "rad", "raw", "score", "bright", "contrast",
            "chroma", "band", "angle", "tram", "confirmed")
QUANTILES = np.arange(0.05, 0.96, 0.05)


def score(rows, keep):
    """keep is a boolean mask over rows: what the rule would let through."""
    kinds = np.array([r["kind"] for r in rows])
    real = int(((kinds == "scratch") | (kinds == "dirt"))[keep].sum())
    fake = int((kinds == "false")[keep].sum())
    scr_all = int((kinds == "scratch").sum())
    scr_kept = int((kinds == "scratch")[keep].sum())
    if real + fake == 0 or scr_all == 0:
        return None
    return {
        "precision": 100.0 * real / (real + fake),
        "recall": RECALL_NOW * scr_kept / scr_all,
        "kept": int(keep.sum()), "real": real, "false": fake,
        "scr": f"{scr_kept}/{scr_all}",
    }


def conditions(rows):
    """Every single comparison worth trying, as (label, keep-mask)."""
    out = []
    for f in FEATURES:
        v = np.array([r[f] for r in rows], float)
        uniq = np.unique(v)
        cuts = uniq if len(uniq) <= 3 else np.quantile(v, QUANTILES)
        for c in np.unique(np.round(cuts, 4)):
            out.append((f"{f} <= {c:g}", v <= c))
            out.append((f"{f} >= {c:g}", v >= c))
    return out


def main():
    want_p = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    min_r = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0

    rows = json.load(open(SRC, encoding="utf-8"))
    # The radial rule is now part of the detector, so the marks it removes are
    # no longer on the table. Searching with them still in would rediscover the
    # pattern already handled and hide whatever sits underneath it.
    before = len(rows)
    rows = [r for r in rows
            if not (r["angle"] > 83 and r["length"] > 45)]
    print(f"the radial rule already removed {before - len(rows)} of {before}")

    cal = [r for r in rows if r["set"] == "cal"]
    val = [r for r in rows if r["set"] == "val"]
    print(f"calibration {len(cal)}   validation {len(val)}")
    base = score(cal, np.ones(len(cal), bool))
    print(f"today, on calibration: precision {base['precision']:.0f}%"
          f"   recall {base['recall']:.1f}%\n")
    print(f"looking for precision >= {want_p:.0f}% with recall >= {min_r:.0f}%\n")

    conds_cal = conditions(cal)
    print(f"{len(conds_cal)} single conditions, "
          f"{len(conds_cal) * (len(conds_cal) - 1) // 2} pairs\n")

    found = []
    for name, keep in conds_cal:
        s = score(cal, keep)
        if s and s["recall"] >= min_r:
            found.append((s["precision"], name, keep, s))

    # pairs, joined by AND. Only conditions that are not already useless on
    # their own are combined, or the search spends its time on noise.
    usable = [(n, k) for n, k in conds_cal
              if (s := score(cal, k)) and s["recall"] >= min_r - 5]
    for (n1, k1), (n2, k2) in itertools.combinations(usable, 2):
        k = k1 & k2
        s = score(cal, k)
        if s and s["recall"] >= min_r:
            found.append((s["precision"], f"{n1}  AND  {n2}", k, s))

    found.sort(reverse=True, key=lambda t: t[0])
    print(f"{'rule':<52}{'prec':>6}{'recall':>8}{'kept':>7}{'scratches':>11}")
    print("-" * 84)
    seen, shown = set(), 0
    for prec, name, keep, s in found:
        key = tuple(np.flatnonzero(keep)[:40])
        if key in seen:
            continue
        seen.add(key)
        print(f"{name[:50]:<52}{s['precision']:>5.0f}%{s['recall']:>7.1f}%"
              f"{s['kept']:>7}{s['scr']:>11}")
        shown += 1
        if shown == 12:
            break
    if not found:
        print("  nothing reaches the recall floor at all")

    # the honest part: re-score the survivors where they were not chosen
    print(f"\n{'the same rules on VALIDATION':<52}{'prec':>6}{'recall':>8}"
          f"{'kept':>7}{'scratches':>11}")
    print("-" * 84)
    vbase = score(val, np.ones(len(val), bool))
    print(f"{'today':<52}{vbase['precision']:>5.0f}%{vbase['recall']:>7.1f}%"
          f"{vbase['kept']:>7}{vbase['scr']:>11}")
    for prec, name, _keep, s in found[:12]:
        parts = name.split("  AND  ")
        keep_v = np.ones(len(val), bool)
        ok = True
        for part in parts:
            f, op, c = part.rsplit(" ", 2)
            if f not in FEATURES:
                ok = False
                break
            v = np.array([r[f] for r in val], float)
            keep_v &= (v <= float(c)) if op == "<=" else (v >= float(c))
        if not ok:
            continue
        sv = score(val, keep_v)
        if sv:
            print(f"{name[:50]:<52}{sv['precision']:>5.0f}%{sv['recall']:>7.1f}%"
                  f"{sv['kept']:>7}{sv['scr']:>11}")

    print(f"\n  A rule is real only if both tables agree. One that shines on")
    print(f"  calibration and collapses on validation was chance, not signal.")


if __name__ == "__main__":
    main()
