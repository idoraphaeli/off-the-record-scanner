# -*- coding: utf-8 -*-
"""
Precision on the locked test set, scored the same way as the other two.

Detections that landed on a pen mark are correct by construction and were never
shown for labelling, so they are counted from the tool's own index rather than
from the verdicts. Dirt counts as a correct call: a dirty record really is in
worse condition, and the scanner is meant to say so. "Unsure" is dropped from
both sides rather than guessed at.

All three sets are recomputed here from their own label files, so the comparison
is like for like and does not lean on figures quoted earlier.

Usage:  python analyse_test.py [path to labels_test.json]
"""

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")

GOOD = ("scratch", "dirt")
SETS = ("cal", "val", "test")


def find_labels(which, override=None):
    if override and os.path.exists(override):
        return override
    for p in (os.path.join(HERE, f"label_tool_{which}", f"labels_{which}.json"),
              os.path.join(DOWNLOADS, f"labels_{which}.json")):
        if os.path.exists(p):
            return p
    return None


def matched_count(which):
    """Detections that fell on a pen mark, from the build the labels came from."""
    p = os.path.join(HERE, f"label_tool_{which}", "index.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8")).get("matched")


def tally(rows):
    c = collections.Counter(r["label"] for r in rows if r.get("label"))
    return c


def precision(c, matched):
    good = matched + c["scratch"] + c["dirt"]
    bad = c["false"]
    return 100.0 * good / max(good + bad, 1), good, bad


def main():
    override = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"{'set':<8}{'on a pen mark':>15}{'scratch':>9}{'dirt':>7}"
          f"{'false':>8}{'unsure':>8}{'PRECISION':>12}")
    print("-" * 68)
    summary = {}
    for which in SETS:
        path = find_labels(which, override if which == "test" else None)
        m = matched_count(which)
        if not path or m is None:
            print(f"{which:<8}  (no labels found)")
            continue
        data = json.load(open(path, encoding="utf-8"))
        rows = data["rows"]
        c = tally(rows)
        p, good, bad = precision(c, m)
        summary[which] = (p, c, m, rows)
        print(f"{which:<8}{m:>15}{c['scratch']:>9}{c['dirt']:>7}"
              f"{c['false']:>8}{c['unsure']:>8}{p:>11.1f}%")

    if "test" not in summary:
        sys.exit("\nno test labels found")

    p, c, m, rows = summary["test"]
    n_lab = sum(c.values())
    print(f"\nTEST SET, in detail")
    print(f"  detections the model reported : {m + n_lab}")
    print(f"  of them, on a pen mark        : {m}")
    print(f"  judged by you                 : {n_lab}"
          f"   ({c['unsure']} left as unsure)")
    print(f"\n  counted as CORRECT : {m} on a mark + {c['scratch']} scratch"
          f" + {c['dirt']} dirt = {m + c['scratch'] + c['dirt']}")
    print(f"  counted as WRONG   : {c['false']}")
    print(f"  PRECISION          : {p:.1f}%")

    # how much of the answer rests on dirt being counted as correct
    strict = 100.0 * (m + c["scratch"]) / max(m + c["scratch"] + c["dirt"]
                                              + c["false"], 1)
    print(f"\n  if dirt were counted as a MISTAKE instead: {strict:.1f}%")
    print(f"  (it is not — a dirty record is genuinely in worse condition)")

    by_rec = collections.defaultdict(collections.Counter)
    for r in rows:
        if r.get("label"):
            by_rec[r["record"]][r["label"]] += 1
    print(f"\n{'record':<26}{'scratch':>9}{'dirt':>7}{'false':>8}"
          f"{'unsure':>8}{'prec of judged':>16}")
    print("-" * 74)
    for rec in sorted(by_rec, key=lambda k: -by_rec[k]["false"]):
        d = by_rec[rec]
        g, b = d["scratch"] + d["dirt"], d["false"]
        pr = 100.0 * g / max(g + b, 1)
        print(f"{rec[:24]:<26}{d['scratch']:>9}{d['dirt']:>7}{d['false']:>8}"
              f"{d['unsure']:>8}{pr:>15.0f}%")
    print(f"\n  the last column excludes the on-a-mark detections, so it is")
    print(f"  harsher than the overall figure — it is only for comparing records")

    # scratches you had not marked in pen: these were missing from recall
    extra_scr = c["scratch"]
    print(f"\n  {extra_scr} real scratches were found that you had NOT marked in")
    print(f"  pen. Recall was scored against the pen marks only, so the true")
    print(f"  recall is a little better than the 56.9% reported.")


if __name__ == "__main__":
    main()
