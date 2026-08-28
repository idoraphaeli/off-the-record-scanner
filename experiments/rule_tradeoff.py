# -*- coding: utf-8 -*-
"""
Turn the feature separations into decisions.

An AUC says a feature carries signal; it does not say what a rule built on it
would cost. This sweeps a threshold across the promising features and reports
the only three numbers that matter for each cut:

    false removed    the gain
    scratch removed  the price, and the one that must stay near zero
    dirt removed     also a price, but a much cheaper one — dirt is a real
                     finding that lowers a record's grade, just not damage

Rules are only worth adopting where the gain is several times the scratch cost.
Note the sample: 414 false and 985 dirt, but only 55 scratches, so every
scratch-side figure moves by ~2% per record and should be read as a direction
rather than a measurement.

Usage:  python rule_tradeoff.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# feature, direction, thresholds. "above" means the rule REJECTS values above
# the threshold, which is where the false detections were found to sit.
CANDIDATES = [
    ("rad", "above", (0.72, 0.76, 0.80, 0.84, 0.88)),
    ("thick", "above", (4.0, 5.0, 6.0, 8.0, 10.0)),
    ("area", "above", (150, 200, 300, 500, 800)),
    ("length", "above", (35, 45, 60, 90, 130)),
    ("band", "above", (4, 8, 14, 22, 35)),
    ("contrast", "above", (18, 24, 30, 40, 55)),
    ("bright", "above", (70, 80, 95, 110, 130)),
]

# Stacked with OR, not AND: each of these cuts a different slice of the false
# detections at near-zero cost on its own, so the question is whether they
# overlap or add up.
COMBOS = [
    ("rad>0.80 AND band>8", lambda r: r["rad"] > 0.80 and r["band"] > 8),
    ("rad>0.80 AND thick>4", lambda r: r["rad"] > 0.80 and r["thick"] > 4),
    ("band>8 AND thick>4", lambda r: r["band"] > 8 and r["thick"] > 4),
    ("OVERSIZE: len>60 | area>500 | thick>8",
     lambda r: r["length"] > 60 or r["area"] > 500 or r["thick"] > 8),
    ("OVERSIZE, harder: len>45 | area>300 | thick>6",
     lambda r: r["length"] > 45 or r["area"] > 300 or r["thick"] > 6),
    ("OVERSIZE + rim: above | rad>0.88",
     lambda r: r["length"] > 60 or r["area"] > 500 or r["thick"] > 8
     or r["rad"] > 0.88),
    ("OVERSIZE + bright band: above | band>22",
     lambda r: r["length"] > 60 or r["area"] > 500 or r["thick"] > 8
     or r["band"] > 22),
]


def score(rows, drops):
    """drops(row) -> True when the rule would reject that detection."""
    out = {}
    for kind in ("false", "dirt", "scratch"):
        grp = [r for r in rows if r["kind"] == kind]
        n = sum(1 for r in grp if drops(r))
        out[kind] = (n, len(grp))
    return out


def line(name, s):
    f, ft = s["false"]
    d, dt = s["dirt"]
    c, ct = s["scratch"]
    ratio = f / c if c else float("inf")
    flag = "  <-- worth it" if c and ratio >= 6 and f >= 40 else \
           ("  <-- clean" if c == 0 and f >= 20 else "")
    print(f"{name:<34}{f:>4}/{ft} ({100*f/ft:>4.0f}%)"
          f"{d:>6}/{dt} ({100*d/dt:>4.0f}%)"
          f"{c:>5}/{ct} ({100*c/ct:>4.0f}%)"
          f"{('%.0f' % ratio) if c else '-':>7}{flag}")


def main():
    rows = json.load(open(os.path.join(HERE, "false_features.json"),
                          encoding="utf-8"))
    n_f = sum(1 for r in rows if r["kind"] == "false")
    n_d = sum(1 for r in rows if r["kind"] == "dirt")
    n_s = sum(1 for r in rows if r["kind"] == "scratch")
    print(f"{len(rows)} labelled detections   "
          f"false {n_f}   dirt {n_d}   scratch {n_s}\n")

    head = (f"{'rule (reject when...)':<34}{'FALSE cut':>16}"
            f"{'dirt cut':>17}{'SCRATCH cut':>16}{'gain/cost':>7}")
    print(head)
    print("-" * len(head))

    for feat, _, thresholds in CANDIDATES:
        for t in thresholds:
            line(f"{feat} > {t}", score(rows, lambda r, f=feat, t=t: r[f] > t))
        print()

    print("combined rules")
    print("-" * len(head))
    for name, fn in COMBOS:
        line(name, score(rows, fn))


if __name__ == "__main__":
    main()
