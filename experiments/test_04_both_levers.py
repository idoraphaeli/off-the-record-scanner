# -*- coding: utf-8 -*-
"""TEST 4 -- both levers at once, to find the operating point to ship.

Tests 1 and 2 each open the detector up, and on the full calibration set they
land in the same place at the same cost: at 30 detections a photo, lowering the
thresholds gives 60.7% recall and dropping the three anti-glare rules gives
59.1%. Applying BOTH has never been measured, and it cannot be assumed to be the
sum of the two -- test 1 already showed the curve turning over at its loosest
level, where recall kept climbing while the confirmed scratches fell.

So the rules come off and the thresholds are swept underneath them. What is
being looked for is the point where recall is still rising and the confirmed
scratches have not yet started to fall, because that is the operating point
worth writing into the detector.

Usage:  python test_04_both_levers.py [how many records]
"""

import collections
import os
import shutil
import sys

from detector import P
from test_01_loosen_then_confirm import LEVELS, TESTS
from test_01_precision_with_dirt import load_labels
from test_02_drop_glare_rules import (OFF_GROOVE, OFF_PATCH, OFF_RADIAL,
                                      pick_sides, run)
import test_02_drop_glare_rules as t2

OFF_ALL = {**OFF_RADIAL, **OFF_PATCH, **OFF_GROOVE}
LEVEL = dict(LEVELS)

CONFIGS = [
    ("1_baseline",          {}),
    ("2_rules_off",         OFF_ALL),
    ("3_rules_off_open",    {**OFF_ALL, **LEVEL["2_open"]}),
    ("4_rules_off_wide",    {**OFF_ALL, **LEVEL["3_wide"]}),
    ("5_rules_off_loose",   {**OFF_ALL, **LEVEL["4_loose"]}),
]


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    keys = set().union(*(set(p) for _, p in CONFIGS)) | set(LEVEL["1_current"])
    baseline = {k: P[k] for k in keys}

    # run() writes its pictures under test 2's folder; point it at this one
    t2.OUT = os.path.join(TESTS, "04_both_levers")
    if os.path.isdir(t2.OUT):
        shutil.rmtree(t2.OUT)
    os.makedirs(os.path.join(t2.OUT, "photos"), exist_ok=True)
    for name, _ in CONFIGS:
        os.makedirs(os.path.join(t2.OUT, f"config_{name}"), exist_ok=True)

    labels = load_labels()
    chosen = pick_sides(want)
    render, seen = {}, set()
    for rec, side, _ in chosen:
        if rec not in seen:
            seen.add(rec)
            render[(rec, side)] = len(seen)
    print(f"{len(chosen)} sides from {len(seen)} records\n")

    report = []
    try:
        for name, params in CONFIGS:
            P.update(baseline)
            report.append(run(name, params, chosen, render, labels))
    finally:
        P.update(baseline)

    head = (f"\n{'config':<22}{'per photo':>11}{'confirmed':>11}{'recall':>9}"
            f"{'scratches':>13}{'blank':>7}{'PRECISION':>12}{'unjudged':>10}")
    body = [head, "-" * len(head)]
    for r in report:
        kept = f"{r['found_kept']} of {r['zones']}"
        body.append(f"{r['name']:<22}{r['per_photo']:>11.1f}{r['conf_photo']:>11.1f}"
                    f"{r['recall']:>8.1f}%{kept:>13}{r['blank']:>7}"
                    f"{r['labelled']:>11.1f}%{r['unjudged']:>10}")
    print("\n".join(body))

    with open(os.path.join(t2.OUT, "results.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\n")
        fh.write("\n".join(body) + "\n")
        for r in report:
            fh.write(f"\n\nconfig {r['name']}   {r['params'] or 'nothing changed'}\n")
            fh.write("\n".join(r["lines"]) + "\n")
    print(f"\nwritten to {t2.OUT}")


if __name__ == "__main__":
    main()
