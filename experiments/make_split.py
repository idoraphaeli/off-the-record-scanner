# -*- coding: utf-8 -*-
"""
Stage 0c — one-time calibration/test split (frozen after creation).
Stratified so the locked test set contains both marked and unmarked images:
marked pairs 5 cal / 2 test, unmarked 5 cal / 1 test  ->  10 cal / 3 test.
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(HERE, "gt")
OUT = os.path.join(HERE, "split.json")

summary = json.load(open(os.path.join(GT_DIR, "gt_summary.json"), encoding="utf-8"))
marked = sorted(n for n, v in summary.items() if v["marked"])
unmarked = sorted(n for n, v in summary.items() if not v["marked"])
print(f"{len(marked)} marked, {len(unmarked)} unmarked")

rng = random.Random(42)
rng.shuffle(marked)
rng.shuffle(unmarked)
# ~20% to the locked test set; take unmarked negatives too when they exist
n_test_m = 2 if unmarked else 3
n_test_u = 1 if unmarked else 0
split = {
    "cal": sorted(marked[n_test_m:] + unmarked[n_test_u:]),
    "test": sorted(marked[:n_test_m] + unmarked[:n_test_u]),
}
json.dump(split, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("cal:", len(split["cal"]), "| test:", len(split["test"]))
for n in split["test"]:
    print("  TEST:", n)
