# -*- coding: utf-8 -*-
"""
Which individual PHOTOS produce the genuinely-false detections.

The per-record view is too coarse to act on: a record has four photos, two per
side, and the labelling showed false positives concentrating in a handful of
shots rather than spreading across a record. Naming the exact file is what makes
the next step possible — open that photo and see what is different about it.

Usage:  python worst_photos.py [cal|val]
"""

import collections
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "gt_new")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "val"
    tool = os.path.join(HERE, f"label_tool_{which}")
    labels = os.path.join(tool, f"labels_{which}.json")
    if not os.path.exists(labels):
        sys.exit(f"no labels at {labels}")

    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        gt = {r["pair"]: r for r in csv.DictReader(fh)}
    rows = json.load(open(labels, encoding="utf-8"))["rows"]

    per = collections.Counter(r["pair"] for r in rows if r["label"] == "false")
    total = sum(per.values())
    print(f"SET = {which}   {total} genuinely-false detections\n")
    print(f"{'photo file':<54}{'side':>5}{'shot':>6}{'marks':>7}{'false':>7}")
    print("-" * 80)

    run = 0
    for pair, n in per.most_common():
        r = gt.get(pair)
        if r is None:
            continue
        run += n
        tag = "  <- you marked this photo clean" if int(r["marks"]) == 0 else ""
        print(f"{r['photo_file'][:52]:<54}{r['side']:>5}{r['shot']:>6}"
              f"{r['marks']:>7}{n:>7}{tag}")

    print(f"\ntop 3 photos account for "
          f"{100 * sum(n for _, n in per.most_common(3)) / max(total, 1):.0f}% "
          f"of all false detections in this set")


if __name__ == "__main__":
    main()
