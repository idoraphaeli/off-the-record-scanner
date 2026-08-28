# -*- coding: utf-8 -*-
"""
Read back the hand-labelled detections and say what the model is actually doing.

Every unmatched detection was counted as a false positive up to now, which
assumes dirt and dust are errors. They are not: a record photographed dirty is a
record in worse condition, and the user is asked to clean it first. So this
answers how the "false positives" really split, and what precision looks like
once dirt counts as a correct call.

Precision needs the matched detections too — the ones that landed on a pen mark
and were never shown for labelling, because they are correct by construction.
Those are counted from the tool's own index, not assumed.

The calibration set was labelled across three nested operating points and its
entries carry a tier; sets built afterwards carry tier 0, meaning "whatever the
detector holds", and the per-tier breakdown is then skipped as meaningless.

Usage:  python analyse_labels.py [cal|val|test] [path to labels json]
"""

import collections
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "gt_new")

TIER_NAME = {4: "recommended", 6: "+ mild easing", 8: "+ full easing"}
REAL = ("scratch", "dirt")


def load(which, path):
    tool = os.path.join(HERE, f"label_tool_{which}")
    index = json.load(open(os.path.join(tool, "index.json"), encoding="utf-8"))
    if path is None:
        for cand in (os.path.join(tool, f"labels_{which}.json"),
                     os.path.join(HERE, "label_tool_cal", f"labels_{which}.json")):
            if os.path.exists(cand):
                path = cand
                break
    if path is None or not os.path.exists(path):
        sys.exit(f"no labels found for {which} — export them from the tool first")
    return index, json.load(open(path, encoding="utf-8")), path


def block(title, rows, matched, clean_pairs, clean_photos):
    real = sum(1 for r in rows if r["label"] in REAL)
    false = sum(1 for r in rows if r["label"] == "false")
    unsure = sum(1 for r in rows if r["label"] == "unsure")
    judged = matched + real + false
    prec = 100.0 * (matched + real) / judged if judged else 0.0
    print(f"\n{title}")
    print(f"  on a pen mark already (correct)  : {matched}")
    print(f"  labelled real (dirt or scratch)  : {real}")
    print(f"  labelled genuinely false         : {false}")
    print(f"  unsure (left out of precision)   : {unsure}")
    print(f"  PRECISION                        : {prec:.0f}%")

    sel = [r for r in rows if r["pair"] in clean_pairs]
    if sel and clean_photos:
        cr = sum(1 for r in sel if r["label"] in REAL) / clean_photos
        cf = sum(1 for r in sel if r["label"] == "false") / clean_photos
        print(f"  on the {clean_photos} photos you marked clean: "
              f"{cr:.1f} real, {cf:.1f} truly false per photo")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "cal"
    path = sys.argv[2] if len(sys.argv) > 2 else None
    index, done, path = load(which, path)
    by_id = {e["id"]: e for e in index["items"]}

    rows = [r for r in done["rows"] if r["label"]]
    print(f"SET = {which}   {len(rows)} of {index['total']} detections labelled")
    print(f"  {path}")
    if not rows:
        return

    tally = collections.Counter(r["label"] for r in rows)
    print()
    for k in ("scratch", "dirt", "false", "unsure"):
        n = tally[k]
        bar = "#" * int(round(40 * n / max(tally.values())))
        print(f"  {k:<10}{n:>5}{100*n/len(rows):>7.1f}%  {bar}")

    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        gt_rows = [r for r in csv.DictReader(fh)]
    pairs_here = {e["pair"] for e in index["items"]}
    clean_pairs = {r["pair"] for r in gt_rows if int(r["marks"]) == 0}
    # count clean photos from the split, not from the ones that happened to
    # produce a detection, or a clean photo the model left alone would vanish
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    in_set = set(split[which])
    clean_photos = sum(1 for r in gt_rows
                       if r["record"] in in_set and int(r["marks"]) == 0)

    tiers = {by_id[r["id"]]["tier"] for r in rows if r["id"] in by_id}
    matched_all = index.get("matched")

    if tiers == {0}:
        if matched_all is None:
            sys.exit("index.json has no 'matched' count — rebuild the tool, "
                     "otherwise precision would silently omit the detections "
                     "that landed on your own marks")
        block("whole set, at the detector's current settings",
              rows, matched_all, clean_pairs, clean_photos)
    else:
        cum = []
        for tier in (4, 6, 8):
            cum.append(tier)
            sel = [r for r in rows if by_id[r["id"]]["tier"] in cum]
            if sel:
                block(f"operating point: {TIER_NAME[tier]}", sel,
                      MATCHED_BY_TIER[tier], clean_pairs, clean_photos)

    worst = collections.Counter(r["record"] for r in rows if r["label"] == "false")
    if worst:
        print("\nrecords producing the most genuinely false detections:")
        for rec, n in worst.most_common(8):
            print(f"  {rec[:44]:<46}{n:>4}")


# Measured when the calibration tool was built, per nested operating point:
# total detections shown (715 / 973 / 1219) minus those offered for labelling
# (459 / 691 / 894). Its index predates the "matched" field.
MATCHED_BY_TIER = {4: 256, 6: 282, 8: 325}


if __name__ == "__main__":
    main()
