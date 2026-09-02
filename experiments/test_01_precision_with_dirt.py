# -*- coding: utf-8 -*-
"""TEST 1, second half -- precision of the confirmed set, with dirt counted as
a correct call.

The sweep scored precision against the pen marks, which were drawn on scratches
only, so dirt came out as a mistake and the numbers read far below the ones we
normally quote. That is not the rule we grade by: a dirty record really is in
worse condition and the scanner is meant to say so.

The obstacle is that the hand labels describe the detections the CURRENT
thresholds produce. A looser level invents detections nobody has judged. So each
confirmed detection is placed in one of three boxes:

    on a pen mark   correct by construction, and never shown for labelling
    hand-labelled   matched by position to a verdict you gave: scratch/dirt/false
    unjudged        new at this level, and honestly unknown

Precision is computed over the first two and the third is reported beside it, so
it is visible how much of the answer rests on detections nobody has looked at.
That share grows as the gates open, which is exactly the caveat that has to
travel with the number.

Usage:  python test_01_precision_with_dirt.py [how many records]
"""

import collections
import json
import os
import sys

import cv2
import numpy as np

import detector
from cross_shot import VIEW_W, label_profile, rotation_from_label
from detector import P
from evaluate_frozen import TOLERANCE
from test_01_loosen_then_confirm import (LEVELS, OUT, WINDOW, analysed, gt_for,
                                         mask_of, pick_sides)
from tune_alignment import offsets, refine

HERE = os.path.dirname(os.path.abspath(__file__))
JOIN_PX = 25          # how close a detection must be to inherit a verdict


SOURCES = [
    os.path.join(HERE, "label_tool_cal", "labels_cal.json"),
    # the round done specifically on level 4's confirmed detections. Merged in
    # so that every level is scored against the same body of verdicts -- without
    # it, level 4 would be judged on evidence the other levels never saw.
    os.path.join(HERE, "label_tool_test01", "labels_test01.json"),
    os.path.join(os.path.expanduser("~"), "Downloads", "labels_test01.json"),
]


def load_labels():
    """Every verdict given on these records, from every round, by photo."""
    out = collections.defaultdict(list)
    seen = set()
    for path in SOURCES:
        if not os.path.exists(path):
            continue
        for r in json.load(open(path, encoding="utf-8"))["rows"]:
            if r.get("label") in ("scratch", "dirt", "false", "unsure") \
                    and r["id"] not in seen:
                seen.add(r["id"])
                out[r["pair"]].append(r)
    return out


def classify(shot, kept, gt, verdicts):
    """Each confirmed detection, as one of: on a mark, a verdict, or unjudged."""
    near_gt = (cv2.dilate((gt > 127).astype(np.uint8),
                          np.ones((TOLERANCE, TOLERANCE), np.uint8)) > 0
               if gt is not None else None)
    scale = VIEW_W / float(shot["img"].shape[1])
    arr = np.array([[r["cx"], r["cy"]] for r in verdicts], float) \
        if verdicts else np.zeros((0, 2))

    c = collections.Counter()
    for k in kept:
        m = shot["marks"][k]
        cy, cx = int(round(m["cy"])), int(round(m["cx"]))
        if near_gt is not None and 0 <= cy < near_gt.shape[0] \
                and 0 <= cx < near_gt.shape[1] and near_gt[cy, cx]:
            c["on_mark"] += 1
            continue
        if len(arr):
            d = np.hypot(arr[:, 0] - cx * scale, arr[:, 1] - cy * scale)
            j = int(d.argmin())
            if d[j] <= JOIN_PX:
                c[verdicts[j]["label"]] += 1
                continue
        c["unjudged"] += 1
    return c


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    baseline = {k: P[k] for k in ("PCT_STRONG", "PCT_WEAK", "THR_FLOOR")}
    labels = load_labels()
    chosen = pick_sides(want)
    print(f"{len(chosen)} sides\n")

    report = []
    try:
        for name, params in LEVELS:
            P.update(params)
            c = collections.Counter()
            for rec, side, shots in chosen:
                (pair_a, path_a), (_, path_b) = shots
                try:
                    a, b = analysed(path_a), analysed(path_b)
                    delta, _ = rotation_from_label(label_profile(path_a),
                                                   label_profile(path_b))
                except Exception:
                    continue
                if delta is None:
                    continue
                fixed, _, _, _ = refine(a["marks"], b["marks"], delta)
                kept = [k for k, d in
                        enumerate(offsets(a["marks"], b["marks"], fixed, WINDOW))
                        if d is not None]
                c += classify(a, kept, gt_for(pair_a, a["img"].shape),
                              labels.get(pair_a, []))
            report.append((name, params, c))
            good = c["on_mark"] + c["scratch"] + c["dirt"]
            bad = c["false"]
            p = 100.0 * good / max(good + bad, 1)
            tot = sum(c.values())
            print(f"level {name}: {p:.1f}%   "
                  f"(judged {good+bad} of {tot}, unjudged {c['unjudged']})")
    finally:
        P.update(baseline)

    head = (f"\n{'level':<12}{'confirmed':>11}{'on a mark':>11}{'scratch':>9}"
            f"{'dirt':>7}{'false':>7}{'unsure':>8}{'unjudged':>10}"
            f"{'PRECISION':>12}{'scratch only':>14}")
    body = [head, "-" * len(head)]
    for name, params, c in report:
        good = c["on_mark"] + c["scratch"] + c["dirt"]
        bad = c["false"]
        p = 100.0 * good / max(good + bad, 1)
        # the same set scored the other way, so both readings sit side by side
        strict = 100.0 * (c["on_mark"] + c["scratch"]) \
            / max(c["on_mark"] + c["scratch"] + c["dirt"] + bad, 1)
        body.append(f"{name:<12}{sum(c.values()):>11}{c['on_mark']:>11}"
                    f"{c['scratch']:>9}{c['dirt']:>7}{c['false']:>7}"
                    f"{c['unsure']:>8}{c['unjudged']:>10}{p:>11.1f}%"
                    f"{strict:>13.1f}%")
    print("\n".join(body))
    print("\nPRECISION counts dirt as a correct call; the last column counts it")
    print("as a mistake. Neither column can see the unjudged detections.")

    with open(os.path.join(OUT, "precision_with_dirt.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(__doc__ + "\n")
        fh.write("\n".join(body) + "\n")


if __name__ == "__main__":
    main()
