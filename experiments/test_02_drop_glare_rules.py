# -*- coding: utf-8 -*-
"""TEST 2 -- switch off the anti-reflection rules and let the gate cover them.

Three rules were added over the past months, each trading recall for precision
against glare:

  radial beam   a mark aimed almost dead at the centre AND long is the streak a
                lamp throws off thousands of concentric grooves. Removes a third
                of all false detections -- and a needle dropped on a record cuts
                exactly that shape.
  bright patch  in the outer ring, a mark sitting inside a pool of light is part
                of the pool. Tuned to keep only 90% of the scratches, so it is
                KNOWN to cost up to a tenth of them.
  groove line   a mark lying along the grooves is the disc's own geometry
                catching the light, unless it is longer than 250px.

Every one of them fights something that MOVES when the record is tilted, and the
cross-shot gate kills exactly that: measured, 184 confirmed detections at loose
thresholds contained 7 false ones. So the rules may now be paying for something
we already get, and paying in scratches.

Each rule is switched off on its own and then all three together, at the current
thresholds, so the effect of each is separable. Recall is scored against the pen
marks and needs no fresh judgement; precision is scored twice -- against the pen
marks (dirt counts as a mistake) and against every verdict given so far, with
the unjudged share reported beside it, since a relaxed rule invents detections
nobody has looked at.

Usage:  python test_02_drop_glare_rules.py [how many records]
"""

import collections
import os
import shutil
import sys

import numpy as np

import detector
from cross_shot import VIEW_W, label_profile, rotation_from_label
from detector import P
from test_01_loosen_then_confirm import (TESTS, WINDOW, analysed, gt_for,
                                         inside_disc, mask_of, measure, paint,
                                         pick_sides, save)
from test_01_precision_with_dirt import classify, load_labels
from tune_alignment import offsets, refine

OUT = os.path.join(TESTS, "02_drop_glare_rules")

# Each rule is disabled by pushing its threshold past what the measurement can
# ever reach, rather than by branching in the detector: the code that ships stays
# the code that is being measured.
OFF_RADIAL = dict(RADIAL_TOL_DEG=91.0)       # an axis angle never exceeds 90
OFF_PATCH = dict(GLARE_PATCH_MAX=256)        # a mean of 8-bit pixels never does
OFF_GROOVE = dict(GROOVE_TOL_DEG=-1.0)       # nor is an angle ever below zero

CONFIGS = [
    ("1_baseline", {}),
    ("2_no_radial_beam", OFF_RADIAL),
    ("3_no_bright_patch", OFF_PATCH),
    ("4_no_groove_line", OFF_GROOVE),
    ("5_none_of_the_three", {**OFF_RADIAL, **OFF_PATCH, **OFF_GROOVE}),
]


def run(name, params, chosen, render, labels):
    P.update(params)
    t = collections.Counter()
    verdicts = collections.Counter()
    lines = []

    for rec, side, shots in chosen:
        (pair_a, path_a), (_, path_b) = shots
        try:
            a, b = analysed(path_a), analysed(path_b)
            delta, _ = rotation_from_label(label_profile(path_a),
                                           label_profile(path_b))
        except Exception as exc:
            lines.append(f"  {rec} {side}: failed ({type(exc).__name__})")
            continue
        gt = gt_for(pair_a, a["img"].shape)
        if gt is None:
            continue

        if delta is None:
            kept = []
        else:
            fixed, _, _, _ = refine(a["marks"], b["marks"], delta)
            kept = [k for k, d in
                    enumerate(offsets(a["marks"], b["marks"], fixed, WINDOW))
                    if d is not None]

        all_mask = mask_of(a, range(len(a["marks"])))
        keep_mask = mask_of(a, kept)
        f0, zones, m0, s0 = measure(all_mask, gt)
        f1, _, m1, s1 = measure(keep_mask, gt)
        verdicts += classify(a, kept, gt, labels.get(pair_a, []))

        for k, v in (("zones", zones), ("found_all", f0), ("shown_all", s0),
                     ("miss_all", m0), ("found_kept", f1), ("shown_kept", s1),
                     ("miss_kept", m1), ("sides", 1),
                     ("blank", 1 if not kept else 0)):
            t[k] += v

        if (rec, side) in render:
            stem = f"{render[(rec, side)]:02d}_{rec[:26]}_{side}"
            ins = inside_disc(a["center"], a["radius"], a["img"].shape)
            save(paint(a["img"], keep_mask, ins),
                 os.path.join(OUT, f"config_{name}", f"{stem}_confirmed.jpg"))
            save(paint(a["img"], all_mask, ins),
                 os.path.join(OUT, f"config_{name}", f"{stem}_all.jpg"))
        lines.append(f"  {rec[:26]:<28}{side}  found {s0:>4} -> kept {s1:>3}"
                     f"   scratches {f0:>2}/{zones:<3} -> {f1:>2}")

    good = verdicts["on_mark"] + verdicts["scratch"] + verdicts["dirt"]
    bad = verdicts["false"]
    row = dict(
        name=name, params=params, lines=lines,
        per_photo=t["shown_all"] / max(t["sides"], 1),
        conf_photo=t["shown_kept"] / max(t["sides"], 1),
        recall=100.0 * t["found_all"] / max(t["zones"], 1),
        recall_gate=100.0 * t["found_kept"] / max(t["zones"], 1),
        prec=100.0 * (t["shown_all"] - t["miss_all"]) / max(t["shown_all"], 1),
        prec_gate=100.0 * (t["shown_kept"] - t["miss_kept"]) / max(t["shown_kept"], 1),
        found_kept=t["found_kept"], zones=t["zones"], blank=t["blank"],
        judged=good + bad, unjudged=verdicts["unjudged"],
        labelled=100.0 * good / max(good + bad, 1),
        scratch=verdicts["scratch"], dirt=verdicts["dirt"], false=bad,
        on_mark=verdicts["on_mark"])
    print(f"  {name:<22} recall {row['recall']:>5.1f}%   confirmed "
          f"{row['conf_photo']:>5.1f}/photo   scratches kept "
          f"{row['found_kept']:>3}/{row['zones']}")
    return row


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    keys = set(OFF_RADIAL) | set(OFF_PATCH) | set(OFF_GROOVE)
    baseline = {k: P[k] for k in keys}

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "photos"), exist_ok=True)
    for name, _ in CONFIGS:
        os.makedirs(os.path.join(OUT, f"config_{name}"), exist_ok=True)

    labels = load_labels()
    chosen = pick_sides(want)
    render, seen = {}, set()
    for rec, side, _ in chosen:
        if rec not in seen:
            seen.add(rec)
            render[(rec, side)] = len(seen)
    for rec, side, shots in chosen:
        if (rec, side) in render:
            save(detector.load_image(shots[0][1]),
                 os.path.join(OUT, "photos",
                              f"{render[(rec, side)]:02d}_{rec[:26]}_{side}.jpg"))
    print(f"{len(chosen)} sides from {len(seen)} records\n")

    report = []
    try:
        for name, params in CONFIGS:
            P.update(baseline)      # each config is measured against a clean slate
            report.append(run(name, params, chosen, render, labels))
    finally:
        P.update(baseline)

    head = (f"\n{'config':<22}{'per photo':>11}{'confirmed':>11}{'recall':>9}"
            f"{'after gate':>12}{'scratches':>11}{'blank':>7}"
            f"{'PRECISION':>12}{'unjudged':>10}")
    body = [head, "-" * len(head)]
    for r in report:
        kept = f"{r['found_kept']} of {r['zones']}"
        body.append(
            f"{r['name']:<22}{r['per_photo']:>11.1f}{r['conf_photo']:>11.1f}"
            f"{r['recall']:>8.1f}%{r['recall_gate']:>11.1f}%{kept:>11}"
            f"{r['blank']:>7}{r['labelled']:>11.1f}%{r['unjudged']:>10}")
    print("\n".join(body))
    print("\nPRECISION is over the confirmed set, dirt counted as a correct call,")
    print("and it can only see detections that have been judged -- the last")
    print("column says how many it could not.")

    with open(os.path.join(OUT, "results.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\n")
        fh.write("\n".join(body) + "\n")
        for r in report:
            fh.write(f"\n\nconfig {r['name']}   {r['params'] or 'nothing changed'}\n")
            fh.write(f"  confirmed: {r['on_mark']} on a pen mark, "
                     f"{r['scratch']} scratch, {r['dirt']} dirt, "
                     f"{r['false']} false, {r['unjudged']} unjudged\n")
            fh.write("\n".join(r["lines"]) + "\n")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
