# -*- coding: utf-8 -*-
"""
Does a mark have company pointing the same way?

A lamp does not throw one streak. Tilt a record towards a bulb and a family of
parallel highlights appears together, because thousands of grooves are all
catching the same light at once. Damage has no such habit.

Ido's caution is the reason this counts siblings rather than merely noticing
them: a record dragged across a surface really does pick up several parallel
scratches, so "has a neighbour at the same angle" cannot separate on its own.
The question is how MANY — if reflections come in fives and eights while
scratches come in twos, the count still carries the signal even though the plain
yes/no does not.

Neighbours are counted in disc coordinates, so "nearby" means nearby on the
record itself rather than in a photograph taken from who knows what distance.

Usage:  python probe_siblings.py [cal|val|both]
"""

import collections
import csv
import json
import math
import os
import sys

import cv2
import numpy as np

import detector
from detector import P
from evaluate_frozen import MIN_EXTRA_AREA

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
VIEW_W = 1100

ANG_SAME = 15.0        # degrees: close enough to call two marks parallel
NEAR_FRAC = 0.25       # fraction of the disc radius that counts as nearby


def auc(pos, neg):
    if len(pos) < 5 or len(neg) < 5:
        return 0.5
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) \
        / (len(pos) * len(neg))


def photo_marks(path):
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner, outer = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer]
    radial, tram = detector.scratch_map(ring)[:2]

    k = VIEW_W / gray.shape[1]
    marks = []
    for smap, min_len in ((radial, None), (tram, P["TRAM_MIN_LEN"])):
        mask, _ = detector.extract(smap, min_len)
        n, lb, stats, cent = cv2.connectedComponentsWithStats(
            (mask > 127).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < MIN_EXTRA_AREA:
                continue
            comp = (lb[y:y + h, x:x + w] == i).astype(np.uint8)
            ang_pos = float(cent[i][0]) / P["POLAR_STEPS"] * 2 * math.pi
            rr = float(cent[i][1]) + inner
            marks.append({
                "orient": detector._axis_angle_deg(comp),
                "px": center[0] + rr * math.cos(ang_pos),
                "py": center[1] + rr * math.sin(ang_pos),
                "radius": radius,
                "vx": (center[0] + rr * math.cos(ang_pos)) * k,
                "vy": (center[1] + rr * math.sin(ang_pos)) * k,
            })

    near_px = NEAR_FRAC * radius
    for m in marks:
        same_near = same_any = 0
        for o in marks:
            if o is m:
                continue
            if abs(o["orient"] - m["orient"]) > ANG_SAME:
                continue
            same_any += 1
            if math.hypot(o["px"] - m["px"], o["py"] - m["py"]) <= near_px:
                same_near += 1
        m["siblings_near"] = float(same_near)
        m["siblings_any"] = float(same_any)
        m["neighbours"] = float(sum(
            1 for o in marks if o is not m and
            math.hypot(o["px"] - m["px"], o["py"] - m["py"]) <= near_px))
        m["marks_on_photo"] = float(len(marks))
    return marks


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    sets = ("cal", "val") if which == "both" else (which,)
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    records = set()
    for s in sets:
        records |= set(split[s])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in records]

    labels = collections.defaultdict(list)
    for s in sets:
        p = os.path.join(HERE, f"label_tool_{s}", f"labels_{s}.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p, encoding="utf-8"))["rows"]:
            if r["label"] in ("scratch", "dirt", "false"):
                labels[r["pair"]].append(r)

    out = []
    for r in rows:
        rows_l = labels.get(r["pair"])
        path = os.path.join(PHOTOS, r["photo_file"])
        if not rows_l or not os.path.exists(path):
            continue
        try:
            marks = photo_marks(path)
        except Exception:
            continue
        if not marks:
            continue
        arr = np.array([[m["vx"], m["vy"]] for m in marks], float)
        for lr in rows_l:
            d = np.hypot(arr[:, 0] - lr["cx"], arr[:, 1] - lr["cy"])
            j = int(d.argmin())
            if d[j] > 30:
                continue
            rec = {k: v for k, v in marks[j].items()
                   if k in ("siblings_near", "siblings_any", "neighbours",
                            "marks_on_photo", "orient")}
            rec["kind"] = lr["label"]
            out.append(rec)
        print(f"  {r['pair'][:46]:<48}{len(rows_l):>4}")

    if not out:
        sys.exit("nothing matched")
    fake = [r for r in out if r["kind"] == "false"]
    real = [r for r in out if r["kind"] in ("dirt", "scratch")]
    scr = [r for r in out if r["kind"] == "scratch"]
    print(f"\nmatched {len(out)}   real {len(real)}   false {len(fake)}"
          f"   scratches {len(scr)}")

    keys = ("siblings_near", "siblings_any", "neighbours", "marks_on_photo")
    for pos, name in ((real, "real"), (scr, "scratch")):
        if len(pos) < 10:
            continue
        print(f"\n[{name} vs reflections]   {len(pos)} vs {len(fake)}")
        print(f"{'':<16}{name[:8]+' med':>13}{'refl med':>11}{'AUC':>7}   verdict")
        for f in keys:
            p = np.array([r[f] for r in pos], float)
            q = np.array([r[f] for r in fake], float)
            a = auc(p, q)
            sep = abs(a - 0.5)
            v = "USEFUL" if sep >= 0.15 else ("weak" if sep >= 0.08 else "nothing")
            print(f"{f:<16}{np.median(p):>13.1f}{np.median(q):>11.1f}{a:>7.2f}   {v}")

    print(f"\nhow many parallel companions each kind keeps nearby:")
    for name, grp in (("scratch", scr), ("dirt",
                      [r for r in out if r["kind"] == "dirt"]),
                      ("reflection", fake)):
        if not grp:
            continue
        v = np.array([r["siblings_near"] for r in grp], float)
        print(f"   {name:<11}none {100*(v == 0).mean():>4.0f}%"
              f"   one or two {100*((v >= 1) & (v <= 2)).mean():>4.0f}%"
              f"   three or more {100*(v >= 3).mean():>4.0f}%")

    with open(os.path.join(HERE, "sibling_features.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"\nwrote sibling_features.json ({len(out)} rows)")


if __name__ == "__main__":
    main()
