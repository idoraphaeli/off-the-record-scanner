# -*- coding: utf-8 -*-
"""
Does a mark's WIDTH stay steady along its length?

Ido's observation, from looking through the painted false positives: a real
scratch holds roughly one width from end to end, while many of the false ones
start narrow and flare, or wander in and out along their length.

The physics is on his side. A scratch is cut — its width is set by whatever was
dragged across the vinyl and has no reason to change halfway. A reflection's
edge is set by the geometry of a lamp, a surface and a viewing angle, and
nothing holds it to a constant width.

Average thickness was measured before and separated nothing; averaging is
exactly what destroys this signal. Here each mark is straightened along its own
axis and its width counted at every step, giving a profile rather than a number:

    spread   how much the width varies against its own average
    ratio    the widest part over the narrowest
    jitter   how much it changes from one step to the next — closest to
             "wobbly along its length"

Everything is measured in the unwrapped ring, where the detector actually found
the mark; mapping back to the photograph stretches shapes and would change the
very thing being measured.

Usage:  python probe_width.py [cal|val|both]
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
FEATURES = ("spread", "ratio", "jitter", "wmean", "steps")


def auc(pos, neg):
    if not len(pos) or not len(neg):
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


def width_profile(comp):
    """Widths measured along the mark's own long axis.

    The component is rotated so that axis lies horizontal, after which the width
    at each step is simply how many pixels that column holds. Rotating first is
    what makes the numbers comparable between a mark lying flat and one standing
    on end.
    """
    pts = np.column_stack(np.nonzero(comp)).astype(np.float32)   # (row, col)
    if len(pts) < 12:
        return None
    mean = pts.mean(axis=0)
    _, eig = cv2.PCACompute(pts, mean.reshape(1, -1), maxComponents=1)
    ang = math.degrees(math.atan2(float(eig[0][0]), float(eig[0][1])))

    pad = int(max(comp.shape) * 1.5) + 8
    big = cv2.copyMakeBorder(comp, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    c = (big.shape[1] / 2.0, big.shape[0] / 2.0)
    M = cv2.getRotationMatrix2D(c, -ang, 1.0)
    rot = cv2.warpAffine(big, M, (big.shape[1], big.shape[0]),
                         flags=cv2.INTER_NEAREST)

    w = rot.sum(axis=0).astype(np.float32)
    w = w[w > 0]
    if len(w) < 5:
        return None
    m = float(w.mean())
    if m <= 0:
        return None
    return {
        "spread": float(w.std() / m),
        "ratio": float(np.percentile(w, 90) / max(np.percentile(w, 10), 1.0)),
        "jitter": float(np.abs(np.diff(w)).mean() / m),
        "wmean": m,
        "steps": float(len(w)),
    }


def ring_components(path):
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner, outer = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer]
    radial, tram = detector.scratch_map(ring)[:2]

    out = []
    for smap, min_len in ((radial, None), (tram, P["TRAM_MIN_LEN"])):
        mask, _ = detector.extract(smap, min_len)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(
            (mask > 127).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < MIN_EXTRA_AREA:
                continue
            prof = width_profile((lab[y:y + h, x:x + w] == i).astype(np.uint8))
            if prof is None:
                continue
            ang = cent[i][0] / P["POLAR_STEPS"] * 2 * math.pi
            rad = cent[i][1] + inner
            k = VIEW_W / gray.shape[1]
            prof["vx"] = (center[0] + rad * math.cos(ang)) * k
            prof["vy"] = (center[1] + rad * math.sin(ang)) * k
            out.append(prof)
    return out


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
            cands = ring_components(path)
        except Exception:
            continue
        if not cands:
            continue
        arr = np.array([[c["vx"], c["vy"]] for c in cands], float)
        for lr in rows_l:
            d = np.hypot(arr[:, 0] - lr["cx"], arr[:, 1] - lr["cy"])
            k = int(d.argmin())
            if d[k] > 30:
                continue
            rec = dict(cands[k])
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

    def table(pos, name):
        if len(pos) < 12:
            print(f"\n[{name}] only {len(pos)} — too few to judge")
            return
        print(f"\n[{name} vs false]   {len(pos)} vs {len(fake)}")
        print(f"{'':<9}{name[:8]+' med':>13}{'false med':>12}{'AUC':>7}   verdict")
        for f in FEATURES:
            p = np.array([r[f] for r in pos], float)
            q = np.array([r[f] for r in fake], float)
            a = auc(p, q)
            sep = abs(a - 0.5)
            v = "USEFUL" if sep >= 0.15 else ("weak" if sep >= 0.08 else "nothing")
            print(f"{f:<9}{np.median(p):>13.2f}{np.median(q):>12.2f}{a:>7.2f}   {v}")

    table(real, "real")
    table(scr, "scratch")

    # Split by length. A speck of dust is a handful of pixels — its "width along
    # its length" is two or three measurements and means nothing, so applying a
    # steadiness test to it would only throw away real dirt. Reflections are the
    # long marks, so the question is where along the length scale the test
    # starts to earn its place.
    bands = [(0, 25), (25, 40), (40, 60), (60, 10 ** 9)]
    print(f"\n{'length (px)':<14}{'real':>6}{'false':>7}"
          f"{'spread':>9}{'jitter':>9}{'ratio':>9}")
    print("-" * 54)
    for lo, hi in bands:
        r = [x for x in real if lo <= x["steps"] < hi]
        f_ = [x for x in fake if lo <= x["steps"] < hi]
        if len(r) < 10 or len(f_) < 10:
            print(f"{f'{lo}-{hi if hi < 10**9 else ''}':<14}"
                  f"{len(r):>6}{len(f_):>7}   too few to judge")
            continue
        cells = ""
        for key in ("spread", "jitter", "ratio"):
            a = auc(np.array([x[key] for x in r], float),
                    np.array([x[key] for x in f_], float))
            cells += f"{a:>8.2f}{'*' if abs(a - 0.5) >= 0.15 else ' '}"
        print(f"{f'{lo}-{hi if hi < 10**9 else ''}':<14}"
              f"{len(r):>6}{len(f_):>7}{cells}")
    print("\n  * marks a separation worth building a rule on.")
    print("  Below 0.50 means the value runs HIGHER on the false ones,")
    print("  which is the direction the observation predicts.")

    with open(os.path.join(HERE, "width_features.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"\nwrote width_features.json ({len(out)} rows)")


if __name__ == "__main__":
    main()
