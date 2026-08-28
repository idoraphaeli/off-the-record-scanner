# -*- coding: utf-8 -*-
"""
WHY did the reflections fire? Not what they look like — what in the pipeline let
them through.

Everything measured so far described the false detections from outside: their
size, colour, position. This opens the detector instead and records, for each
labelled detection, the two numbers that decide its fate:

    raw     the ridge response before normalisation — how bright the mark
            actually is against its groove row
    noise   the local noise scale it is divided by
    score   raw / noise, which is what the threshold sees

The suspicion is that reflections are not strong, they are QUIET. The last step
divides by the local noise, and the divisor cannot go below NOISE_FLOOR. A
reflection is a smooth, featureless patch, so its neighbourhood has almost no
noise, the divisor bottoms out on that floor, and a weak ripple inside it is
multiplied into a confident-looking score. If that is what is happening, false
detections will show LOW raw and a divisor sitting ON the floor, while real
damage shows high raw — and the fix is a parameter, not a new feature.

Usage:  python why_false_fires.py [cal|val|both]
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


def internals(path):
    """Re-run scratch_map's radial channel, keeping the intermediates."""
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner, outer = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer]

    flat = ring.astype(np.float32)
    flat -= cv2.blur(flat, (P["ROW_FLATTEN"], 1))
    flat = cv2.GaussianBlur(flat, (3, 3), 0)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (P["TOPHAT_W"], 1))
    ridge = cv2.morphologyEx(flat, cv2.MORPH_TOPHAT, se)
    chan = np.zeros_like(ridge)
    for k in detector._line_kernels(P["BOOST_LEN"], P["ANGLES"]):
        chan = np.maximum(chan, cv2.filter2D(ridge, -1, k))

    # the same normalisation the detector applies, with its parts kept
    win = (P["NOISE_WIN"], P["NOISE_WIN"])
    med = cv2.medianBlur(np.clip(chan, 0, 255).astype(np.uint8), 51).astype(np.float32)
    dev = np.abs(chan - med)
    mad = cv2.blur(dev, win)
    mad_robust = cv2.blur(np.minimum(dev, 2.0 * mad), win)
    noise = np.maximum(1.4826 * mad_robust, P["NOISE_FLOOR"])
    raw = chan - med
    score = raw / noise

    smap, _ = detector.extract(np.clip(score * 10, 0, 255).astype(np.uint8))
    return dict(ring=ring, raw=raw, noise=noise, score=score, mask=smap,
                inner=inner, radius=radius, center=center,
                shape=gray.shape, w=gray.shape[1])


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    sets = ("cal", "val") if which == "both" else (which,)

    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    records = set()
    for s in sets:
        records |= set(split[s])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in records]

    labels = {}
    for s in sets:
        tool = os.path.join(HERE, f"label_tool_{s}")
        p = os.path.join(tool, f"labels_{s}.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p, encoding="utf-8"))["rows"]:
            if r["label"] in ("scratch", "dirt", "false"):
                labels.setdefault(r["pair"], []).append(r)

    out = []
    for r in rows:
        rows_l = labels.get(r["pair"])
        path = os.path.join(PHOTOS, r["photo_file"])
        if not rows_l or not os.path.exists(path):
            continue
        try:
            d = internals(path)
        except Exception:
            continue

        n, lab, stats, cent = cv2.connectedComponentsWithStats(
            (d["mask"] > 127).astype(np.uint8), connectivity=8)
        cands = []
        for i in range(1, n):
            if stats[i][4] < MIN_EXTRA_AREA:
                continue
            sel = lab == i
            # judge each blob at its strongest pixel, which is what carried it
            # over the threshold in the first place
            j = int(np.argmax(np.where(sel, d["score"], -1e9)))
            rr, cc = np.unravel_index(j, d["score"].shape)
            ang = cc / P["POLAR_STEPS"] * 2 * math.pi
            rad = rr + d["inner"]
            x = d["center"][0] + rad * math.cos(ang)
            y = d["center"][1] + rad * math.sin(ang)
            k = VIEW_W / d["w"]
            cands.append({
                "vx": x * k, "vy": y * k,
                "raw": float(d["raw"][rr, cc]),
                "noise": float(d["noise"][rr, cc]),
                "score": float(d["score"][rr, cc] * 10),
                "on_floor": float(d["noise"][rr, cc] <= P["NOISE_FLOOR"] + 1e-6),
                "bright": float(d["ring"][rr, cc]),
            })
        if not cands:
            continue
        arr = np.array([[c["vx"], c["vy"]] for c in cands], float)
        for lr in rows_l:
            dd = np.hypot(arr[:, 0] - lr["cx"], arr[:, 1] - lr["cy"])
            k = int(dd.argmin())
            if dd[k] > 30:
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
          f"   (scratches {len(scr)})")

    def table(pos, name):
        if len(pos) < 12:
            return
        print(f"\n[{name} vs false]   {len(pos)} vs {len(fake)}")
        print(f"{'':<10}{name[:8]+' med':>13}{'false med':>12}{'AUC':>7}")
        for f in ("raw", "noise", "score", "bright", "on_floor"):
            p = np.array([r[f] for r in pos], float)
            q = np.array([r[f] for r in fake], float)
            print(f"{f:<10}{np.median(p):>13.1f}{np.median(q):>12.1f}"
                  f"{auc(p, q):>7.2f}")

    table(real, "real")
    table(scr, "scratch")

    for name, grp in (("real", real), ("scratch", scr), ("false", fake)):
        if grp:
            share = 100.0 * np.mean([r["on_floor"] for r in grp])
            print(f"\n  {name:<8}sitting ON the noise floor: {share:.0f}%")

    with open(os.path.join(HERE, "fire_features.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"\nwrote fire_features.json ({len(out)} rows)")


if __name__ == "__main__":
    main()
