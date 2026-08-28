# -*- coding: utf-8 -*-
"""
Is the groove texture still there underneath the mark?

This is the one property that follows from what the two things ARE, rather than
from how they look. A scratch is a cut: where it crosses, the grooves are gone.
A reflection is light lying on an undamaged surface: the grooves underneath it
are untouched, merely lit.

In the unwrapped ring the grooves run horizontally, so their texture shows up as
ripple along the VERTICAL direction. For every mark, that ripple is measured
inside it and in the vinyl immediately either side of it, over the same rows:

    ripple_ratio   how much of the texture survives inside the mark
                   near 1 means the grooves are intact -> light on a good surface
                   well under 1 means they are flattened -> something cut them
    texture_corr   whether the pattern inside still matches the pattern beside
                   it, groove for groove

A caution worth stating: an LP holds several hundred grooves per side and the
detector works at 1600px, so individual grooves are close to the resolution
limit. What is being measured is the texture in aggregate, not single grooves,
and marks spanning too few rows are reported as unmeasurable rather than guessed
at.

Usage:  python probe_texture.py [cal|val|both]
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
MIN_ROWS = 9          # fewer rows than this and there is no ripple to measure
FEATURES = ("ripple_ratio", "texture_corr", "ripple_in", "ripple_out")


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


def ripple(profile):
    """The groove component of a vertical brightness profile.

    Subtracting a smoothed copy removes the slow shading — a mark is brighter
    overall, and that brightness is not texture and must not be counted as it.
    """
    if len(profile) < MIN_ROWS:
        return None, None
    p = profile.astype(np.float32)
    smooth = cv2.GaussianBlur(p.reshape(-1, 1), (1, 7), 0).ravel()
    hp = p - smooth
    return float(hp.std()), hp


def texture_of(ring, x, y, w, h):
    """Groove ripple inside the mark against the vinyl either side of it."""
    H, W = ring.shape
    if h < MIN_ROWS:
        return None
    inside = ring[y:y + h, x:x + w].mean(axis=1)
    pad = max(w, 4)
    lx0, lx1 = max(x - 3 * pad, 0), max(x - pad, 0)
    rx0, rx1 = min(x + w + pad, W), min(x + w + 3 * pad, W)
    flanks = []
    if lx1 - lx0 >= 2:
        flanks.append(ring[y:y + h, lx0:lx1].mean(axis=1))
    if rx1 - rx0 >= 2:
        flanks.append(ring[y:y + h, rx0:rx1].mean(axis=1))
    if not flanks:
        return None

    r_in, hp_in = ripple(inside)
    outs = [ripple(f) for f in flanks]
    outs = [(a, b) for a, b in outs if a is not None]
    if r_in is None or not outs:
        return None
    r_out = float(np.mean([a for a, _ in outs]))
    hp_out = np.mean([b for _, b in outs], axis=0)

    denom = (np.std(hp_in) * np.std(hp_out))
    corr = float(np.mean((hp_in - hp_in.mean()) * (hp_out - hp_out.mean())) / denom) \
        if denom > 1e-6 else 0.0
    return {"ripple_in": r_in, "ripple_out": r_out,
            "ripple_ratio": r_in / max(r_out, 1e-3),
            "texture_corr": corr}


def photo_marks(path):
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner, outer = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer].astype(np.float32)
    radial, tram = detector.scratch_map(ring.astype(np.uint8))[:2]

    k = VIEW_W / gray.shape[1]
    out = []
    for smap, min_len in ((radial, None), (tram, P["TRAM_MIN_LEN"])):
        mask, _ = detector.extract(smap, min_len)
        n, lb, stats, cent = cv2.connectedComponentsWithStats(
            (mask > 127).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < MIN_EXTRA_AREA:
                continue
            t = texture_of(ring, x, y, w, h)
            if t is None:
                continue
            ang = float(cent[i][0]) / P["POLAR_STEPS"] * 2 * math.pi
            rr = float(cent[i][1]) + inner
            t["vx"] = (center[0] + rr * math.cos(ang)) * k
            t["vy"] = (center[1] + rr * math.sin(ang)) * k
            out.append(t)
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

    out, skipped = [], 0
    for r in rows:
        rows_l = labels.get(r["pair"])
        path = os.path.join(PHOTOS, r["photo_file"])
        if not rows_l or not os.path.exists(path):
            continue
        try:
            cands = photo_marks(path)
        except Exception:
            continue
        if not cands:
            continue
        arr = np.array([[c["vx"], c["vy"]] for c in cands], float)
        for lr in rows_l:
            d = np.hypot(arr[:, 0] - lr["cx"], arr[:, 1] - lr["cy"])
            k = int(d.argmin())
            if d[k] > 30:
                skipped += 1
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
    print(f"\nmeasurable {len(out)}   real {len(real)}   false {len(fake)}"
          f"   scratches {len(scr)}")
    print(f"  ({skipped} too short to have a texture worth measuring)")

    def table(pos, name):
        if len(pos) < 10:
            print(f"\n[{name}] only {len(pos)} — too few")
            return
        print(f"\n[{name} vs reflections]   {len(pos)} vs {len(fake)}")
        print(f"{'':<15}{name[:8]+' med':>13}{'refl med':>11}{'AUC':>7}   verdict")
        for f in FEATURES:
            p = np.array([r[f] for r in pos], float)
            q = np.array([r[f] for r in fake], float)
            a = auc(p, q)
            sep = abs(a - 0.5)
            v = "USEFUL" if sep >= 0.15 else ("weak" if sep >= 0.08 else "nothing")
            print(f"{f:<15}{np.median(p):>13.2f}{np.median(q):>11.2f}{a:>7.2f}   {v}")

    table(real, "real")
    table(scr, "scratch")

    with open(os.path.join(HERE, "texture_features.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"\nwrote texture_features.json ({len(out)} rows)")


if __name__ == "__main__":
    main()
