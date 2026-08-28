# -*- coding: utf-8 -*-
"""Fix the outer-ring brightness rule in the space it will actually run in.

The rule came out of features.json, where "bright" is the mean of a window in the
PHOTOGRAPH around each mark. The detector has no photograph in hand at that point
— it works on the unwrapped ring — so a threshold measured in one space cannot be
copied into the other and assumed to mean the same thing. This measures it in the
ring, which is where the code will read it.

Everything here is chosen on the CALIBRATION records. Validation is scored once,
at the end, and takes no part in the choice.

Usage:  python tune_glare_rule.py
"""

import collections
import csv
import json
import os

import cv2
import numpy as np

import detector
from detector import P
from evaluate_frozen import MIN_EXTRA_AREA

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
VIEW_W = 1100
WIN = 45                 # half-width of the patch measured around a mark
ON_MARK_SLACK = 9

RADII = (0.55, 0.60, 0.65, 0.70, 0.75)
BRIGHTS = (55, 60, 65, 70, 75, 80, 85, 90, 95)
MIN_RECALL_KEPT = 0.90   # keep at least this share of the scratches we find today


def per_photo(path, gt_path):
    """Every detection with the two things the rule needs, plus what it is."""
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner = int(P["LABEL_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:int(P["OUTER_R"] * radius)]
    rad_map, tram = detector.scratch_map(ring)[:2]

    gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    ink = cv2.dilate((gt > 127).astype(np.uint8),
                     np.ones((ON_MARK_SLACK, ON_MARK_SLACK), np.uint8))

    H, W = ring.shape
    out = []
    for smap, min_len in ((rad_map, None), (tram, P["TRAM_MIN_LEN"])):
        mask, _ = detector.extract(smap, min_len)
        n, lb, stats, cent = cv2.connectedComponentsWithStats(
            (mask > 127).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < MIN_EXTRA_AREA:
                continue
            # the patch this mark sits in, measured in the ring
            y0, y1 = max(y - WIN, 0), min(y + h + WIN, H)
            x0, x1 = max(x - WIN, 0), min(x + w + WIN, W)
            patch = ring[y0:y1, x0:x1]
            rr = float(cent[i][1]) + inner
            ang = float(cent[i][0]) / P["POLAR_STEPS"] * 2 * np.pi
            px = int(center[0] + rr * np.cos(ang))
            py = int(center[1] + rr * np.sin(ang))
            out.append({
                "rad": rr / max(radius, 1),
                "ring_bright": float(patch.mean()) if patch.size else 0.0,
                "on_mark": bool(ink[min(max(py, 0), gt.shape[0] - 1),
                                    min(max(px, 0), gt.shape[1] - 1)]),
                "vx": px * VIEW_W / gray.shape[1],
                "vy": py * VIEW_W / gray.shape[1],
            })
    return out


def load_labels(which):
    p = os.path.join(HERE, f"label_tool_{which}", f"labels_{which}.json")
    out = collections.defaultdict(list)
    if os.path.exists(p):
        for r in json.load(open(p, encoding="utf-8"))["rows"]:
            if r.get("label") in ("scratch", "dirt", "false"):
                out[r["pair"]].append(r)
    return out


def gather(which):
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    records = set(split[which])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in records]
    labels = load_labels(which)

    got = []
    for r in rows:
        photo = os.path.join(PHOTOS, r["photo_file"])
        gtp = os.path.join(GT, r["pair"] + ".png")
        if not (os.path.exists(photo) and os.path.exists(gtp)):
            continue
        try:
            cands = per_photo(photo, gtp)
        except Exception:
            continue
        if not cands:
            continue
        arr = np.array([[c["vx"], c["vy"]] for c in cands], float)
        taken = set()
        for lr in labels.get(r["pair"], []):
            d = np.hypot(arr[:, 0] - lr["cx"], arr[:, 1] - lr["cy"])
            j = int(d.argmin())
            if d[j] > 30 or j in taken:
                continue
            taken.add(j)
            rec = dict(cands[j]); rec["kind"] = lr["label"]; got.append(rec)
        for j, c in enumerate(cands):
            if j in taken or not c["on_mark"]:
                continue
            rec = dict(c); rec["kind"] = "scratch"; got.append(rec)
    return got


def score(rows, rad_cut, bright_cut):
    keep = [r for r in rows if r["rad"] < rad_cut or r["ring_bright"] <= bright_cut]
    k = collections.Counter(r["kind"] for r in keep)
    a = collections.Counter(r["kind"] for r in rows)
    good, bad = k["scratch"] + k["dirt"], k["false"]
    if good + bad == 0 or a["scratch"] == 0:
        return None
    return (100.0 * good / (good + bad), k["scratch"] / a["scratch"],
            a["false"] - k["false"], a["scratch"] - k["scratch"])


def main():
    cal = gather("cal")
    print(f"calibration: {len(cal)} detections  "
          f"({collections.Counter(r['kind'] for r in cal)})\n")

    base = score(cal, 9, 999)
    print(f"today on calibration: precision {base[0]:.1f}%\n")
    print(f"{'inner passes below':>19}{'bright cut':>12}{'precision':>11}"
          f"{'scratches kept':>16}{'false removed':>15}")
    print("-" * 74)
    best = None
    for rc in RADII:
        for bc in BRIGHTS:
            s = score(cal, rc, bc)
            if not s or s[1] < MIN_RECALL_KEPT:
                continue
            print(f"{rc:>19.2f}{bc:>12}{s[0]:>10.1f}%{100*s[1]:>15.0f}%{s[2]:>15}")
            if best is None or s[0] > best[0][0]:
                best = (s, rc, bc)

    if not best:
        print("\nnothing keeps enough of the scratches")
        return
    s, rc, bc = best
    print(f"\nchosen on calibration: inner {rc:.2f} passes, "
          f"outer needs a patch at or below {bc}")
    print(f"  precision {base[0]:.1f}% -> {s[0]:.1f}%   "
          f"scratches kept {100*s[1]:.0f}%   false removed {s[2]}")

    val = gather("val")
    vb = score(val, 9, 999)
    vs = score(val, rc, bc)
    print(f"\nscored once on VALIDATION, which took no part in the choice:")
    print(f"  precision {vb[0]:.1f}% -> {vs[0]:.1f}%   "
          f"scratches kept {100*vs[1]:.0f}%   false removed {vs[2]}")

    json.dump({"rad": rc, "bright": bc}, open(os.path.join(HERE, "glare_rule.json"), "w"))


if __name__ == "__main__":
    main()
