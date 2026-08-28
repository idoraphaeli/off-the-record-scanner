# -*- coding: utf-8 -*-
"""
Look at the rotation estimate instead of arguing about it.

Two shots of one side, each with a line drawn from the centre along ITS OWN
anchor angle. If the anchor is doing its job, both lines land on the same
printed feature of the label — that is the whole claim, and it is decidable by
eye in a second.

The truth is available too, and costs nothing: the same physical scratches were
marked by hand in BOTH shots, so the rotation that maps one set of marks onto
the other is the real one. Each pair is scored against it, so "how close was the
alignment" is a number as well as a picture.

Usage:  python show_alignment.py [cal|val|both] [how many pictures]
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
from cross_shot import (rotation_from_label, label_profile, LABEL_LO, LABEL_HI,
                        BIN_DEG, RAD_TOL)

# The line is drawn at this angle on the first shot, and at this angle plus the
# measured rotation on the second. Any value works — what is being checked is
# whether the two land on the same printed feature.
REF_DEG = 40.0

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
OUT = os.path.join(os.path.dirname(HERE), "Model_Alignment")

VIEW_W = 900
ORANGE = (0, 150, 255)
BLUE = (255, 170, 60)
WHITE = (245, 245, 245)
BAR = 128


def gt_points(pair, center, radius, shape):
    """Hand-marked scratches of one photo, in disc coordinates."""
    path = os.path.join(GT, pair + ".png")
    if not os.path.exists(path):
        return []
    m = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    n, _, stats, cent = cv2.connectedComponentsWithStats(
        (m > 127).astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i][4] < 40:
            continue
        dx, dy = float(cent[i][0]) - center[0], float(cent[i][1]) - center[1]
        out.append({"rad": math.hypot(dx, dy) / max(radius, 1),
                    "ang": math.degrees(math.atan2(dy, dx)) % 360.0,
                    "x": float(cent[i][0]), "y": float(cent[i][1])})
    return out


def true_rotation(a, b):
    """The rotation the hand marks agree on — the answer to check against."""
    if len(a) < 3 or len(b) < 3:
        return None, 0
    nb = int(round(360 / BIN_DEG))
    hist = np.zeros(nb)
    for p in a:
        for q in b:
            if abs(p["rad"] - q["rad"]) <= RAD_TOL:
                hist[int(((q["ang"] - p["ang"]) % 360.0) / BIN_DEG) % nb] += 1
    if hist.sum() < 4:
        return None, 0
    sm = hist + np.roll(hist, 1) + np.roll(hist, -1)
    k = int(sm.argmax())
    if sm[k] < 4 or sm[k] < 2.5 * sm.mean():
        return None, int(sm[k])
    return (k * BIN_DEG) % 360.0, int(sm[k])


def draw(path, pair, ang, marks, center, radius):
    img = detector.load_image(path)
    k = VIEW_W / img.shape[1]
    vis = cv2.resize(img, (VIEW_W, int(round(img.shape[0] * k))),
                     interpolation=cv2.INTER_AREA)
    cx, cy = center[0] * k, center[1] * k
    r = radius * k

    for f in (LABEL_LO, LABEL_HI):
        cv2.circle(vis, (int(cx), int(cy)), int(r * f), (70, 70, 70), 1, cv2.LINE_AA)
    for m in marks:
        cv2.circle(vis, (int(m["x"] * k), int(m["y"] * k)), 16, (10, 10, 10), 5,
                   cv2.LINE_AA)
        cv2.circle(vis, (int(m["x"] * k), int(m["y"] * k)), 16, BLUE, 2, cv2.LINE_AA)

    if ang is not None:
        t = math.radians(ang)
        p1 = (int(cx + r * LABEL_LO * 0.2 * math.cos(t)),
              int(cy + r * LABEL_LO * 0.2 * math.sin(t)))
        p2 = (int(cx + r * 0.98 * math.cos(t)), int(cy + r * 0.98 * math.sin(t)))
        cv2.line(vis, p1, p2, (10, 10, 10), 7, cv2.LINE_AA)
        cv2.line(vis, p1, p2, ORANGE, 3, cv2.LINE_AA)
        tip = (int(cx + r * LABEL_HI * math.cos(t)),
               int(cy + r * LABEL_HI * math.sin(t)))
        cv2.circle(vis, tip, 11, (10, 10, 10), -1, cv2.LINE_AA)
        cv2.circle(vis, tip, 11, ORANGE, 3, cv2.LINE_AA)
    return vis


def caption(img, lines):
    strip = np.full((BAR, img.shape[1], 3), 22, np.uint8)
    y = 32
    for text, colour, sz in lines:
        cv2.putText(strip, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, sz, colour,
                    2, cv2.LINE_AA)
        y += int(34 * (sz / 0.62))
    return np.vstack([strip, img])


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    os.makedirs(OUT, exist_ok=True)

    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    sets = ("cal", "val") if which == "both" else (which,)
    records = set()
    for s in sets:
        records |= set(split[s])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in records]

    sides = collections.defaultdict(list)
    for r in rows:
        sides[(r["record"], r["side"])].append(r)

    results = []
    for key, shots in sorted(sides.items()):
        if len(shots) < 2:
            continue
        a, b = shots[0], shots[1]
        info = []
        ok = True
        for r in (a, b):
            path = os.path.join(PHOTOS, r["photo_file"])
            if not os.path.exists(path):
                ok = False
                break
            img = detector.load_image(path)
            center, radius = detector.find_disc(img)
            marks = gt_points(r["pair"], center, radius, img.shape[:2])
            info.append([r, path, center, radius, None, 0.0, marks,
                         label_profile(path)])
        if not ok or len(info) < 2:
            continue

        d_anchor, conf = rotation_from_label(info[0][7], info[1][7])
        # the reference ray on shot 1, and where the measured rotation says the
        # same physical spot has moved to on shot 2
        info[0][4] = REF_DEG
        info[1][4] = None if d_anchor is None else (REF_DEG + d_anchor) % 360.0
        info[0][5] = info[1][5] = conf
        d_true, votes = true_rotation(info[0][6], info[1][6])
        gap = None
        if d_anchor is not None and d_true is not None:
            gap = abs((d_anchor - d_true + 180.0) % 360.0 - 180.0)
        results.append((key, info, d_anchor, d_true, votes, gap))

    scored = [r for r in results if r[5] is not None]
    print(f"sides with two shots            : {len(results)}")
    print(f"sides where BOTH answers exist  : {len(scored)}\n")
    if scored:
        gaps = np.array([r[5] for r in scored])
        for t in (5, 10, 20, 45):
            print(f"  anchor within {t:>2} deg of the truth : "
                  f"{int((gaps <= t).sum())} of {len(gaps)}"
                  f"   ({100*(gaps<=t).mean():.0f}%)")
        print(f"  median error                     : {np.median(gaps):.0f} deg")
        print(f"  (a random guess would average 90 deg)")

    # picture the worst and best, so the failure mode is visible too
    scored.sort(key=lambda r: r[5])
    picks = scored[:want // 2] + scored[-(want - want // 2):] if scored else results[:want]
    print(f"\n{'record':<34}{'side':>5}{'anchor':>9}{'truth':>9}{'error':>8}")
    for key, info, d_anchor, d_true, votes, gap in picks:
        panels = []
        for (r, path, center, radius, ang, conf, marks, _prof) in info:
            v = draw(path, r["pair"], ang, marks, center, radius)
            panels.append(caption(v, [
                (f"shot {r['shot']}   marks {len(marks)}", WHITE, 0.62),
                ("the SAME spot on the disc, per the measured rotation"
                 if ang is not None else "no rotation found", ORANGE, 0.58),
            ]))
        h = min(p.shape[0] for p in panels)
        gapimg = np.full((h, 14, 3), 22, np.uint8)
        combo = np.hstack([panels[0][:h], gapimg, panels[1][:h]])
        combo = caption(combo, [
            (f"{key[0][:40]}  side {key[1]}", WHITE, 0.7),
            (f"label says {d_anchor:.0f} deg   |   your marks say {d_true:.0f} deg"
             f"   |   error {gap:.0f} deg", ORANGE, 0.66),
        ])
        name = f"align_{int(gap):03d}deg_{key[0][:24]}_{key[1]}.jpg"
        cv2.imencode(".jpg", combo, [int(cv2.IMWRITE_JPEG_QUALITY), 86])[1].tofile(
            os.path.join(OUT, name))
        print(f"{key[0][:32]:<34}{key[1]:>5}{d_anchor:>8.0f}"
              f"{d_true:>9.0f}{gap:>8.0f}")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
