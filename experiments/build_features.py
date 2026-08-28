# -*- coding: utf-8 -*-
"""
One table holding every property measured today, for every hand-labelled
detection.

Each feature has so far been tested on its own, and each was weak. Weak
features can still combine — "long AND bright AND unconfirmed" may separate
where none of the three does alone — but that can only be tried once they sit in
the same table. This builds it.

Everything is measured in the unwrapped ring, where the detector actually works;
mapping shapes back to the photograph stretches them and changes what is being
measured.

Outputs features.json — one row per labelled detection.

Usage:  python build_features.py [cal|val|both]
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
import cross_shot as cs
from probe_width import width_profile

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
VIEW_W = 1100
WIN = 45
RAD_TOL, ANG_IN, ANG_OUT = 0.025, 12.0, 2.0
INNER, OUTER = 0.36, 0.95


def ang_tol(r):
    t = min(max((r - INNER) / (OUTER - INNER), 0.0), 1.0)
    return ANG_IN + t * (ANG_OUT - ANG_IN)


def photo_features(path):
    """Every detection on one photo, with everything measurable about it."""
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.sqrt((lab[:, :, 1] - 128) ** 2 + (lab[:, :, 2] - 128) ** 2)
    center, radius = detector.find_disc(img)
    inner, outer = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer]

    # the pipeline's own intermediates, so "how strong was it really" is
    # available alongside "how strong did the threshold think it was"
    flat = ring.astype(np.float32)
    flat -= cv2.blur(flat, (P["ROW_FLATTEN"], 1))
    flat = cv2.GaussianBlur(flat, (3, 3), 0)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (P["TOPHAT_W"], 1))
    ridge = cv2.morphologyEx(flat, cv2.MORPH_TOPHAT, se)
    boosted = np.zeros_like(ridge)
    for k in detector._line_kernels(P["BOOST_LEN"], P["ANGLES"]):
        boosted = np.maximum(boosted, cv2.filter2D(ridge, -1, k))
    med = cv2.medianBlur(np.clip(boosted, 0, 255).astype(np.uint8), 51).astype(np.float32)
    raw_map = boosted - med

    radial, tram = detector.scratch_map(ring)[:2]
    H, W = gray.shape
    k_view = VIEW_W / W
    out = []

    for smap, min_len, is_tram in ((radial, None, 0.0),
                                   (tram, P["TRAM_MIN_LEN"], 1.0)):
        mask, _ = detector.extract(smap, min_len)
        n, lb, stats, cent = cv2.connectedComponentsWithStats(
            (mask > 127).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < MIN_EXTRA_AREA:
                continue
            comp = (lb[y:y + h, x:x + w] == i).astype(np.uint8)
            prof = width_profile(comp)
            if prof is None:
                continue
            sel = lb == i
            rr = float(cent[i][1]) + inner
            ang = float(cent[i][0]) / P["POLAR_STEPS"] * 2 * math.pi
            px = center[0] + rr * math.cos(ang)
            py = center[1] + rr * math.sin(ang)

            y0, y1 = max(int(py) - WIN, 0), min(int(py) + WIN, H)
            x0, x1 = max(int(px) - WIN, 0), min(int(px) + WIN, W)
            win_g = gray[y0:y1, x0:x1]
            win_c = chroma[y0:y1, x0:x1]

            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            per = max((cv2.arcLength(c, True) for c in contours), default=0)
            length = max(per / 2, max(w, h))
            thick = area / max(length, 1)

            out.append({
                "length": float(length), "thick": float(thick),
                "elong": float(length / max(thick, 1)), "area": float(area),
                "steps": prof["steps"], "wmean": prof["wmean"],
                "spread": prof["spread"], "ratio": prof["ratio"],
                "jitter": prof["jitter"],
                "rad": float(rr) / max(radius, 1),
                "raw": float(raw_map[sel].max()),
                "score": float(smap[sel].max()),
                "bright": float(win_g.mean()) if win_g.size else 0.0,
                "contrast": float(win_g.std()) if win_g.size else 0.0,
                "chroma": float(win_c.mean()) if win_c.size else 0.0,
                "band": float(gray[int(py) % H, int(px) % W]
                              - (win_g.mean() if win_g.size else 0)),
                "tram": is_tram,
                "angle": float(detector._axis_angle_deg(comp)),
                # kept for the cross-shot step and for joining to the labels
                "_rad": float(rr) / max(radius, 1),
                "_ang": math.degrees(ang) % 360.0,
                "vx": px * k_view, "vy": py * k_view,
            })
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    sets = ("cal", "val") if which == "both" else (which,)

    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    which_of = {}
    for s in sets:
        for rec in split[s]:
            which_of[rec] = s
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in which_of]

    labels = collections.defaultdict(list)
    for s in sets:
        p = os.path.join(HERE, f"label_tool_{s}", f"labels_{s}.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p, encoding="utf-8"))["rows"]:
            if r["label"] in ("scratch", "dirt", "false"):
                labels[r["pair"]].append(r)

    per_photo, profiles = {}, {}
    for r in rows:
        path = os.path.join(PHOTOS, r["photo_file"])
        if not os.path.exists(path):
            continue
        try:
            per_photo[r["pair"]] = photo_features(path)
            profiles[r["pair"]] = cs.label_profile(path)
        except Exception:
            continue
        print(f"  {r['pair'][:46]:<48}{len(per_photo[r['pair']]):>4}")

    # cross-shot confirmation: does the same spot on the disc show up in the
    # other photograph of this side, once the rotation is taken out
    side_of = collections.defaultdict(list)
    for r in rows:
        if r["pair"] in per_photo:
            side_of[(r["record"], r["side"])].append(r["pair"])
    confirmed = {}
    for key, pairs in side_of.items():
        for a in pairs:
            flags = [False] * len(per_photo[a])
            for b in pairs:
                if a == b:
                    continue
                d, _ = cs.rotation_from_label(profiles.get(a), profiles.get(b))
                if d is None:
                    continue
                for i, p in enumerate(per_photo[a]):
                    if flags[i]:
                        continue
                    want = (p["_ang"] + d) % 360.0
                    for q in per_photo[b]:
                        if abs(p["_rad"] - q["_rad"]) > RAD_TOL:
                            continue
                        if abs((q["_ang"] - want + 180.0) % 360.0 - 180.0) \
                                <= ang_tol(p["_rad"]):
                            flags[i] = True
                            break
            confirmed[a] = flags

    out = []
    for pair, rows_l in labels.items():
        feats = per_photo.get(pair)
        if not feats:
            continue
        flags = confirmed.get(pair, [False] * len(feats))
        arr = np.array([[f["vx"], f["vy"]] for f in feats], float)
        rec_name = rows_l[0]["record"]
        for lr in rows_l:
            d = np.hypot(arr[:, 0] - lr["cx"], arr[:, 1] - lr["cy"])
            j = int(d.argmin())
            if d[j] > 30:
                continue
            row = {k: v for k, v in feats[j].items() if not k.startswith("_")}
            row.pop("vx", None)
            row.pop("vy", None)
            row["confirmed"] = 1.0 if flags[j] else 0.0
            row["kind"] = lr["label"]
            row["record"] = rec_name
            row["set"] = which_of.get(rec_name, "?")
            out.append(row)

    with open(os.path.join(HERE, "features.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    c = collections.Counter((r["set"], r["kind"]) for r in out)
    print(f"\n{len(out)} labelled detections with full features")
    for s in sets:
        print(f"  {s}: scratch {c[(s,'scratch')]}   dirt {c[(s,'dirt')]}"
              f"   false {c[(s,'false')]}")
    print(f"\nwrote features.json")


if __name__ == "__main__":
    main()
