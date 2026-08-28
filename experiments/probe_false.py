# -*- coding: utf-8 -*-
"""
What separates the genuinely-false detections from the real ones?

Every earlier attempt to cut false positives was a guess dressed as a rule,
because there was no ground truth for what "false" meant — dirt was counted as
an error, so any rule that removed dirt scored as an improvement. That is fixed:
1828 detections across both sets now carry a verdict from the person who owns
the grading standard.

Three comparisons are run, and the third is the one that decides anything:

    real vs false     everything the model got right against everything it did not
    dirt vs false     dominated by dirt, which is 63% of the real calls
    scratch vs false  the only comparison that matters for a RULE. A feature
                      that separates dirt from glare but not scratches from
                      glare cannot be filtered on — the rule would delete
                      exactly the damage the scanner exists to find.

Colour and sharpness are included because looking at the worst record showed
what the errors are: it was photographed in direct sun, and the failures are
broad soft reflection bands and coloured sky glare. The detector works on
greyscale, so saturation has never been tested, and a reflection is a smooth
gradient where a scratch is a sharp step.

AUC 0.5 means the feature is noise. Above 0.5 means the value runs HIGHER on the
first group, below means higher on the second.

Usage:  python probe_false.py [cal|val|both]
"""

import collections
import csv
import json
import os
import sys

import cv2
import numpy as np

import detector
from detector import P
from evaluate_frozen import MIN_EXTRA_AREA, detect

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")

VIEW_W = 1100
MATCH_PX = 40          # full-res px within which a label belongs to a blob
WIN = 45               # half-size of the neighbourhood used for context features

FEATURES = ("length", "thick", "elong", "area", "rad",
            "bright", "contrast", "sat", "sat_win", "sharp", "band", "tram",
            "chroma", "chroma_rel", "hue_shift")


def chroma_map(img):
    """Colourfulness per pixel, in Lab rather than HSV.

    HSV saturation is the wrong instrument here. It is a ratio, so on near-black
    vinyl it is dominated by sensor noise — the darker the pixel, the wilder the
    number. Lab separates lightness from colour outright, and the distance from
    the neutral axis, sqrt(a^2 + b^2), is a stable measure of how far a pixel is
    from grey at any brightness. Grey, white and black all sit near 0; a warm
    lamp or a patch of sky does not.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0
    return np.sqrt(a * a + b * b), a, b


def detect_channels(path):
    """The two channels kept apart instead of merged.

    The tramline channel exists for scratches running ALONG the grooves, and it
    pays for that by accepting anything groove-parallel and long. A reflection
    band on a tilted disc is exactly that, so the suspicion is that this channel
    produces a large share of the false detections while contributing little.
    Merging the masks, as the detector does, hides which one fired.
    """
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner, outer = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer]
    radial, tram = detector.scratch_map(ring)[:2]
    m1, _ = detector.extract(radial)
    m2, _ = detector.extract(tram, min_len=P["TRAM_MIN_LEN"])
    rw = lambda m: detector.rewrap(m, inner, center, radius, gray.shape)
    return rw(m1), rw(m2), rw(cv2.bitwise_or(m1, m2)), img, center, radius


def auc(pos, neg):
    """Probability a random member of `pos` scores above a random member of
    `neg`. Rank-based, so the scale and skew of the feature do not matter."""
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
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _hue_shift(ch_a, ch_b, sel, y0, y1, x0, x1):
    """Angle between the detection's mean colour direction and its background's.

    Averaged as vectors, not as angles: hue wraps at 360, so a plain mean of
    angles is meaningless near the wrap point.
    """
    da, db = float(ch_a[sel].mean()), float(ch_b[sel].mean())
    if y1 <= y0 or x1 <= x0:
        return 0.0
    wa = float(ch_a[y0:y1, x0:x1].mean())
    wb = float(ch_b[y0:y1, x0:x1].mean())
    n1, n2 = np.hypot(da, db), np.hypot(wa, wb)
    if n1 < 1e-3 or n2 < 1e-3:
        return 0.0
    cos = np.clip((da * wa + db * wb) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def photo_features(path):
    """Every detection on one photo, with the numbers that might tell the two
    groups apart, keyed by full-resolution centroid."""
    m_radial, m_tram, det, img, center, radius = detect_channels(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    # |Laplacian| is large on a step and small on a gradient, which is exactly
    # the difference between a scratch edge and a reflection falling off
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    chroma, ch_a, ch_b = chroma_map(img)
    tram_b = m_tram > 127
    H, W = gray.shape

    n, labels, stats, cent = cv2.connectedComponentsWithStats(
        (det > 127).astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_EXTRA_AREA:
            continue
        comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        per = max((cv2.arcLength(c, True) for c in contours), default=0)
        length = max(per / 2, max(w, h))
        thick = area / max(length, 1)
        sel = labels == i
        cx, cy = float(cent[i][0]), float(cent[i][1])

        y0, y1 = max(int(cy) - WIN, 0), min(int(cy) + WIN, H)
        x0, x1 = max(int(cx) - WIN, 0), min(int(cx) + WIN, W)
        win_g = gray[y0:y1, x0:x1]
        win_s = sat[y0:y1, x0:x1]

        out.append({
            "cx": cx, "cy": cy,
            "length": float(length),
            "thick": float(thick),
            "elong": float(length / max(thick, 1)),
            "area": float(area),
            "rad": float(np.hypot(cx - center[0], cy - center[1])) / max(radius, 1),
            "bright": float(win_g.mean()) if win_g.size else 0.0,
            "contrast": float(win_g.std()) if win_g.size else 0.0,
            "sat": float(sat[sel].mean()),
            "sat_win": float(win_s.mean()) if win_s.size else 0.0,
            "sharp": float(lap[sel].mean()),
            # how much brighter the detection is than its surroundings: a broad
            # reflection barely stands out, a scratch does
            "band": float(gray[sel].mean() - (win_g.mean() if win_g.size else 0)),
            # 1 when the groove-parallel channel is what reported this
            "tram": float(np.count_nonzero(tram_b[sel]) > 0.5 * area),
            "chroma": float(chroma[sel].mean()),
            # the measure that matters: colourfulness ABOVE the surrounding
            # vinyl. A disc lit by a blue lamp is blue everywhere, so absolute
            # colour says nothing — only the excess does.
            "chroma_rel": float(chroma[sel].mean() -
                                (chroma[y0:y1, x0:x1].mean()
                                 if y1 > y0 and x1 > x0 else 0.0)),
            # how far the detection's colour DIRECTION swings away from its
            # surroundings, in degrees around the neutral axis: a reflection of
            # a coloured light points somewhere the vinyl does not
            "hue_shift": _hue_shift(ch_a, ch_b, sel, y0, y1, x0, x1),
        })
    return out, VIEW_W / float(W)


def collect(which):
    tool = os.path.join(HERE, f"label_tool_{which}")
    labels_path = os.path.join(tool, f"labels_{which}.json")
    if not os.path.exists(labels_path):
        return []
    rows = [r for r in json.load(open(labels_path, encoding="utf-8"))["rows"]
            if r["label"] in ("scratch", "dirt", "false")]
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        source = {r["pair"]: r["photo_file"] for r in csv.DictReader(fh)}

    by_pair = collections.defaultdict(list)
    for r in rows:
        by_pair[r["pair"]].append(r)

    out = []
    for pair, group in sorted(by_pair.items()):
        name = source.get(pair)
        if not name:
            continue
        path = os.path.join(PHOTOS, name)
        if not os.path.exists(path):
            continue
        feats, scale = photo_features(path)
        if not feats:
            continue
        arr = np.array([[f["cx"], f["cy"]] for f in feats], float)
        for r in group:
            tx, ty = r["cx"] / scale, r["cy"] / scale
            d = np.hypot(arr[:, 0] - tx, arr[:, 1] - ty)
            j = int(d.argmin())
            if d[j] > MATCH_PX:
                continue
            rec = dict(feats[j])
            rec["kind"] = r["label"]
            rec["record"] = r["record"]
            out.append(rec)
        print(f"  {pair[:46]:<48}{len(group):>4}")
    return out


def report(pos, neg, pos_name, neg_name):
    if len(pos) < 15 or len(neg) < 15:
        print(f"\n[{pos_name} vs {neg_name}] too few ({len(pos)} vs {len(neg)})")
        return
    print(f"\n[{pos_name} vs {neg_name}]   {len(pos)} vs {len(neg)}")
    print(f"{'feature':<10}{pos_name[:8]+' med':>13}{neg_name[:8]+' med':>13}"
          f"{'AUC':>7}   verdict")
    ranked = []
    for k in FEATURES:
        p = np.array([r[k] for r in pos], float)
        q = np.array([r[k] for r in neg], float)
        a = auc(p, q)
        sep = abs(a - 0.5)
        verdict = ("USEFUL" if sep >= 0.15 else
                   "weak" if sep >= 0.08 else "nothing")
        ranked.append((sep, k, np.median(p), np.median(q), a, verdict))
    for sep, k, mp, mq, a, verdict in sorted(ranked, reverse=True):
        higher = "higher on " + (pos_name if a > 0.5 else neg_name)
        print(f"{k:<10}{mp:>13.2f}{mq:>13.2f}{a:>7.2f}   {verdict:<8}"
              f"{'  ' + higher if verdict != 'nothing' else ''}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    sets = ("cal", "val") if which == "both" else (which,)
    rows = []
    for s in sets:
        rows += collect(s)

    if not rows:
        sys.exit("nothing matched")
    fake = [r for r in rows if r["kind"] == "false"]
    real = [r for r in rows if r["kind"] in ("dirt", "scratch")]
    dirt = [r for r in rows if r["kind"] == "dirt"]
    scratch = [r for r in rows if r["kind"] == "scratch"]

    print(f"\nmatched {len(rows)} detections   "
          f"real {len(real)} (dirt {len(dirt)}, scratch {len(scratch)})   "
          f"false {len(fake)}")

    report(real, fake, "real", "false")
    report(dirt, fake, "dirt", "false")
    report(scratch, fake, "scratch", "false")

    with open(os.path.join(HERE, "false_features.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    print(f"\nwrote false_features.json ({len(rows)} rows)")


if __name__ == "__main__":
    main()
