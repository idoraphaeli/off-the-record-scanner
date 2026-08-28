# -*- coding: utf-8 -*-
"""
Re-test width steadiness at the photograph's own resolution.

Ido's observation — a scratch holds one width, a reflection wobbles — separated
dirt from reflections but not scratches, and the likely reason was resolution.
The detector works at 1600px, where these marks are three to five pixels across:
a single pixel of change reads as a quarter of the width, so genuine wobble and
measurement noise are indistinguishable.

Raising MAX_DIM is not the way to check that. Every pixel-denominated parameter
in the detector — filter widths, minimum lengths, noise windows — is tuned for
1600, and doubling the image silently invalidates all of them. So detection is
left exactly as it is, and only the MEASUREMENT moves to full resolution.

The measurement is also better than before. Instead of counting mask pixels,
each step along the mark takes a slice across it and measures the width of the
bright band at half its height — the standard way to size a ridge, and one that
gives fractions of a pixel rather than whole ones.

Usage:  python probe_width_hires.py [cal|val|both]
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
HALF = 14          # how far either side of the mark to sample, in fine pixels
MIN_STEPS = 6
FEATURES = ("spread", "jitter", "ratio", "wmean", "steps")


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


def full_res(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(path)
    return img


def fwhm(slice_vals):
    """Width of the bright band at half its height.

    The ends of the slice are vinyl, so their level is the baseline; the width
    is where the profile sits above halfway between that and the peak. Crossings
    are interpolated, which is what buys the fractions of a pixel that counting
    mask pixels cannot give.
    """
    v = slice_vals.astype(np.float32)
    if len(v) < 7:
        return None
    edge = max(2, len(v) // 6)
    base = float(np.median(np.concatenate([v[:edge], v[-edge:]])))
    mid = len(v) // 2
    peak_i = int(np.argmax(v[mid - 3:mid + 4])) + mid - 3
    peak = float(v[peak_i])
    if peak - base < 2.0:
        return None
    half = base + 0.5 * (peak - base)

    def cross(idx, step):
        i = idx
        while 0 < i < len(v) - 1 and v[i] > half:
            i += step
        if v[i] > half:
            return float(i)
        a, b = v[i], v[i - step]
        if abs(b - a) < 1e-6:
            return float(i)
        return i + step * (half - a) / (b - a) * -1.0

    left = cross(peak_i, -1)
    right = cross(peak_i, +1)
    w = abs(right - left)
    return w if 0.5 <= w <= len(v) else None


def sample_line(ring, r0, c0, dr, dc, n):
    """Bilinear samples along a straight line in the unwrapped ring."""
    H, W = ring.shape
    rows = r0 + dr * np.arange(-n, n + 1, dtype=np.float32)
    cols = c0 + dc * np.arange(-n, n + 1, dtype=np.float32)
    rows = np.clip(rows, 0, H - 1)
    cols = np.clip(cols, 0, W - 1)
    return cv2.remap(ring, cols.reshape(1, -1), rows.reshape(1, -1),
                     cv2.INTER_LINEAR).ravel()


def photo_marks(path):
    small = detector.load_image(path)
    gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    c_s, r_s = detector.find_disc(small)
    inner_s = int(P["LABEL_R"] * r_s)
    ring_s = detector.unwrap(gray_s, c_s, r_s)[inner_s:int(P["OUTER_R"] * r_s)]

    big = full_res(path)
    gray_b = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    c_b, r_b = detector.find_disc(big)
    scale = r_b / max(r_s, 1)
    # matching the angular step to the radial one keeps the fine ring a plain
    # enlargement of the coarse one, so a direction measured in one is the same
    # direction in the other
    steps_b = int(round(P["POLAR_STEPS"] * scale))
    polar = cv2.warpPolar(gray_b, (r_b, steps_b), c_b, r_b, cv2.WARP_POLAR_LINEAR)
    ring_b = cv2.transpose(polar)[int(P["LABEL_R"] * r_b):
                                  int(P["OUTER_R"] * r_b)].astype(np.float32)

    radial, tram = detector.scratch_map(ring_s)[:2]
    k_view = VIEW_W / gray_s.shape[1]
    out = []
    for smap, min_len in ((radial, None), (tram, P["TRAM_MIN_LEN"])):
        mask, _ = detector.extract(smap, min_len)
        n, lb, stats, cent = cv2.connectedComponentsWithStats(
            (mask > 127).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < MIN_EXTRA_AREA:
                continue
            pts = np.column_stack(np.nonzero(lb == i)).astype(np.float32)
            if len(pts) < 12:
                continue
            mean = pts.mean(axis=0)
            _, eig = cv2.PCACompute(pts, mean.reshape(1, -1), maxComponents=1)
            dr, dc = float(eig[0][0]), float(eig[0][1])
            norm = math.hypot(dr, dc) or 1.0
            dr, dc = dr / norm, dc / norm
            length = max(w, h)

            widths = []
            n_steps = int(min(length, 90))
            for t in np.linspace(-length / 2, length / 2, max(n_steps, 4)):
                rr = (mean[0] + dr * t) * scale
                cc = (mean[1] + dc * t) * scale
                if not (0 <= rr < ring_b.shape[0] and 0 <= cc < ring_b.shape[1]):
                    continue
                prof = sample_line(ring_b, rr, cc, -dc, dr, HALF)
                fw = fwhm(prof)
                if fw is not None:
                    widths.append(fw)
            if len(widths) < MIN_STEPS:
                continue
            v = np.array(widths, float)
            m = float(v.mean())
            if m <= 0:
                continue
            ang_pos = float(cent[i][0]) / P["POLAR_STEPS"] * 2 * math.pi
            rr_pos = float(cent[i][1]) + inner_s
            out.append({
                "spread": float(v.std() / m),
                "jitter": float(np.abs(np.diff(v)).mean() / m),
                "ratio": float(np.percentile(v, 90) / max(np.percentile(v, 10), .3)),
                "wmean": m, "steps": float(len(v)),
                "vx": (c_s[0] + rr_pos * math.cos(ang_pos)) * k_view,
                "vy": (c_s[1] + rr_pos * math.sin(ang_pos)) * k_view,
            })
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
            cands = photo_marks(path)
        except Exception as exc:
            print(f"  skip {r['pair'][:40]}: {type(exc).__name__}")
            continue
        if not cands:
            continue
        arr = np.array([[c["vx"], c["vy"]] for c in cands], float)
        for lr in rows_l:
            d = np.hypot(arr[:, 0] - lr["cx"], arr[:, 1] - lr["cy"])
            j = int(d.argmin())
            if d[j] > 30:
                continue
            rec = dict(cands[j])
            rec["kind"] = lr["label"]
            out.append(rec)
        print(f"  {r['pair'][:46]:<48}{len(rows_l):>4}")

    if not out:
        sys.exit("nothing matched")
    fake = [r for r in out if r["kind"] == "false"]
    real = [r for r in out if r["kind"] in ("dirt", "scratch")]
    scr = [r for r in out if r["kind"] == "scratch"]
    print(f"\nmeasured {len(out)}   real {len(real)}   false {len(fake)}"
          f"   scratches {len(scr)}")
    w = np.array([r["wmean"] for r in out])
    print(f"  typical width at full resolution: {np.median(w):.1f} px"
          f"   (about {np.median(w) / 4:.1f} at the detector's 1600)")

    def table(pos, name):
        if len(pos) < 10:
            print(f"\n[{name}] only {len(pos)} — too few")
            return
        print(f"\n[{name} vs reflections]   {len(pos)} vs {len(fake)}")
        print(f"{'':<9}{name[:8]+' med':>13}{'refl med':>11}{'AUC':>7}   verdict")
        for f in FEATURES:
            p = np.array([r[f] for r in pos], float)
            q = np.array([r[f] for r in fake], float)
            a = auc(p, q)
            sep = abs(a - 0.5)
            v = "USEFUL" if sep >= 0.15 else ("weak" if sep >= 0.08 else "nothing")
            print(f"{f:<9}{np.median(p):>13.2f}{np.median(q):>11.2f}{a:>7.2f}   {v}")

    table(real, "real")
    table(scr, "scratch")

    with open(os.path.join(HERE, "width_hires.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"\nwrote width_hires.json ({len(out)} rows)")


if __name__ == "__main__":
    main()
