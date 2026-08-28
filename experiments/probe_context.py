# -*- coding: utf-8 -*-
"""
Two things never yet measured: what a mark looks like NEXT TO the other marks on
its own photograph, and how it ends.

Everything tried so far judged each detection alone, against a fixed number. But
a reflection is not a property of one spot — it is a property of the whole
photograph. A lamp at an unlucky angle throws twenty streaks at once; a clean
record photographed well gives two faint ones. The detector already knows this,
because it produced all of them, and that knowledge has been thrown away every
time. So each mark is here described by where it sits among its own siblings:
how many there are, how it ranks, how it compares to the middling one.

The second measurement is about what the two things ARE. A reflection is a
gradual event — the lighting angle changes smoothly along the record, so the
streak swells in the middle and fades away at both ends. A scratch is a cut of
roughly even depth: it should be about equally bright end to end, and it should
stop where the cut stops rather than fading.

    taper       middle brightness against the two ends
                well above 1 means it swells and fades -> lighting
                near 1 means it holds -> a cut of even depth
    flatness    how much the brightness wanders along its own length
    end_drop    how abruptly it stops at its ends

Both families are measured in the unwrapped ring, where the detector works.

Usage:  python probe_context.py [cal|val|both]
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
BINS = 12              # slices along the mark's own length
MIN_PTS = 24

CONTEXT = ("n_marks", "rank_score", "rank_len", "rel_score", "rel_len",
           "rel_bright", "photo_len_med")
SHAPE = ("taper", "flatness", "end_drop", "hump")


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


def along_axis(flat, pts, mean, direction):
    """Brightness along the mark's own length, in equal slices.

    Every pixel of the mark is placed by how far along it lies, then the slices
    are averaged. Using the mark's own pixels rather than a straight line keeps
    a slightly bent mark from wandering off itself.
    """
    rel = pts - mean
    t = rel @ np.array(direction, np.float32)
    span = t.max() - t.min()
    if span < 4:
        return None
    idx = np.clip(((t - t.min()) / span * BINS).astype(int), 0, BINS - 1)
    vals = flat[pts[:, 0].astype(int), pts[:, 1].astype(int)]
    prof = np.zeros(BINS, np.float32)
    for b in range(BINS):
        sel = idx == b
        if not sel.any():
            return None
        prof[b] = float(vals[sel].mean())
    return prof


def shape_of(prof):
    m = float(prof.mean())
    if m <= 1e-3:
        return None
    edge = max(BINS // 4, 1)
    mid = prof[edge:BINS - edge]
    ends = np.concatenate([prof[:edge], prof[BINS - edge:]])
    if len(mid) == 0 or ends.mean() <= 1e-3:
        return None
    # an inverted U, for comparison: a reflection should follow it closely
    x = np.linspace(-1, 1, BINS)
    arch = 1.0 - x ** 2
    p = prof - prof.mean()
    a = arch - arch.mean()
    denom = float(np.std(p) * np.std(a))
    return {
        "taper": float(mid.mean() / ends.mean()),
        "flatness": float(prof.std() / m),
        "end_drop": float((prof[1:3].mean() + prof[-3:-1].mean()) / 2
                          / max(prof.max(), 1e-3)),
        "hump": float(np.mean(p * a) / denom) if denom > 1e-6 else 0.0,
    }


def photo_marks(path):
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner, outer = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner:outer]

    # brightness above the local vinyl, which is what a mark actually is
    flat = ring.astype(np.float32)
    flat = np.clip(flat - cv2.blur(flat, (P["ROW_FLATTEN"], 1)), 0, None)

    radial, tram = detector.scratch_map(ring)[:2]
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
            pts = np.column_stack(np.nonzero(lb == i)).astype(np.float32)
            if len(pts) < MIN_PTS:
                continue
            mean = pts.mean(axis=0)
            _, eig = cv2.PCACompute(pts, mean.reshape(1, -1), maxComponents=1)
            d = np.array(eig[0], np.float32)
            d /= (np.linalg.norm(d) or 1.0)
            prof = along_axis(flat, pts, mean, d)
            if prof is None:
                continue
            sh = shape_of(prof)
            if sh is None:
                continue
            sel = lb == i
            rr = float(cent[i][1]) + inner
            ang = float(cent[i][0]) / P["POLAR_STEPS"] * 2 * math.pi
            rec = dict(sh)
            rec.update({
                "score": float(smap[sel].max()),
                "length": float(max(w, h)),
                "bright": float(flat[sel].mean()),
                "vx": (center[0] + rr * math.cos(ang)) * k,
                "vy": (center[1] + rr * math.sin(ang)) * k,
            })
            out.append(rec)

    if not out:
        return out
    # now the part that needed the whole photograph
    sc = np.array([r["score"] for r in out], float)
    ln = np.array([r["length"] for r in out], float)
    br = np.array([r["bright"] for r in out], float)
    for r in out:
        r["n_marks"] = float(len(out))
        r["rank_score"] = float((sc < r["score"]).mean())
        r["rank_len"] = float((ln < r["length"]).mean())
        r["rel_score"] = float(r["score"] / max(np.median(sc), 1e-3))
        r["rel_len"] = float(r["length"] / max(np.median(ln), 1e-3))
        r["rel_bright"] = float(r["bright"] / max(np.median(br), 1e-3))
        r["photo_len_med"] = float(np.median(ln))
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

    out = []
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
            j = int(d.argmin())
            if d[j] > 30:
                continue
            rec = {k: v for k, v in cands[j].items() if k not in ("vx", "vy")}
            rec["kind"] = lr["label"]
            rec["pair"] = r["pair"]
            rec["record"] = r["record"]
            rec["set"] = which_of[r["record"]]
            out.append(rec)
        print(f"  {r['pair'][:46]:<48}{len(cands):>4} marks")

    if not out:
        sys.exit("nothing matched")
    fake = [r for r in out if r["kind"] == "false"]
    real = [r for r in out if r["kind"] in ("dirt", "scratch")]
    scr = [r for r in out if r["kind"] == "scratch"]
    print(f"\nmeasured {len(out)}   real {len(real)}   false {len(fake)}"
          f"   scratches {len(scr)}")

    def table(pos, name, keys, title):
        if len(pos) < 10:
            return
        print(f"\n[{title}]  {name} vs reflections   {len(pos)} vs {len(fake)}")
        print(f"{'':<15}{name[:8]+' med':>13}{'refl med':>11}{'AUC':>7}   verdict")
        for f in keys:
            p = np.array([r[f] for r in pos], float)
            q = np.array([r[f] for r in fake], float)
            a = auc(p, q)
            sep = abs(a - 0.5)
            v = "USEFUL" if sep >= 0.15 else ("weak" if sep >= 0.08 else "nothing")
            print(f"{f:<15}{np.median(p):>13.2f}{np.median(q):>11.2f}{a:>7.2f}   {v}")

    for keys, title in ((CONTEXT, "the photograph as context"),
                        (SHAPE, "how the mark ends")):
        table(real, "real", keys, title)
        table(scr, "scratch", keys, title)

    # what a photograph's own mix looks like, by how busy it is
    per_photo = collections.defaultdict(list)
    for r in out:
        per_photo[r["pair"]].append(r)
    print(f"\nhow a photograph's crop of marks changes as it gets busier:")
    print(f"{'marks on photo':<18}{'photos':>8}{'labelled':>10}{'precision':>11}")
    for lo, hi, lbl in ((0, 8, "up to 8"), (9, 16, "9 to 16"),
                        (17, 30, "17 to 30"), (31, 9999, "31 and up")):
        grp = [rs for rs in per_photo.values()
               if lo <= rs[0]["n_marks"] <= hi]
        flat_rows = [r for rs in grp for r in rs]
        if not flat_rows:
            continue
        good = sum(1 for r in flat_rows if r["kind"] != "false")
        print(f"{lbl:<18}{len(grp):>8}{len(flat_rows):>10}"
              f"{100.0*good/len(flat_rows):>10.0f}%")

    with open(os.path.join(HERE, "context_features.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"\nwrote context_features.json ({len(out)} rows)")


if __name__ == "__main__":
    main()
