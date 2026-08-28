# -*- coding: utf-8 -*-
"""
Stage 0 for the new dataset: build ground truth, then split it. Run once.

Which half of a pair carries the pen is now DECLARED, not inferred. The photos
are filed under "Records_Data_New_jpg - AGAIN" in two folders:

    IDO    marks were drawn on the ORIGINAL  (the file without "(1)")
    AMIT   marks were drawn on the COPY      (the file with "(1)")

Inferring it was the single worst defect this evaluation had. The old code chose
a clean half correctly but then recorded the OTHER name in the index, so 60% of
marked photos were fed to the detector WITH the ink on them — blue ink on black
vinyl is a bright thin line, exactly what the detector hunts, so it was being
shown the answer sheet and recall was meaningless. Declaring the rule removes
the inference entirely; the folders are only read to learn the rule, while the
images themselves still come from the flat folder, so nothing downstream moves.

A pixel counts as pen where the pair DIFFERS, the marked half is pen-blue, and
the clean half is NOT. The last clause matters: several records are blue-lit or
carry a turquoise label, so a colour test alone passes across the whole disc and
re-encoding makes those pixels differ — scoring 60,000-90,000 "blue changes" on
a pair nobody drew on.

The split is BY RECORD, never by photo. Two shots of one side are near
duplicates; with them on opposite sides of the split the test set would be
contaminated and every score would read too well. An existing split.json is
REUSED rather than redrawn, so the locked test set stays locked even when the
marks behind it change; pass --resplit to draw a new one deliberately.

Outputs into gt_new/:  <pair>.png  masks
                       split.json  calibration / validation / locked test
                       gt_index.csv  one row per pair
"""

import collections
import csv
import glob
import json
import os
import random
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCANNER = os.path.dirname(HERE)
PHOTOS = os.path.join(SCANNER, "Records_Data_New_jpg")
OUT = os.path.join(HERE, "gt_new")

NAME_RE = re.compile(r"^(?P<rec>.+?)_side(?P<side>[AB])_shot(?P<shot>\d+)"
                     r"_tilt(?P<tilt>[\d\-]+)", re.I)
COPY_RE = re.compile(r"\(1\)$")

DIFF_SAME, DIFF_RESIZED = 30, 55
PEN_HSV_LO, PEN_HSV_HI = (95, 120, 90), (125, 255, 255)
MIN_PEN_WIDTH, MIN_MARK_AREA = 3, 120

# A pair is dropped when the ink says the opposite of its folder this loudly.
# Two pairs do that, and they were filed in the wrong folder rather than being
# genuinely ambiguous; excluding them beats silently trusting either source.
CONTRADICT_INK = 800

SEED = 42
FRAC = {"cal": 0.55, "val": 0.20, "test": 0.25}
RULE = {"IDO": "original", "AMIT": "copy"}


def stem(name):
    return re.sub(r"\.jpe?g$", "", name, flags=re.I).strip()


def key(name):
    return COPY_RE.sub("", stem(name)).strip().lower()


def read(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def blue(img):
    return cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
                       PEN_HSV_LO, PEN_HSV_HI) > 0


def declared_pairs():
    """pair -> (folder, marked filename, clean filename), from the AGAIN tree.

    The folder name begins with two U+200F marks that Windows adds to Hebrew
    paths, so it cannot be opened by typing the visible name.
    """
    roots = [h for h in glob.glob(os.path.join(SCANNER, "*AGAIN*"))
             if os.path.isdir(h)]
    if not roots:
        sys.exit("could not find the AGAIN folder")

    out, unpaired = {}, []
    for folder, rule in RULE.items():
        path = os.path.join(roots[0], folder)
        if not os.path.isdir(path):
            sys.exit(f"missing folder {folder}")
        groups = collections.defaultdict(list)
        for f in os.listdir(path):
            if f.lower().endswith((".jpg", ".jpeg")):
                groups[key(f)].append(f)
        for k, group in groups.items():
            orig = [f for f in group if not COPY_RE.search(stem(f))]
            copy = [f for f in group if COPY_RE.search(stem(f))]
            if len(orig) != 1 or len(copy) != 1:
                unpaired.append((folder, k))
                continue
            marked, clean = (orig[0], copy[0]) if rule == "original" \
                else (copy[0], orig[0])
            out[k] = (folder, marked, clean)
    return out, unpaired


def extract(marked, clean):
    """The pen strokes, as a mask aligned to the CLEAN image.

    Aligned to the clean one because that is the image the detector reads; a
    mask drawn at the other file's resolution would sit a few pixels off.
    """
    resized = marked.shape != clean.shape
    if resized:
        marked = cv2.resize(marked, (clean.shape[1], clean.shape[0]),
                            interpolation=cv2.INTER_AREA)
    thr = DIFF_RESIZED if resized else DIFF_SAME

    changed = cv2.absdiff(clean, marked).max(axis=2) > thr
    pen = (changed & blue(marked) & ~blue(clean)).astype(np.uint8) * 255
    pen = cv2.morphologyEx(pen, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    pen = cv2.morphologyEx(pen, cv2.MORPH_OPEN,
                           np.ones((MIN_PEN_WIDTH, MIN_PEN_WIDTH), np.uint8))
    ink_here = int(np.count_nonzero(pen))

    # the same measurement the other way round: how much ink the file we were
    # told is clean carries. Loud disagreement means the pair is misfiled.
    other = (changed & blue(clean) & ~blue(marked)).astype(np.uint8) * 255
    other = cv2.morphologyEx(other, cv2.MORPH_OPEN,
                             np.ones((MIN_PEN_WIDTH, MIN_PEN_WIDTH), np.uint8))
    ink_other = int(np.count_nonzero(other))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(pen, connectivity=8)
    mask = np.zeros_like(pen)
    count = 0
    for i in range(1, n):
        if stats[i][4] >= MIN_MARK_AREA:
            mask[labels == i] = 255
            count += 1
    return mask, count, ink_here, ink_other, "resized" if resized else "same size"


def build_split(rows, per_record):
    records = sorted(per_record)

    def band(r):
        n = per_record[r]["marks"]
        return "clean" if n == 0 else ("light" if n <= 6 else
                                       "medium" if n <= 20 else "heavy")

    groups = collections.defaultdict(list)
    for r in records:
        groups[band(r)].append(r)

    rng = random.Random(SEED)
    split = {"cal": [], "val": [], "test": []}
    for b in sorted(groups):
        g = groups[b][:]
        rng.shuffle(g)
        n = len(g)
        n_cal = round(n * FRAC["cal"])
        n_val = round(n * FRAC["val"])
        split["cal"] += g[:n_cal]
        split["val"] += g[n_cal:n_cal + n_val]
        split["test"] += g[n_cal + n_val:]
    return {k: sorted(v) for k, v in split.items()}, groups


def main():
    resplit = "--resplit" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    declared, unpaired = declared_pairs()

    rows, dropped = [], []
    per_record = collections.defaultdict(lambda: {"pairs": 0, "marks": 0})

    for pair in sorted(declared):
        folder, marked_name, clean_name = declared[pair]
        m = NAME_RE.match(pair)
        if m is None:
            continue
        marked = read(os.path.join(PHOTOS, marked_name))
        clean = read(os.path.join(PHOTOS, clean_name))
        if marked is None or clean is None:
            dropped.append((pair, folder, "unreadable"))
            continue

        mask, count, ink_here, ink_other, note = extract(marked, clean)
        if ink_other > CONTRADICT_INK and ink_other > 4 * max(ink_here, 1):
            dropped.append((pair, folder,
                            f"ink contradicts folder ({ink_other} vs {ink_here})"))
            continue

        cv2.imencode(".png", mask)[1].tofile(os.path.join(OUT, pair + ".png"))
        rec = m.group("rec")
        rows.append({"pair": pair, "record": rec, "side": m.group("side"),
                     "shot": int(m.group("shot")), "tilt": m.group("tilt"),
                     "photo_file": clean_name, "marks": count,
                     "size_note": note, "folder": folder,
                     "width": clean.shape[1], "height": clean.shape[0]})
        per_record[rec]["pairs"] += 1
        per_record[rec]["marks"] += count

    split_path = os.path.join(OUT, "split.json")
    reused = os.path.exists(split_path) and not resplit
    if reused:
        split = json.load(open(split_path, encoding="utf-8"))
        groups = None
    else:
        split, groups = build_split(rows, per_record)
        with open(split_path, "w", encoding="utf-8") as fh:
            json.dump(split, fh, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT, "gt_index.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair", "record", "side", "shot", "tilt",
                                           "photo_file", "marks", "size_note",
                                           "folder", "width", "height"])
        w.writeheader()
        w.writerows(rows)

    total = sum(r["marks"] for r in rows)
    print(f"{len(rows)} pairs, {len(per_record)} records, {total} marked scratches")
    print(f"  declared by folder: {len(declared)}   "
          f"dropped: {len(dropped)}   unpaired (never entered): {len(unpaired)}")
    for pair, folder, why in dropped:
        print(f"    dropped  [{folder}] {pair[:46]:<48}{why}")
    for folder, k in unpaired:
        print(f"    unpaired [{folder}] {k[:46]}")

    if groups:
        print(f"\n{'band':<10}{'records':>9}   split (cal/val/test)")
        for b in ("clean", "light", "medium", "heavy"):
            g = groups.get(b, [])
            c = sum(1 for r in g if r in split["cal"])
            v = sum(1 for r in g if r in split["val"])
            t = sum(1 for r in g if r in split["test"])
            print(f"    {b:<10}{len(g):>5}      {c} / {v} / {t}")
    else:
        print("\n  split.json REUSED — the locked test set is unchanged")

    print()
    for k in ("cal", "val", "test"):
        recs = set(split[k])
        pairs = [r for r in rows if r["record"] in recs]
        marks = sum(r["marks"] for r in pairs)
        clean = sum(1 for r in pairs if r["marks"] == 0)
        print(f"  {k:<5}{len(recs):>3} records{len(pairs):>5} pairs"
              f"{marks:>6} scratches{clean:>5} clean photos")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
