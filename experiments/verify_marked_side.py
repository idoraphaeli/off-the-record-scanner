# -*- coding: utf-8 -*-
"""
Check the claim that each folder has a consistent rule for which half of a pair
carries the blue marks:

    IDO    marks drawn on the ORIGINAL      (the file WITHOUT "(1)")
    AMIT   marks drawn on the COPY          (the file WITH "(1)")

If that holds, ground truth no longer has to guess which file is clean — the
guess is what corrupted the whole evaluation before, because it silently pointed
the detector at the inked photo in 60% of marked pairs.

Blue is counted only where the two files DIFFER. Counting blue over the whole
frame is useless here: black vinyl carries a bluish sheen worth tens of
thousands of pixels, while a pen stroke is worth a few thousand, so the sheen
decides the answer. Where they differ, only ink (and resampling noise) remains.

Reported per folder: pairs that follow the rule, pairs that break it, pairs with
no marks at all, and pairs where BOTH files carry ink — those have no clean half
and no naming rule can rescue them.

Usage:  python verify_marked_side.py
"""

import collections
import glob
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCANNER = os.path.dirname(HERE)

PEN_LO, PEN_HI = (95, 120, 90), (125, 255, 255)
DIFF_SAME, DIFF_RESIZED = 30, 55

NO_PEN = 200        # fewer changed-and-blue pixels than this: nothing was drawn
BOTH_MIN = 500      # both halves need at least this much before calling it "both"
BOTH_RATIO = 0.25   # ...and the quieter half must be this share of the louder

COPY_RE = re.compile(r"\(1\)$")
EXPECT = {"IDO": "original", "AMIT": "copy"}


def stem(name):
    return re.sub(r"\.jpe?g$", "", name, flags=re.I).strip()


def key(name):
    return COPY_RE.sub("", stem(name)).strip().lower()


def read(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def blue(img):
    return cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), PEN_LO, PEN_HI) > 0


def pen_scores(orig, copy):
    """Pixels where one half carries ink and the other does not.

    "Changed AND blue" is not enough on its own. Several of these records are
    blue-lit or have a turquoise label, so a large part of the disc is blue in
    BOTH files, and re-encoding makes those same pixels differ slightly — which
    scores tens of thousands of "blue changes" on a pair nobody drew on. The
    giveaway was that both halves scored almost the same number.

    Ink is blue in one file and NOT blue in the other, so requiring that asks
    the question directly and leaves the record's own colour out of it.
    """
    if orig.shape != copy.shape:
        copy = cv2.resize(copy, (orig.shape[1], orig.shape[0]),
                          interpolation=cv2.INTER_AREA)
        thr = DIFF_RESIZED
    else:
        thr = DIFF_SAME
    changed = cv2.absdiff(orig, copy).max(axis=2) > thr
    b_o, b_c = blue(orig), blue(copy)
    ink_o = changed & b_o & ~b_c
    ink_c = changed & b_c & ~b_o
    # a stroke is a solid run, not scattered speckle: an opening drops the
    # single-pixel noise that survives along every blue edge
    k = np.ones((3, 3), np.uint8)
    ink_o = cv2.morphologyEx(ink_o.astype(np.uint8), cv2.MORPH_OPEN, k)
    ink_c = cv2.morphologyEx(ink_c.astype(np.uint8), cv2.MORPH_OPEN, k)
    return int(np.count_nonzero(ink_o)), int(np.count_nonzero(ink_c))


def find_root():
    # the folder name begins with two U+200F marks that Windows adds to Hebrew
    # paths, so it cannot be matched by typing the visible name
    hits = glob.glob(os.path.join(SCANNER, "*AGAIN*"))
    hits = [h for h in hits if os.path.isdir(h)]
    if not hits:
        sys.exit("could not find the AGAIN folder")
    return hits[0]


def main():
    root = find_root()
    print(f"root: {os.path.basename(root)}\n")

    grand = collections.Counter()
    for folder in ("IDO", "AMIT"):
        path = os.path.join(root, folder)
        if not os.path.isdir(path):
            print(f"[{folder}] missing\n")
            continue

        files = [f for f in os.listdir(path)
                 if f.lower().endswith((".jpg", ".jpeg"))]
        groups = collections.defaultdict(list)
        for f in files:
            groups[key(f)].append(f)

        rule = EXPECT[folder]
        ok, broken, clean, both, unpaired = 0, [], 0, [], []

        for k, group in sorted(groups.items()):
            orig = [f for f in group if not COPY_RE.search(stem(f))]
            copy = [f for f in group if COPY_RE.search(stem(f))]
            if len(orig) != 1 or len(copy) != 1:
                unpaired.append((k, len(orig), len(copy)))
                continue
            a, b = read(os.path.join(path, orig[0])), read(os.path.join(path, copy[0]))
            if a is None or b is None:
                unpaired.append((k, -1, -1))
                continue

            s_orig, s_copy = pen_scores(a, b)
            hi, lo = max(s_orig, s_copy), min(s_orig, s_copy)
            if hi < NO_PEN:
                clean += 1
                continue
            if lo > BOTH_MIN and lo > BOTH_RATIO * hi:
                both.append((k, s_orig, s_copy))
                continue
            found = "original" if s_orig > s_copy else "copy"
            if found == rule:
                ok += 1
            else:
                broken.append((k, s_orig, s_copy, found))

        total = len(groups)
        print(f"[{folder}]  rule: marks on the {rule.upper()}   {total} pairs, "
              f"{len(files)} files")
        print(f"   follows the rule        : {ok}")
        print(f"   BREAKS the rule         : {len(broken)}")
        print(f"   no marks at all         : {clean}")
        print(f"   ink on BOTH halves      : {len(both)}")
        print(f"   unpaired / unreadable   : {len(unpaired)}")

        for k, so, sc, found in broken[:25]:
            print(f"      breaks: {k[:52]:<54} orig {so:>6}  copy {sc:>6}"
                  f"  -> {found}")
        if len(broken) > 25:
            print(f"      ... and {len(broken) - 25} more")
        for k, so, sc in both[:25]:
            print(f"      both  : {k[:52]:<54} orig {so:>6}  copy {sc:>6}")
        if len(both) > 25:
            print(f"      ... and {len(both) - 25} more")
        for k, no, nc in unpaired[:25]:
            print(f"      unpaired: {k[:50]:<52} originals {no}  copies {nc}")
        print()

        grand["pairs"] += total
        grand["ok"] += ok
        grand["broken"] += len(broken)
        grand["clean"] += clean
        grand["both"] += len(both)
        grand["unpaired"] += len(unpaired)

    print(f"TOTAL  {grand['pairs']} pairs   "
          f"rule holds {grand['ok']}   breaks {grand['broken']}   "
          f"no marks {grand['clean']}   both inked {grand['both']}   "
          f"unpaired {grand['unpaired']}")


if __name__ == "__main__":
    main()
