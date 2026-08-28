# -*- coding: utf-8 -*-
"""
Audit Records_Data_New_jpg: what is actually in it, before any tuning is designed.

Pairs are "<name>.jpg" and "<name>(1).jpg". Which one carries the pen is NOT
consistent across records, so it is decided by measurement, not by name.

Marks are found by DIFFERENCING the pair rather than by colour: the blue of the
pen and the bluish diffraction sheen of black vinyl overlap badly, and a colour
search returned dozens of components per photo where the annotator drew three.
Where the two files were saved at different sizes the copy is resampled first,
and the difference bar is raised to absorb the resampling noise.

Read-only apart from a few check images in audit_out/.

Usage:  python audit_new.py [folder]
"""

import collections
import csv
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
OUT = os.path.join(HERE, "audit_out")

NAME_RE = re.compile(r"^(?P<rec>.+?)_side(?P<side>[AB])_shot(?P<shot>\d+)"
                     r"_tilt(?P<tilt>[\d\-]+)", re.I)
COPY_RE = re.compile(r"\(1\)$")

DIFF_SAME = 30      # difference threshold when both files are the same size
DIFF_RESIZED = 55   # higher when one had to be resampled onto the other
MIN_MARK_AREA = 120
CHECK_IMAGES = 5

# The editor's pen is one fixed colour (about RGB 0,122,255) drawn at a fixed
# minimum width. Both are used as filters, because differencing alone also
# catches resampling noise: 183 of 200 copies were re-saved at another size.
PEN_HSV_LO = (95, 120, 90)      # OpenCV hue scale; blue sits near 105
PEN_HSV_HI = (125, 255, 255)
MIN_PEN_WIDTH = 3               # px: a drawn stroke is never one pixel wide


def key(name):
    stem = re.sub(r"\.jpe?g$", "", name, flags=re.I).strip()
    return COPY_RE.sub("", stem).lower()


def parse(name):
    m = NAME_RE.match(key(name))
    return (m.group("rec"), m.group("side"), int(m.group("shot")),
            m.group("tilt")) if m else None


def read(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def pen_colour(img):
    """Pixels holding the editor's pen colour."""
    return cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), PEN_HSV_LO, PEN_HSV_HI)


def find_marks(a, b):
    """Return (plain, marked, mask, area, n_marks, note).

    A pixel counts as pen only if the pair DIFFERS there AND the drawn-on image
    holds the pen colour there. Either test alone is unreliable: differencing
    also fires on resampling noise, and the pen's blue overlaps the bluish
    diffraction sheen of black vinyl.
    """
    note = "same size"
    if a.shape != b.shape:
        note = "resized"
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    thr = DIFF_SAME if note == "same size" else DIFF_RESIZED

    changed = cv2.absdiff(a, b).max(axis=2) > thr
    blue_a, blue_b = pen_colour(a) > 0, pen_colour(b) > 0

    # the drawn-on image is the one carrying pen colour where the two differ
    if np.count_nonzero(changed & blue_b) >= np.count_nonzero(changed & blue_a):
        plain, marked, is_pen = a, b, blue_b
    else:
        plain, marked, is_pen = b, a, blue_a

    pen = (changed & is_pen).astype(np.uint8) * 255
    pen = cv2.morphologyEx(pen, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # a real stroke has body; single-pixel filaments are compression artefacts
    pen = cv2.morphologyEx(pen, cv2.MORPH_OPEN,
                           np.ones((MIN_PEN_WIDTH, MIN_PEN_WIDTH), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(pen, connectivity=8)
    mask = np.zeros_like(pen)
    count = 0
    for i in range(1, n):
        if stats[i][4] < MIN_MARK_AREA:
            continue
        mask[labels == i] = 255
        count += 1
    return plain, marked, mask, int(np.count_nonzero(mask)), count, note


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    files = sorted(f for f in os.listdir(root) if f.lower().endswith((".jpg", ".jpeg")))
    plains = [f for f in files if not COPY_RE.search(
        re.sub(r"\.jpe?g$", "", f, flags=re.I))]
    copies = {key(f): f for f in files if COPY_RE.search(
        re.sub(r"\.jpe?g$", "", f, flags=re.I))}

    print(f"{len(files)} files: {len(plains)} plain, {len(copies)} copies")
    lonely = [f for f in plains if key(f) not in copies]
    if lonely:
        print(f"{len(lonely)} without a pair (excluded): {[f[:40] for f in lonely]}")
    print()

    os.makedirs(OUT, exist_ok=True)
    rows, saved = [], 0
    per_record = collections.defaultdict(
        lambda: {"pairs": 0, "with_marks": 0, "marks": 0})

    for f in plains:
        meta = parse(f)
        if meta is None or key(f) not in copies:
            continue
        rec, side, shot, tilt = meta
        a = read(os.path.join(root, f))
        b = read(os.path.join(root, copies[key(f)]))
        if a is None or b is None:
            rows.append({"pair": key(f), "record": rec, "status": "unreadable"})
            continue

        plain, marked, mask, area, count, note = find_marks(a, b)
        rows.append({"pair": key(f), "record": rec, "side": side, "shot": shot,
                     "tilt": tilt, "status": "ok", "size_note": note,
                     "marks": count, "mark_px": area,
                     "width": plain.shape[1], "height": plain.shape[0]})

        d = per_record[rec]
        d["pairs"] += 1
        if count:
            d["with_marks"] += 1
            d["marks"] += count

        if count and saved < CHECK_IMAGES:
            vis = plain.copy()
            band = cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0
            vis[band] = (0, 255, 255)
            cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(
                os.path.join(OUT, f"check_{saved+1}_{rec[:24]}_{side}{shot}.jpg"))
            saved += 1

    ok = [r for r in rows if r["status"] == "ok"]
    marked = [r for r in ok if r["marks"] > 0]
    clean = [r for r in ok if r["marks"] == 0]
    total = sum(r["marks"] for r in marked)

    print("=" * 64)
    print(f"records                  : {len(per_record)}")
    print(f"usable pairs             : {len(ok)}")
    print(f"pairs WITH marks         : {len(marked)}")
    print(f"pairs with NO marks      : {len(clean)}    <- clean records")
    print(f"MARKED SCRATCHES total   : {total}")
    if marked:
        c = [r["marks"] for r in marked]
        print(f"  per marked photo       : median {int(np.median(c))},"
              f" range {min(c)}-{max(c)}")
    print(f"size notes               : "
          f"{dict(collections.Counter(r['size_note'] for r in ok))}")

    print("\nper record  (pairs / with marks / scratches):")
    for rec in sorted(per_record):
        d = per_record[rec]
        print(f"    {rec[:36]:<38}{d['pairs']:>3}{d['with_marks']:>8}{d['marks']:>9}")

    with open(os.path.join(OUT, "audit_new.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair", "record", "side", "shot", "tilt",
                                           "status", "size_note", "marks",
                                           "mark_px", "width", "height"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}  ({saved} check images + audit_new.csv)")


if __name__ == "__main__":
    main()
