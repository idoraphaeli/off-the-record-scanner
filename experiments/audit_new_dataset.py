# -*- coding: utf-8 -*-
"""
Audit Records_Data_New before any tuning is designed.

Layout is flat: "<record>_side<A|B>_shot<n>_tilt<a-b>.jpg" plus a "(1)" copy on
which scratches were drawn in blue DIRECTLY ALONG the scratch (not circled, as
in the earlier set). That changes the ground truth: the pen stroke *is* the
scratch, so it is used as-is with a small tolerance for hand wobble, rather than
being treated as a boundary to fill.

Reports what exists before deciding how to split it: how many records, how many
photos carry marks, how many distinct scratches were marked, and how many photos
are clean (which are the pure negatives).

Read-only. Writes nothing except a handful of check images into audit_out/.

Usage:  python audit_new_dataset.py [folder]
"""

import collections
import csv
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(os.path.dirname(HERE), "Records_Data_New")
OUT = os.path.join(HERE, "audit_out")

NAME_RE = re.compile(r"^(?P<rec>.+?)_side(?P<side>[AB])_shot(?P<shot>\d+)"
                     r"_tilt(?P<tilt>[\d\-]+)", re.I)
COPY_RE = re.compile(r"\(1\)\.jpe?g$", re.I)


def key(name):
    """Matching key for a photo and its copy: the "(1)" marker and the extension
    are stripped, and case is normalised — plain files arrive as both .JPG and
    .jpg while every copy is .jpg, so a case-sensitive match pairs almost none."""
    stem = COPY_RE.sub("", name)
    stem = re.sub(r"\.jpe?g$", "", stem, flags=re.I)
    return stem.strip().lower()

DIFF_THR = 25          # per-pixel difference counted as pen
MIN_STROKE_AREA = 60   # px: below this is JPEG noise, not a drawn line
CHECK_IMAGES = 4


def parse(name):
    m = NAME_RE.match(key(name))
    if not m:
        return None
    return m.group("rec"), m.group("side"), int(m.group("shot")), m.group("tilt")


def read(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def blue_mask(img):
    """Saturated blue, which is the pen and essentially nothing else on a photo
    of black vinyl."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (95, 90, 60), (140, 255, 255))


def pick_marked(a, b):
    """Return (original, marked). Which file of a pair carries the pen is not
    consistent across records, so it is decided by measurement: whichever holds
    materially more blue is the marked one."""
    ba, bb = int(np.count_nonzero(blue_mask(a))), int(np.count_nonzero(blue_mask(b)))
    return (a, b, bb) if bb >= ba else (b, a, ba)


def strokes(marked):
    """Pen strokes as drawn. No filling: the line IS the scratch.

    Found by COLOUR, not by differencing the pair — the copies were re-saved at
    various sizes, which makes a pixel difference unusable for most of them.
    """
    pen = blue_mask(marked)
    pen_px = int(np.count_nonzero(pen))

    # join a stroke broken by JPEG artefacts, then drop specks
    pen = cv2.morphologyEx(pen, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pen, connectivity=8)
    keep = np.zeros_like(pen)
    count = 0
    for i in range(1, n):
        if stats[i][4] < MIN_STROKE_AREA:
            continue
        keep[labels == i] = 255
        count += 1
    return keep, pen_px, count


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    files = [f for f in sorted(os.listdir(root))
             if f.lower().endswith((".jpg", ".jpeg"))]

    plains = [f for f in files if not COPY_RE.search(f)]
    copies = {key(f): f for f in files if COPY_RE.search(f)}

    print(f"{len(files)} files: {len(plains)} plain, {len(copies)} copies\n")

    plain_keys = {key(f) for f in plains}
    unpaired = [f for f in plains if key(f) not in copies]
    orphan = [c for k, c in copies.items() if k not in plain_keys]
    if unpaired:
        print(f"WARNING  {len(unpaired)} photos with no copy: {unpaired[:5]}")
    if orphan:
        print(f"WARNING  {len(orphan)} copies with no original: {orphan[:5]}")
    unparsed = [f for f in plains if parse(f) is None]
    if unparsed:
        print(f"WARNING  {len(unparsed)} names did not parse: {unparsed[:5]}")
    print()

    os.makedirs(OUT, exist_ok=True)
    rows, saved = [], 0
    per_record = collections.defaultdict(lambda: {"photos": 0, "marked": 0, "scratches": 0})

    for f in plains:
        meta = parse(f)
        if meta is None or key(f) not in copies:
            continue
        rec, side, shot, tilt = meta
        a = read(os.path.join(root, f))
        b = read(os.path.join(root, copies[key(f)]))
        if a is None or b is None:
            rows.append({"file": f, "record": rec, "status": "unreadable"})
            continue

        plain, marked, _ = pick_marked(a, b)
        status = "ok" if plain.shape == marked.shape else "sizes differ"
        mask, pen_px, count = strokes(marked)
        rows.append({"file": f, "record": rec, "side": side, "shot": shot,
                     "tilt": tilt, "status": status, "pen_px": pen_px,
                     "scratches": count,
                     "width": plain.shape[1], "height": plain.shape[0]})

        r = per_record[rec]
        r["photos"] += 1
        if count:
            r["marked"] += 1
            r["scratches"] += count

        if count and saved < CHECK_IMAGES:
            vis = marked.copy()
            band = cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0
            vis[band] = (0, 255, 255)
            cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(
                os.path.join(OUT, f"check_{saved+1}_{rec}_{side}{shot}.jpg"))
            saved += 1

    status_counts = collections.Counter(r.get("status") for r in rows)
    print("pair status:", dict(status_counts), "\n")

    ok = [r for r in rows if r.get("status") in ("ok", "sizes differ")]
    marked_photos = [r for r in ok if r["scratches"] > 0]
    clean_photos = [r for r in ok if r["scratches"] == 0]
    total_scratches = sum(r["scratches"] for r in marked_photos)

    print("=" * 62)
    print(f"records                    : {len(per_record)}")
    print(f"photo pairs usable         : {len(ok)}")
    print(f"photos WITH marks          : {len(marked_photos)}")
    print(f"photos with NO marks       : {len(clean_photos)}   <- pure negatives")
    print(f"MARKED SCRATCHES in total  : {total_scratches}")
    if marked_photos:
        c = [r["scratches"] for r in marked_photos]
        print(f"  per marked photo         : median {int(np.median(c))},"
              f" range {min(c)}-{max(c)}")
    sizes = collections.Counter((r["width"], r["height"]) for r in ok)
    print(f"image sizes                : {dict(sizes)}")

    print("\nper record (photos / marked photos / scratches):")
    for rec in sorted(per_record):
        d = per_record[rec]
        print(f"    {rec[:34]:<36}{d['photos']:>3}{d['marked']:>8}{d['scratches']:>10}")

    with open(os.path.join(OUT, "audit.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "record", "side", "shot", "tilt",
                                           "status", "pen_px", "scratches",
                                           "width", "height"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}  ({saved} check images + audit.csv)")


if __name__ == "__main__":
    main()
