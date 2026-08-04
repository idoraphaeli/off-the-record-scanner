# -*- coding: utf-8 -*-
"""
Scan a folder of record photos, one photo at a time (no pair comparison).

For every image it writes a copy with the detected marks highlighted in
translucent yellow, and a CSV summarising what was found. Two extra columns
matter when comparing lighting setups: `judged_pct` says how much of the playing
surface was bright enough to assess at all, and `groove_rejected` counts marks
thrown out for running along the grooves (lamp reflections rather than damage).
A photo with a low judged_pct has not been shown to be clean -- most of it was
simply never examined.

Usage:
    python scan_folder.py <folder>
    python scan_folder.py <folder> --out <output folder>
"""

import csv
import os
import sys

import cv2
import numpy as np

import detector
from detector import P

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MARK_ALPHA = 0.45
MARK_HALO = 9


def paint(img, det):
    """Translucent highlight: findable at page scale, sheer enough that the
    scratch itself is still visible underneath."""
    vis = img.copy().astype(np.float32)
    band = cv2.dilate((det > 127).astype(np.uint8),
                      np.ones((MARK_HALO, MARK_HALO), np.uint8))
    band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.0)
    a = np.clip(band, 0, 1)[..., None] * MARK_ALPHA
    return (vis * (1 - a) + np.array([90, 255, 255], np.float32) * a).astype(np.uint8)


def scan_one(path):
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner_px, outer_px = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner_px:outer_px]

    glare = detector.glare_mask(ring) > 0
    unlit = detector.unlit_mask(ring) > 0
    judged = 100.0 * (~(glare | unlit)).mean()

    radial, tram = detector.scratch_map(ring)
    mask_a, sa = detector.extract(radial)
    mask_b, sb = detector.extract(tram, min_len=P["TRAM_MIN_LEN"])
    mask = cv2.bitwise_or(mask_a, mask_b)
    marks = sa + sb

    det = detector.rewrap(mask, inner_px, center, radius, gray.shape)
    return dict(img=img, det=det, marks=marks, radius=radius, judged=judged,
                glare=100.0 * glare.mean(), unlit=100.0 * unlit.mean(),
                ring_mean=float(ring.mean()))


def main():
    folder = sys.argv[1]
    out = folder
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    out = os.path.join(out, "scan")
    os.makedirs(out, exist_ok=True)

    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXT))
    if not names:
        raise SystemExit(f"no images found in {folder}")
    print(f"scanning {len(names)} photos from {folder}\n")
    print(f"{'photo':<34}{'marks':>6}{'judged%':>9}{'unlit%':>8}{'glare%':>8}"
          f"{'longest':>9}")
    print("-" * 74)

    rows = []
    for name in names:
        try:
            r = scan_one(os.path.join(folder, name))
        except Exception as exc:                    # keep going through the batch
            print(f"{name[:32]:<34}  FAILED: {exc}")
            rows.append({"photo": name, "marks": "", "judged_pct": "",
                         "unlit_pct": "", "glare_pct": "", "longest_px": "",
                         "groove_rejected": "", "note": f"failed: {exc}"})
            continue

        longest = max((m["length"] for m in r["marks"]), default=0)
        stem = os.path.splitext(name)[0]
        cv2.imencode(".jpg", paint(r["img"], r["det"]))[1].tofile(
            os.path.join(out, stem + "_marked.jpg"))

        note = ""
        if r["judged"] < 55:
            note = "LOW COVERAGE - most of the disc was too dark/bright to judge"
        print(f"{name[:32]:<34}{len(r['marks']):>6}{r['judged']:>9.1f}"
              f"{r['unlit']:>8.1f}{r['glare']:>8.1f}{longest:>9}"
              + (f"   << {note}" if note else ""))
        rows.append({"photo": name, "marks": len(r["marks"]),
                     "judged_pct": round(r["judged"], 1),
                     "unlit_pct": round(r["unlit"], 1),
                     "glare_pct": round(r["glare"], 1),
                     "longest_px": longest,
                     "ring_brightness": round(r["ring_mean"], 1),
                     "note": note})

    with open(os.path.join(out, "scan_summary.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}  (marked images + scan_summary.csv)")


if __name__ == "__main__":
    main()
