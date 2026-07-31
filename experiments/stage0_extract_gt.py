# -*- coding: utf-8 -*-
"""
Stage 0a — Ground-truth extraction (run once, then frozen).

The marked photos are byte-copies of the clean photos with blue pen loops drawn
around visible scratches. Since the underlying photo is identical, the marks are
recovered EXACTLY as the per-pixel difference between the two files — far more
robust than color thresholding (backgrounds/labels can also contain blue).

Each closed pen loop is filled; every filled region = one ground-truth scratch
zone. Outputs, per marked pair:
  gt/<name>_mask.png     binary mask of all scratch zones (original resolution)
  gt/<name>_overlay.jpg  green zones over the clean photo, for human approval
Plus gt_summary.json with per-image region counts, and pairs_report.txt.
"""

import json
import os

import cv2
import numpy as np

BASE = r"C:\Users\User\Desktop\Ido private\Computer science\off-the-record\record_pics"


def _locate(substrings):
    # folder names may carry invisible RTL marks -- match by substring, not equality
    for d in os.listdir(BASE):
        full = os.path.join(BASE, d)
        if os.path.isdir(full) and all(s in d for s in substrings):
            return full
    raise SystemExit(f"folder not found: {substrings}")


CLEAN_DIR = _locate(["ללא סימונים", "חדש"])
MARKED_DIR = _locate(["עם סימון", "חדש"])
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt")

DIFF_THRESHOLD = 25    # per-pixel |marked-clean| above this = pen stroke
STROKE_CLOSE = 11      # px: bridge small gaps so loops close before filling


def imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def extract_regions(clean, marked):
    diff = cv2.absdiff(clean, marked).max(axis=2)
    stroke = (diff > DIFF_THRESHOLD).astype(np.uint8) * 255
    stroke = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE,
                              np.ones((STROKE_CLOSE, STROKE_CLOSE), np.uint8))

    # fill the interiors of the closed loops: flood from the border, anything
    # the flood can't reach and isn't stroke = enclosed interior
    h, w = stroke.shape
    ff = stroke.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ff_mask, (0, 0), 255)
    interior = cv2.bitwise_not(ff) & cv2.bitwise_not(stroke)

    regions = cv2.bitwise_or(interior, stroke)  # zone = loop + its interior
    # drop tiny specks (JPEG noise along the diff)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(regions, connectivity=8)
    mask = np.zeros_like(regions)
    kept, warn = 0, 0
    for i in range(1, n):
        area = stats[i][4]
        if area < 300:
            continue
        comp_interior = interior[labels == i]
        if np.count_nonzero(comp_interior) == 0:
            warn += 1   # a stroke with no interior = unclosed loop, still kept
        mask[labels == i] = 255
        kept += 1
    return mask, kept, warn


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary, report = {}, []
    for name in sorted(os.listdir(CLEAN_DIR)):
        cp, mp = os.path.join(CLEAN_DIR, name), os.path.join(MARKED_DIR, name)
        if not os.path.exists(mp):
            report.append(f"UNPAIRED: {name}")
            continue
        clean, marked = imread(cp), imread(mp)
        if clean is None or marked is None:
            report.append(f"BAD PAIR (unreadable): {name}")
            continue
        pad = None   # (top, left) offset of the cropped window inside the clean image
        if clean.shape != marked.shape:
            dh = clean.shape[0] - marked.shape[0]
            dw = clean.shape[1] - marked.shape[1]
            if 0 <= dh <= 2 and 0 <= dw <= 2:
                # the editor shaved a pixel or two off the marked copy. Resampling
                # would smear noise everywhere; instead find WHICH rows/cols were
                # shaved by trying every crop offset and keeping the best match.
                best = None
                mh, mw = marked.shape[:2]
                for oy in range(dh + 1):
                    for ox in range(dw + 1):
                        crop = clean[oy:oy + mh, ox:ox + mw]
                        score = float(cv2.absdiff(crop, marked).mean())
                        if best is None or score < best[0]:
                            best = (score, oy, ox)
                _, oy, ox = best
                clean_win = clean[oy:oy + mh, ox:ox + mw]
                pad = (oy, ox)
                report.append(f"ALIGNED BY CROP (offset y={oy}, x={ox}): {name}")
            else:
                report.append(f"BAD PAIR (size mismatch {clean.shape} vs {marked.shape}): {name}")
                continue
        else:
            clean_win = clean
        if np.array_equal(clean_win, marked):
            summary[name] = {"regions": 0, "marked": False}
            report.append(f"NOT MARKED (identical files): {name}")
            continue

        mask, kept, warn = extract_regions(clean_win, marked)
        if pad is not None:
            # place the mask back into full clean-image coordinates
            full = np.zeros(clean.shape[:2], np.uint8)
            full[pad[0]:pad[0] + mask.shape[0], pad[1]:pad[1] + mask.shape[1]] = mask
            mask = full
        summary[name] = {"regions": kept, "marked": True, "unclosed_loops": warn}
        report.append(f"OK: {name} -> {kept} scratch zones"
                      + (f" ({warn} unclosed loops)" if warn else ""))

        stem = os.path.splitext(name)[0]
        cv2.imencode(".png", mask)[1].tofile(os.path.join(OUT_DIR, stem + "_mask.png"))
        overlay = clean.copy()
        zone = mask > 0
        overlay[zone] = (0.4 * overlay[zone] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.imencode(".jpg", overlay)[1].tofile(os.path.join(OUT_DIR, stem + "_overlay.jpg"))

    with open(os.path.join(OUT_DIR, "gt_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "pairs_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
