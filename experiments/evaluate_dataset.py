# -*- coding: utf-8 -*-
"""
Score the detector against the hand-marked copies in Records_Data.

Each photo has a companion "<name> - עותק.jpeg" with scratches circled in pen.
The marks are recovered as the pixel difference between the two files, which is
exact — colour thresholding would also pick up blue elsewhere in the frame.

Two numbers come out, and they answer different questions:

  recall    — of the scratches the annotator could see, how many did we find
  extra     — detections that fall outside every marked zone

`extra` is deliberately NOT called "false positives". The annotator marked what
was visible to the naked eye, so a detection outside a zone may be a real
scratch they missed. Those need a human verdict; the script only counts and
locates them.

Usage:  python evaluate_dataset.py [path-to-Records_Data]
"""

import csv
import os
import sys

import cv2
import numpy as np

import detector
from detector import P

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(os.path.dirname(HERE), "Records_Data")
EXT = (".jpg", ".jpeg", ".png")
GENERATED = ("overlap", "confirmed", "judge_", "eval_")
COPY_HINT = "עותק"

DIFF_THRESHOLD = 25     # per-pixel |marked-original| above this is pen
STROKE_CLOSE = 11       # bridge small gaps so a loop closes before filling
MIN_ZONE_AREA = 300
MIN_HIT_PX = 20         # detection pixels inside a zone for it to count as found
MIN_EXTRA_AREA = 50     # smaller stray detections are ignored as speck-level


# ------------------------------------------------------------------ ground truth
def zones_from_pair(original, marked):
    """Filled regions enclosed by the pen loops, in original-image coordinates."""
    if original.shape != marked.shape:
        dh = original.shape[0] - marked.shape[0]
        dw = original.shape[1] - marked.shape[1]
        if not (0 <= dh <= 2 and 0 <= dw <= 2):
            return None, "size mismatch", 0
        original = original[:marked.shape[0], :marked.shape[1]]

    diff = cv2.absdiff(original, marked).max(axis=2)
    pen_px = int(np.count_nonzero(diff > DIFF_THRESHOLD))
    stroke = (diff > DIFF_THRESHOLD).astype(np.uint8) * 255
    stroke = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE,
                              np.ones((STROKE_CLOSE, STROKE_CLOSE), np.uint8))

    h, w = stroke.shape
    ff = stroke.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    interior = cv2.bitwise_not(ff) & cv2.bitwise_not(stroke)
    regions = cv2.bitwise_or(interior, stroke)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(regions, connectivity=8)
    mask = np.zeros_like(regions)
    kept = 0
    for i in range(1, n):
        if stats[i][4] < MIN_ZONE_AREA:
            continue
        mask[labels == i] = 255
        kept += 1
    return mask, kept, pen_px


# ------------------------------------------------------------------ detection
def detect(path):
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner_px, outer_px = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner_px:outer_px]
    radial, tram = detector.scratch_map(ring)
    m1, k1 = detector.extract(radial)
    m2, k2 = detector.extract(tram, min_len=P["TRAM_MIN_LEN"])
    ring_mask = cv2.bitwise_or(m1, m2)
    det = detector.rewrap(ring_mask, inner_px, center, radius, gray.shape)
    return img, det, k1 + k2


def score(det, gt):
    """Zones found, and detection components lying outside every zone."""
    det_b = (det > 127).astype(np.uint8)
    gt_b = (gt > 127).astype(np.uint8)

    n, labels, _, _ = cv2.connectedComponentsWithStats(gt_b, connectivity=8)
    found = sum(1 for i in range(1, n)
                if np.count_nonzero(det_b[labels == i]) >= MIN_HIT_PX)

    outside = det_b & (gt_b == 0)
    nd, dl, ds, dc = cv2.connectedComponentsWithStats(outside, connectivity=8)
    extra = []
    for i in range(1, nd):
        x, y, w, h, area = ds[i]
        if area < MIN_EXTRA_AREA:
            continue
        # a component that mostly sits inside a zone but spills out is not extra
        if np.count_nonzero(det_b[y:y + h, x:x + w] & gt_b[y:y + h, x:x + w]) >= MIN_HIT_PX:
            continue
        extra.append((int(dc[i][0]), int(dc[i][1]), int(area)))
    return found, n - 1, extra


# ------------------------------------------------------------------ drawing
def overlay(img, det, gt, extra):
    vis = img.astype(np.float32)
    band = cv2.dilate((det > 127).astype(np.uint8), np.ones((9, 9), np.uint8))
    band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.0)
    a = np.clip(band, 0, 1)[..., None] * 0.5
    vis = (vis * (1 - a) + np.array([90, 255, 255], np.float32) * a).astype(np.uint8)

    cs, _ = cv2.findContours((gt > 127).astype(np.uint8),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cs, -1, (80, 220, 80), 3)
    for cx, cy, _ in extra:
        cv2.circle(vis, (cx, cy), 34, (60, 60, 255), 3, cv2.LINE_AA)
    return vis


# ------------------------------------------------------------------ main
def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    rows = []

    print(f"{'photo':<46}{'found':>7}{'zones':>7}{'extra':>7}  note")
    print("-" * 78)

    for rec in sorted(os.listdir(root)):
        rp = os.path.join(root, rec)
        if not os.path.isdir(rp):
            continue
        for sub in sorted(os.listdir(rp)):
            sp = os.path.join(rp, sub)
            if not os.path.isdir(sp):
                continue
            files = [f for f in sorted(os.listdir(sp))
                     if f.lower().endswith(EXT) and not f.startswith(GENERATED)]
            originals = [f for f in files if COPY_HINT not in f]
            copies = [f for f in files if COPY_HINT in f]

            for orig in originals:
                stem = os.path.splitext(orig)[0]
                match = next((c for c in copies if stem in c), None)
                short = f"{sub}/{orig}"[:44]
                if match is None:
                    print(f"{short:<46}{'-':>7}{'-':>7}{'-':>7}  no marked copy")
                    rows.append({"folder": sub, "photo": orig, "status": "no marked copy"})
                    continue

                o = cv2.imdecode(np.fromfile(os.path.join(sp, orig), np.uint8),
                                 cv2.IMREAD_COLOR)
                m = cv2.imdecode(np.fromfile(os.path.join(sp, match), np.uint8),
                                 cv2.IMREAD_COLOR)
                gt_full, kept, pen_px = zones_from_pair(o, m)
                if gt_full is None:
                    print(f"{short:<46}{'-':>7}{'-':>7}{'-':>7}  {kept}")
                    rows.append({"folder": sub, "photo": orig, "status": kept})
                    continue

                img, det, marks = detect(os.path.join(sp, orig))
                gt = cv2.resize(gt_full, (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
                found, zones, extra = score(det, gt)

                out = os.path.join(sp, "eval_" + stem[:40] + ".jpg")
                cv2.imencode(".jpg", overlay(img, det, gt, extra),
                             [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(out)

                # distinguish "annotator marked nothing" from "the pen was not found"
                if zones:
                    note = ""
                elif pen_px < 200:
                    note = "copy is identical - nothing was drawn"
                else:
                    note = f"pen found ({pen_px} px) but no closed zone"
                print(f"{short:<46}{found:>7}{zones:>7}{len(extra):>7}  {note}")
                rows.append({"folder": sub, "photo": orig, "status": "ok",
                             "zones": zones, "found": found,
                             "recall_pct": round(100.0 * found / zones, 1) if zones else "",
                             "extra": len(extra), "detections_total": len(marks),
                             "pen_px": pen_px})

    ok = [r for r in rows if r.get("status") == "ok"]
    tz = sum(r["zones"] for r in ok)
    tf = sum(r["found"] for r in ok)
    te = sum(r["extra"] for r in ok)
    td = sum(r["detections_total"] for r in ok)

    print("-" * 78)
    print(f"{len(ok)} photos scored")
    print(f"  scratches marked by you : {tz}")
    print(f"  of those, model found   : {tf}   ({100.0 * tf / max(tz,1):.0f}% recall)")
    print(f"  detections outside zones: {te}   ({te / max(len(ok),1):.1f} per photo)")
    print(f"  total detections shown  : {td}")
    if td:
        print(f"  -> {100.0 * (td - te) / td:.0f}% of what the model shows lands on a marked scratch")
    print("\n  'outside' is not proven wrong — you marked what you could see by eye.")
    print("  Open the eval_*.jpg files: green outline = your marks, yellow = model,")
    print("  red circle = a detection you did not mark.")

    out = os.path.join(root, "evaluation.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "photo", "status", "zones",
                                          "found", "recall_pct", "extra",
                                          "detections_total", "pen_px"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
