# -*- coding: utf-8 -*-
"""
Produce sheets for judging detections by hand.

The point is to measure precision — what fraction of what the model shows is
really a scratch — without depending on the two-shot cross-check, which we know
is currently mis-aligning and therefore cannot be used as a measuring stick.

Per sub-folder, using the FIRST photo only (so the job is one sheet per side):

  judge_overview.jpg  the whole disc, every detection highlighted and NUMBERED
  judge_crops.jpg     a contact sheet: each numbered detection zoomed in, so a
                      hairline can actually be seen rather than guessed at
  judge_marks.csv     one row per number, with its measurements

You then send back, per folder, the numbers that are NOT scratches.

Usage:  python make_annotation_sheets.py [path-to-Records_Data]
"""

import csv
import os
import sys

import cv2
import numpy as np

import detector
from detector import P
from multishot import analyse_photo

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(os.path.dirname(HERE), "Records_Data")
EXT = (".jpg", ".jpeg", ".png")
SKIP = ("overlap", "confirmed", "judge_")

CROP = 220          # px around each mark in the contact sheet
COLS = 5            # crops per row
CELL = 260          # rendered cell size
MARK_ALPHA = 0.42


def paint(img, mask, colour=(90, 255, 255), alpha=MARK_ALPHA, halo=9):
    vis = img.astype(np.float32)
    band = cv2.dilate((mask > 127).astype(np.uint8), np.ones((halo, halo), np.uint8))
    band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.0)
    a = np.clip(band, 0, 1)[..., None] * alpha
    return (vis * (1 - a) + np.array(colour, np.float32) * a).astype(np.uint8)


def label_at(img, pt, n, taken):
    """Draw a numbered badge near a point, nudged away from badges already placed
    so two close detections stay separately identifiable."""
    x, y = pt
    for dx, dy in ((46, -46), (-46, -46), (46, 46), (-46, 46), (0, -66), (0, 66)):
        cx, cy = x + dx, y + dy
        if all((cx - tx) ** 2 + (cy - ty) ** 2 > 62 ** 2 for tx, ty in taken):
            break
    cx = int(np.clip(cx, 30, img.shape[1] - 30))
    cy = int(np.clip(cy, 30, img.shape[0] - 30))
    taken.append((cx, cy))
    cv2.line(img, (x, y), (cx, cy), (30, 30, 30), 3, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 24, (25, 25, 25), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 24, (120, 255, 255), 3, cv2.LINE_AA)
    txt = str(n)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.putText(img, txt, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


def contact_sheet(img, boxes, folder_name):
    """Zoomed crops of every detection, numbered to match the overview."""
    if not boxes:
        return None
    rows = (len(boxes) + COLS - 1) // COLS
    sheet = np.full((rows * CELL + 46, COLS * CELL, 3), 24, np.uint8)
    cv2.putText(sheet, folder_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (200, 210, 220), 2, cv2.LINE_AA)

    for i, (n, cx, cy) in enumerate(boxes):
        half = CROP // 2
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(img.shape[1], cx + half), min(img.shape[0], cy + half)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (CELL - 12, CELL - 12), interpolation=cv2.INTER_CUBIC)
        r, c = divmod(i, COLS)
        oy, ox = r * CELL + 46, c * CELL
        sheet[oy + 6:oy + CELL - 6, ox + 6:ox + CELL - 6] = crop
        cv2.circle(sheet, (ox + 30, oy + 30), 20, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(sheet, (ox + 30, oy + 30), 20, (120, 255, 255), 2, cv2.LINE_AA)
        t = str(n)
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(sheet, t, (ox + 30 - tw // 2, oy + 30 + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return sheet


def process(folder):
    paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(EXT) and not f.startswith(SKIP))
    if not paths:
        return None
    shot = analyse_photo(paths[0])

    mask_a, marks_a = detector.extract(shot["radial"])
    mask_b, marks_b = detector.extract(shot["tram"], min_len=P["TRAM_MIN_LEN"])
    ring_mask = cv2.bitwise_or(mask_a, mask_b)

    det = detector.rewrap(ring_mask, shot["inner_px"], shot["center"],
                          shot["radius"], shot["img"].shape[:2])
    vis = paint(shot["img"], det)

    # number each detection in the ORIGINAL photo's coordinates
    n, labels, stats, cent = cv2.connectedComponentsWithStats(
        (det > 127).astype(np.uint8), connectivity=8)
    order = sorted((i for i in range(1, n) if stats[i][4] >= 25),
                   key=lambda i: (cent[i][1], cent[i][0]))   # top-to-bottom

    taken, boxes, rows = [], [], []
    clean = shot["img"].copy()
    for k, i in enumerate(order, 1):
        cx, cy = int(cent[i][0]), int(cent[i][1])
        label_at(vis, (cx, cy), k, taken)
        boxes.append((k, cx, cy))
        x, y, w, h, area = stats[i]
        rows.append({"mark": k, "x": cx, "y": cy, "area_px": int(area),
                     "bbox_w": int(w), "bbox_h": int(h)})

    write = lambda p, im: cv2.imencode(".jpg", im,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(p)
    write(os.path.join(folder, "judge_overview.jpg"), vis)

    sheet = contact_sheet(paint(clean, det, alpha=0.30, halo=7), boxes,
                          os.path.basename(folder))
    if sheet is not None:
        write(os.path.join(folder, "judge_crops.jpg"), sheet)

    with open(os.path.join(folder, "judge_marks.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["mark", "x", "y", "area_px",
                                          "bbox_w", "bbox_h"])
        w.writeheader()
        w.writerows(rows)

    return {"folder": os.path.basename(folder), "photo": os.path.basename(paths[0]),
            "marks": len(rows)}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    results = []
    for rec in sorted(os.listdir(root)):
        rp = os.path.join(root, rec)
        if not os.path.isdir(rp):
            continue
        for sub in sorted(os.listdir(rp)):
            sp = os.path.join(rp, sub)
            if not os.path.isdir(sp):
                continue
            try:
                r = process(sp)
            except Exception as exc:
                print(f"{sub:<22} FAILED: {exc}")
                continue
            if r:
                results.append(r)
                print(f"{r['folder']:<22} {r['marks']:>3} marks")

    total = sum(r["marks"] for r in results)
    print(f"\n{len(results)} sheets, {total} marks to judge in total")
    print("in each folder: judge_overview.jpg, judge_crops.jpg, judge_marks.csv")


if __name__ == "__main__":
    main()
