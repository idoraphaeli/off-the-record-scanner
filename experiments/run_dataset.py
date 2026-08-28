# -*- coding: utf-8 -*-
"""
Run the detector over Records_Data: one sub-folder = one side of one record,
holding two photos of that side.

For each pair it measures how far apart the two shots really are, aligns them,
and reports which detections appear in both. The alignment is measured in three
stages because an offset can come from three different causes, and they are not
fixed the same way:

  1. rotation  -- the record turned between shots. Measured on the printed LABEL,
                  which turns rigidly with the record and cannot be confused with
                  the lighting. (Groove texture was tried and failed: it reported
                  1.4 deg for a disc that had actually turned 79.5.)
  2. residual  -- what is left afterwards, in angle and radius. Measured by phase
                  correlation on lighting-flattened rings, sub-pixel.
  3. per-sector -- the same residual measured in 12 slices around the disc. A
                  CONSTANT residual means one global correction is enough; a
                  VARYING one means the two shots found different disc centres,
                  which a global shift cannot fix. That case is reported rather
                  than silently producing a confident-looking wrong answer.

Usage:  python run_dataset.py [path-to-Records_Data]
"""

import csv
import os
import sys

import cv2
import numpy as np

import detector
from detector import P
from multishot import analyse_photo, estimate_offset

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(os.path.dirname(HERE), "Records_Data")
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

SECTORS = 12
TOLERANCES = (5, 15, 30)      # report sensitivity instead of guessing one number
MAIN_TOL = 15
MARK_ALPHA = 0.5
MARK_HALO = 9


# ----------------------------------------------------------------- alignment
def crop_pair(x, y):
    """Trim two ring-shaped arrays to a common height.

    The two shots almost never yield exactly the same disc radius, so their ring
    bands differ by a row or two and cannot be compared directly. Both bands start
    at the same FRACTION of the radius, so trimming the taller one at the outer
    edge keeps them registered while making the shapes match.
    """
    h = min(x.shape[0], y.shape[0])
    return np.ascontiguousarray(x[:h]), np.ascontiguousarray(y[:h])


def flatten(ring):
    """Strip the lighting so correlation locks onto the record, not the lamp."""
    f = ring.astype(np.float32)
    f -= cv2.blur(f, (P["ROW_FLATTEN"], 1))    # radial gradient
    f -= cv2.blur(f, (1, 151))                 # angular gradient (the lit sector)
    return np.ascontiguousarray(f)


def phase_shift(a, b):
    """Sub-pixel (dx, dy) aligning b onto a. dx = angle, dy = radius."""
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a * win, b * win)
    return dx, dy, response


def measure_offset(a, b):
    """Full three-stage measurement. Returns a dict of everything observed."""
    width = a["ring"].shape[1]
    deg = 360.0 / P["POLAR_STEPS"]

    la, lb = crop_pair(a["label"], b["label"])
    rot_cols, sharp = estimate_offset(la, lb)
    fa, fb = crop_pair(flatten(a["ring"]),
                       flatten(np.roll(b["ring"], rot_cols, axis=1)))
    dx, dy, resp = phase_shift(fa, fb)

    step = width // SECTORS
    dxs, dys = [], []
    for s in range(SECTORS):
        sl = slice(s * step, (s + 1) * step)
        sdx, sdy, _ = phase_shift(np.ascontiguousarray(fa[:, sl]),
                                  np.ascontiguousarray(fb[:, sl]))
        dxs.append(sdx)
        dys.append(sdy)
    dxs, dys = np.array(dxs), np.array(dys)
    uniform = dxs.std() < 6 and dys.std() < 6

    return {
        "rotation_deg": rot_cols * deg,
        "rotation_sharpness": sharp,
        "residual_dx_px": dx, "residual_dy_px": dy,
        "residual_conf": resp,
        "sector_dx_spread": float(dxs.std()),
        "sector_dy_spread": float(dys.std()),
        "offset_uniform": bool(uniform),
        "total_shift_cols": int(round(rot_cols + dx)),
        "radial_shift_px": int(round(dy)),
        "radius_diff_px": b["radius"] - a["radius"],
        "centre_diff_px": float(np.hypot(b["center"][0] - a["center"][0],
                                         b["center"][1] - a["center"][1])),
    }


# ----------------------------------------------------------------- detection
def detections(shot):
    """Run extraction on both channels; return the polar mask and the marks."""
    mask_a, marks_a = detector.extract(shot["radial"])
    mask_b, marks_b = detector.extract(shot["tram"], min_len=P["TRAM_MIN_LEN"])
    return cv2.bitwise_or(mask_a, mask_b), marks_a + marks_b


def confirm(mask_a, mask_b_aligned, tol):
    """Keep components of A that have any B detection within `tol` px."""
    near_b = cv2.dilate((mask_b_aligned > 0).astype(np.uint8),
                        np.ones((tol, tol), np.uint8))
    n, labels, _, _ = cv2.connectedComponentsWithStats(
        (mask_a > 0).astype(np.uint8), connectivity=8)
    yes = np.zeros_like(mask_a)
    no = np.zeros_like(mask_a)
    n_yes = 0
    for i in range(1, n):
        sel = labels == i
        if np.count_nonzero(near_b[sel]):
            yes[sel] = 255
            n_yes += 1
        else:
            no[sel] = 255
    return yes, no, n_yes, n - 1


# ----------------------------------------------------------------- drawing
def paint(img, mask, colour, alpha=MARK_ALPHA, halo=MARK_HALO):
    vis = img.astype(np.float32)
    band = cv2.dilate((mask > 127).astype(np.uint8), np.ones((halo, halo), np.uint8))
    band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.0)
    a = np.clip(band, 0, 1)[..., None] * alpha
    return (vis * (1 - a) + np.array(colour, np.float32) * a).astype(np.uint8)


def write(path, img):
    cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(path)


# ----------------------------------------------------------------- per pair
def process(folder):
    paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(EXT)
                   and not f.startswith(("overlap", "confirmed")))
    if len(paths) < 2:
        return {"folder": os.path.basename(folder), "status": f"skipped ({len(paths)} photo)"}

    a, b = analyse_photo(paths[0]), analyse_photo(paths[1])
    off = measure_offset(a, b)

    mask_a, marks_a = detections(a)
    mask_b, marks_b = detections(b)

    # apply the measured offset: angle by rolling columns, radius by rolling rows
    aligned = np.roll(mask_b, off["total_shift_cols"], axis=1)
    if off["radial_shift_px"]:
        aligned = np.roll(aligned, off["radial_shift_px"], axis=0)
    mask_a, aligned = crop_pair(mask_a, aligned)

    counts = {}
    for tol in TOLERANCES:
        _, _, n_yes, n_tot = confirm(mask_a, aligned, tol)
        counts[tol] = n_yes
    yes, no, n_conf, n_tot = confirm(mask_a, aligned, MAIN_TOL)

    # --- overlap map: red = shot 1, green = shot 2 aligned, yellow = agreement
    k = np.ones((5, 5), np.uint8)
    overlap = np.zeros((*mask_a.shape, 3), np.uint8)
    overlap[:, :, 2] = cv2.dilate((mask_a > 0).astype(np.uint8), k) * 255
    overlap[:, :, 1] = cv2.dilate((aligned > 0).astype(np.uint8), k) * 255
    write(os.path.join(folder, "overlap.jpg"), overlap)

    # --- confirmed: shot 1, confirmed in yellow, unconfirmed in grey
    cart_yes = detector.rewrap(yes, a["inner_px"], a["center"], a["radius"],
                               a["img"].shape[:2])
    cart_no = detector.rewrap(no, a["inner_px"], a["center"], a["radius"],
                              a["img"].shape[:2])
    vis = paint(a["img"], cart_no, (140, 140, 140), alpha=0.32, halo=7)
    vis = paint(vis, cart_yes, (90, 255, 255), alpha=0.55, halo=11)
    write(os.path.join(folder, "confirmed.jpg"), vis)

    # --- text report
    lines = [
        f"folder      : {os.path.basename(folder)}",
        f"photo 1     : {os.path.basename(paths[0])}",
        f"photo 2     : {os.path.basename(paths[1])}",
        "",
        "-- geometry --",
        f"disc 1      : centre={a['center']} r={a['radius']}",
        f"disc 2      : centre={b['center']} r={b['radius']}",
        f"radius diff : {off['radius_diff_px']} px",
        f"centre diff : {off['centre_diff_px']:.1f} px",
        "",
        "-- alignment --",
        f"rotation    : {off['rotation_deg']:.1f} deg  (label sharpness {off['rotation_sharpness']:.1f})",
        f"residual    : dx={off['residual_dx_px']:+.1f} px  dy={off['residual_dy_px']:+.1f} px"
        f"  (confidence {off['residual_conf']:.3f})",
        f"sector spread: dx {off['sector_dx_spread']:.1f} px   dy {off['sector_dy_spread']:.1f} px",
        f"offset is   : {'UNIFORM - one global correction is valid'if off['offset_uniform'] else 'VARYING - the two shots found different disc centres; a global shift cannot fully fix this'}",
        "",
        "-- detections --",
        f"photo 1 alone : {len(marks_a)}",
        f"photo 2 alone : {len(marks_b)}",
        f"confirmed     : {n_conf} of {n_tot}   (tolerance {MAIN_TOL} px)",
        "",
        "confirmed vs tolerance: " + "   ".join(f"{t}px -> {c}" for t, c in counts.items()),
    ]
    with open(os.path.join(folder, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {
        "folder": os.path.basename(folder),
        "status": "ok",
        "photo1_marks": len(marks_a),
        "photo2_marks": len(marks_b),
        "confirmed": n_conf,
        "confirmed_pct": round(100.0 * n_conf / max(n_tot, 1), 1),
        "rotation_deg": round(off["rotation_deg"], 1),
        "rotation_sharpness": round(off["rotation_sharpness"], 1),
        "residual_dx_px": round(off["residual_dx_px"], 1),
        "residual_dy_px": round(off["residual_dy_px"], 1),
        "sector_dx_spread": round(off["sector_dx_spread"], 1),
        "sector_dy_spread": round(off["sector_dy_spread"], 1),
        "offset_uniform": off["offset_uniform"],
        "centre_diff_px": round(off["centre_diff_px"], 1),
        "radius_diff_px": off["radius_diff_px"],
        "conf_tol5": counts[5], "conf_tol15": counts[15], "conf_tol30": counts[30],
    }


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    subs = []
    for rec in sorted(os.listdir(root)):
        rec_path = os.path.join(root, rec)
        if not os.path.isdir(rec_path):
            continue
        for sub in sorted(os.listdir(rec_path)):
            sub_path = os.path.join(rec_path, sub)
            if os.path.isdir(sub_path):
                subs.append(sub_path)

    print(f"{len(subs)} sub-folders under {root}\n")
    print(f"{'folder':<22}{'p1':>4}{'p2':>4}{'conf':>6}{'rot':>8}{'sharp':>7}"
          f"{'dx':>7}{'dy':>7}  offset")
    print("-" * 82)

    rows = []
    for sub in subs:
        try:
            r = process(sub)
        except Exception as exc:
            print(f"{os.path.basename(sub)[:20]:<22}  FAILED: {exc}")
            rows.append({"folder": os.path.basename(sub), "status": f"failed: {exc}"})
            continue
        rows.append(r)
        if r["status"] != "ok":
            print(f"{r['folder'][:20]:<22}  {r['status']}")
            continue
        print(f"{r['folder'][:20]:<22}{r['photo1_marks']:>4}{r['photo2_marks']:>4}"
              f"{r['confirmed']:>6}{r['rotation_deg']:>8.1f}"
              f"{r['rotation_sharpness']:>7.1f}"
              f"{r['residual_dx_px']:>7.1f}{r['residual_dy_px']:>7.1f}"
              f"  {'uniform' if r['offset_uniform'] else 'VARYING'}")

    fields = ["folder", "status", "photo1_marks", "photo2_marks", "confirmed",
              "confirmed_pct", "rotation_deg", "rotation_sharpness",
              "residual_dx_px", "residual_dy_px", "sector_dx_spread",
              "sector_dy_spread", "offset_uniform", "centre_diff_px",
              "radius_diff_px", "conf_tol5", "conf_tol15", "conf_tol30"]
    out = os.path.join(root, "summary.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("status") == "ok"]
    if ok:
        p1 = sum(r["photo1_marks"] for r in ok)
        cf = sum(r["confirmed"] for r in ok)
        varying = sum(1 for r in ok if not r["offset_uniform"])
        print("-" * 74)
        print(f"{len(ok)} pairs processed")
        print(f"detections in photo 1: {p1}   confirmed by photo 2: {cf}"
              f"   ({100.0 * cf / max(p1, 1):.0f}% survived)")
        print(f"pairs with a VARYING offset (different disc centres): {varying}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
