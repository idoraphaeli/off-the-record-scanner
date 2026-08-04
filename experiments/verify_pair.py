# -*- coding: utf-8 -*-
"""
Two-shot verification, step by step and visible at every stage.

  1. run the detector on each photo separately, mark its findings in
     translucent yellow on the original
  2. unwrap both to polar and align them (anchored on the printed label, which
     turns with the record and cannot be confused with the lighting)
  3. a mark counts as CERTAIN when it appears in both shots at the same place on
     the record, within a tolerance -- a scratch is fixed to the vinyl, while a
     reflection sits wherever the lamp is and moves when the record turns

Usage: python verify_pair.py <folder> [tolerance_px]
"""

import os
import sys

import cv2
import numpy as np

import detector
from detector import P
from multishot import label_strip, estimate_offset

TOL = 25          # px: how far apart the same scratch may land in the two shots
MIN_SHARP = 3.0


def detect_polar(path):
    """Run the normal detector but keep the result in polar coordinates."""
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    inner_px, outer_px = int(P["LABEL_R"] * radius), int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner_px:outer_px]
    radial, tram = detector.scratch_map(ring)
    mask_a, sa = detector.extract(radial)
    mask_b, sb = detector.extract(tram, min_len=P["TRAM_MIN_LEN"])
    mask = cv2.bitwise_or(mask_a, mask_b)
    return dict(path=path, img=img, center=center, radius=radius, inner_px=inner_px,
                ring=ring, mask=mask, marks=sa + sb,
                label=label_strip(gray, center, radius))


def paint(img, det_cart, color=(90, 255, 255), alpha=0.45, halo=9):
    """Translucent highlight: wide enough to find, sheer enough to still see the
    scratch through it."""
    vis = img.copy().astype(np.float32)
    band = cv2.dilate((det_cart > 127).astype(np.uint8), np.ones((halo, halo), np.uint8))
    band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.0)
    a = np.clip(band, 0, 1)[..., None] * alpha
    return (vis * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


def main():
    folder = sys.argv[1]
    tol = int(sys.argv[2]) if len(sys.argv) > 2 else TOL
    out = os.path.join(folder, "verify")
    os.makedirs(out, exist_ok=True)

    paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if len(paths) != 2:
        raise SystemExit(f"need exactly 2 photos, found {len(paths)}")

    # --- step 1: detect on each shot on its own ---
    print("STEP 1 - detect on each photo separately")
    shots = []
    for i, p in enumerate(paths, 1):
        s = detect_polar(p)
        det = detector.rewrap(s["mask"], s["inner_px"], s["center"], s["radius"],
                              s["img"].shape[:2])
        cv2.imencode(".jpg", paint(s["img"], det))[1].tofile(
            os.path.join(out, f"step1_shot{i}_detections.jpg"))
        print(f"  shot {i}: {len(s['marks'])} marks  ({os.path.basename(p)[-22:]})")
        shots.append(s)

    a, b = shots
    width = a["ring"].shape[1]

    # --- step 2: align the two unwrapped shots ---
    print("\nSTEP 2 - align the two unwrapped shots")
    shift, sharp = estimate_offset(a["label"], b["label"])
    print(f"  offset {360.0*shift/width:.1f} deg   sharpness {sharp:.1f}"
          f"{'   << WEAK, verdict unreliable' if sharp < MIN_SHARP else ''}")
    sep = np.full((8, width), 255, np.uint8)
    cv2.imencode(".jpg", np.vstack([a["ring"], sep, np.roll(b["ring"], shift, axis=1)]))[
        1].tofile(os.path.join(out, "step2_rings_aligned.jpg"))

    # --- step 3: which marks appear in BOTH ---
    print(f"\nSTEP 3 - agreement between the shots (tolerance {tol}px)")
    b_mask = np.roll(b["mask"], shift, axis=1)
    b_near = cv2.dilate((b_mask > 0).astype(np.uint8), np.ones((tol, tol), np.uint8))
    a_near = cv2.dilate((a["mask"] > 0).astype(np.uint8), np.ones((tol, tol), np.uint8))

    certain = np.zeros_like(a["mask"])
    single = np.zeros_like(a["mask"])
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (a["mask"] > 0).astype(np.uint8), connectivity=8)
    n_certain = 0
    for i in range(1, n):
        sel = labels == i
        if np.count_nonzero(b_near[sel]) > 0:
            certain[sel] = 255
            n_certain += 1
        else:
            single[sel] = 255
    # marks seen only by shot B still deserve to be reported as unconfirmed
    nb, lb, sb_stats, _ = cv2.connectedComponentsWithStats(
        (b_mask > 0).astype(np.uint8), connectivity=8)
    for i in range(1, nb):
        sel = lb == i
        if np.count_nonzero(a_near[sel]) == 0:
            single[sel] = 255

    print(f"  CERTAIN (in both shots): {n_certain}")
    print(f"  unconfirmed (one shot only): "
          f"{cv2.connectedComponents(((single > 0).astype(np.uint8)))[0] - 1}")

    cert_cart = detector.rewrap(certain, a["inner_px"], a["center"], a["radius"],
                                a["img"].shape[:2])
    sing_cart = detector.rewrap(single, a["inner_px"], a["center"], a["radius"],
                                a["img"].shape[:2])
    vis = paint(a["img"], sing_cart, color=(120, 120, 120), alpha=0.30, halo=7)
    vis = paint(vis, cert_cart, color=(90, 255, 255), alpha=0.55, halo=11)
    cv2.imencode(".jpg", vis)[1].tofile(os.path.join(out, "step3_certain.jpg"))

    overlay = np.zeros((*a["mask"].shape, 3), np.uint8)
    k = np.ones((5, 5), np.uint8)
    overlay[:, :, 2] = cv2.dilate((a["mask"] > 0).astype(np.uint8), k) * 255
    overlay[:, :, 1] = cv2.dilate((b_mask > 0).astype(np.uint8), k) * 255
    cv2.imencode(".jpg", overlay)[1].tofile(os.path.join(out, "step3_overlap.jpg"))
    print(f"\n  wrote {out}")
    print("    step1_shot1/2_detections.jpg  what each photo found on its own")
    print("    step2_rings_aligned.jpg       the two unwrapped discs, aligned")
    print("    step3_certain.jpg             yellow = certain, grey = one shot only")
    print("    step3_overlap.jpg             red = shot A, green = shot B, yellow = both")


if __name__ == "__main__":
    main()
