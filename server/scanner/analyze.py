# -*- coding: utf-8 -*-
"""High-level analysis: bytes in, a result dict out."""

import base64
import time

import cv2
import numpy as np

from . import detector
from .detector import P

MARK_ALPHA = 0.45     # translucent, so the scratch stays readable under the mark
MARK_HALO = 9
LOW_COVERAGE = 55.0   # % below which the photo has not really been assessed

# Damage index -> Goldmine band. UNCALIBRATED: these cut-points are reasonable
# but were never fitted to human-graded records, so the grade is a suggestion,
# not a measurement. Calibrating them needs a set of records graded by a person.
GRADE_BANDS = ((0.5, "Near Mint (NM)"), (2.0, "Very Good Plus (VG+)"),
               (6.0, "Very Good (VG)"), (15.0, "Good Plus (G+)"),
               (30.0, "Good (G)"), (60.0, "Fair (F)"))
WORST_GRADE = "Poor (P)"


def _paint(img, det_mask):
    vis = img.copy().astype(np.float32)
    band = cv2.dilate((det_mask > 127).astype(np.uint8),
                      np.ones((MARK_HALO, MARK_HALO), np.uint8))
    band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.0)
    a = np.clip(band, 0, 1)[..., None] * MARK_ALPHA
    return (vis * (1 - a) + np.array([90, 255, 255], np.float32) * a).astype(np.uint8)


def _grade(marks, ring_area):
    """Severity weights length and thickness rather than counting pixels: one
    long deep gouge matters more than many hairlines covering the same area."""
    if not marks:
        return "Near Mint (NM)", 0.0
    severity = sum(m["length_px"] * max(m["thickness_px"], 1.0) for m in marks)
    index = 1000.0 * severity / max(ring_area, 1)
    for limit, name in GRADE_BANDS:
        if index <= limit:
            return name, round(index, 2)
    return WORST_GRADE, round(index, 2)


def analyze(image_bytes, want_overlay=True):
    started = time.time()
    img = detector.decode_image(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius, how = detector.find_disc(img)

    inner_px = int(P["LABEL_R"] * radius)
    outer_px = int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner_px:outer_px]

    radial, tram, dead = detector.scratch_map(ring)
    mask_a, marks_a = detector.extract(radial)
    mask_b, marks_b = detector.extract(tram, min_len=P["TRAM_MIN_LEN"])
    ring_mask = cv2.bitwise_or(mask_a, mask_b)
    marks = marks_a + marks_b

    # report each mark's position on the RECORD, not in the photo, so it stays
    # meaningful across shots taken at different angles
    for m in marks:
        m["radius_frac"] = round((inner_px + m.pop("polar_row")) / radius, 3)
        m["angle_deg"] = round(360.0 * m.pop("polar_col") / P["POLAR_STEPS"], 1)

    judged_pct = 100.0 * float((~dead).mean())
    glare_pct = 100.0 * float((detector.glare_mask(ring) > 0).mean())
    unlit_pct = 100.0 * float((detector.unlit_mask(ring) > 0).mean())
    grade, index = _grade(marks, ring.shape[0] * ring.shape[1])

    warnings = []
    if judged_pct < LOW_COVERAGE:
        warnings.append(
            f"only {judged_pct:.0f}% of the playing surface could be assessed; "
            "most of it was too dark or too blown-out. Re-shoot with one lamp at "
            "a low angle in a dim room.")
    if how == "fallback_centered":
        warnings.append("the record outline was not found; results are unreliable. "
                        "Shoot the disc alone on a plain dark surface.")

    result = {
        "marks": marks,
        "mark_count": len(marks),
        "grade": grade,
        "damage_index": index,
        "grade_is_calibrated": False,
        "coverage": {"judged_pct": round(judged_pct, 1),
                     "unlit_pct": round(unlit_pct, 1),
                     "glare_pct": round(glare_pct, 1)},
        "disc": {"center_x": center[0], "center_y": center[1],
                 "radius_px": radius, "found_by": how},
        "warnings": warnings,
        "elapsed_ms": int(1000 * (time.time() - started)),
    }

    if want_overlay:
        det = detector.rewrap(ring_mask, inner_px, center, radius, gray.shape)
        ok, buf = cv2.imencode(".jpg", _paint(img, det),
                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            result["overlay_png"] = ("data:image/jpeg;base64,"
                                     + base64.b64encode(buf).decode("ascii"))
    return result
