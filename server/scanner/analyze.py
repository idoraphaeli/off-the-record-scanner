# -*- coding: utf-8 -*-
"""High-level analysis: photographs in, marked photographs and a grade out.

A record is graded from four photographs — two of each side, the disc tilted
differently between the two so the lamp rakes across it from a different angle.
The two shots of a side are analysed together: marks appearing in both are far
more likely to be real damage than a reflection, and a mark the lamp missed in
one shot is still on the record, so both shots contribute.

    analyze_record   four photographs -> four marked photographs, a grade per
                     side, and a grade for the record, which is the worse side
    analyze          one side on its own, with an optional second shot

Why the record takes the worse of its two sides: a buyer plays both. A side that
skips is not redeemed by the other side being clean, so the sides are not
averaged.
"""

import base64
import ctypes
import gc
import time

import cv2
import numpy as np

from . import crossshot, detector
from .detector import P

MARK_ALPHA = 0.45     # translucent, so the scratch stays readable under the mark
MARK_HALO = 9
LOW_COVERAGE = 55.0   # % below which the photo has not really been assessed


def _release_memory():
    """Hand freed memory back to the operating system, not just to Python.

    Python frees a large array to its allocator, and glibc keeps the pages in
    case they are wanted again. On a desktop that is a sensible trade. In a
    512 MB container it is fatal: measured, one analysis request succeeded and
    the next was killed mid-flight, because the first request's arrays were
    still counted against the limit even though nothing referenced them.

    malloc_trim asks glibc to return what it is sitting on. It exists only
    there, so anywhere else this is a no-op and the caller carries on.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass          # not glibc (Windows, macOS, musl) — nothing to trim

# ---------------------------------------------------------------------------
# How much each mark counts towards the grade.
#
# The detector reports every bright thin thing it finds, and those are not
# equally bad for a listener. Three separate questions decide the weight, and
# each is answered by something measurable.
# ---------------------------------------------------------------------------

# 1. IS IT A CUT OR A LUMP? Dirt sits on the surface and is a blob; a scratch is
#    cut into it and is a line. Measured over 1062 hand-labelled marks
#    (reflections excluded): elongation — length over thickness — separates the
#    two better than anything else tried, median 7.1 for scratches against 4.4
#    for dirt. It is a soft signal, not a classifier: a cut at elongation 3.9
#    keeps 94% of scratches but also 58% of dirt. That is exactly why it is used
#    as a WEIGHT and never as a decision — no mark is discarded for looking like
#    dirt, it simply counts for less.
DIRT_ELONG = 4.0       # measured: dirt's median. At or below, weigh as dirt.
CUT_ELONG = 8.0        # measured: just above the scratch median. Weigh in full.
DIRT_WEIGHT = 0.30     # A CHOICE, not a measurement. Dirt is largely cleanable,
                       # so it should count for something but not for much.

# 2. HOW MANY TRACKS DOES IT CROSS? A record is a single spiral groove, so a
#    mark reaching across the disc puts a tick into every track it passes, while
#    a mark of the same area sitting between two grooves spoils one. Area alone
#    cannot tell those apart, so the radial reach is counted a second time on
#    purpose.
SPAN_GAIN = 2.0        # A CHOICE: a mark spanning the whole playing surface
                       # weighs three times one that spans almost none of it.

# 3. IS IT REALLY THERE? Measured over the hand-labelled sets: 89% of marks
#    confirmed in the other shot of the same side were real, against 71% of
#    marks seen in one shot only, against 77% overall. The weights are those
#    reliabilities divided by the overall figure, so a lone photograph — all the
#    scanner could work from until now — keeps a weight of exactly 1 and its
#    grades stay on the same scale.
#
#    Note what this does NOT do: confirmation says a mark is really there, not
#    that it is a scratch rather than dirt. Measured, it does not separate those
#    two at all. Question 1 is what handles that.
CONF_SEEN_TWICE = 1.16     # 0.89 / 0.77
CONF_SEEN_ONCE = 0.92      # 0.71 / 0.77 — the other shot was checked and did
                           # not show it
CONF_NO_SECOND = 1.00      # nothing to check against: no evidence either way

# ---------------------------------------------------------------------------
# From damage to a grade, in two steps.
#
# The bands are the published Goldmine scale and are not ours to invent, so the
# scanner reports a quality score out of 100 and reads the grade off that.
# ---------------------------------------------------------------------------
GRADE_BANDS = ((98, "M (Mint)"),
               (90, "NM / M- (Near Mint)"),
               (80, "VG+ (Very Good Plus)"),
               (65, "VG (Very Good)"),
               (50, "G / G+ (Good / Good Plus)"),
               (0,  "P / F (Poor / Fair)"))

# The one number still chosen rather than measured: how much damage costs half
# the score. The shape is deliberate — a score that halves at a fixed amount of
# damage never runs off the bottom, so a badly marked record degrades rather
# than falling off a cliff.
#
# Set by two measurements over the whole collection, in that order.
#
# First, what a believable shelf looks like. Grading all 198 photographs one at
# a time and asking where a given value puts them: too low and two sides in five
# land in Good, too high and one in seven reaches Near Mint. Neither is credible
# for played second-hand LPs, which should sit mostly in VG and VG+ with a short
# tail either way.
#
# Second, and this is what fixes the value: A RECORD MUST NOT CHANGE GRADE
# BECAUSE IT WAS PHOTOGRAPHED TWICE. Two shots of a side find marks the one shot
# missed, so the same physical record scores a higher damage index — measured
# over 95 sides, the median index rises from 7.6 on one photograph to 9.7 on two.
# Left alone that would drop honest records a whole band for no reason but extra
# evidence. So this number is set to hold the median score steady across the two
# routes: one photograph at 25 gives a median of 77%, two photographs at 35 give
# 78%, and the band distribution comes out 2% Near Mint, 33% VG+, 61% VG, 4%
# Good.
#
# This is evidence, not calibration. That the collection's shape is right is no
# proof the absolute placement is, and the honest fix is unchanged: grade 20-30
# records by eye, see what index each gets, and pin this number to that.
SCORE_HALF_AT = 35.0


def _paint(img, det_mask, keep_inside=None):
    """Colour the marks onto the photograph.

    The highlight is deliberately drawn wider than the mark it covers, so that a
    hairline is visible at all. That halo has to be clipped too: a mark lying
    against the rim would otherwise paint over the sleeve, and a colour on the
    sleeve reads as the model claiming damage there.
    """
    vis = img.copy().astype(np.float32)
    band = cv2.dilate((det_mask > 127).astype(np.uint8),
                      np.ones((MARK_HALO, MARK_HALO), np.uint8))
    band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.0)
    if keep_inside is not None:
        band = band * (keep_inside > 0)
    a = np.clip(band, 0, 1)[..., None] * MARK_ALPHA
    return (vis * (1 - a) + np.array([90, 255, 255], np.float32) * a).astype(np.uint8)


def _overlay(img, ring_mask, inner_px, center, radius, shape):
    """The photograph with every mark the model found painted onto it."""
    det = detector.rewrap(ring_mask, inner_px, center, radius, shape)
    inside = (detector.disc_mask(center, radius, shape)
              if detector.CLAMP_TO_DISC else None)
    ok, buf = cv2.imencode(".jpg", _paint(img, det, inside),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def _looks_like_a_cut(mark):
    """0.30 for a blob, 1.00 for a clean line, sliding between."""
    elong = mark["length_px"] / max(mark["thickness_px"], 1.0)
    t = (elong - DIRT_ELONG) / (CUT_ELONG - DIRT_ELONG)
    return DIRT_WEIGHT + (1.0 - DIRT_WEIGHT) * min(max(t, 0.0), 1.0)


def _tracks_crossed(mark):
    """1.0 for a mark crossing almost no grooves, 3.0 for one crossing them all."""
    return 1.0 + SPAN_GAIN * min(max(mark["radial_span_frac"], 0.0), 1.0)


def _band(score):
    for floor, name in GRADE_BANDS:
        if score >= floor:
            return name
    return GRADE_BANDS[-1][1]


def _score_of(marks):
    """Weighted damage over a set of marks, and the grade it comes to.

    Every mark already carries its area as a fraction of ITS OWN photograph's
    playing surface, so marks gathered from two shots taken at different
    distances can be added together without either one counting for more simply
    because the phone was closer.
    """
    if not marks:
        return _band(100.0), 100, 0.0, 0.0
    severity = sum(m["area_frac"] * m["weight"] for m in marks)
    raw = sum(m["area_frac"] for m in marks)
    index = 1000.0 * severity
    score = 100.0 / (1.0 + index / SCORE_HALF_AT)
    return _band(score), round(score), round(index, 2), round(1000.0 * raw, 2)


def _detect(image_bytes, index, want_overlay):
    """One photograph, reduced to the small things the rest of the code needs.

    Everything heavy — the decoded image, the unwrapped ring, the response maps,
    the component labels — is finished with and released before this returns.
    That matters: a whole record is four photographs, and holding all four sets
    of intermediates at once exhausted the 512 MB the service runs in, which
    killed the process mid-request and returned a 502. Nothing outside this
    function ever sees a numpy array now, so peak memory is one photograph
    rather than four.
    """
    img = detector.decode_image(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius, how = detector.find_disc(img)

    inner_px = int(P["LABEL_R"] * radius)
    outer_px = int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner_px:outer_px]
    ring_h, ring_area = ring.shape[0], ring.shape[0] * ring.shape[1]

    radial, tram, dead = detector.scratch_map(ring)
    mask_r, marks_r = detector.extract(radial, None, ring, inner_px, radius)
    mask_t, marks_t = detector.extract(tram, P["TRAM_MIN_LEN"], ring, inner_px, radius)
    marks = marks_r + marks_t
    del radial, tram

    for m in marks:
        # position on the RECORD, not in the photo, so it stays meaningful
        # across shots taken at different angles — and so the other shot has
        # something it can be compared against
        m["radius_frac"] = round((inner_px + m.pop("polar_row")) / radius, 3)
        m["angle_deg"] = round(360.0 * m.pop("polar_col") / P["POLAR_STEPS"], 1)
        # both sizes as fractions of this photo's own playing surface, so the
        # phone's distance cannot change how much a mark counts
        m["radial_span_frac"] = round(m.pop("radial_span_px") / max(ring_h, 1), 3)
        m["area_frac"] = (m["length_px"] * max(m["thickness_px"], 1.0)) / max(ring_area, 1)

    judged = 100.0 * float((~dead).mean())
    warnings = []
    if judged < LOW_COVERAGE:
        warnings.append(
            f"only {judged:.0f}% of the playing surface could be assessed in "
            f"photo {index}; most of it was too dark or too blown-out. Re-shoot "
            "with one lamp at a low angle in a dim room.")
    if how == "fallback_centered":
        warnings.append(f"the record outline was not found in photo {index}; "
                        "results are unreliable. Shoot the disc alone, whole and "
                        "square-on, on a plain dark surface.")

    report = {
        "photo": index,
        "mark_count": len(marks),
        "coverage": {
            "judged_pct": round(judged, 1),
            "unlit_pct": round(100.0 * float((detector.unlit_mask(ring) > 0).mean()), 1),
            "glare_pct": round(100.0 * float((detector.glare_mask(ring) > 0).mean()), 1)},
        "disc": {"center_x": center[0], "center_y": center[1],
                 "radius_px": radius, "found_by": how},
        "warnings": warnings,
    }
    if want_overlay:
        report["overlay_png"] = _overlay(img, cv2.bitwise_or(mask_r, mask_t),
                                         inner_px, center, radius, gray.shape)

    # the label strip is the one array that has to outlive this call, and it is
    # a few hundred kilobytes rather than tens of megabytes
    profile = crossshot.label_profile(gray, center, radius)

    del img, gray, ring, dead, mask_r, mask_t
    _release_memory()
    return {"marks": marks, "report": report, "profile": profile}


def _grade_side(shots, want_overlay=True, side_name=None):
    """One side, from one or two photographs of it."""
    a = shots[0]
    b = shots[1] if len(shots) > 1 else None

    cross = {"used": False, "rotation_deg": None, "confirmed": 0,
             "note": "only one photograph of this side was supplied"}
    marks = []
    warnings = []

    if b is None:
        for m in a["marks"]:
            m["seen_in_both_shots"] = None
            m["photo"] = 1
        marks = list(a["marks"])
    else:
        delta, ratio = crossshot.rotation_from_label(a["profile"], b["profile"])
        if delta is None:
            # A label with no print cannot say how the disc turned between the
            # two shots. Without that, a mark in one photo cannot be matched to
            # a mark in the other, so the second shot is set aside entirely:
            # counting its marks as new would double every defect, and calling
            # them unconfirmed would punish each mark for the label rather than
            # for anything about itself.
            cross["note"] = ("the two photos of this side could not be lined up: "
                             "this label has too little print to show how the "
                             "disc turned, so only the first was graded")
            warnings.append(cross["note"])
            for m in a["marks"]:
                m["seen_in_both_shots"] = None
                m["photo"] = 1
            marks = list(a["marks"])
        else:
            in_b = crossshot.confirm(a["marks"], b["marks"], delta)
            in_a = crossshot.confirm(b["marks"], a["marks"], -delta)
            for m, seen in zip(a["marks"], in_b):
                m["seen_in_both_shots"] = bool(seen)
                m["photo"] = 1
                marks.append(m)
            # A defect the lamp missed in the first shot is still on the record,
            # so the second shot's own finds are added — but only those that did
            # NOT match something already counted, or every confirmed mark would
            # be counted twice.
            for m, seen in zip(b["marks"], in_a):
                if seen:
                    continue
                m["seen_in_both_shots"] = False
                m["photo"] = 2
                marks.append(m)
            cross.update({
                "used": True, "rotation_deg": round(delta, 1),
                "alignment_confidence": round(ratio, 1),
                "confirmed": int(sum(in_b)),
                "only_in_photo_1": int(len(in_b) - sum(in_b)),
                "only_in_photo_2": int(len(in_a) - sum(in_a)),
                "note": "both photos were graded together"})

    for m in marks:
        conf = (CONF_NO_SECOND if m["seen_in_both_shots"] is None
                else (CONF_SEEN_TWICE if m["seen_in_both_shots"] else CONF_SEEN_ONCE))
        cut = _looks_like_a_cut(m)
        m["weight"] = round(cut * _tracks_crossed(m) * conf, 3)
        m["looks_like"] = "a cut" if cut > 0.75 else ("dirt" if cut < 0.45 else "unclear")
        m["area_frac"] = round(m["area_frac"], 6)

    grade, score, index, raw_index = _score_of(marks)
    photos = [s["report"] for s in shots]
    for p in photos:
        warnings.extend(p["warnings"])

    out = {
        "grade": grade,
        "quality_score": score,
        "damage_index": index,
        "damage_index_unweighted": raw_index,
        "grade_is_calibrated": False,   # the bands are the standard; the curve
                                        # that reaches them is not yet pinned
        "mark_count": len(marks),
        "marks": marks,
        "photos": photos,
        "cross_shot": cross,
        "warnings": warnings,
    }
    if side_name:
        out["side"] = side_name
    return out


def analyze(image_bytes, want_overlay=True, second_image_bytes=None):
    """One side of a record, from one photograph or two."""
    started = time.time()
    shots = [_detect(image_bytes, 1, want_overlay)]
    if second_image_bytes:
        shots.append(_detect(second_image_bytes, 2, want_overlay))
    result = _grade_side(shots, want_overlay)
    # the first photo's overlay is also offered at the top level, so a caller
    # that only ever sends one image does not have to walk the photos list
    if want_overlay and result["photos"]:
        result["overlay_png"] = result["photos"][0].get("overlay_png")
    result["elapsed_ms"] = int(1000 * (time.time() - started))
    return result


def analyze_record(sides, want_overlay=True):
    """A whole record: a mapping of side name -> list of one or two photographs.

    The record's grade is the WORSE of its sides, never their average. A buyer
    plays both sides, and a side that skips is not redeemed by the other side
    being clean.
    """
    started = time.time()
    graded = {}
    for name, images in sides.items():
        if not images:
            continue
        # one photograph at a time, and the side is reduced to its result before
        # the next side starts. Four photographs' worth of intermediates held at
        # once does not fit in the memory this service runs in.
        shots = []
        for i, data in enumerate(images, 1):
            shots.append(_detect(data, i, want_overlay))
        graded[name] = _grade_side(shots, want_overlay, side_name=name)
        del shots
        _release_memory()

    if not graded:
        raise ValueError("no photographs were supplied")

    worst = min(graded.values(), key=lambda s: s["quality_score"])
    return {
        "grade": worst["grade"],
        "quality_score": worst["quality_score"],
        "graded_from_side": worst.get("side"),
        "grade_is_calibrated": False,
        "sides": graded,
        "mark_count": sum(s["mark_count"] for s in graded.values()),
        "warnings": [w for s in graded.values() for w in s["warnings"]],
        "elapsed_ms": int(1000 * (time.time() - started)),
    }
