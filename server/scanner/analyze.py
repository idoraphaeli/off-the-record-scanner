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
#
# Refitted from 35 when the detector changed to report only what BOTH shots saw.
# That paints 9.6 marks a photograph rather than 14.8, so the same record
# produces a smaller index and would come out a band better for no reason but a
# change in what we count: left at 35, 29 of 72 sides come out Near Mint where
# one did before. Measured over calibration and validation, the new index runs at
# 0.385 of the old, so the constant moves with it.
#
# It does not map one model onto the other, and cannot: the ratio's quartiles are
# 0.23 to 0.54, because the new detector disagrees with the old about WHICH
# records are damaged rather than scaling them all down. Thirty-four of 72 sides
# still change band, against 51 unrefitted. This holds the distribution, not the
# individual record.
SCORE_HALF_AT = 13.5


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


def _prepare(image_bytes, index):
    """One photograph, taken as far as it can go ALONE.

    It stops before the threshold, because this detector does not decide
    anything from one photograph. What comes back is the response map — a number
    per point on the disc saying how scratch-like it is — and the two shots of a
    side are combined into one map before anything is called a mark.

    That means two photographs' maps have to be alive at the same time, where the
    old arrangement only ever held one. Two is affordable and four is not, which
    is why a record is still walked one SIDE at a time and everything is dropped
    between sides.
    """
    img = detector.decode_image(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius, how = detector.find_disc(img)

    inner_px = int(P["LABEL_R"] * radius)
    outer_px = int(P["OUTER_R"] * radius)
    ring = detector.unwrap(gray, center, radius)[inner_px:outer_px]
    radial, tram, dead = detector.scratch_map(ring)

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
        "coverage": {
            "judged_pct": round(judged, 1),
            "unlit_pct": round(100.0 * float((detector.unlit_mask(ring) > 0).mean()), 1),
            "glare_pct": round(100.0 * float((detector.glare_mask(ring) > 0).mean()), 1)},
        "disc": {"center_x": center[0], "center_y": center[1],
                 "radius_px": radius, "found_by": how},
        "warnings": warnings,
    }
    profile = crossshot.label_profile(gray, center, radius)
    del dead
    return {"img": img, "shape": gray.shape, "ring": ring, "radial": radial,
            "tram": tram, "inner_px": inner_px, "center": center,
            "radius": radius, "report": report, "profile": profile}


def _marks_of(ring_mask, marks, inner_px, radius, ring):
    """The detector's raw marks, given the sizes the grade needs."""
    ring_h, ring_area = ring.shape[0], ring.shape[0] * ring.shape[1]
    for m in marks:
        # position on the RECORD, not in the photo, so it stays meaningful
        # across shots taken at different angles
        m["radius_frac"] = round((inner_px + m.pop("polar_row")) / radius, 3)
        m["angle_deg"] = round(360.0 * m.pop("polar_col") / P["POLAR_STEPS"], 1)
        # both sizes as fractions of the photo's own playing surface, so the
        # phone's distance cannot change how much a mark counts
        m["radial_span_frac"] = round(m.pop("radial_span_px") / max(ring_h, 1), 3)
        m["area_frac"] = (m["length_px"] * max(m["thickness_px"], 1.0)) / max(ring_area, 1)
        # every mark this detector reports was present in both shots; that is
        # what being on the combined map means
        m["seen_in_both_shots"] = True
    return marks


def _into_frame_of(ring_mask, shot, delta):
    """The combined mask, which lives in the FIRST shot's frame, moved into the
    second's — so both photographs come back marked in the same places on the
    record rather than each carrying its own separate findings."""
    h = shot["ring"].shape[0]
    m = ring_mask
    if m.shape[0] != h:
        m = cv2.resize(m, (m.shape[1], h), interpolation=cv2.INTER_NEAREST)
    return np.roll(m, int(round(delta / 360.0 * P["POLAR_STEPS"])), axis=1)


STATUS_OK = "ok"
STATUS_NOT_ALIGNED = "alignment_failed"
STATUS_NEEDS_TWO = "needs_two_photos"

RETAKE_NOTE = ("the two photos of this side could not be matched to each other "
               "— please take them again")


def _unjudged(shots, status, note, want_overlay, side_name):
    """A side we could not read, said so plainly.

    Every mark this detector reports was seen in BOTH photographs, so when the
    two cannot be matched there is nothing to report — and an empty result is
    not the same statement as a clean record. Returning one would grade a
    scratched disc perfect, and the buyer is the person who finds out. So there
    is no grade and no score here at all: the caller has to handle the status,
    and the photographs go back exactly as they came in, unmarked.
    """
    out = {
        "status": status,
        "grade": None,
        "quality_score": None,
        "mark_count": 0,
        "marks": [],
        "needs_retake": True,
        "message": note,
        "photos": [],
        "cross_shot": {"used": False, "rotation_deg": None, "confirmed": 0,
                       "note": note},
        "warnings": [note] + [w for s in shots for w in s["report"]["warnings"]],
    }
    for s in shots:
        report = dict(s["report"])
        report["mark_count"] = 0
        if want_overlay:
            # the plain photograph: an empty mask paints nothing
            report["overlay_png"] = _overlay(
                s["img"], np.zeros_like(s["ring"]), s["inner_px"],
                s["center"], s["radius"], s["shape"])
        out["photos"].append(report)
    if side_name:
        out["side"] = side_name
    return out


def _grade_side(shots, want_overlay=True, side_name=None):
    """One side, from the two photographs of it.

    The two response maps are brought into the same frame and the pixelwise
    minimum is taken, so a point survives only if BOTH shots saw it. A lamp's
    reflection moves when the disc is tilted, so the other shot has nothing
    where it was and the minimum cuts it to nothing; a scratch is in the vinyl
    and stays put. Only then is a threshold applied, and what comes out is one
    set of marks for the side rather than two sets to be reconciled.

    Measured over 30 calibration records with every mark hand-judged: 96%
    precision against 87% for thresholding each shot separately, and 21 outright
    false marks in 528 against 71. It costs recall — 34% against 46% — which is
    the trade this detector is making on purpose.
    """
    a = shots[0]
    b = shots[1] if len(shots) > 1 else None
    if b is None:
        return _unjudged(shots, STATUS_NEEDS_TWO,
                         "this side needs two photographs, taken with the disc "
                         "tilted differently between them",
                         want_overlay, side_name)

    delta, ratio, how = crossshot.align(a["radial"], b["radial"],
                                        a["profile"], b["profile"])
    if delta is None:
        return _unjudged(shots, STATUS_NOT_ALIGNED, RETAKE_NOTE,
                         want_overlay, side_name)

    steps = P["POLAR_STEPS"]
    rad = crossshot.combine(a["radial"], b["radial"], delta, steps)
    tra = crossshot.combine(a["tram"], b["tram"], delta, steps)
    mask_r, marks_r = detector.extract(rad, None, a["ring"], a["inner_px"],
                                       a["radius"])
    mask_t, marks_t = detector.extract(tra, P["TRAM_MIN_LEN"], a["ring"],
                                       a["inner_px"], a["radius"])
    del rad, tra
    ring_mask = cv2.bitwise_or(mask_r, mask_t)
    marks = _marks_of(ring_mask, marks_r + marks_t, a["inner_px"], a["radius"],
                      a["ring"])
    for m in marks:
        m["photo"] = 1

    for m in marks:
        cut = _looks_like_a_cut(m)
        m["weight"] = round(cut * _tracks_crossed(m) * CONF_SEEN_TWICE, 3)
        m["looks_like"] = "a cut" if cut > 0.75 else ("dirt" if cut < 0.45 else "unclear")
        m["area_frac"] = round(m["area_frac"], 6)

    grade, score, index, raw_index = _score_of(marks)
    photos, warnings = [], []
    for i, s in enumerate(shots):
        report = dict(s["report"])
        report["mark_count"] = len(marks)
        if want_overlay:
            # both photographs carry the SAME marks, each drawn where they fall
            # in that photograph — they are one set of findings about one side,
            # not two separate opinions
            here = ring_mask if i == 0 else _into_frame_of(ring_mask, s, delta)
            report["overlay_png"] = _overlay(s["img"], here, s["inner_px"],
                                             s["center"], s["radius"], s["shape"])
        photos.append(report)
        warnings.extend(report["warnings"])

    out = {
        "status": STATUS_OK,
        "grade": grade,
        "quality_score": score,
        "damage_index": index,
        "damage_index_unweighted": raw_index,
        "grade_is_calibrated": False,   # the bands are the standard; the curve
                                        # that reaches them is not yet pinned
        "needs_retake": False,
        "message": "",
        "mark_count": len(marks),
        "marks": marks,
        "photos": photos,
        "cross_shot": {
            "used": True, "rotation_deg": round(delta, 1),
            "aligned_by": how, "alignment_confidence": round(ratio, 1),
            "confirmed": len(marks),
            "note": "both photos were combined before anything was called a mark"},
        "warnings": warnings,
    }
    if side_name:
        out["side"] = side_name
    return out


def analyze(image_bytes, want_overlay=True, second_image_bytes=None):
    """One side of a record, from one photograph or two."""
    started = time.time()
    shots = [_prepare(image_bytes, 1)]
    if second_image_bytes:
        shots.append(_prepare(second_image_bytes, 2))
    try:
        result = _grade_side(shots, want_overlay)
    finally:
        del shots
        _release_memory()
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
        # Both photographs of a side at once, because they are combined before
        # anything is decided -- but one SIDE at a time, and everything dropped
        # before the next. Two photographs' worth of intermediates fits; four
        # does not, and holding four is what used to kill the process mid-request
        # and return a 502.
        shots = [_prepare(data, i) for i, data in enumerate(images, 1)]
        try:
            graded[name] = _grade_side(shots, want_overlay, side_name=name)
        finally:
            del shots
            _release_memory()

    if not graded:
        raise ValueError("no photographs were supplied")

    # A side we could not read takes the whole record with it. Grading the other
    # side alone and calling that the record's condition answers a question we
    # did not manage to ask: the side we could not read might be the damaged one.
    retake = [name for name, s in graded.items() if s.get("needs_retake")]
    if retake:
        which = " and ".join(sorted(retake))
        return {
            "status": STATUS_NOT_ALIGNED,
            "grade": None,
            "quality_score": None,
            "needs_retake": True,
            "sides_to_retake": sorted(retake),
            "message": (f"side {which}: " + RETAKE_NOTE.split("— ")[0].strip()
                        + " — please photograph "
                        + ("that side" if len(retake) == 1 else "those sides")
                        + " again"),
            "sides": graded,
            "mark_count": 0,
            "warnings": [w for s in graded.values() for w in s["warnings"]],
            "elapsed_ms": int(1000 * (time.time() - started)),
        }

    worst = min(graded.values(), key=lambda s: s["quality_score"])
    return {
        "status": STATUS_OK,
        "grade": worst["grade"],
        "quality_score": worst["quality_score"],
        "graded_from_side": worst.get("side"),
        "grade_is_calibrated": False,
        "needs_retake": False,
        "sides_to_retake": [],
        "message": "",
        "sides": graded,
        "mark_count": sum(s["mark_count"] for s in graded.values()),
        "warnings": [w for s in graded.values() for w in s["warnings"]],
        "elapsed_ms": int(1000 * (time.time() - started)),
    }
