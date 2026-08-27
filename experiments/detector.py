# -*- coding: utf-8 -*-
"""
Experimental scratch detector (Agent A works here; pipeline.py stays untouched).
All tunables live in the P dict so iterations are traceable in the changelog.
"""

import os

import cv2
import numpy as np

P = dict(
    MAX_DIM=1600,
    # Analysed band, as fractions of the disc radius. 0.40/0.93 was cropping
    # away scratches nobody ever looked at: 15% of all misses lay outside it.
    # Widening inward is free -- there is no hard edge towards the centre. Going
    # outward is not: measured, 0.95 bought 2.7 points of recall for 0.2 extra
    # false detections per clean photo, while the further step to 0.97 bought
    # 0.6 points for 0.4 -- the rim itself is a bright line, and find_disc is
    # only approximate, so past 0.95 the cut starts reading the edge as damage.
    LABEL_R=0.36,        # inner mask: fraction of disc radius (label + run-out)
    OUTER_R=0.95,        # outer mask: fraction of disc radius (rim + lead-in)
    POLAR_STEPS=3600,
    ROW_FLATTEN=151,     # background flattening window along each groove row
    TOPHAT_W=15,         # ridge width limit: wider-than-this bright = not a scratch
    BOOST_LEN=21,        # line-averaging length
    # 10deg steps, not 20: a long scratch curves along its length, and a segment
    # falling between two kernel angles gets smeared instead of boosted -- which
    # is where long scratches were breaking apart.
    ANGLES=tuple(range(-70, 71, 10)),
    # Groove-parallel ("tramline") channel: scratches running ALONG the grooves
    # are invisible to the channel above (a horizontal top-hat deletes anything
    # horizontally long). This channel uses a VERTICAL top-hat instead: it keeps
    # marks that are brighter than the grooves immediately above and below them.
    TRAM_H=9,            # vertical neighbourhood height for the top-hat
    TRAM_ANGLES=(75, 90, 105),
    TRAM_LEN=61,         # tramlines are long -- average over a long window
    TRAM_WEIGHT=1.0,     # scale of this channel relative to the radial one
    TRAM_MIN_LEN=80,     # a groove-parallel mark must be long to outrank a groove
    # Hysteresis thresholds as PERCENTILES of each image's own judgeable-area
    # response, not fixed levels: the noise floor moves image to image (measured
    # p99.9 spans 18-29 across the set), so a constant either floods or starves.
    # Eased from 99.8/99.5, chosen on the calibration set and confirmed on
    # validation. Hand-labelling 894 detections showed the easing does NOT
    # degrade what the model reports: precision held at 79-81%.
    #
    # Eased again after the miss diagnosis found 80% of missed scratches
    # produced a response that was measured and then discarded, while only 2 of
    # 527 left no trace at all. Chosen from a nine-point sweep ON CALIBRATION:
    # 49.0% -> 62.5% recall. An earlier attempt swept the same nine points on
    # validation and was reverted — picking the best of nine by looking at the
    # held-out set spends the only clean estimate there is on the choice itself.
    #
    # These and THR_FLOOR are gates in SERIES: opening either alone gained 3-10
    # points, opening both gained 21, which is why every earlier attempt to move
    # one of them went nowhere. On calibration the FLOOR is the heavier of the
    # two — dropping it 30 -> 25 was worth 5.0 points, while moving these across
    # the same span was worth 3.3.
    PCT_STRONG=99.3,
    PCT_WEAK=98.7,
    # Absolute floor on the normalised map (units: 10x local sigma). This is what
    # keeps a CLEAN record clean -- percentiles alone would always "find" the top
    # 0.1% of pure noise.
    THR_FLOOR=25,
    GLARE_BRIGHT=200,
    GLARE_MARGIN=11,
    LIT_MIN=14,          # ring areas dimmer than this (local mean) are unjudgeable
    LIT_WIN=51,
    # Local noise normalisation: response is divided by the local noise scale, so
    # a single threshold means the same thing in a bright, busy area and in a dim,
    # quiet one. Without it a global percentile is set by the noisiest sector and
    # starves every other sector (measured: 16 of 48 marked zones had real signal
    # sitting under the global threshold).
    NOISE_WIN=201,
    NOISE_FLOOR=1.5,     # keeps quiet areas from dividing by ~0 and exploding
    CLOSE=(15, 5),       # fragment-linking kernel (tall x wide)
    LINK_DIST=45,        # px: max gap to rejoin two fragments of one scratch
    LINK_COS=0.85,       # how collinear they must be (cos of angle, 1 = perfect)
    # Groove-direction rejection. A highlight running along a groove is the disc's
    # own geometry catching the light, not damage. In the unwrapped image grooves
    # are horizontal, so a detection whose axis sits within GROOVE_TOL_DEG of
    # horizontal is groove-aligned and is dropped -- unless it is long enough to
    # be a genuine tramline, which no single groove highlight reaches.
    GROOVE_TOL_DEG=12,
    GROOVE_KEEP_LEN=250,
    # The opposite rejection, and the one that pays: a mark aimed almost exactly
    # at the centre of the disc, and long. That is not damage, it is the beam a
    # lamp throws off thousands of concentric grooves — the streak of light you
    # see by eye when a record is tilted towards a bulb. Measured over 1527
    # hand-labelled detections: half of all reflections point within 10 degrees
    # of dead radial, against one scratch in nine, and NO marked scratch in the
    # set is both radial and long.
    #
    # Both conditions are required. The angle alone removes short radial
    # scratches too and costs 6 points of recall; adding the length spares them
    # and the rule becomes free. 83/45 is the tightest pair that deletes not one
    # marked scratch on either the calibration or the validation records, and it
    # removes a third of all false detections: precision 69->77 on calibration,
    # 73->80 on validation, with recall unmoved at 62.5% on both.
    RADIAL_TOL_DEG=83,
    RADIAL_MIN_LEN=45,
    # Shape limits. why_missed.py attributed 15.6% of all missed hand-marked
    # scratches to MIN_LEN alone, and those misses clustered at length 15-28
    # against a bar of 30 that had no measurement behind it.
    #
    # Dropping the bar for everyone also admits every short stubby speck, so the
    # rule is graded instead: below SHORT_LEN a component must clear the much
    # stricter SHORT_ELONG. A short real scratch is a thin sharp line; a short
    # piece of dirt is a blob. Measured on the calibration set, the graded form
    # found the same scratches as a flat MIN_LEN=15 while reporting 160 fewer
    # detections.
    MIN_LEN=15,
    SHORT_LEN=30,
    SHORT_ELONG=6.0,
    MAX_THICK=16,
    MIN_ELONG=2.5,
)


def load_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read {path}")
    h, w = img.shape[:2]
    scale = P["MAX_DIM"] / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def find_disc(bgr):
    """Real detection (no framing guide). Primary: Hough circle on the rim edge
    -- robust to the disc's own shadow, which fools blob-based methods by
    attaching to the dark disc and dragging its center/area. Fallback: dark-blob
    contour, then centered guess."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    m = min(h, w)
    blur = cv2.medianBlur(gray, 7)

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=2, minDist=m,
                               param1=120, param2=50,
                               minRadius=int(0.33 * m), maxRadius=int(0.48 * m))
    if circles is not None:
        cx, cy, r = circles[0][0]
        return (int(round(cx)), int(round(cy))), int(round(r))

    thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark = (blur < thr).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    best = None
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.05 * h * w:
            continue
        (cx, cy), r_enc = cv2.minEnclosingCircle(c)
        if r_enc <= 0:
            continue
        fill = area / (np.pi * r_enc ** 2)
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(bh, 1)
        if fill < 0.55 or not (0.75 <= aspect <= 1.33):
            continue
        if best is None or area > best[0]:
            mm = cv2.moments(c)
            best = (area, (mm["m10"] / mm["m00"], mm["m01"] / mm["m00"]),
                    np.sqrt(area / np.pi))
    if best is None:
        return (w // 2, h // 2), int(0.45 * m)
    (_, (cx, cy), r) = best
    return (int(round(cx)), int(round(cy))), int(round(r))


def unwrap(gray, center, radius):
    polar = cv2.warpPolar(gray, (radius, P["POLAR_STEPS"]), center, radius,
                          cv2.WARP_POLAR_LINEAR)
    return cv2.transpose(polar)


def _line_kernels(length, angles):
    size = length | 1
    ks = []
    for ang in angles:
        k = np.zeros((size, size), np.float32)
        cv2.line(k, (size // 2, 0), (size // 2, size - 1), 1.0, 1)
        M = cv2.getRotationMatrix2D(((size - 1) / 2, (size - 1) / 2), ang, 1.0)
        k = cv2.warpAffine(k, M, (size, size))
        k /= max(k.sum(), 1e-6)
        ks.append(k)
    return ks


def glare_mask(ring):
    bright = (ring > P["GLARE_BRIGHT"]).astype(np.uint8) * 255
    broad = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return cv2.dilate(broad, np.ones((P["GLARE_MARGIN"], P["GLARE_MARGIN"]), np.uint8))


def unlit_mask(ring):
    """Side lighting only lights part of the disc; the rest carries no signal.
    Judging it produces noise-driven detections, so mark it unjudgeable."""
    local = cv2.blur(ring.astype(np.float32), (P["LIT_WIN"], P["LIT_WIN"]))
    return (local < P["LIT_MIN"]).astype(np.uint8) * 255


def scratch_map(ring):
    flat = ring.astype(np.float32)
    flat -= cv2.blur(flat, (P["ROW_FLATTEN"], 1))
    flat = cv2.GaussianBlur(flat, (3, 3), 0)

    # channel A -- radial scratches (cross the grooves): horizontally narrow ridges
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (P["TOPHAT_W"], 1))
    ridge = cv2.morphologyEx(flat, cv2.MORPH_TOPHAT, se)
    best = np.zeros_like(ridge)
    for k in _line_kernels(P["BOOST_LEN"], P["ANGLES"]):
        best = np.maximum(best, cv2.filter2D(ridge, -1, k))

    # channel B -- tramlines (run along the grooves): vertically narrow ridges,
    # i.e. brighter than the grooves directly above and below. Groove texture is
    # regular so it cancels; an anomalous streak survives.
    se_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, P["TRAM_H"]))
    ridge_v = cv2.morphologyEx(flat, cv2.MORPH_TOPHAT, se_v)
    tram = np.zeros_like(ridge_v)
    for k in _line_kernels(P["TRAM_LEN"], P["TRAM_ANGLES"]):
        tram = np.maximum(tram, cv2.filter2D(ridge_v, -1, k))

    # Keep the channels SEPARATE: they have different response magnitudes, and a
    # merged map lets the louder channel raise the shared adaptive threshold and
    # starve the quieter one. Each is thresholded on its own distribution.
    dead = (glare_mask(ring) > 0) | (unlit_mask(ring) > 0)
    out = []
    for chan, weight in ((best, 1.0), (tram, P["TRAM_WEIGHT"])):
        z = _local_normalize(chan * weight)
        z[dead] = 0
        out.append(np.clip(z * 10, 0, 255).astype(np.uint8))   # x10 -> integer grid
    return out[0], out[1]


def _local_normalize(chan):
    """Divide by the local noise scale so one threshold works everywhere.

    The noise estimate must be ROBUST: a plain local std includes the scratch
    itself, so a long bright mark inflates its own denominator and suppresses
    itself, leaving only its brightest fragment (measured: robust estimation
    recovers 3.3x more of each scratch's length). Deviations are clipped at
    twice the local average before being averaged again, which bounds how much
    any bright outlier can contribute to its own noise floor.
    """
    win = (P["NOISE_WIN"], P["NOISE_WIN"])
    med = cv2.medianBlur(np.clip(chan, 0, 255).astype(np.uint8), 51).astype(np.float32)
    dev = np.abs(chan - med)
    mad = cv2.blur(dev, win)
    mad_robust = cv2.blur(np.minimum(dev, 2.0 * mad), win)
    return (chan - med) / np.maximum(1.4826 * mad_robust, P["NOISE_FLOOR"])


def _link_collinear(binary):
    """Rejoin fragments of one scratch that a mask or a faint stretch split.

    A morphological close only bridges gaps smaller than its kernel and would
    also merge unrelated neighbours. This instead joins two fragments only when
    they are close AND lie along the same axis AND that axis points from one to
    the other -- i.e. they look like two pieces of a single line.
    """
    n, labels, stats, cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:
        return binary

    info = []
    for i in range(1, n):
        if stats[i][4] < 15:
            continue
        pts = np.column_stack(np.nonzero(labels == i)).astype(np.float32)  # (row, col)
        mean = pts.mean(axis=0)
        _, eigvec = cv2.PCACompute(pts, mean.reshape(1, -1), maxComponents=1)
        info.append((i, mean, eigvec[0] / (np.linalg.norm(eigvec[0]) + 1e-9)))

    out = binary.copy()
    for a in range(len(info)):
        ia, ca, va = info[a]
        for b in range(a + 1, len(info)):
            ib, cb, vb = info[b]
            delta = cb - ca
            dist = float(np.linalg.norm(delta))
            if dist > P["LINK_DIST"] or dist < 1:
                continue
            u = delta / dist
            # both fragments must run along the connecting direction
            if (abs(float(np.dot(va, u))) < P["LINK_COS"]
                    or abs(float(np.dot(vb, u))) < P["LINK_COS"]):
                continue
            cv2.line(out, (int(ca[1]), int(ca[0])), (int(cb[1]), int(cb[0])), 255, 2)
    return out


def extract(smap, min_len=None):
    min_len = P["MIN_LEN"] if min_len is None else min_len
    judgeable = smap[smap > 0]          # zeros are masked-out glare/unlit areas
    if judgeable.size < 1000:
        return np.zeros_like(smap), []
    thr_strong = max(float(np.percentile(judgeable, P["PCT_STRONG"])), P["THR_FLOOR"])
    thr_weak = max(float(np.percentile(judgeable, P["PCT_WEAK"])), P["THR_FLOOR"] / 2)

    weak = (smap > thr_weak).astype(np.uint8)
    strong = smap >= thr_strong
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    seeds = set(np.unique(labels[strong])) - {0}
    binary = np.isin(labels, list(seeds)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              np.ones(P["CLOSE"], np.uint8))
    binary = _link_collinear(binary)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros_like(binary)
    scratches = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = max((cv2.arcLength(c, True) for c in contours), default=0)
        length = max(perimeter / 2, max(w, h))
        thickness = area / max(length, 1)
        if length < min_len or thickness > P["MAX_THICK"]:
            continue
        # Short components are held to a stricter elongation than long ones: a
        # short real scratch is a thin sharp line, a short piece of dirt is a
        # blob, and one flat threshold cannot ask that of both.
        need = P["SHORT_ELONG"] if length < P["SHORT_LEN"] else P["MIN_ELONG"]
        if length / max(thickness, 1) < need:
            continue

        angle = _axis_angle_deg(comp)          # 0 = along the grooves, 90 = across
        if angle < P["GROOVE_TOL_DEG"] and length < P["GROOVE_KEEP_LEN"]:
            continue                            # groove highlight, not damage
        if angle > P["RADIAL_TOL_DEG"] and length > P["RADIAL_MIN_LEN"]:
            continue                            # the lamp's beam off the grooves

        mask[labels == i] = 255
        scratches.append({"length": int(length), "thickness": round(thickness, 1),
                          "angle_to_groove": round(angle, 1)})
    return mask, scratches


def _axis_angle_deg(comp):
    """Angle of a component's long axis away from the groove direction, in the
    unwrapped image where grooves run horizontally. 0 = parallel to the grooves,
    90 = crossing them."""
    pts = np.column_stack(np.nonzero(comp)).astype(np.float32)   # (row, col)
    if len(pts) < 5:
        return 90.0
    mean = pts.mean(axis=0)
    _, eigvec = cv2.PCACompute(pts, mean.reshape(1, -1), maxComponents=1)
    d_row, d_col = float(eigvec[0][0]), float(eigvec[0][1])
    return float(np.degrees(np.arctan2(abs(d_row), abs(d_col))))


def rewrap(ring_mask, inner_px, center, radius, out_shape):
    """Put the cropped-ring mask back into a full cartesian mask."""
    full_polar = np.zeros((radius, P["POLAR_STEPS"]), np.uint8)
    full_polar[inner_px:inner_px + ring_mask.shape[0]] = ring_mask
    polar = cv2.transpose(full_polar)
    return cv2.warpPolar(polar, (out_shape[1], out_shape[0]), center, radius,
                         cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)


def detect(path):
    img = load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = find_disc(img)
    polar = unwrap(gray, center, radius)
    inner_px = int(P["LABEL_R"] * radius)
    outer_px = int(P["OUTER_R"] * radius)
    ring = polar[inner_px:outer_px]

    radial_map, tram_map = scratch_map(ring)
    mask_a, scr_a = extract(radial_map)
    mask_b, scr_b = extract(tram_map, min_len=P["TRAM_MIN_LEN"])
    ring_mask = cv2.bitwise_or(mask_a, mask_b)
    scratches = scr_a + scr_b
    det = rewrap(ring_mask, inner_px, center, radius, gray.shape)

    overlay = img.copy()
    on = det > 0
    overlay[on] = (0.25 * overlay[on] + 0.75 * np.array([0, 255, 255])).astype(np.uint8)
    cv2.circle(overlay, center, radius, (0, 200, 0), 2)
    cv2.circle(overlay, center, inner_px, (0, 200, 0), 1)
    info = {"center": center, "radius": radius, "n_scratches": len(scratches),
            "scratches": scratches}
    return det, overlay, info
