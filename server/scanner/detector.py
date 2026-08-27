# -*- coding: utf-8 -*-
"""
Vinyl scratch detector -- production copy.

This is a deliberate snapshot of experiments/detector.py, not an import of it.
The experiments folder is a sandbox that changes between runs; promoting a
version here is an explicit decision, so a deployed server never shifts under
you because someone was mid-experiment. To ship an improvement, copy it over and
note the change in CHANGELOG.

Every constant in P was set by measurement, not by taste; the comments record
what each one was measured against.

Promoted 2026-08-27 from experiments/detector.py. Measured over 53 records split
BY RECORD into calibration (30), validation (11) and a test set (12) that was
opened only once, at the end:

               recall   precision
  calibration   60.7%      75.9%
  validation    65.2%      78.1%
  test          56.9%      77.2%

Precision counts dirt as a correct call, because a dirty record genuinely is in
worse condition. It was measured by hand-labelling all 2757 detections.

One consequence to be aware of: this build reports roughly three times as many
marks per photo as the one it replaces, and analyze._grade sums length x
thickness over them. The GRADE_BANDS there were never fitted to human-graded
records, and they now sit against a much larger damage index -- so every record
will grade lower than before. The bands need refitting before the grade is
shown to anyone as a number.
"""

import cv2
import numpy as np

P = dict(
    MAX_DIM=1600,
    # Analysed band, as fractions of the disc radius. 0.40/0.93 was cropping away
    # scratches nobody had ever looked at: 15% of all misses lay outside it.
    # Widening inward is free -- there is no hard edge towards the centre. Going
    # outward is not: measured, 0.95 bought 2.7 points of recall for 0.2 extra
    # false detections per clean photo, while the further step to 0.97 bought 0.6
    # points for 0.4 -- the rim itself is a bright line, and find_disc is only
    # approximate, so past 0.95 the cut starts reading the edge as damage.
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
    # marks brighter than the grooves immediately above and below them.
    TRAM_H=9,
    TRAM_ANGLES=(75, 90, 105),
    TRAM_LEN=61,
    TRAM_WEIGHT=1.0,
    TRAM_MIN_LEN=80,     # a groove-parallel mark must be long to outrank a groove
    # Hysteresis thresholds as PERCENTILES of each image's own judgeable-area
    # response, not fixed levels: the noise floor moves image to image (measured
    # p99.9 spans 18-29 across the set), so a constant either floods or starves.
    #
    # Eased from 99.8/99.5 after a miss diagnosis found that 80% of missed
    # scratches produced a response that was measured and then discarded, while
    # only 2 of 527 left no trace at all. Chosen from a nine-point sweep on the
    # CALIBRATION records alone: 49.0% -> 62.5% recall.
    #
    # These and THR_FLOOR are gates in SERIES: opening either alone gained 3-10
    # points, opening both gained 21, which is why every earlier attempt to move
    # just one of them went nowhere.
    PCT_STRONG=99.3,
    PCT_WEAK=98.7,
    # Absolute floor on the normalised map (units: 10x local sigma). This is what
    # keeps a CLEAN record clean -- percentiles alone would always "find" the top
    # 0.1% of pure noise. On calibration this is the heavier of the two gates:
    # dropping it 35 -> 25 was worth 5.0 points, while moving the percentiles
    # across the same span was worth 3.3.
    THR_FLOOR=25,
    GLARE_BRIGHT=200,
    GLARE_MARGIN=11,
    LIT_MIN=14,          # ring areas dimmer than this (local mean) are unjudgeable
    LIT_WIN=51,
    # Local noise normalisation: response divided by the local noise scale, so one
    # threshold means the same thing in a bright busy area and a dim quiet one.
    # Without it a global percentile is set by the noisiest sector and starves the
    # rest (measured: 16 of 48 marked zones had real signal under the threshold).
    NOISE_WIN=201,
    NOISE_FLOOR=1.5,
    CLOSE=(15, 5),       # fragment-linking kernel (tall x wide)
    LINK_DIST=45,        # px: max gap to rejoin two fragments of one scratch
    LINK_COS=0.85,       # how collinear they must be (cos of angle, 1 = perfect)
    # Groove-direction rejection. A highlight running along a groove is the disc's
    # own geometry catching the light, not damage. Long marks are exempt so that
    # genuine tramlines survive. Cut false positives 55% at no cost to recall.
    GROOVE_TOL_DEG=12,
    GROOVE_KEEP_LEN=250,
    # The opposite rejection, and the one that pays: a mark aimed almost exactly
    # at the centre of the disc, and long. That is not damage, it is the beam a
    # lamp throws off thousands of concentric grooves -- the streak of light you
    # see by eye when a record is tilted towards a bulb. Measured over 1527
    # hand-labelled detections: half of all reflections point within 10 degrees of
    # dead radial, against one scratch in nine, and NO marked scratch in the set
    # is both radial and long.
    #
    # Both conditions are required. The angle alone removes short radial
    # scratches too and costs 6 points of recall; adding the length spares them
    # and the rule becomes free. 83/45 is the tightest pair that deletes not one
    # marked scratch on either the calibration or the validation records, and it
    # removes a third of all false detections: precision 69->77 on calibration,
    # 73->80 on validation, with recall unmoved.
    RADIAL_TOL_DEG=83,
    RADIAL_MIN_LEN=45,
    # Shape limits. A miss diagnosis attributed 15.6% of all missed hand-marked
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


def decode_image(data):
    """Decode uploaded bytes and scale to the working size."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a readable image")
    h, w = img.shape[:2]
    scale = P["MAX_DIM"] / max(h, w)
    if scale < 1.0:      # shrink only; upscaling invents detail that isn't there
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def find_disc(bgr):
    """Locate the record. Primary: Hough circle on the rim -- robust to the disc's
    own shadow, which fools blob methods by attaching to the dark disc and
    dragging its centre. Fallbacks: dark-blob contour, then a centred guess."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    m = min(h, w)
    blur = cv2.medianBlur(gray, 7)

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=2, minDist=m,
                               param1=120, param2=50,
                               minRadius=int(0.33 * m), maxRadius=int(0.48 * m))
    if circles is not None:
        cx, cy, r = circles[0][0]
        return (int(round(cx)), int(round(cy))), int(round(r)), "hough"

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
        if fill < 0.55 or not (0.75 <= bw / max(bh, 1) <= 1.33):
            continue
        if best is None or area > best[0]:
            mm = cv2.moments(c)
            best = (area, (mm["m10"] / mm["m00"], mm["m01"] / mm["m00"]),
                    np.sqrt(area / np.pi))
    if best is None:
        return (w // 2, h // 2), int(0.45 * m), "fallback_centered"
    (_, (cx, cy), r) = best
    return (int(round(cx)), int(round(cy))), int(round(r)), "contour"


def unwrap(gray, center, radius):
    """Re-sample the disc by (angle, distance) so grooves become horizontal
    lines and radial scratches become vertical ones."""
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
    """Blown-out lamp reflections: very bright AND wide. Opening erases thin
    bright lines first, so a scratch's own glint is not mistaken for glare."""
    bright = (ring > P["GLARE_BRIGHT"]).astype(np.uint8) * 255
    broad = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return cv2.dilate(broad, np.ones((P["GLARE_MARGIN"], P["GLARE_MARGIN"]), np.uint8))


def unlit_mask(ring):
    """Side lighting reaches only part of the disc; the rest carries no signal,
    and judging it produces noise-driven detections."""
    local = cv2.blur(ring.astype(np.float32), (P["LIT_WIN"], P["LIT_WIN"]))
    return (local < P["LIT_MIN"]).astype(np.uint8) * 255


def _local_normalize(chan):
    """Divide by the local noise scale so one threshold works everywhere.

    The estimate must be ROBUST: a plain local std includes the scratch itself,
    so a long bright mark inflates its own denominator and survives only as a
    fragment (measured: robust estimation recovers 3.3x more of its length).
    """
    win = (P["NOISE_WIN"], P["NOISE_WIN"])
    med = cv2.medianBlur(np.clip(chan, 0, 255).astype(np.uint8), 51).astype(np.float32)
    dev = np.abs(chan - med)
    mad = cv2.blur(dev, win)
    mad_robust = cv2.blur(np.minimum(dev, 2.0 * mad), win)
    return (chan - med) / np.maximum(1.4826 * mad_robust, P["NOISE_FLOOR"])


def scratch_map(ring):
    """Two response maps: radial scratches, and groove-parallel tramlines."""
    flat = ring.astype(np.float32)
    flat -= cv2.blur(flat, (P["ROW_FLATTEN"], 1))
    flat = cv2.GaussianBlur(flat, (3, 3), 0)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (P["TOPHAT_W"], 1))
    ridge = cv2.morphologyEx(flat, cv2.MORPH_TOPHAT, se)
    best = np.zeros_like(ridge)
    for k in _line_kernels(P["BOOST_LEN"], P["ANGLES"]):
        best = np.maximum(best, cv2.filter2D(ridge, -1, k))

    se_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, P["TRAM_H"]))
    ridge_v = cv2.morphologyEx(flat, cv2.MORPH_TOPHAT, se_v)
    tram = np.zeros_like(ridge_v)
    for k in _line_kernels(P["TRAM_LEN"], P["TRAM_ANGLES"]):
        tram = np.maximum(tram, cv2.filter2D(ridge_v, -1, k))

    # Channels stay SEPARATE: merged, the louder one raises the shared adaptive
    # threshold and starves the quieter one.
    dead = (glare_mask(ring) > 0) | (unlit_mask(ring) > 0)
    out = []
    for chan, weight in ((best, 1.0), (tram, P["TRAM_WEIGHT"])):
        z = _local_normalize(chan * weight)
        z[dead] = 0
        out.append(np.clip(z * 10, 0, 255).astype(np.uint8))
    return out[0], out[1], dead


def _link_collinear(binary):
    """Rejoin fragments of one scratch split by a mask or a faint stretch.

    Unlike a morphological close, which also merges unrelated neighbours, two
    fragments are joined only when they are close AND both run along the axis
    connecting them -- i.e. they look like two pieces of one line.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:
        return binary

    info = []
    for i in range(1, n):
        if stats[i][4] < 15:
            continue
        pts = np.column_stack(np.nonzero(labels == i)).astype(np.float32)
        mean = pts.mean(axis=0)
        _, eigvec = cv2.PCACompute(pts, mean.reshape(1, -1), maxComponents=1)
        info.append((mean, eigvec[0] / (np.linalg.norm(eigvec[0]) + 1e-9)))

    out = binary.copy()
    for a in range(len(info)):
        ca, va = info[a]
        for b in range(a + 1, len(info)):
            cb, vb = info[b]
            delta = cb - ca
            dist = float(np.linalg.norm(delta))
            if dist > P["LINK_DIST"] or dist < 1:
                continue
            u = delta / dist
            if (abs(float(np.dot(va, u))) < P["LINK_COS"]
                    or abs(float(np.dot(vb, u))) < P["LINK_COS"]):
                continue
            cv2.line(out, (int(ca[1]), int(ca[0])), (int(cb[1]), int(cb[0])), 255, 2)
    return out


def _axis_angle_deg(comp):
    """Angle of a component's long axis away from the groove direction:
    0 = along the grooves, 90 = crossing them."""
    pts = np.column_stack(np.nonzero(comp)).astype(np.float32)
    if len(pts) < 5:
        return 90.0
    mean = pts.mean(axis=0)
    _, eigvec = cv2.PCACompute(pts, mean.reshape(1, -1), maxComponents=1)
    return float(np.degrees(np.arctan2(abs(float(eigvec[0][0])),
                                       abs(float(eigvec[0][1])))))


def extract(smap, min_len=None):
    """Threshold with hysteresis, then keep only long, thin, groove-crossing
    components. Returns the mask and one record per surviving mark."""
    min_len = P["MIN_LEN"] if min_len is None else min_len
    judgeable = smap[smap > 0]
    if judgeable.size < 1000:
        return np.zeros_like(smap), []
    thr_strong = max(float(np.percentile(judgeable, P["PCT_STRONG"])), P["THR_FLOOR"])
    thr_weak = max(float(np.percentile(judgeable, P["PCT_WEAK"])), P["THR_FLOOR"] / 2)

    weak = (smap > thr_weak).astype(np.uint8)
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    seeds = set(np.unique(labels[smap >= thr_strong])) - {0}
    binary = np.isin(labels, list(seeds)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones(P["CLOSE"], np.uint8))
    binary = _link_collinear(binary)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros_like(binary)
    marks = []
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
        angle = _axis_angle_deg(comp)      # 0 = along the grooves, 90 = across
        if angle < P["GROOVE_TOL_DEG"] and length < P["GROOVE_KEEP_LEN"]:
            continue                        # groove highlight, not damage
        if angle > P["RADIAL_TOL_DEG"] and length > P["RADIAL_MIN_LEN"]:
            continue                        # the lamp's beam off the grooves
        mask[labels == i] = 255
        marks.append({"length_px": int(length),
                      "thickness_px": round(float(thickness), 1),
                      "angle_to_groove_deg": round(angle, 1),
                      # rows of the unwrapped ring ARE the radius, so the
                      # component's height is how far across the record it
                      # reaches -- i.e. how many grooves, and so how many
                      # tracks, it crosses. Used to weight the grade.
                      "radial_span_px": int(h),
                      "polar_row": int(y + h / 2), "polar_col": int(x + w / 2)})
    return mask, marks


def rewrap(ring_mask, inner_px, center, radius, out_shape):
    """Map a ring mask back onto the original photo."""
    full_polar = np.zeros((radius, P["POLAR_STEPS"]), np.uint8)
    full_polar[inner_px:inner_px + ring_mask.shape[0]] = ring_mask
    polar = cv2.transpose(full_polar)
    return cv2.warpPolar(polar, (out_shape[1], out_shape[0]), center, radius,
                         cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP
                         | cv2.INTER_NEAREST)
