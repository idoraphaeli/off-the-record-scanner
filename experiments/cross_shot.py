# -*- coding: utf-8 -*-
"""
Does a defect appearing in BOTH shots of a side tell real damage from glare?

A scratch is a groove cut into the vinyl: it sits at a fixed radius and a fixed
angle on the disc, and it is there no matter how the record is held. A
reflection is a property of the light, not the record — move the disc and it
lands somewhere else. So "seen twice, in the same place on the disc" should be
strong evidence of real damage, and that is exactly the discriminator that
colour, edge sharpness and channel all failed to provide.

The earlier attempt at this tried to ALIGN the two photographs and gave up:
every pair reported the same ~280 degrees with residuals in the thousands of
pixels. It failed for one reason — a record is very nearly rotationally
symmetric, thousands of near-identical circular grooves, so there is almost
nothing for an image aligner to lock onto. Worse, the one asymmetric, printed,
high-contrast thing on the disc is the centre label, and the detector crops it
away before any of this runs.

This does not align images at all. Radius is invariant between shots and we
already measure it precisely; the whole rotation is a single unknown angle. So
every pair of detections at a MATCHING RADIUS votes for the rotation that would
map one onto the other. Real defects all vote for the same angle; noise spreads
out. The peak is the rotation, and no pixel ever has to be registered.

Usage:  python cross_shot.py [cal|val|both]
"""

import collections
import csv
import json
import math
import os
import sys

import cv2
import numpy as np

import detector
from evaluate_frozen import MIN_EXTRA_AREA, detect

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
VIEW_W = 1100

BIN_DEG = 3.0        # rotation histogram resolution
RAD_TOL = 0.025      # radius agreement, as a fraction of the disc radius
ANG_TOL = 6.0        # angular agreement once the rotation is known, degrees
# Two shots of a side share only a handful of defects — most dirt moves and a
# faint scratch shows at one light angle and not the other. The first run
# demanded 12 votes in the winning bin, which is more than the signal that
# exists, and 73 of 74 sides were thrown out. What matters is how far the peak
# stands above the background, not its absolute height.
MIN_PEAK_RATIO = 3.0
MIN_VOTES = 3

# The label is the only thing on a record that gives away its rotation: printed,
# asymmetric, high contrast, and turning rigidly with the disc. The detector
# crops it away before analysis, which is precisely why the earlier alignment
# attempt had nothing to lock onto. Here it is the primary anchor and the
# detection vote is only the fallback.
LABEL_LO, LABEL_HI = 0.12, 0.33   # fractions of the disc radius
POLAR_STEPS = 1440                # 0.25 degrees per column
LABEL_MIN_RATIO = 4.0             # correlation peak over its own background
ANCHOR_MIN_CONF = 1.8             # first harmonic over the rest of the spectrum
# Blur wide enough to erase printed letters but not the lighting gradient, which
# spans the whole strip. 60 columns of 1440 is 15 degrees of arc.
BLUR_W = 60


def polar_detections(path):
    """Every detection on one photo, in disc coordinates.

    Radius is normalised by the disc radius, so it is comparable across photos
    taken at different distances; the angle is measured from the disc centre.
    """
    img, det = detect(path)
    center, radius = detector.find_disc(img)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(
        (det > 127).astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i][4] < MIN_EXTRA_AREA:
            continue
        cx, cy = float(cent[i][0]), float(cent[i][1])
        dx, dy = cx - center[0], cy - center[1]
        out.append({
            "rad": math.hypot(dx, dy) / max(radius, 1),
            "ang": math.degrees(math.atan2(dy, dx)) % 360.0,
            "vx": cx * VIEW_W / img.shape[1],
            "vy": cy * VIEW_W / img.shape[1],
        })
    return out


def label_profile(path):
    """The centre label, unrolled into a strip indexed by angle, with the
    lighting removed so that only the PRINT is left.

    Unrolling first turns a rotation of the disc into a sideways slide of the
    strip, which is the whole point — after this there is nothing to search for
    but an offset.

    Then the lighting has to go. A label is matte paper and carries no
    reflection, but a lamp standing to one side still makes the side facing it
    brighter, exactly as it would a sheet of paper on a desk. That gradient is
    the strongest thing in the strip and it belongs to the ROOM, not the record:
    measuring it returned the same ~280 degrees for every record in the set, and
    it is what defeated the first attempt at this two months ago as well.
    Lighting varies SLOWLY across the strip while print makes sharp jumps, so
    blurring hard leaves only the lighting, and subtracting that leaves only the
    print. Rows are then normalised individually so exposure cannot creep back
    in.
    """
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    polar = cv2.warpPolar(gray, (radius, POLAR_STEPS), center, radius,
                          cv2.WARP_POLAR_LINEAR)
    strip = cv2.transpose(polar)[int(LABEL_LO * radius):int(LABEL_HI * radius)]
    if strip.size < POLAR_STEPS:
        return None
    s = strip.astype(np.float32)
    # wrap the ends round before blurring: the strip is a circle cut open, and a
    # blur that treats its edges as edges invents a seam that never existed
    pad = BLUR_W
    wide = np.hstack([s[:, -pad:], s, s[:, :pad]])
    lighting = cv2.GaussianBlur(wide, (BLUR_W * 2 + 1, 1), 0)[:, pad:pad + s.shape[1]]
    s = s - lighting
    s -= s.mean(axis=1, keepdims=True)
    return s / np.maximum(s.std(axis=1, keepdims=True), 1e-3)


def anchor_angle(strip):
    """One agreed-upon reference direction per photo, read off the label.

    Rather than comparing two photos to each other, each photo is reduced to a
    single number that turns with the disc. Take the label's brightness as it
    varies around the circle and keep the FIRST harmonic — the component that
    completes exactly one cycle per revolution. Its phase points somewhere
    specific on the label and nowhere else, so it is a mark in the sense asked
    for: fixed to the record, identical in every photo of it, and computed
    before the label is cropped away.

    The first harmonic is the right one because it is the only one without an
    ambiguity: a k-th harmonic repeats k times per turn and so cannot say which
    of its k identical positions you are looking at.

    Confidence is that harmonic's strength against the rest of the spectrum. A
    plain single-colour label has no direction to give and is rejected here
    rather than answering at random.
    """
    if strip is None:
        return None, 0.0
    prof = strip.mean(axis=0)
    prof = prof - prof.mean()
    F = np.fft.rfft(prof)
    mag = np.abs(F)
    if mag.size < 12 or mag[1] <= 0:
        return None, 0.0
    conf = float(mag[1] / max(mag[2:12].mean(), 1e-6))
    if conf < ANCHOR_MIN_CONF:
        return None, conf
    return float(np.degrees(np.angle(F[1])) % 360.0), conf


def rotation_from_label(pa, pb):
    """Circular cross-correlation of the two label strips, via FFT.

    A rotation of the disc is a cyclic shift of the strip, so the correlation
    peak IS the rotation — no search, no optimiser to get stuck at the edge of
    its range the way the first attempt did.
    """
    if pa is None or pb is None:
        return None, 0.0
    # The strips are as tall as the disc's radius IN PIXELS, which changes with
    # how close the phone was held — a median of two rows out of a few hundred
    # between two shots of the same side. Refusing to compare on that threw out
    # 54 of 74 sides, and every side that got past it aligned. Stretch to a
    # common height instead; the radius axis is only there to give the match
    # more evidence, so a couple of rows of resampling costs nothing.
    if pa.shape[0] != pb.shape[0]:
        h = min(pa.shape[0], pb.shape[0])
        pa = cv2.resize(pa, (pa.shape[1], h), interpolation=cv2.INTER_AREA)
        pb = cv2.resize(pb, (pb.shape[1], h), interpolation=cv2.INTER_AREA)
    corr = np.fft.irfft(np.fft.rfft(pb, axis=1) *
                        np.conj(np.fft.rfft(pa, axis=1)), axis=1).sum(axis=0)
    k = int(corr.argmax())
    peak = float(corr[k])
    # judge the peak against the rest of the curve, with its own neighbourhood
    # excluded so a broad peak does not dilute its own background
    mask = np.ones(corr.size, bool)
    w = max(corr.size // 60, 3)
    mask[(np.arange(corr.size) - k) % corr.size <= w] = False
    mask[(k - np.arange(corr.size)) % corr.size <= w] = False
    bg, sd = corr[mask].mean(), corr[mask].std()
    if sd <= 0:
        return None, 0.0
    ratio = (peak - bg) / sd
    if ratio < LABEL_MIN_RATIO:
        return None, ratio
    return (k * 360.0 / corr.size) % 360.0, ratio


def find_rotation(a, b):
    """The angle that maps photo A's disc onto photo B's, by voting.

    Only pairs at the same radius may vote, which is what keeps the histogram
    from filling with noise: two defects at different radii cannot be the same
    defect however well their angles happen to line up.
    """
    nbins = int(round(360 / BIN_DEG))
    hist = np.zeros(nbins)
    for p in a:
        for q in b:
            if abs(p["rad"] - q["rad"]) > RAD_TOL:
                continue
            hist[int(((q["ang"] - p["ang"]) % 360.0) / BIN_DEG) % nbins] += 1
    if hist.sum() < MIN_VOTES:
        return None, 0.0, 0
    # a defect's angle is only good to a couple of degrees, so let a vote count
    # for its neighbours too rather than falling off a bin edge
    smooth = hist + np.roll(hist, 1) + np.roll(hist, -1)
    k = int(smooth.argmax())
    peak = float(smooth[k])
    ratio = peak / max(smooth.mean(), 1e-6)
    if ratio < MIN_PEAK_RATIO or peak < MIN_VOTES:
        return None, ratio, int(hist.sum())
    return (k * BIN_DEG) % 360.0, ratio, int(hist.sum())


def confirmed(p, others, delta):
    """Is there a detection in the other shot at the same spot on the disc?"""
    want = (p["ang"] + delta) % 360.0
    for q in others:
        if abs(p["rad"] - q["rad"]) > RAD_TOL:
            continue
        d = abs((q["ang"] - want + 180.0) % 360.0 - 180.0)
        if d <= ANG_TOL:
            return True
    return False


def load_labels(which):
    tool = os.path.join(HERE, f"label_tool_{which}")
    path = os.path.join(tool, f"labels_{which}.json")
    if not os.path.exists(path):
        return {}
    out = {}
    for r in json.load(open(path, encoding="utf-8"))["rows"]:
        if r["label"] in ("scratch", "dirt", "false"):
            out.setdefault(r["pair"], []).append(r)
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    sets = ("cal", "val") if which == "both" else (which,)

    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        index = list(csv.DictReader(fh))
    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    wanted = set()
    for s in sets:
        wanted |= set(split[s])
    rows = [r for r in index if r["record"] in wanted]

    labels = {}
    for s in sets:
        labels.update(load_labels(s))

    # detections per photo, grouped by the SIDE they belong to
    sides = collections.defaultdict(list)
    for r in rows:
        path = os.path.join(PHOTOS, r["photo_file"])
        if not os.path.exists(path):
            continue
        try:
            pts = polar_detections(path)
            prof = label_profile(path)
        except Exception as exc:
            print(f"  skip {r['pair'][:40]}: {type(exc).__name__}")
            continue
        sides[(r["record"], r["side"])].append((r["pair"], pts, prof))
        print(f"  {r['pair'][:46]:<48}{len(pts):>4} detections")

    tally = collections.Counter()
    how_count = collections.Counter()
    agree = collections.Counter()
    failed, singles = 0, 0
    conf_flag = {}

    for key, shots in sorted(sides.items()):
        if len(shots) < 2:
            singles += len(shots)
            continue
        for i, (pair_i, pts_i, prof_i) in enumerate(shots):
            ok = [False] * len(pts_i)
            best = "none"
            for j, (pair_j, pts_j, prof_j) in enumerate(shots):
                if i == j:
                    continue
                # Matching the printed pattern is the primary method and the
                # only one measured against ground truth: on every side where
                # the hand marks could also answer, it agreed within 8 degrees,
                # median 4. anchor_angle is deliberately NOT used — it reads a
                # single harmonic, which on a high-passed strip is noise, and
                # before high-passing it was reading the room's lighting.
                # Matching the printed pattern is the ONLY method used. The
                # detection vote was dropped: it was never checked against the
                # hand marks, it aligned one side in 74 on its own, and a wrong
                # rotation turns "confirmed" into coincidence, which would drag
                # a verified result down to meet an unverified one.
                delta, ratio = rotation_from_label(prof_i, prof_j)
                how = "print"
                if i < j:
                    if delta is None:
                        failed += 1
                    else:
                        how_count[how] += 1
                if delta is None:
                    continue
                # remember the STRONGEST evidence this photo was aligned by, so
                # the result can be read separately for the method that was
                # checked against ground truth and the one that was not
                if best != "print":
                    best = how
                for k, p in enumerate(pts_i):
                    if not ok[k] and confirmed(p, pts_j, delta):
                        ok[k] = True
            conf_flag[pair_i] = (pts_i, ok, best)

    # join the confirmation flags onto the hand-labelled verdicts by position
    for pair, rows_l in labels.items():
        if pair not in conf_flag:
            continue
        pts, ok, how = conf_flag[pair]
        if not pts:
            continue
        arr = np.array([[p["vx"], p["vy"]] for p in pts], float)
        for r in rows_l:
            d = np.hypot(arr[:, 0] - r["cx"], arr[:, 1] - r["cy"])
            k = int(d.argmin())
            if d[k] > 25:
                continue
            real = r["label"] in ("scratch", "dirt")
            # a side whose label could not be aligned says nothing either way —
            # counting it as "not confirmed" would punish a detection for the
            # label's lack of print rather than for anything about itself
            g = "unknown" if how == "none" else ("confirmed" if ok[k] else "once")
            tally[(g, "real" if real else "false")] += 1
            if r["label"] == "scratch":
                tally[(g, "scratch")] += 1

    print(f"\nsides with two or more shots : "
          f"{sum(1 for v in sides.values() if len(v) > 1)}")
    for k in ("print", "vote"):
        print(f"  rotation from the {k:<10}        : {how_count[k]}")
    print(f"  no rotation found                 : {failed}")
    print(f"  photos with no partner shot       : {singles}")

    head = f"\n{'group':<26}{'real':>7}{'false':>7}{'total':>7}{'PRECISION':>12}"
    print(head)
    print("-" * len(head))
    groups = (("confirmed", "seen in BOTH shots"), ("once", "seen once only"),
              ("unknown", "side not alignable"))
    for g, name in groups:
        real, fake = tally[(g, "real")], tally[(g, "false")]
        tot = real + fake
        if not tot:
            continue
        print(f"{name:<26}{real:>7}{fake:>7}{tot:>7}{100*real/tot:>11.0f}%")

    both = sum(tally[(g, x)] for g, _ in groups for x in ("real", "false"))
    r_all = sum(tally[(g, "real")] for g, _ in groups)
    if both:
        print(f"{'everything (today)':<26}{r_all:>7}{both-r_all:>7}{both:>7}"
              f"{100*r_all/both:>11.0f}%")
    dec = sum(tally[(g, x)] for g in ("confirmed", "once") for x in ("real", "false"))
    if dec:
        r_dec = sum(tally[(g, "real")] for g in ("confirmed", "once"))
        print(f"\n  of the {dec} detections on sides we COULD align, "
              f"{100*tally[('confirmed','real')]/max(r_dec,1):.0f}% of the real ones "
              f"were confirmed")
    print(f"\nhand-marked scratches kept by the cross-check : "
          f"{tally[('confirmed','scratch')]} of "
          f"{tally[('confirmed','scratch')] + tally[('once','scratch')]}")



if __name__ == "__main__":
    main()
