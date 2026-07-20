import cv2
import numpy as np

MAX_DIM = 1600

# Must match the framing guide circles in the app, as a fraction of half the
# image's shorter side. DISC = outer guide; LABEL = inner guide (of disc radius).
DISC_RADIUS_RATIO = 0.95
LABEL_RADIUS_RATIO = 0.40
POLAR_ANGLE_STEPS = 3600   # angular resolution of the unwrap: one column per 0.1°
OUTER_TRIM_RATIO = 0.97
GROOVE_SUPPRESS = 0.5   # how hard to cancel the groove direction (explained below)
ALIGN_SEARCH_DEG = 10    # search window (deg) around 180deg for the fine alignment
SCRATCH_THRESHOLD = 25   # a candidate-map pixel counts as a scratch above this


def load_image(path):
    # fromfile + imdecode (not imread) so non-ASCII paths work on Windows
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read image: {path}")

    h, w = img.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1.0:  # shrink only, never upscale
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def _looks_like_disc(gray, center, inner_r, outer_r):
    # Build a mask of the grooved ring: fill to the outer edge, then punch the
    # label hole back out.
    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, center, outer_r, 255, -1)
    cv2.circle(mask, center, inner_r, 0, -1)
    ring_pixels = gray[mask == 255]

    # A real record's grooves make this region textured (high spread); a blank
    # image is flat (near-zero spread). 8 is a tunable threshold.
    return ring_pixels.std() > 8


def find_disc(gray):
    h, w = gray.shape

    center = (w // 2, h // 2)
    outer_radius = int(DISC_RADIUS_RATIO * min(h, w) / 2)
    inner_radius = int(LABEL_RADIUS_RATIO * outer_radius)

    # Server-side check: don't trust the client blindly — confirm a record is
    # actually present before returning a grade.
    if not _looks_like_disc(gray, center, inner_radius, outer_radius):
        raise ValueError("no record detected — image does not match framing")

    return center, inner_radius, outer_radius

def unwrap(gray, center, outer_radius):
    # warpPolar re-samples the disc by (angle, distance-from-center) instead of
    # (x, y), so every concentric groove becomes a straight horizontal line.
    polar = cv2.warpPolar(gray, (outer_radius, POLAR_ANGLE_STEPS),
                          center, outer_radius, cv2.WARP_POLAR_LINEAR)
    # Transpose so rows = radius (center at top), cols = angle. This is what
    # turns grooves horizontal and radial scratches vertical.
    return cv2.transpose(polar)

def crop_ring(polar, inner_radius):
    # In the unwrapped image the row index IS the distance from the center in
    # pixels, so cropping the label and the rim is just slicing rows: keep from
    # the label edge out to just short of the disc edge.
    outer_radius = polar.shape[0]
    return polar[inner_radius:int(outer_radius * OUTER_TRIM_RATIO)]

def scratch_map(ring):
    # Local contrast equalization: evens out uneven lighting so a bright patch
    # doesn't fire stronger than a dark one.
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ring)
    eq = cv2.GaussianBlur(eq, (3, 3), 0)   # small denoise before measuring gradients

    # Sobel measures brightness change in one direction:
    #   gx (dx=1) reacts to VERTICAL lines   -> scratches
    #   gy (dy=1) reacts to HORIZONTAL lines  -> grooves
    gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)

    # Scratch response minus PART of the groove response. Partial (0.5), not full,
    # so a diagonal scratch (which excites both directions) still survives.
    scratch = np.clip(np.abs(gx) - GROOVE_SUPPRESS * np.abs(gy), 0, None)

    # Scale to contrast units (a 3x3 Sobel amplifies ~4x). Do NOT normalize to the
    # image max — that would blow up noise on a clean record.
    return np.clip(scratch / 4.0, 0, 255).astype(np.uint8)

def _best_shift(ring_a, ring_b):
    # The two photos are ~180deg apart, so the true horizontal shift is near half
    # the width. We search only a small window around it (NOT globally) -- this is
    # what stops us from locking onto the glare, which aligns at shift 0.
    # NOTE: we align on the RINGS (groove structure), never on the scratch maps.
    width = ring_a.shape[1]
    base = width // 2
    span = int(width * ALIGN_SEARCH_DEG / 360)
    a = ring_a.astype(np.float32) - float(ring_a.mean())
    b = ring_b.astype(np.float32) - float(ring_b.mean())
    best_shift, best_score = base, -1e18
    for s in range(base - span, base + span + 1, 2):
        score = float(np.sum(a * np.roll(b, s, axis=1)))  # higher = better match
        if score > best_score:
            best_score, best_shift = score, s
    return best_shift


def align_and_confirm(ring_a, map_a, ring_b, map_b):
    shift = _best_shift(ring_a, ring_b)
    map_b_aligned = np.roll(map_b, shift, axis=1)
    # A real scratch appears in BOTH maps at the same spot -> logical AND.
    # Glare appears in only one -> dropped.
    confirmed = (map_a > SCRATCH_THRESHOLD) & (map_b_aligned > SCRATCH_THRESHOLD)
    return confirmed.astype(np.uint8) * 255, shift

if __name__ == "__main__":
    import sys

    def process_image(path):
        img = load_image(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        center, inner_r, outer_r = find_disc(gray)
        ring = crop_ring(unwrap(gray, center, outer_r), inner_r)
        return ring, scratch_map(ring)

    ring_a, map_a = process_image(sys.argv[1])
    ring_b, map_b = process_image(sys.argv[2])
    confirmed, shift = align_and_confirm(ring_a, map_a, ring_b, map_b)
    print("shift:", shift, "  confirmed pixels:", int(np.count_nonzero(confirmed)))
    cv2.imencode(".jpg", confirmed)[1].tofile("confirmed_check.jpg")
    print("saved confirmed_check.jpg")