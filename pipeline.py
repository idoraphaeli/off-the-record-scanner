import cv2
import numpy as np

MAX_DIM = 1600

# Must match the framing guide circles in the app, as a fraction of half the
# image's shorter side. DISC = outer guide; LABEL = inner guide (of disc radius).
DISC_RADIUS_RATIO = 0.95
LABEL_RADIUS_RATIO = 0.40
POLAR_ANGLE_STEPS = 3600   # angular resolution of the unwrap: one column per 0.1°
OUTER_TRIM_RATIO = 0.97


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


if __name__ == "__main__":
    import sys
    img = load_image(sys.argv[1])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    center, inner_r, outer_r = find_disc(gray)
    print("center:", center, "inner:", inner_r, "outer:", outer_r)

    polar = unwrap(gray, center, outer_r)
    print("polar shape:", polar.shape, "(rows=radius, cols=angle)")
    ring = crop_ring(polar, inner_r)
    print("ring shape:", ring.shape)
    cv2.imencode(".jpg", ring)[1].tofile("crop_check.jpg")
    print("saved crop_check.jpg")