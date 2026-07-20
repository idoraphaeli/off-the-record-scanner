import cv2
import numpy as np

MAX_DIM = 1600

# Must match the framing guide circles in the app, as a fraction of half the
# image's shorter side. DISC = outer guide; LABEL = inner guide (of disc radius).
DISC_RADIUS_RATIO = 0.95
LABEL_RADIUS_RATIO = 0.40


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


if __name__ == "__main__":
    import sys
    img = load_image(sys.argv[1])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    center, inner_r, outer_r = find_disc(gray)
    print("center:", center, "inner:", inner_r, "outer:", outer_r)

    cv2.circle(img, center, outer_r, (0, 255, 0), 3)     # green: disc edge
    cv2.circle(img, center, inner_r, (0, 165, 255), 3)   # orange: label edge
    cv2.drawMarker(img, center, (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    cv2.imencode(".jpg", img)[1].tofile("disc_check.jpg")
    print("saved disc_check.jpg")