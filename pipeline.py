import cv2
import numpy as np

MAX_DIM = 1600

# --- app framing contract ---
DISC_RADIUS_RATIO = 0.95
LABEL_RADIUS_RATIO = 0.40

# --- geometry ---
POLAR_ANGLE_STEPS = 3600   # angular resolution of the unwrap: one column per 0.1 deg
OUTER_TRIM_RATIO = 0.97

# --- scratch detection ---
ROW_FLATTEN_LEN = 151      # px: window for removing each row's local background
TOPHAT_WIDTH = 15          # px: anything wider than this horizontally is NOT a scratch
LINE_BOOST_LEN = 21        # px: how far to average along a line to lift it above noise
LINE_ANGLES = (-60, -40, -20, 0, 20, 40, 60)   # orientations to search, deg from vertical
SCRATCH_THRESHOLD = 6      # threshold on the boosted map
GLARE_BRIGHTNESS = 200     # gray level above which a pixel may be lamp glare
GLARE_MARGIN = 11          # px: safety margin erased around confirmed glare blobs
SCRATCH_WEAK = 2           # weak threshold: could-be-scratch, kept only if connected to a strong seed (Canny-style hysteresis)

# --- shape filter ---
MIN_SCRATCH_LEN = 40       # px: shorter than this is dust/noise, not a scratch
MAX_AVG_THICKNESS = 8      # px: fatter than this is a blob/glare, not a scratch
MIN_ELONGATION = 4         # length/thickness: a scratch is long and thin

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
    # 1. flatten each row: removes ring-to-ring brightness and uneven lighting
    flat = ring.astype(np.float32)
    flat -= cv2.blur(flat, (ROW_FLATTEN_LEN, 1))
    flat = cv2.GaussianBlur(flat, (3, 3), 0)

    # 2. white top-hat with a WIDE-flat element: keeps only what is bright AND
    #    horizontally narrow. Grooves (horizontally long) are erased; the edge of
    #    a broad glare blob is erased; a thin scratch line survives. This is a
    #    true LINE detector, unlike Sobel which also fires on edges of blobs.
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_WIDTH, 1))
    ridge = cv2.morphologyEx(flat, cv2.MORPH_TOPHAT, se)

    # 3. multi-orientation line boost: average along 7 directions, keep the best.
    #    A wavy scratch matches a different direction on each segment; noise
    #    matches none and gets crushed by the averaging.
    best = np.zeros_like(ridge)
    for k in _line_kernels(LINE_BOOST_LEN, LINE_ANGLES):
        best = np.maximum(best, cv2.filter2D(ridge, -1, k))

    # 4. erase confirmed lamp-glare areas
    best[glare_mask(ring) > 0] = 0
    return np.clip(best, 0, 255).astype(np.uint8)

def fuse_stack(maps):
    # Each lighting direction reveals different scratches (a scratch only glints
    # when the light hits it at the right angle). Per-pixel MAX across all the
    # scratch maps keeps every scratch that lit up in ANY of the shots.
    # Glare doesn't survive this well: it moves with the lamp between shots and
    # is soft and blob-shaped, so the shape filter drops what's left of it.
    return np.max(np.stack(maps), axis=0)

def extract_scratches(smap):
    # --- hysteresis: strong seeds + weak flood ---
    weak = (smap > SCRATCH_WEAK).astype(np.uint8)
    strong = smap >= SCRATCH_THRESHOLD
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    seed_labels = set(np.unique(labels[strong])) - {0}   # weak blobs containing a seed
    binary = np.isin(labels, list(seed_labels)).astype(np.uint8) * 255

    # reconnect nearby fragments of the same scratch (bigger kernel than before,
    # because faint scratches break into pieces)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((15, 5), np.uint8))

    # --- shape filter: keep only long-and-thin components ---
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
        if length < MIN_SCRATCH_LEN or thickness > MAX_AVG_THICKNESS:
            continue
        if length / max(thickness, 1) < MIN_ELONGATION:
            continue
        mask[labels == i] = 255
        scratches.append({"length": int(length), "thickness": round(thickness, 1)})
    return mask, scratches

def _line_kernels(length, angles):
    # Build a fan of thin-line averaging kernels, one per orientation.
    size = length | 1  # force odd size
    kernels = []
    for ang in angles:
        k = np.zeros((size, size), np.float32)
        cv2.line(k, (size // 2, 0), (size // 2, size - 1), 1.0, 1)
        M = cv2.getRotationMatrix2D(((size - 1) / 2, (size - 1) / 2), ang, 1.0)
        k = cv2.warpAffine(k, M, (size, size))
        k /= max(k.sum(), 1e-6)   # normalize so output stays in contrast units
        kernels.append(k)
    return kernels

def glare_mask(ring):
    # Lamp glare = pixels that are both VERY BRIGHT and part of a WIDE blob.
    # Opening with a 7x7 block erases thin bright lines (scratch glints survive
    # as scratches!) but keeps wide blobs -> what remains is confirmed glare.
    bright = (ring > GLARE_BRIGHTNESS).astype(np.uint8) * 255
    broad = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return cv2.dilate(broad, np.ones((GLARE_MARGIN, GLARE_MARGIN), np.uint8))

if __name__ == "__main__":
    import sys, os

    def analyze_photo(path):
        # full single-photo analysis: disc -> unwrap -> crop -> map -> scratches
        tag = os.path.splitext(os.path.basename(path))[0]
        img = load_image(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        center, inner_r, outer_r = find_disc(gray)
        ring = crop_ring(unwrap(gray, center, outer_r), inner_r)
        smap = scratch_map(ring)
        mask, scratches = extract_scratches(smap)

        cv2.imencode(".jpg", ring)[1].tofile(f"{tag}_ring.jpg")
        cv2.imencode(".jpg", smap)[1].tofile(f"{tag}_scratchmap.jpg")
        vis = cv2.cvtColor(ring, cv2.COLOR_GRAY2BGR)
        vis[mask > 0] = (0, 255, 255)
        cv2.imencode(".jpg", vis)[1].tofile(f"{tag}_scratches.jpg")

        total_len = sum(s["length"] for s in scratches)
        print(f"{tag}: {len(scratches)} scratches, total length {total_len}px")
        for s in scratches:
            print(f"    length={s['length']}px  thickness={s['thickness']}px")
        return {"tag": tag, "scratches": scratches, "total_len": total_len}

    # --- run: python pipeline.py <photo1> <photo2> ---
    # The two photos are independent views of the same record (handheld, record
    # rotated freely between shots). No pixel alignment is attempted: each photo
    # is analyzed on its own, and the WORSE result wins -- a scratch visible in
    # either photo counts, without double-counting the ones visible in both.
    paths = sys.argv[1:]
    if len(paths) != 2:
        raise SystemExit("usage: python pipeline.py <photo1> <photo2>")

    results = [analyze_photo(p) for p in paths]
    worst = max(results, key=lambda r: r["total_len"])
    print(f"\nfinal (worse of the two photos): {worst['tag']}"
          f" -> {len(worst['scratches'])} scratches, total {worst['total_len']}px")