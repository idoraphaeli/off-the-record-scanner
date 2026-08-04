# -*- coding: utf-8 -*-
"""Cross-check the angular offset using the LABEL as the anchor.

The printed label rotates rigidly with the record and carries high-contrast text,
so it is a far stronger alignment signal than groove texture -- and unlike the
grooves it cannot be confused with a lighting pattern.
"""

import os
import sys

import cv2
import numpy as np

import detector
from detector import P

HERE = os.path.dirname(os.path.abspath(__file__))


def label_strip(path):
    img = detector.load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, radius = detector.find_disc(img)
    polar = detector.unwrap(gray, center, radius)
    lo, hi = int(0.12 * radius), int(0.36 * radius)   # inside the label only
    strip = polar[lo:hi].astype(np.float32)
    strip -= cv2.blur(strip, (61, 1))                 # drop lighting gradient
    return strip, radius


def main():
    folder = sys.argv[1]
    paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    a, _ = label_strip(paths[0])
    b, _ = label_strip(paths[1])
    w = a.shape[1]

    pa = np.abs(a).mean(axis=0)
    pb = np.abs(b).mean(axis=0)
    pa = (pa - pa.mean()) / (pa.std() + 1e-9)
    pb = (pb - pb.mean()) / (pb.std() + 1e-9)
    scores = np.array([float(np.dot(pa, np.roll(pb, s))) for s in range(0, w, 2)])
    best = int(np.argmax(scores)) * 2
    sharp = (scores.max() - scores.mean()) / (scores.std() + 1e-9)
    print(f"label-based offset: {360.0*best/w:6.1f} deg   peak sharpness {sharp:.1f}")

    # full 2D check on the label strip itself, which uses the text layout too
    scores2 = np.array([float((a * np.roll(b, s, axis=1)).mean())
                        for s in range(0, w, 4)])
    best2 = int(np.argmax(scores2)) * 4
    sharp2 = (scores2.max() - scores2.mean()) / (scores2.std() + 1e-9)
    print(f"label 2D offset   : {360.0*best2/w:6.1f} deg   peak sharpness {sharp2:.1f}")

    os.makedirs(os.path.join(folder, "analysis"), exist_ok=True)
    vis = np.vstack([
        np.clip(np.abs(a) * 4, 0, 255).astype(np.uint8),
        np.full((6, w), 255, np.uint8),
        np.clip(np.abs(np.roll(b, best2, axis=1)) * 4, 0, 255).astype(np.uint8)])
    cv2.imencode(".jpg", vis)[1].tofile(
        os.path.join(folder, "analysis", "label_align.jpg"))
    print("wrote analysis/label_align.jpg (top: shot A, bottom: shot B aligned)")


if __name__ == "__main__":
    main()
