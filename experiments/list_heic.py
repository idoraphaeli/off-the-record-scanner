# -*- coding: utf-8 -*-
"""List the files whose CONTENT is not JPEG, whatever their name says.

Windows shows the type from the extension; this reads the first bytes of each
file, which is what actually decides whether OpenCV can open it. iPhone editing
commonly rewrites a photo as HEIC while leaving the .jpg name in place.

Writes heic_files.txt next to the dataset so the list can be acted on.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(HERE), "Records_Data_New")

MAGIC = [(b"\xff\xd8\xff", "JPEG"), (b"\x89PNG", "PNG"),
         (b"GIF8", "GIF"), (b"BM", "BMP"), (b"RIFF", "WEBP")]


def fmt(path):
    with open(path, "rb") as fh:
        head = fh.read(16)
    for magic, name in MAGIC:
        if head.startswith(magic):
            return name
    if head[4:8] == b"ftyp":
        return "HEIC/" + head[8:12].decode("ascii", "replace")
    return "unknown"


files = sorted(f for f in os.listdir(ROOT) if f.lower().endswith((".jpg", ".jpeg")))
bad = [(f, fmt(os.path.join(ROOT, f))) for f in files]
bad = [(f, k) for f, k in bad if not k.startswith("JPEG")]

out = os.path.join(ROOT, "heic_files.txt")
with open(out, "w", encoding="utf-8") as fh:
    fh.write(f"{len(bad)} files whose name says .jpg but whose content is not JPEG\n")
    fh.write("OpenCV cannot read these; they need converting.\n\n")
    for f, k in bad:
        fh.write(f"{k:<12}{f}\n")

print(f"{len(bad)} of {len(files)} files are not really JPEG\n")
for f, k in bad:
    print(f"  {k:<12}{f}")
print(f"\nwrote {out}")
