# -*- coding: utf-8 -*-
"""
Build a working copy of Records_Data_New in which every file is真 JPEG.

Some of the photos are HEIC with a .jpg name — the iPhone's own format, kept
when the files were transferred in a way that did not re-encode them. Windows
apps open them because Windows ships a HEIF codec, but OpenCV brings its own
decoders and has none for HEIC, so they were unreadable to the pipeline.

Originals are never touched: everything is written to a sibling folder, with
already-JPEG files copied verbatim so they are not re-compressed a second time.

Usage:  python convert_heic.py [source] [destination]
"""

import os
import shutil
import sys

import numpy as np
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(HERE), "Records_Data_New")
DST = sys.argv[2] if len(sys.argv) > 2 else SRC + "_jpg"

QUALITY = 95        # high: the marks we are after are hairline-width
MAGIC_JPEG = b"\xff\xd8\xff"


def is_jpeg(path):
    with open(path, "rb") as fh:
        return fh.read(3) == MAGIC_JPEG


def main():
    os.makedirs(DST, exist_ok=True)
    files = sorted(f for f in os.listdir(SRC)
                   if f.lower().endswith((".jpg", ".jpeg")))

    copied = converted = failed = 0
    sizes_changed = []

    for f in files:
        src = os.path.join(SRC, f)
        dst = os.path.join(DST, f)
        if is_jpeg(src):
            shutil.copy2(src, dst)
            copied += 1
            continue
        try:
            with Image.open(src) as im:
                before = im.size
                im = im.convert("RGB")          # HEIC can carry alpha or 10-bit
                im.save(dst, "JPEG", quality=QUALITY, subsampling=0)
            with Image.open(dst) as chk:
                if chk.size != before:
                    sizes_changed.append((f, before, chk.size))
            converted += 1
        except Exception as exc:
            print(f"  FAILED {f}: {exc}")
            failed += 1

    print(f"\n{len(files)} files")
    print(f"  copied unchanged (already JPEG): {copied}")
    print(f"  converted from HEIC            : {converted}")
    print(f"  failed                         : {failed}")
    if sizes_changed:
        print(f"  WARNING: {len(sizes_changed)} changed size during conversion")
        for row in sizes_changed[:5]:
            print(f"     {row}")
    else:
        print("  every converted file kept its original dimensions")

    left = [f for f in sorted(os.listdir(DST)) if not is_jpeg(os.path.join(DST, f))]
    print(f"  non-JPEG remaining in destination: {len(left)}")
    print(f"\nwrote {DST}")


if __name__ == "__main__":
    main()
