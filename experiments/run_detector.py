# -*- coding: utf-8 -*-
"""Run the experimental detector on a split set. Usage: run_detector.py <run_name> [cal|test|all]"""

import json
import os
import sys

import cv2
import numpy as np

import detector

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\User\Desktop\Ido private\Computer science\off-the-record\record_pics"
CLEAN_DIR = next(os.path.join(BASE, d) for d in os.listdir(BASE)
                 if os.path.isdir(os.path.join(BASE, d))
                 and "ללא סימונים" in d and "חדש" in d)


def main():
    run_name = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "cal"
    split = json.load(open(os.path.join(HERE, "split.json"), encoding="utf-8"))
    names = split["cal"] + split["test"] if which == "all" else split[which]

    out_dir = os.path.join(HERE, "runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    infos = {}
    for name in names:
        stem = os.path.splitext(name)[0]
        det, overlay, info = detector.detect(os.path.join(CLEAN_DIR, name))
        cv2.imencode(".png", det)[1].tofile(os.path.join(out_dir, stem + "_det.png"))
        cv2.imencode(".jpg", overlay)[1].tofile(os.path.join(out_dir, stem + "_overlay.jpg"))
        infos[stem] = info
        print(f"{stem[-22:]:>24} | disc r={info['radius']} | {info['n_scratches']} scratches")
    with open(os.path.join(out_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"params": detector.P, "images": infos}, f, ensure_ascii=False,
                  indent=2, default=str)


if __name__ == "__main__":
    main()
