# -*- coding: utf-8 -*-
"""
Build a browser tool for hand-labelling what the detector reports.

Two modes, because there are two different things to check.

  extra    (default) the detections that did NOT land on a pen mark. Those are
           what the evaluator counts as false positives, and the question is how
           many are real — dirt and dust that legitimately lower a record's
           grade — versus glare, groove highlights and rim edges that are
           nothing.

  matched  the detections that DID land on a pen mark. Until now these were
           assumed correct and never looked at, which quietly trusts two things
           at once: that the mask really marks pen, and that the detection
           really sits on it. Ground truth was extracted by differencing the two
           copies of a pair and keeping what is pen-blue, and blue-lit vinyl and
           cyan labels both pass a colour test — so a mask can contain things
           that were never drawn. Anything wrong here inflates recall directly.
           This mode shows the plain photo and the pen-marked copy of the same
           spot, side by side, so both assumptions can be judged at once.

Detections come from whatever settings detector.py currently holds, so the tool
always asks about the model that would actually ship. Each set and mode gets its
own folder — labelling one must never overwrite another.

Usage:  python build_label_tool.py [cal|val|test] [extra|matched]
"""

import csv
import hashlib
import json
import os
import re
import shutil
import sys

import cv2
import numpy as np

import detector
from detector import P
from evaluate_frozen import TOLERANCE, MIN_EXTRA_AREA, detect

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")
GT = os.path.join(HERE, "gt_new")
TEMPLATE = os.path.join(HERE, "label_page.html")

VIEW_W = 1100          # width of the context photo served to the browser
CROP_OUT = 460         # zoom crop is rendered at this size
CROP_MIN = 190         # never crop tighter than this, or context is lost
INHERIT_PX = 20        # view px within which a previous verdict still applies


def load_previous(out, which):
    """Verdicts from an earlier build of this same tool, keyed by photo.

    A label is a judgement about a spot on a record, so it stays true when the
    detector is rebuilt — only the detection it was attached to moves. Measured
    across both sets, 86% of the verdicts land on a detection the model still
    reports, so re-asking for them would be throwing away most of a day's work.
    Read before anything is written, since the folder is about to be rewritten.
    """
    idx_path = os.path.join(out, "index.json")
    lab_path = os.path.join(out, f"labels_{which}.json")
    if not (os.path.exists(idx_path) and os.path.exists(lab_path)):
        return {}
    index = json.load(open(idx_path, encoding="utf-8"))
    by_id = {e["id"]: e for e in index["items"]}
    prev = {}
    for r in json.load(open(lab_path, encoding="utf-8"))["rows"]:
        e = by_id.get(r["id"])
        if e and r.get("label"):
            prev.setdefault(r["pair"], []).append((e["cx"], e["cy"], r["label"]))
    return prev


def inherit(prev_for_pair, cx, cy):
    best, bd = None, INHERIT_PX ** 2
    for px, py, lab in prev_for_pair:
        d = (px - cx) ** 2 + (py - cy) ** 2
        if d <= bd:
            best, bd = lab, d
    return best

COPY_RE = re.compile(r"\(1\)$")
PEN_LO, PEN_HI = (95, 120, 90), (125, 255, 255)


def stem(name):
    return re.sub(r"\.jpe?g$", "", name, flags=re.I).strip()


def pair_key(name):
    return COPY_RE.sub("", stem(name)).lower()


def blobs(mask):
    n, labels, stats, cent = cv2.connectedComponentsWithStats(
        (mask > 127).astype(np.uint8), connectivity=8)
    return [{"cx": int(cent[i][0]), "cy": int(cent[i][1]),
             "w": int(stats[i][2]), "h": int(stats[i][3])}
            for i in range(1, n) if stats[i][4] >= MIN_EXTRA_AREA]


def save_jpg(img, path, quality=86):
    cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tofile(path)


def find_pen_copy(index_by_pair, pair, plain_name, shape):
    """The half of the pair that carries the pen, resized onto the plain one.

    Which half it is cannot be assumed from the filename: the annotator drew on
    whichever copy he happened to open. Blue is counted only where the two files
    DIFFER, because black vinyl has a bluish sheen worth far more pixels than a
    pen stroke and would otherwise decide the answer.
    """
    others = [f for f in index_by_pair.get(pair, []) if f != plain_name]
    if not others:
        return None
    other = detector.load_image(os.path.join(PHOTOS, others[0]))
    plain = detector.load_image(os.path.join(PHOTOS, plain_name))
    if other is None or plain is None:
        return None
    if other.shape != plain.shape:
        other = cv2.resize(other, (plain.shape[1], plain.shape[0]),
                           interpolation=cv2.INTER_AREA)
    changed = cv2.absdiff(plain, other).max(axis=2) > 30
    blue_o = cv2.inRange(cv2.cvtColor(other, cv2.COLOR_BGR2HSV),
                         PEN_LO, PEN_HI) > 0
    blue_p = cv2.inRange(cv2.cvtColor(plain, cv2.COLOR_BGR2HSV),
                         PEN_LO, PEN_HI) > 0
    penned = other if np.count_nonzero(changed & blue_o) >= \
        np.count_nonzero(changed & blue_p) else plain
    if penned.shape[:2] != shape:
        penned = cv2.resize(penned, (shape[1], shape[0]),
                            interpolation=cv2.INTER_AREA)
    return penned


def crop_window(b, H, W):
    cx, cy = b["cx"], b["cy"]
    half = max(CROP_MIN, int(max(b["w"], b["h"]) * 1.6)) // 2
    return (max(cy - half, 0), min(cy + half, H),
            max(cx - half, 0), min(cx + half, W))


def ringed_crop(img, b, win, thickness=3):
    """A zoom on one detection, with a ring marking the pixels the model fired
    on. Drawn on the crop rather than the full frame so it stays legible."""
    y0, y1, x0, x1 = win
    crop = cv2.resize(img[y0:y1, x0:x1], (CROP_OUT, CROP_OUT),
                      interpolation=cv2.INTER_CUBIC)
    k = CROP_OUT / max(x1 - x0, 1)
    pt = (int((b["cx"] - x0) * k),
          int((b["cy"] - y0) * CROP_OUT / max(y1 - y0, 1)))
    r = int(max(b["w"], b["h"]) * 0.5 * k) + 18
    cv2.circle(crop, pt, r, (15, 15, 15), thickness + 4, cv2.LINE_AA)
    cv2.circle(crop, pt, r, (0, 150, 255), thickness, cv2.LINE_AA)
    return crop


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "cal"
    mode = sys.argv[2] if len(sys.argv) > 2 else "extra"
    if mode not in ("extra", "matched"):
        sys.exit("mode must be extra or matched")

    suffix = "" if mode == "extra" else "_matched"
    out = os.path.join(HERE, f"label_tool_{which}{suffix}")
    previous = load_previous(out, which)     # read BEFORE the folder is rewritten
    os.makedirs(os.path.join(out, "photos"), exist_ok=True)
    os.makedirs(os.path.join(out, "crops"), exist_ok=True)

    split = json.load(open(os.path.join(GT, "split.json"), encoding="utf-8"))
    if which not in split:
        sys.exit(f"unknown set {which}")
    records = set(split[which])
    with open(os.path.join(GT, "gt_index.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record"] in records]

    by_pair = {}
    for f in os.listdir(PHOTOS):
        if f.lower().endswith((".jpg", ".jpeg")):
            by_pair.setdefault(pair_key(f), []).append(f)

    entries, n_photos, skipped = [], 0, 0

    for r in rows:
        photo = os.path.join(PHOTOS, r["photo_file"])
        gt_path = os.path.join(GT, r["pair"] + ".png")
        if not (os.path.exists(photo) and os.path.exists(gt_path)):
            continue

        img, det = detect(photo)
        gt = cv2.imdecode(np.fromfile(gt_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        near_gt = cv2.dilate((gt > 127).astype(np.uint8),
                             np.ones((TOLERANCE, TOLERANCE), np.uint8))

        H, W = img.shape[:2]
        wanted = [b for b in blobs(det)
                  if bool(near_gt[min(b["cy"], H - 1), min(b["cx"], W - 1)])
                  == (mode == "matched")]
        skipped += len(blobs(det)) - len(wanted)
        if not wanted:
            continue

        penned = find_pen_copy(by_pair, r["pair"], r["photo_file"], (H, W)) \
            if mode == "matched" else None

        scale = VIEW_W / float(W)
        view = cv2.resize(img, (VIEW_W, int(round(H * scale))),
                          interpolation=cv2.INTER_AREA)
        view_name = f"photos/{r['pair']}.jpg"
        save_jpg(view, os.path.join(out, view_name.replace("/", os.sep)), 82)
        n_photos += 1

        for b in wanted:
            uid = hashlib.md5(
                f"{r['pair']}|{b['cx']}|{b['cy']}".encode()).hexdigest()[:12]
            win = crop_window(b, H, W)
            crop_name = f"crops/{uid}.jpg"
            save_jpg(ringed_crop(img, b, win),
                     os.path.join(out, crop_name.replace("/", os.sep)))

            entry = {
                "id": uid, "tier": 0,
                "pair": r["pair"], "record": r["record"],
                "side": r["side"], "shot": r["shot"],
                "photo": view_name, "crop": crop_name,
                "cx": round(b["cx"] * scale, 1), "cy": round(b["cy"] * scale, 1),
                "r": round((max(b["w"], b["h"]) * 0.5 + 22) * scale, 1),
                "vw": view.shape[1], "vh": view.shape[0],
            }
            if penned is not None:
                pen_name = f"crops/{uid}_pen.jpg"
                save_jpg(ringed_crop(penned, b, win, thickness=2),
                         os.path.join(out, pen_name.replace("/", os.sep)))
                entry["crop2"] = pen_name

            carried = inherit(previous.get(r["pair"], []), entry["cx"], entry["cy"])
            if carried:
                entry["prefill"] = carried
            entries.append(entry)
        print(f"  {r['pair'][:44]:<46}{len(wanted):>4} to label")

    # grouped by photo, so the eye stays in one place instead of jumping records
    entries.sort(key=lambda e: (e["pair"], e["cy"], e["cx"]))
    for i, e in enumerate(entries):
        e["n"] = i + 1

    data = {"set": which, "mode": mode, "total": len(entries),
            # needed later to compute precision: detections on a pen mark are
            # correct by construction and never appear in the "extra" listing
            "matched": skipped if mode == "extra" else len(entries),
            "items": entries}
    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    # served as a script assignment, not JSON: a browser refuses fetch() on file://
    with open(os.path.join(out, "index.js"), "w", encoding="utf-8") as fh:
        fh.write("window.LABEL_DATA = ")
        json.dump(data, fh, ensure_ascii=False)
        fh.write(";\n")
    shutil.copyfile(TEMPLATE, os.path.join(out, "label.html"))

    missing = sum(1 for e in entries if "crop2" not in e)
    carried = sum(1 for e in entries if "prefill" in e)
    print(f"\nSET = {which}   MODE = {mode}   {n_photos} photos")
    print(f"  detections to label : {len(entries)}")
    print(f"  the other kind      : {skipped}   (not shown in this mode)")
    if previous:
        print(f"  carried over from your earlier labelling : {carried}")
        print(f"  LEFT FOR YOU TO JUDGE                    : "
              f"{len(entries) - carried}")
    if mode == "matched" and missing:
        print(f"  WARNING: {missing} have no pen-marked copy to compare against")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
