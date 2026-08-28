# -*- coding: utf-8 -*-
"""
Show what the model got WRONG on one record, using the hand-labelled verdicts.

One record can dominate the error count — on the validation set a single one
produced 36 of 96 genuinely-false detections across all four of its photos, both
sides. That is not a bad shot, it is something about the record itself, and the
only way to find out what is to look at the pixels the model fired on.

Two outputs per record:
  overview   each photo, with the false detections boxed and numbered and the
             correct ones drawn faintly, so the errors can be seen in context
  closeups   a contact sheet of the same detections zoomed in, numbered to
             match, since a false detection is usually a few pixels wide and
             invisible at full-frame scale

Shapes carry the meaning, not colour alone: a WRONG call is a square, a correct
one is a circle.

Usage:  python show_record_errors.py <record> [cal|val]
"""

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(os.path.dirname(HERE), "Model_Errors")

WHITE = (245, 245, 245)
ORANGE = (0, 150, 255)
BAR = 96
COLS = 6


def label_bubble(vis, x, y, text, colour):
    cv2.circle(vis, (x, y), 19, (15, 15, 15), -1, cv2.LINE_AA)
    cv2.circle(vis, (x, y), 19, colour, 3, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(vis, text, (x - tw // 2, y + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)


def caption(img, lines):
    w = img.shape[1]
    strip = np.full((BAR, w, 3), 22, np.uint8)
    scale = max(w / 1200.0, 0.55)
    y = int(38 * scale) + 8
    for text, colour in lines:
        cv2.putText(strip, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72 * scale, colour, max(1, int(2 * scale)), cv2.LINE_AA)
        y += int(32 * scale)
    return np.vstack([strip, img])


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python show_record_errors.py <record> [cal|val]")
    record = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "val"
    tool = os.path.join(HERE, f"label_tool_{which}")
    out = os.path.join(OUT_ROOT, record[:40])
    os.makedirs(out, exist_ok=True)

    index = json.load(open(os.path.join(tool, "index.json"), encoding="utf-8"))
    by_id = {e["id"]: e for e in index["items"]}
    rows = json.load(open(os.path.join(tool, f"labels_{which}.json"),
                          encoding="utf-8"))["rows"]

    mine = [r for r in rows if r["record"] == record and r["label"]]
    if not mine:
        sys.exit(f"no labelled detections for {record} in {which}")

    by_pair = {}
    for r in mine:
        by_pair.setdefault(r["pair"], []).append(r)

    n_false = 0
    crops = []
    for pair in sorted(by_pair):
        group = by_pair[pair]
        e0 = by_id[group[0]["id"]]
        view = cv2.imdecode(np.fromfile(
            os.path.join(tool, e0["photo"].replace("/", os.sep)), np.uint8),
            cv2.IMREAD_COLOR)
        if view is None:
            continue
        vis = view.copy()

        # correct calls first and faint, so they sit behind the errors
        for r in group:
            if r["label"] in ("dirt", "scratch"):
                e = by_id[r["id"]]
                cv2.circle(vis, (int(e["cx"]), int(e["cy"])),
                           int(e["r"]) + 6, ORANGE, 1, cv2.LINE_AA)

        wrong = [r for r in group if r["label"] == "false"]
        for r in wrong:
            e = by_id[r["id"]]
            n_false += 1
            x, y, rad = int(e["cx"]), int(e["cy"]), int(e["r"]) + 10
            cv2.rectangle(vis, (x - rad, y - rad), (x + rad, y + rad),
                          (15, 15, 15), 7, cv2.LINE_AA)
            cv2.rectangle(vis, (x - rad, y - rad), (x + rad, y + rad),
                          WHITE, 3, cv2.LINE_AA)
            label_bubble(vis, x + rad + 4, y - rad - 4, str(n_false), WHITE)

            crop = cv2.imdecode(np.fromfile(
                os.path.join(tool, e["crop"].replace("/", os.sep)), np.uint8),
                cv2.IMREAD_COLOR)
            if crop is not None:
                crops.append((n_false, crop))

        n_ok = len(group) - len(wrong)
        vis = caption(vis, [
            (f"{record}   side {group[0]['side']}   shot {group[0]['shot']}", WHITE),
            (f"[] SQUARE = the model was WRONG here  ({len(wrong)})", WHITE),
            (f"O  circle = correct call, dirt or a scratch  ({n_ok})", ORANGE),
        ])
        name = f"photo_{group[0]['side']}{group[0]['shot']}.jpg"
        cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(
            os.path.join(out, name))
        print(f"  {name:<20}{len(wrong):>4} wrong,{n_ok:>4} correct")

    if crops:
        size = 300
        tiles = []
        for k, crop in crops:
            t = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
            strip = np.full((34, size, 3), 22, np.uint8)
            cv2.putText(strip, str(k), (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, WHITE, 2, cv2.LINE_AA)
            tiles.append(np.vstack([strip, t]))
        while len(tiles) % COLS:
            tiles.append(np.full_like(tiles[0], 22))
        sheet = np.vstack([np.hstack(tiles[i:i + COLS])
                           for i in range(0, len(tiles), COLS)])
        sheet = caption(sheet, [
            (f"{record} - every detection you marked as WRONG, zoomed", WHITE),
            ("the orange ring shows exactly which pixels fired", ORANGE),
        ])
        cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(
            os.path.join(out, "closeups.jpg"))

    print(f"\n{n_false} wrong detections on {record}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
