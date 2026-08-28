# -*- coding: utf-8 -*-
"""
Gather every hand-labelled detection into folders by verdict, so the false ones
can be looked through together.

Patterns in false positives are easier to see side by side than one at a time in
a tool that shows a single crop per screen. The crops already exist — this only
sorts them, names them after the record they came from, and lays contact sheets
out grouped by record, since the labelling showed the errors concentrate in a
few discs rather than spreading evenly.

Outputs into Model_Labels/:  false/ dirt/ scratch/ unsure/  and sheets/

Usage:  python collect_labels.py
"""

import collections
import json
import os
import re
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "Model_Labels")

COLS, ROWS = 6, 4
TILE = 300
CAP = 26
BAR = 46
SETS = ("cal", "val")


def safe(name):
    return re.sub(r'[\\/:*?"<>|]', "_", name)[:34]


def main():
    for sub in ("false", "dirt", "scratch", "unsure", "sheets"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    items = []
    for which in SETS:
        tool = os.path.join(HERE, f"label_tool_{which}")
        idx_path = os.path.join(tool, "index.json")
        lab_path = os.path.join(tool, f"labels_{which}.json")
        if not (os.path.exists(idx_path) and os.path.exists(lab_path)):
            print(f"  no labels for {which}, skipping")
            continue
        index = json.load(open(idx_path, encoding="utf-8"))
        by_id = {e["id"]: e for e in index["items"]}
        for r in json.load(open(lab_path, encoding="utf-8"))["rows"]:
            e = by_id.get(r["id"])
            if not e or not r.get("label"):
                continue
            src = os.path.join(tool, e["crop"].replace("/", os.sep))
            if os.path.exists(src):
                items.append({"label": r["label"], "src": src, "set": which,
                              "record": r["record"], "side": r["side"],
                              "shot": r["shot"], "id": r["id"]})

    counts = collections.Counter(i["label"] for i in items)
    print(f"{len(items)} labelled detections\n")

    # grouped by record inside each verdict: the errors were found to cluster on
    # particular discs, so neighbours in a sheet should come from the same one
    items.sort(key=lambda i: (i["label"], i["record"], i["side"], i["shot"]))

    for it in items:
        name = (f"{safe(it['record'])}_{it['side']}{it['shot']}"
                f"_{it['id'][:6]}.jpg")
        dst = os.path.join(OUT, it["label"], name)
        shutil.copyfile(it["src"], dst)

    for label in ("false", "dirt", "scratch", "unsure"):
        group = [i for i in items if i["label"] == label]
        if not group:
            continue
        per = COLS * ROWS
        for page in range((len(group) + per - 1) // per):
            chunk = group[page * per:(page + 1) * per]
            tiles = []
            for it in chunk:
                img = cv2.imdecode(np.fromfile(it["src"], np.uint8),
                                   cv2.IMREAD_COLOR)
                if img is None:
                    continue
                t = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
                strip = np.full((CAP, TILE, 3), 22, np.uint8)
                cv2.putText(strip, f"{it['record'][:22]} {it['side']}{it['shot']}",
                            (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (235, 235, 235), 1, cv2.LINE_AA)
                tiles.append(np.vstack([strip, t]))
            if not tiles:
                continue
            while len(tiles) % COLS:
                tiles.append(np.full_like(tiles[0], 22))
            grid = np.vstack([np.hstack(tiles[i:i + COLS])
                              for i in range(0, len(tiles), COLS)])
            head = np.full((BAR, grid.shape[1], 3), 22, np.uint8)
            cv2.putText(head, f"{label.upper()}   page {page + 1}"
                        f"   ({len(group)} total)", (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2,
                        cv2.LINE_AA)
            sheet = np.vstack([head, grid])
            cv2.imencode(".jpg", sheet,
                         [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(
                os.path.join(OUT, "sheets", f"{label}_{page + 1:02d}.jpg"))
        print(f"  {label:<9}{len(group):>5} crops"
              f"{(len(group) + per - 1) // per:>4} sheets")

    with open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            "Every detection you judged by hand, filed by your verdict.\n\n"
            "  false/    the model was wrong - nothing there\n"
            "  dirt/     real dirt or dust\n"
            "  scratch/  a real scratch you had not marked\n"
            "  unsure/   skipped\n\n"
            "  sheets/   the same crops 24 to a page, grouped by record\n\n"
            "The orange ring in each crop is exactly what the model fired on.\n"
            "Files are named <record>_<side><shot>_<id>, so any crop can be\n"
            "traced back to its photograph.\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
