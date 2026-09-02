# -*- coding: utf-8 -*-
"""Paint one model's marks onto the calibration records, to be judged by eye.

The painting is the server's own -- the same soft yellow, the same translucency,
the same halo drawn wider than the mark so a hairline shows at all, and the same
clip at the edge of the playing surface. What the gallery shows is what the app
would show.

Usage:  python paint_gallery.py <config> [how many records]

        strict   both shots intersected, the three rules left on, the bar as it
                 ships -- the fewest marks of anything measured
        wide     both shots intersected, rules off, bar lowered
        server   what runs today, for reference
"""

import os
import shutil
import sys

import numpy as np

from compare_precision import BAR_NOW, BAR_OPEN, BAR_WIDE, detect_with
from model_v2 import align, maps_of
from test_01_loosen_then_confirm import (TESTS, gt_for, inside_disc, measure,
                                         paint, pick_sides, save)

CONFIGS = {
    #          combine  rules_off  bar        folder
    "strict": (True,    False,     BAR_NOW,   "09_v2_strict_rules_on_bar_now"),
    "opened": (True,    False,     BAR_OPEN,  "10_v2_rules_on_bar_opened"),
    "wide":   (True,    True,      BAR_WIDE,  "08_v2_rules_off_bar_wide"),
    "server": (False,   False,     BAR_NOW,   "11_server_today"),
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "strict"
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    if which not in CONFIGS:
        sys.exit(f"config must be one of {', '.join(CONFIGS)}")
    combine, rules_off, bar, folder = CONFIGS[which]

    out = os.path.join(TESTS, folder)
    if os.path.isdir(out):
        shutil.rmtree(out)
    for sub in ("1_the_record", "2_what_the_server_marks_today",
                "3_what_this_model_marks"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    chosen = pick_sides(want)
    seen, n = set(), 0
    found = zones = shown = photos = 0

    for rec, side, shots in chosen:
        (pair_a, path_a), (_, path_b) = shots
        try:
            a, b = maps_of(path_a), maps_of(path_b)
        except Exception as exc:
            print(f"  {rec} {side}: failed ({type(exc).__name__})")
            continue
        gt = gt_for(pair_a, a["img"].shape)
        if gt is None:
            continue
        delta, _, _ = align(a, b)

        det = detect_with(a, b, delta, combine, rules_off, bar)
        f, z, _, s = measure(det, gt)
        found, zones, shown, photos = found + f, zones + z, shown + s, photos + 1

        if rec in seen:                 # one side per record, as the others do
            continue
        seen.add(rec)
        n += 1
        old = detect_with(a, b, delta, False, False, BAR_NOW)
        ins = inside_disc(a["center"], a["radius"], a["img"].shape)
        stem = f"{n:02d}_{rec[:26]}_{side}"
        save(a["img"], os.path.join(out, "1_the_record", f"{stem}.jpg"))
        save(paint(a["img"], old, ins),
             os.path.join(out, "2_what_the_server_marks_today", f"{stem}.jpg"))
        save(paint(a["img"], det, ins),
             os.path.join(out, "3_what_this_model_marks", f"{stem}.jpg"))
        print(f"  {stem:<40} server {measure(old, gt)[3]:>3} marks"
              f"   this model {s:>3}   scratches {f}/{z}")

    print(f"\n{n} records painted, measured over {photos} sides")
    print(f"  marks a photo : {shown/max(photos,1):.1f}")
    print(f"  recall        : {100*found/max(zones,1):.1f}%"
          f"   ({found} of {zones})")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
