# -*- coding: utf-8 -*-
"""Why does combining the two shots collapse on some records?

On most sides, taking the pixelwise minimum of the two response maps finds more
scratches than either shot alone. On two of ten it returned almost nothing:
masotav_hamoflaaim side a went from 19 marks and 9 of its 14 scratches to a
single mark and none of them.

There are two candidate explanations and they call for opposite fixes.

  the angle is wrong    we are comparing each spot to the wrong spot in the
                        other shot, so a scratch is matched against blank vinyl.
                        Fixable -- align better.
  the shots share       one shot lit a part of the disc the other did not, so
  nothing there         there is genuinely no second opinion to combine with.
                        Not fixable by aligning; the method simply does not
                        apply there.

Telling them apart does not need the pen marks. Slide one map past the other
through every rotation and add up how much they agree at each -- the sum of the
minimum, which is large only where both maps are bright in the same place. If
the angle is wrong, that curve has a clear peak SOMEWHERE ELSE than the angle we
used. If the shots share nothing, the curve is flat and there is no right answer
to find.

Usage:  python diag_combine_failure.py
"""

import glob
import os
import sys

import cv2
import numpy as np

import detector
from cross_shot import label_profile, rotation_from_label
from detector import P
from test_03_combine_before_threshold import (TOL_COLS, TOL_ROWS, maps_of,
                                              marks_in_ring)
from tune_alignment import refine

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(HERE), "Records_Data_New_jpg")

STEP = 2            # columns between samples; 3600 columns to the circle, so
                    # each step is 0.2 degrees
# The whole circle, not a window around zero: the rotation between two shots of
# a side is whatever the record happened to be turned to, and in this set it runs
# around 270 degrees. Sweeping a narrow band near zero measures nothing.

CASES = [("masotav_hamoflaaim", "sideA", "collapsed"),
         ("pierre_et_le_loup", "sideB", "collapsed"),
         ("baldi_olier", "sideA", "improved"),
         ("shirim_bezvaaim_tiviim", "sideA", "improved")]


def shots_of(rec, side):
    fs = sorted(f for f in glob.glob(os.path.join(PHOTOS, f"{rec}_{side}_shot*.jpg"))
                if "(1)" not in f)
    return fs[:2]


def agreement_curve(a, b):
    """How much the two maps agree at every rotation, and where that peaks.

    Dilating first and rolling afterwards is the same as rolling and dilating,
    so the slack only has to be applied once for the whole sweep.
    """
    ma = a["radial"]
    mb = b["radial"]
    if mb.shape[0] != ma.shape[0]:
        mb = cv2.resize(mb, (mb.shape[1], ma.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    mb = cv2.dilate(mb, np.ones((TOL_ROWS, TOL_COLS), np.uint8))

    offs = np.arange(0, P["POLAR_STEPS"], STEP)
    energy = np.array([float(np.minimum(ma, np.roll(mb, int(o), axis=1)).sum())
                       for o in offs])
    return offs, energy


def main():
    for rec, side, what in CASES:
        fs = shots_of(rec, side)
        if len(fs) < 2:
            print(f"{rec} {side}: no pair")
            continue
        a, b = maps_of(fs[0]), maps_of(fs[1])

        delta, ratio = rotation_from_label(label_profile(fs[0]), label_profile(fs[1]))
        used, spread, n = delta, None, 0
        ma_n, mb_n = len(marks_in_ring(a)), len(marks_in_ring(b))
        if delta is not None:
            used, moved, spread, n = refine(marks_in_ring(a), marks_in_ring(b), delta)

        offs, energy = agreement_curve(a, b)
        k = int(energy.argmax())
        # the roll that maximises agreement is -delta, so read it back that way
        best_deg = (-offs[k] * 360.0 / P["POLAR_STEPS"]) % 360.0
        # how much the peak stands above the rest, with its own shoulder excluded
        gap = np.minimum(np.abs(offs - offs[k]),
                         P["POLAR_STEPS"] - np.abs(offs - offs[k]))
        far = gap > 60
        bg = float(np.median(energy[far])) if far.any() else float(np.median(energy))
        sd = float(np.std(energy[far])) if far.any() else 1.0
        sharp = (energy[k] - bg) / max(sd, 1e-9)

        # where the angle we actually used sits on that curve
        at_used = None
        if used is not None:
            col = int(round(-used / 360.0 * P["POLAR_STEPS"])) % P["POLAR_STEPS"]
            j = int(np.argmin(np.abs(offs - col)))
            at_used = (energy[j] - bg) / max(sd, 1e-9)
            miss = (used - best_deg + 180.0) % 360.0 - 180.0

        lit_a = float(np.count_nonzero(a["radial"] > 0)) / a["radial"].size
        lit_b = float(np.count_nonzero(b["radial"] > 0)) / b["radial"].size

        print(f"\n{rec}  {side}   ({what})")
        print(f"  angle from the label   {delta if delta is None else round(delta,1)}"
              f"   confidence {ratio:.1f}")
        print(f"  marks to pair on       shot 1 {ma_n}   shot 2 {mb_n}"
              f"   paired {n}")
        print(f"  after correction       "
              f"{used if used is None else round(used,1)}"
              f"   spread {spread}")
        print(f"  best agreement at      {best_deg:.1f} deg"
              f"   standing {sharp:.1f} sd above the rest")
        if at_used is not None:
            print(f"  the angle we used is   {miss:+.1f} deg off that peak,"
                  f"   where agreement is {at_used:.1f} sd")
        print(f"  judgeable area         shot 1 {100*lit_a:.0f}%"
              f"   shot 2 {100*lit_b:.0f}%")


if __name__ == "__main__":
    main()
