# -*- coding: utf-8 -*-
"""Vinyl scratch scanner: classical computer vision, no machine learning."""

import os

import cv2

# OpenCV starts a worker thread per core and gives each its own scratch buffers,
# which it keeps for the life of the process. On a small instance that is a loss
# twice over: with a fraction of a CPU the extra threads buy no speed at all, and
# their buffers are never handed back. Measured on a 0.1-CPU, 512 MB container,
# two analysis requests in a row exhausted the memory and the process was killed
# mid-flight, returning a 502 on the second.
#
# It is a host decision, not a code one, so the host sets it. One thread is the
# safe default; give it more only where there are cores to use and memory to
# spare.
cv2.setNumThreads(int(os.environ.get("OPENCV_NUM_THREADS", "1")))
cv2.ocl.setUseOpenCL(False)     # no GPU in the container; the runtime probing
                                # for one only allocates

from .analyze import analyze, analyze_record   # noqa: E402  (after the cv2 setup)

__all__ = ["analyze", "analyze_record"]
