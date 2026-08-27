# -*- coding: utf-8 -*-
"""Vinyl scratch scanner: classical computer vision, no machine learning."""

import cv2

# OpenCV starts a worker thread per core and gives each its own scratch buffers,
# which it keeps for the life of the process. On the container this runs in
# that is a straight loss twice over: the instance is allotted a tenth of a CPU,
# so extra threads buy no speed at all, and their buffers are never handed back.
# Measured against the alternative: two analysis requests in a row exhausted the
# 512 MB and the process was killed, returning a 502 on the second.
#
# One thread also makes each request's cost predictable, which matters more here
# than any parallelism would.
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)     # no GPU in the container; the runtime probing
                                # for one only allocates

from .analyze import analyze, analyze_record   # noqa: E402  (after the cv2 setup)

__all__ = ["analyze", "analyze_record"]
