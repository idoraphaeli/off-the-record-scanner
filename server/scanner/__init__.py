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


def source_digest():
    """A short hash of each file in this package, as it exists on disk.

    Two days went into a ring of false marks that the deployed service painted
    and an identical local copy did not. Every check available said the two were
    the same: the build was green and came from the right commit, the parameters
    matched, the library versions matched, the detections matched. None of them
    compared the CODE, so none of them could have caught a stale copy of it.

    This does. A digest per file, so a mismatch also names which file is wrong.
    """
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for name in sorted(os.listdir(here)):
        if not name.endswith(".py"):
            continue
        try:
            with open(os.path.join(here, name), "rb") as fh:
                out[name] = hashlib.md5(fh.read()).hexdigest()[:10]
        except OSError:
            out[name] = "unreadable"
    return out


__all__ = ["analyze", "analyze_record", "source_digest"]
