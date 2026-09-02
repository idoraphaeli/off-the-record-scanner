# -*- coding: utf-8 -*-
"""How much memory a four-photograph request actually costs.

The detector now holds the response maps of BOTH shots of a side at the same
time, where it used to hold one photograph's worth and drop it. That is the one
part of the change with a known way to fail: holding four photographs' worth at
once is what used to exhaust the container and return a 502 mid-request, and
this doubles the old peak on purpose.

The number that matters is the peak across several requests in a row, not one:
the failure was never the first request, it was the second, when the first
request's arrays were still counted against the limit even though nothing
referenced them.

Usage:  python memcheck.py [how many records]
"""

import ctypes
import glob
import os
import sys
import time

from scanner import analyze_record

PHOTOS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "Records_Data_New_jpg")


class _Counters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def memory_mb():
    """Resident and peak-resident, in megabytes.

    Read from the OS rather than from Python, because the thing that killed the
    service was memory Python had freed and the allocator had not returned.
    """
    if os.name != "nt":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        with open("/proc/self/statm") as fh:
            now = int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
        return now, peak

    c = _Counters()
    c.cb = ctypes.sizeof(_Counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    # the same call lives in two places depending on the Windows version, and a
    # failed one leaves the struct zeroed rather than raising -- which reads as
    # "no memory used" and is worse than no measurement at all
    for lib, name in ((ctypes.windll.kernel32, "K32GetProcessMemoryInfo"),
                      (ctypes.windll.psapi, "GetProcessMemoryInfo")):
        fn = getattr(lib, name, None)
        if fn is None:
            continue
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
        fn.restype = ctypes.c_int
        if fn(handle, ctypes.byref(c), c.cb):
            return c.WorkingSetSize / 1e6, c.PeakWorkingSetSize / 1e6
    raise OSError("could not read this process's memory use")


def pair(record, side):
    files = sorted(f for f in glob.glob(
        os.path.join(PHOTOS, f"{record}_{side}_shot*.jpg")) if "(1)" not in f)
    return [open(f, "rb").read() for f in files[:2]]


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    names = []
    for f in sorted(os.listdir(PHOTOS)):
        rec = f.split("_side")[0]
        if "_side" in f and "(1)" not in f and rec not in names:
            names.append(rec)

    print(f"{'record':<28}{'photos':>8}{'seconds':>9}{'resident':>11}{'peak':>9}")
    print("-" * 65)
    done = 0
    for rec in names:
        a, b = pair(rec, "sideA"), pair(rec, "sideB")
        if len(a) < 2 or len(b) < 2:
            continue
        mb = sum(len(x) for x in a + b) / 1e6
        t = time.time()
        analyze_record({"A": a, "B": b}, want_overlay=True)
        now, peak = memory_mb()
        print(f"{rec[:26]:<28}{mb:>7.1f}M{time.time()-t:>9.1f}"
              f"{now:>10.0f}M{peak:>8.0f}M")
        done += 1
        if done >= want:
            break

    now, peak = memory_mb()
    print(f"\npeak across {done} four-photograph requests: {peak:.0f} MB")
    print("Cloud Run is configured with 2048 MB.")


if __name__ == "__main__":
    main()
