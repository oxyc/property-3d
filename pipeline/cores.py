"""How many cores the pipeline may use: `--jobs N` on process.py / merge.py / optimize.py / fuse.py (default 3, so a
performance core stays free on a 4P+6E machine).

Open3D already spreads ICP, normals, voxel downsampling and nearest-neighbour queries over every core it finds, so
`--jobs` first caps that (OMP_NUM_THREADS, read when Open3D loads: entry scripts call `setup()` before importing
anything that imports it; measured, 3 threads cost ~10% on those calls). The rest is serial Python/numpy glue,
JPEG decoding and file reads, which gets `jobs` worker threads or processes (`pmap`).
"""
import multiprocessing
import os
import sys

DEFAULT_JOBS = 3


def setup(argv=None):
    """Read --jobs N from argv (default DEFAULT_JOBS), cap the OpenMP threads Open3D will use, return N."""
    argv = sys.argv if argv is None else argv
    jobs = DEFAULT_JOBS
    for k, a in enumerate(argv):
        if a in ("--jobs", "-j") and k + 1 < len(argv):
            jobs = int(argv[k + 1])
        elif a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
    os.environ.setdefault("OMP_NUM_THREADS", str(jobs))  # an explicit setting in the environment wins
    return max(1, jobs)


def pmap(fn, items, jobs):
    """[fn(x) for x in items] in `jobs` worker processes (spawn, so no Open3D state is forked); items and results are
    pickled, so pass paths and ids and let the worker read from disk. fn must be importable (module-level)."""
    items = list(items)
    if jobs <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    workers = min(jobs, len(items))
    before = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(max(1, jobs // workers))  # the workers share the jobs budget, Open3D threads included
    try:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            return pool.map(fn, items, chunksize=1)
    finally:
        if before is None:
            del os.environ["OMP_NUM_THREADS"]
        else:
            os.environ["OMP_NUM_THREADS"] = before
