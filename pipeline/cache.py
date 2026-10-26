"""Disk cache for work that depends on one walk (and the code that processed it) only, so merging a fourth walk into
three processed ones redoes just the cross-walk work.

Layout (cache_dir is out/cache by default, `--cache DIR` / `--no-cache` on merge.py and process.py):
  <walk name>/unit-<key>/      one unit (a whole walk or one clean segment) optimised: poses.npz, optimize.json, unit.pkl
                               (log lines + the fragment geometry merge.py needs, float32)
  <walk name>/feat-<key>/      SIFT keypoints, descriptors and lidar depths per keyframe (visual.extract)
  pairs/                       results of visual.visual_loops between two units (keyed by both units' poses)

Every key hashes what the entry depends on: the walk's own bytes (frames.jsonl, meta.json, how many depth / rgb files it has,
so an unfinished walk that is copied in later is a new walk), the parameters, CACHE_VERSION and the source of the code that
produced it. Change any of those and the old entry is simply not found (it is not deleted: `rm -r` the cache dir to reclaim
the space, ~1 GB per hour of walking).
Cold and warm runs give identical results: the cached arrays are stored in the precision they are used in, and a cold run
rounds to that precision too.
"""
import hashlib
import inspect
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

CACHE_VERSION = 1
_walks = {}


def sha(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else repr(p).encode())
        h.update(b"\x00")
    return h.hexdigest()


def walk_hash(walk):
    """What identifies a walk: its frame list, meta, and how many depth / rgb files it has (an unfinished copy grows)."""
    walk = Path(walk)
    st = (walk / "frames.jsonl").stat()
    if (walk, st.st_size, st.st_mtime_ns) not in _walks:  # memoised within a process: the files are read once, not per entry
        _walks[(walk, st.st_size, st.st_mtime_ns)] = sha((walk / "frames.jsonl").read_bytes(), (walk / "meta.json").read_bytes(),
                                                         len(os.listdir(walk / "depth")), len(os.listdir(walk / "rgb")))
    return _walks[(walk, st.st_size, st.st_mtime_ns)]


def source(*things):
    """Hash of the source code of modules / functions: any edit to them invalidates what they produced."""
    return sha(*[inspect.getsource(t) for t in things])


def poses_hash(poses):
    if poses is None:
        return "recorded"
    z = np.load(poses)
    return sha(z["i"].tobytes(), z["T"].tobytes())


# ---------------------------------------------------------------- units (optimise + fragments)

def unit_dir(cache_dir, walk, ids, frag_frames, visual, code):
    walk = Path(walk)
    key = sha(walk_hash(walk), CACHE_VERSION, None if ids is None else np.asarray(ids).tobytes(), frag_frames, visual, code)[:16]
    return Path(cache_dir) / walk.name / f"unit-{key}"


def load_unit(entry, out):
    """(lines, fragment data) of a cached unit, with its poses.npz / optimize.json copied into `out`; None on a miss."""
    if not (entry / "done").exists():
        return None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("poses.npz", "optimize.json"):
        shutil.copy(entry / name, out / name)
    with open(entry / "unit.pkl", "rb") as f:
        lines, d = pickle.load(f)
    d["from_cache"] = True
    return lines, d


def store_unit(entry, out, lines, d):
    entry.mkdir(parents=True, exist_ok=True)
    for name in ("poses.npz", "optimize.json"):
        shutil.copy(Path(out) / name, entry / name)
    with open(entry / "unit.pkl", "wb") as f:
        pickle.dump((lines, d), f, protocol=4)
    (entry / "done").write_text("")  # written last: a half-written entry is never read


# ---------------------------------------------------------------- visual.py: features and pairs

def install(cache_dir):
    """Make visual.extract and visual.visual_loops read from / write to the cache (idempotent). Call in every process that
    uses them (worker processes included)."""
    import visual
    if hasattr(visual, "_uncached"):
        visual._cache_dir = Path(cache_dir)
        return
    visual._uncached = (visual.extract, visual.visual_loops)
    visual._cache_dir = Path(cache_dir)
    code = source(visual.extract, visual.sample_depth, visual.lift)
    import cv2
    params = (visual.MATCH_W, visual.N_FEATURES, cv2.__version__, code)

    def extract(walk, kf):
        walk = Path(walk)
        d = visual._cache_dir / walk.name / f"feat-{sha(walk_hash(walk), CACHE_VERSION, params)[:12]}"
        path = d / f"{kf['i']:06d}.npz"
        if path.exists():
            try:
                z = np.load(path)
                return dict(xy=z["xy"].astype(np.float64), desc=z["desc"], d=z["d"].astype(np.float64), scale=float(z["scale"]),
                            size=tuple(int(v) for v in z["size"]))
            except Exception:  # half-written by a killed run
                path.unlink()
        f = visual._uncached[0](walk, kf)
        f = dict(f, xy=f["xy"].astype(np.float32).astype(np.float64), d=f["d"].astype(np.float32).astype(np.float64))  # as stored
        d.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, xy=f["xy"].astype(np.float32), desc=f["desc"], d=f["d"].astype(np.float32), scale=f["scale"], size=np.array(f["size"]))
        tmp.rename(path)
        return f

    loops_code = source(visual)

    def visual_loops(walk_a, walk_b=None, poses_a=None, poses_b=None, init=None, max_pairs=3000, workers=None, log=print, **pair_kw):
        wa = Path(walk_a)
        wb = None if walk_b is None else Path(walk_b)
        key = sha(walk_hash(wa), None if wb is None else walk_hash(wb), poses_hash(poses_a), poses_hash(poses_b),
                  None if init is None else np.round(np.asarray(init), 3).tolist(), max_pairs, sorted(pair_kw.items()), CACHE_VERSION, loops_code)
        path = visual._cache_dir / "pairs" / f"{wa.name}__{'same' if wb is None else wb.name}__{key[:20]}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                out, summary = pickle.load(f)
            log(f"visual: {len(out)}/{summary['pairs']} accepted (from cache)")
            return out, summary
        out, summary = visual._uncached[1](walk_a, walk_b, poses_a, poses_b, init, max_pairs, workers, log, **pair_kw)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump((out, summary), f, protocol=4)
        tmp.rename(path)
        return out, summary

    visual.extract, visual.visual_loops = extract, visual_loops


def uninstall():
    """Put visual.extract / visual.visual_loops back (tests)."""
    import visual
    if hasattr(visual, "_uncached"):
        visual.extract, visual.visual_loops = visual._uncached
        del visual._uncached, visual._cache_dir
