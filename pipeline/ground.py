"""Ground height of a fragment, and how far two fragments that see the same ground disagree about it.

ARKit's gravity is accurate frame by frame, so two fragments of the same ground may differ in height (vertical drift accumulated
along the walk between them) but not in tilt: the offset is one number per pair, measured on the cells both saw (lowest point
per 25 cm cell, cells with tall structure or vegetation left out), robust to cells that really changed (more than 0.5 m from the
median). The pose graph turns these into constraints on height only (merge.py, optimize.py): they bend the chain between the
two fragments instead of tilting frames.
"""
import numpy as np

CELL = 0.25  # m
MIN_POINTS = 5  # per cell
MAX_EXTENT = 0.4  # m: a cell whose points span more than this holds structure or vegetation, not ground
MIN_SHARED = 100  # cells both fragments saw (6 m^2) for an offset to count
REJECT = 0.5  # m: cells whose difference is further than this from the median are real changes
MAX_MAD = 0.10  # m: spread of the differences above which the two fragments do not describe the same surface
CAP_SIGMA_Z = 0.03  # m: the height offset of a pair is never known better than this (the ground itself is 5 cm voxels)

def cells(points):
    """(keys, lowest y, ok) of the 25 cm cells of `points` (N, 3; y up): ok marks ground cells."""
    ix = np.floor(points[:, 0] / CELL).astype(np.int64) + 50000
    iz = np.floor(points[:, 2] / CELL).astype(np.int64) + 50000
    key = ix * 100000 + iz
    y = points[:, 1]
    order = np.lexsort((y, key))
    ks, ys = key[order], y[order]
    first = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    last = np.r_[first[1:] - 1, len(ks) - 1]
    ok = (last - first + 1 >= MIN_POINTS) & (ys[last] - ys[first] <= MAX_EXTENT)
    return ks[first][ok], ys[first][ok]


def edges(cell_sets, centres, usable, max_distance=40.0, max_per_node=12):
    """Height constraints between fragments: [{i, j, dz, mad, n}] for pairs whose centres are within max_distance (xz, m), that
    usable(i, j) accepts and that share enough ground; the pairs with most shared cells first, each fragment in at most
    max_per_node of them (hundreds of pairs of one walk would otherwise outvote its odometry with one piece of evidence)."""
    centres = np.asarray(centres)
    found = []
    for i in range(len(cell_sets)):
        for j in np.flatnonzero(np.linalg.norm(centres[i + 1:] - centres[i], axis=1) < max_distance) + i + 1:
            if usable(i, j):
                r = offset(cell_sets[i], cell_sets[j])
                if r:
                    found.append({"i": i, "j": int(j), **r})
    found.sort(key=lambda e: -e["n"])
    used, out = np.zeros(len(cell_sets), int), []
    for e in found:
        if used[e["i"]] < max_per_node and used[e["j"]] < max_per_node:
            used[e["i"]] += 1
            used[e["j"]] += 1
            out.append(e)
    return out


def info(mad):
    """6x6 pose-graph information (Open3D order: rotation, translation) of a height constraint: only the vertical translation
    is known (to the spread of the cells' differences, never better than CAP_SIGMA_Z); the rest is a whisper that keeps the matrix
    well behaved."""
    m = np.eye(6) * 1e-4
    m[4, 4] = 1 / max(CAP_SIGMA_Z, mad) ** 2
    return m


def translation(dz):
    """The edge measurement of a pair where fragment j's ground is dz above fragment i's: the move that aligns i to j."""
    T = np.eye(4)
    T[1, 3] = dz
    return T


def offset(a, b):
    """Height of b's ground above a's over the cells both have: dict(n, dz, mad) or None. a, b: cells()."""
    common, ia, ib = np.intersect1d(a[0], b[0], assume_unique=True, return_indices=True)
    if len(common) < MIN_SHARED:
        return None
    d = b[1][ib] - a[1][ia]
    med = float(np.median(d))
    d = d[np.abs(d - med) < REJECT]
    if len(d) < MIN_SHARED:
        return None
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    if mad > MAX_MAD:
        return None
    return {"n": int(len(d)), "dz": med, "mad": mad}
