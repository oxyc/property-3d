"""Loop-closure error for walks that start and end on the same marked spot.

usage: closure.py <walk_dir> [poses.npz]

Averages the camera pose over the first and last 2 s (the phone resting on the mark)
and prints how far apart they are, using raw ARKit poses and, if given, corrected poses.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script dir
import fuse  # noqa: E402

STILL = 2.0  # seconds averaged at each end


def ends(frames, T):
    t = np.array([f["t"] for f in frames])
    out = []
    for sel in (t <= t[0] + STILL, t >= t[-1] - STILL):
        pos = T[sel, :3, 3].mean(0)
        fwd = -T[sel, :3, 2].mean(0)  # camera looks along -Z
        out.append((pos, np.degrees(np.arctan2(fwd[0], -fwd[2]))))  # heading, clockwise from -Z
    return out


def report(name, frames, T):
    (p0, h0), (p1, h1) = ends(frames, T)
    d = p1 - p0
    dh = (h1 - h0 + 180) % 360 - 180
    print(f"{name}: horizontal {np.hypot(d[0], d[2]) * 100:.1f} cm, vertical {d[1] * 100:+.1f} cm, heading {dh:+.2f} deg")


walk = Path(sys.argv[1])
_, frames = fuse.read_walk(walk)
frames = [f for f in frames if f["tracking"] == "normal"]
path = np.linalg.norm(np.diff([f["T"][:3, 3] for f in frames], axis=0), axis=1).sum()
print(f"{len(frames)} frames, {frames[-1]['t'] - frames[0]['t']:.0f} s, path {path:.0f} m")
report("raw      ", frames, np.array([f["T"] for f in frames]))
if len(sys.argv) > 2:
    z = np.load(sys.argv[2])
    by_i = dict(zip(z["i"].tolist(), z["T"]))
    report("corrected", frames, np.array([by_i[f["i"]] for f in frames]))
