"""Correct ARKit drift with a fragment pose graph.

usage: optimize.py <walk_dir> <out_dir> [--jobs 3]

Stage 1 removes ARKit relocalisation snaps (single-frame pose jumps faster than 3 m/s): the
poses after a snap are the trustworthy ones, so the jump is ramped in linearly over the frames
since the previous snap (or the walk start). Stage 2, the pose graph: the walk
is cut into fragments of ~100 tracking-normal frames, each fused into a 5 cm
cloud in the ARKit world frame. Consecutive fragments are tied by identity edges (ARKit
is locally accurate). Fragments whose camera paths pass close to each other get a
point-to-plane ICP from identity; validated results become uncertain loop edges. Open3D's
Levenberg-Marquardt pose graph optimiser with a line process prunes bad loops.

ARKit's gravity is accurate, so every correction is yaw about +Y plus a translation (no
tilt), both for loop edges and for the final per-fragment result.

Writes to out_dir:
  poses.npz      i[N], T[N,4,4]: corrected camera-to-world for every frame in frames.jsonl order
                 (the fragment corrections interpolated between fragment centres). Frames not listed in i
                 (dropped bad-tracking stretches, see merge.py) are not used downstream.
  optimize.json  stats
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script dir from sys.path
import cores  # noqa: E402

if __name__ == "__main__":
    cores.setup()  # caps the threads Open3D uses; must happen before it is imported

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

import fuse  # noqa: E402

reg = o3d.pipelines.registration

VOXEL = 0.05
FRAG_FRAMES = 100
LOOP_CAM_DIST = 4.0  # camera paths closer than this make a pair a candidate (m)
MIN_LOOP_PATH = 20.0  # ... if they are at least this far apart along the walk (m)
MIN_OVERLAP = 0.15  # fraction of occupied 1 m voxels shared by the two fragments
SCALES = [(0.2, 1.0), (0.1, 0.4), (0.05, 0.15)]  # (voxel, ICP max correspondence distance)
INLIER_DIST = 0.1  # point pairs closer than this count as aligned
NEAR_DIST = 1.0  # points with a neighbour within this are "in the overlap"
MIN_INLIERS = 2000
MIN_INLIER_RATIO = 0.5  # aligned / overlapping points, after the yaw-only projection
MAX_RMSE = 0.05
MAX_SHIFT, MAX_YAW, MAX_TILT = 3.0, 15.0, 3.0  # plausible correction: m, degrees, degrees (ICP tilt before projection)
ODOM_INFO = 1e4  # per-axis information of consecutive-fragment edges
GAP_INFO = 1.0  # same, across a tracking gap
ANCHOR_INFO = 1e5  # ties a fragment that starts at a snap to the start: no correction beyond stage 1
SNAP_SPEED = 3.0  # m/s between consecutive frames; walking is ~1 m/s, relocalisation snaps are 4-10 m/s
GUARD_SHIFT, GUARD_YAW = 0.5, 3.0  # a loop correcting more than this needs another loop to agree with it
GUARD_AGREE_SHIFT, GUARD_AGREE_YAW = 0.3, 3.0  # m, degrees: two loops agree if their corrections differ by less
GUARD_PATH = 100.0  # m of walking: a loop only counts as corroborating if both its ends are this close
FAIL_SPEED, FAIL_WINDOW = 2.0, 11  # m/s: median step speed over this many frames; a person walks at ~1 m/s
FAIL_JUMP = 3.0  # m: a snap this big cannot be ramped away (the ramp assumes the drift grew steadily)
FAIL_JUMP_YAW, FAIL_JUMP_YAW_SHIFT = 15.0, 1.0  # degrees, with at least this much shift (m): also too big to ramp
RUNAWAY_MIN_STEP, RUNAWAY_GAP = 0.15, 5.0  # m, s: a spike is a step this big (and 3x the steps around it); a run has spikes this close
RUNAWAY_MIN_SPIKES, RUNAWAY_GROWTH = 6, 3.0  # a run of at least this many spikes, never shrinking by more than 10%, whose last is this many times its first
FAIL_TRACKING = 1.0 # seconds of tracking != normal that count as a failure (shorter gaps keep the weak identity edge)
FAIL_PAD = 1.0  # seconds dropped either side of a failure: tracking degrades before and recovers after it
MIN_SEGMENT_SECONDS, MIN_SEGMENT_PATH = 10.0, 5.0  # a clean piece shorter than this is dropped (too little to register)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def yaw_of(R):
    return np.arctan2(R[0, 2], R[0, 0])  # rotation about +Y (up)


def yaw_only(T, c):
    """Drop T's tilt, keeping its yaw and where it sends the point c."""
    R = rot_y(yaw_of(T[:3, :3]))
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, T[:3, :3] @ c + T[:3, 3] - R @ c
    return out


def shift_at(T, c):
    return float(np.linalg.norm(T[:3, :3] @ c + T[:3, 3] - c))


def detect_snaps(frames):
    """Positions n where frame n-1 -> n is not plausible motion: ARKit relocalised and the pose jumped."""
    snaps = []
    for n in range(1, len(frames)):
        a, b = frames[n - 1], frames[n]
        if a["tracking"] != "normal" or b["tracking"] != "normal":
            continue
        dist = float(np.linalg.norm(b["T"][:3, 3] - a["T"][:3, 3]))
        rot = np.degrees(np.arccos(np.clip((np.trace(a["T"][:3, :3].T @ b["T"][:3, :3]) - 1) / 2, -1, 1)))
        if dist / max(b["t"] - a["t"], 1e-3) > SNAP_SPEED:  # not rotation: picking up / putting down the phone spins it fast
            if snaps and snaps[-1]["n"] == n - 1:  # glitches come in bursts of consecutive frames: one event
                s = snaps[-1]
                s.update(n=n, t=b["t"], frames=s["frames"] + 1, jump_m=max(s["jump_m"], dist), rot_deg=max(s["rot_deg"], float(rot)))
            else:
                snaps.append({"n": n, "first": n, "t": b["t"], "frames": 1, "jump_m": dist, "rot_deg": float(rot)})
    return snaps


def snap_offset(frames, s, snaps):
    """The correction (yaw about +Y, translation; applied to world points) that moves the poses before snap s
    onto the post-snap ones. The camera really kept moving as it was (mean velocity and yaw rate of the last
    <= 5 steps before the snap, not reaching back over an earlier snap), so the true pose at the first
    post-snap frame is the last pre-snap one advanced by that motion; the recorded one differs by the snap."""
    a, b = s["first"] - 1, s["n"]
    Ta, Tb = frames[a]["T"], frames[b]["T"]
    lo = max([a - 5] + [p["n"] for p in snaps if p["n"] <= a])
    k = a - lo
    step = (Ta[:3, 3] - frames[lo]["T"][:3, 3]) / k if k else np.zeros(3)
    rate = yaw_of(Ta[:3, :3] @ frames[lo]["T"][:3, :3].T) / k if k else 0.0
    yaw = yaw_of(Tb[:3, :3] @ Ta[:3, :3].T) - rate * (b - a)
    return yaw, Tb[:3, 3] - rot_y(yaw) @ (Ta[:3, 3] + step * (b - a))


def remove_snaps(frames, snaps):
    """Stage 1: ARKit's poses after a snap are the trustworthy ones (it just relocalised), the drift is in
    the poses before it. Ramp the snap offset in linearly over the frames since the previous snap (or the
    walk start), so the trajectory is continuous and the post-snap poses are left as they were. The frames
    inside a burst of snap frames are untrustworthy and are interpolated between the corrected frame before
    and the first one after."""
    offsets = [snap_offset(frames, s, snaps) for s in snaps]  # all from the recorded poses
    start = 0
    for s, (yaw, t) in zip(snaps, offsets):
        span = s["first"] - 1 - start
        for n in range(start, s["first"]):
            w = (n - start) / span if span else 1.0
            C, T = rot_y(w * yaw), frames[n]["T"].copy()
            T[:3, :3], T[:3, 3] = C @ T[:3, :3], C @ T[:3, 3] + w * t
            frames[n]["T"] = T
        Ta, Tb = frames[s["first"] - 1]["T"], frames[s["n"]]["T"]
        d_yaw = yaw_of(Tb[:3, :3] @ Ta[:3, :3].T)
        for n in range(s["first"], s["n"]):
            f = (n - s["first"] + 1) / (s["n"] - s["first"] + 1)
            T = Ta.copy()
            T[:3, :3], T[:3, 3] = rot_y(f * d_yaw) @ Ta[:3, :3], Ta[:3, 3] + f * (Tb[:3, 3] - Ta[:3, 3])
            frames[n]["T"] = T
        start = s["n"]


def find_failures(frames):
    """Bad-tracking intervals, padded by FAIL_PAD: ARKit sliding (tracking drifts for seconds, then relocalises),
    which no snap ramp can fix. Found by (a) a sustained implausible speed (median step speed over FAIL_WINDOW
    frames above FAIL_SPEED: single glitches do not move a median), (b) a snap too big to ramp away, (c) a run of
    jumps that keep growing (the poses between them look normal, but each jump corrects more drift than the last),
    (d) a stretch of tracking != normal of at least FAIL_TRACKING seconds. [{first, last, t0, t1, seconds, reasons}]
    with first/last positions in `frames`, inclusive."""
    n = len(frames)
    t = np.array([f["t"] for f in frames])
    P = np.array([f["T"][:3, 3] for f in frames])
    normal = np.array([f["tracking"] == "normal" for f in frames])
    step = np.full(n, np.nan)  # speed of the step into frame k
    ok = normal[1:] & normal[:-1]
    step[1:][ok] = (np.linalg.norm(np.diff(P, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3))[ok]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # windows without a single valid step
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(step, FAIL_WINDOW // 2, constant_values=np.nan), FAIL_WINDOW)
        median = np.nanmedian(windows, axis=1)
    flags = {"sustained speed": np.zeros(n, bool), "jump too big to ramp": np.zeros(n, bool),
             "growing jumps (runaway)": np.zeros(n, bool), "tracking not normal": np.zeros(n, bool)}
    fast = median > FAIL_SPEED
    flags["sustained speed"] = fast | np.r_[fast[1:], False]  # the step into k touches k-1 as well
    snaps = detect_snaps(frames)
    for s in snaps:
        yaw, shift = snap_offset(frames, s, snaps)
        at = frames[s["first"] - 1]["T"][:3, 3]
        size = float(np.linalg.norm(rot_y(yaw) @ at + shift - at))  # how far the offset moves the camera (it turns about the origin)
        if size > FAIL_JUMP or (abs(np.degrees(yaw)) > FAIL_JUMP_YAW and size > FAIL_JUMP_YAW_SHIFT):
            flags["jump too big to ramp"][s["first"] - 1:s["n"] + 1] = True
    # runaway: pose jumps that keep growing (ARKit is losing track and every correction is bigger than the last).
    # A spike is a step at least RUNAWAY_MIN_STEP and 3x the walking steps around it.
    spikes = []
    for k in range(1, n):
        if normal[k] and normal[k - 1]:
            d = np.linalg.norm(P[k] - P[k - 1])
            near = np.linalg.norm(np.diff(P[max(k - 10, 0):k + 11], axis=0), axis=1)
            if d >= RUNAWAY_MIN_STEP and d >= 3 * np.median(near):
                spikes.append((k, d))
    run = []
    for k, d in spikes + [(None, None)]:
        if run and k is not None and t[k] - t[run[-1][0]] <= RUNAWAY_GAP and d >= 0.9 * run[-1][1]:
            run.append((k, d))
            continue
        if len(run) >= RUNAWAY_MIN_SPIKES and run[-1][1] >= RUNAWAY_GROWTH * run[0][1]:
            flags["growing jumps (runaway)"][run[0][0] - 1:run[-1][0] + 1] = True
        run = [(k, d)] if k is not None else []
    k = 0
    while k < n:
        if not normal[k]:
            j = k
            while j + 1 < n and not normal[j + 1]:
                j += 1
            if t[j] - t[k] >= FAIL_TRACKING:
                flags["tracking not normal"][k:j + 1] = True
            k = j + 1
        else:
            k += 1
    bad = np.any(list(flags.values()), axis=0)
    out = []
    for a in np.flatnonzero(bad & ~np.r_[False, bad[:-1]]):  # starts of runs
        b = a
        while b + 1 < n and bad[b + 1]:
            b += 1
        lo = int(np.searchsorted(t, t[a] - FAIL_PAD, "left"))
        hi = int(np.searchsorted(t, t[b] + FAIL_PAD, "right")) - 1
        reasons = [name for name, fl in flags.items() if fl[a:b + 1].any()]
        if out and lo <= out[-1]["last"] + 1:  # padded intervals that touch become one
            out[-1].update(last=hi, t1=float(t[hi]), reasons=sorted(set(out[-1]["reasons"]) | set(reasons)))
        else:
            out.append({"first": lo, "last": hi, "t0": float(t[lo]), "t1": float(t[hi]), "reasons": reasons})
    for r in out:
        r["seconds"] = r["t1"] - r["t0"]
        r["max_step_m"] = float(np.nanmax(step[r["first"]:r["last"] + 1])) if np.isfinite(step[r["first"]:r["last"] + 1]).any() else 0.0
    return out


def split_walk(frames):
    """(segments, removed, dropped): the failures from find_failures, and the clean pieces between them. A piece
    shorter than MIN_SEGMENT_SECONDS or with less than MIN_SEGMENT_PATH of path is dropped. A walk without
    failures comes back whole as one segment, however short."""
    removed = find_failures(frames)
    if not removed:
        return [_segment(frames, 0, len(frames) - 1)], [], []
    keep = np.ones(len(frames), bool)
    for r in removed:
        keep[r["first"]:r["last"] + 1] = False
    segments, dropped = [], []
    k = 0
    while k < len(frames):
        if keep[k]:
            j = k
            while j + 1 < len(frames) and keep[j + 1]:
                j += 1
            s = _segment(frames, k, j)
            if s["seconds"] < MIN_SEGMENT_SECONDS or s["path_m"] < MIN_SEGMENT_PATH:
                dropped.append({**s, "why": f"too short ({s['seconds']:.0f} s, {s['path_m']:.0f} m of path)"})
            else:
                segments.append(s)
            k = j + 1
        else:
            k += 1
    return segments, removed, dropped


def _segment(frames, a, b):
    P = np.array([f["T"][:3, 3] for f in frames[a:b + 1]])
    return {"first": a, "last": b, "t0": frames[a]["t"], "t1": frames[b]["t"], "seconds": frames[b]["t"] - frames[a]["t"],
            "path_m": float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), "ids": [f["i"] for f in frames[a:b + 1]]}


def fragment_groups(frames, frag_frames, cuts=()):
    """Lists of frame positions: runs of tracking-normal frames (also cut at `cuts`), in ~frag_frames pieces."""
    runs = []
    for n, f in enumerate(frames):
        if f["tracking"] != "normal":
            continue
        if runs and n == runs[-1][-1] + 1 and n not in cuts:
            runs[-1].append(n)
        else:
            runs.append([n])
    groups = []
    for run in runs:
        groups += [g.tolist() for g in np.array_split(run, max(1, round(len(run) / frag_frames)))]
    return [g for g in groups if len(g) >= frag_frames // 4]


def build_fragment(walk, meta, frames, idx):
    """5 cm cloud with normals in the ARKit world frame, streamed 10 frames at a time."""
    acc, pend = o3d.geometry.PointCloud(), []
    for n in idx:
        depth, conf = fuse.read_depth(walk, meta, frames[n]["i"])
        pend.append(fuse.unproject(depth, conf, frames[n], meta))
        if len(pend) == 10 or n == idx[-1]:
            acc = (acc + fuse.to_cloud(np.vstack(pend))).voxel_down_sample(VOXEL)
            pend = []
    acc, _ = acc.remove_radius_outlier(4, 3 * VOXEL)
    acc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(3 * VOXEL, 30))
    return acc


def pyramid(pcd):
    return [pcd.voxel_down_sample(v).normalize_normals() for v, _ in SCALES[:-1]] + [pcd]


def overlap_fraction(a, b):
    def keys(p):  # occupied 1 m voxels as single ints
        v = np.floor(p).astype(np.int64) + 10000
        return np.unique((v[:, 0] * 20000 + v[:, 1]) * 20000 + v[:, 2])
    ka, kb = keys(a), keys(b)
    return np.intersect1d(ka, kb).size / min(len(ka), len(kb))


def gaps(src, tgt, T):
    """Distance from each src point (moved by T) to its nearest tgt point."""
    moved = o3d.geometry.PointCloud(src)
    moved.transform(T)
    return np.asarray(moved.compute_point_cloud_distance(tgt))


def register_pair(src, tgt, pyr_s, pyr_t):
    """Yaw-only src->tgt correction, its quality and the overlap centre (in src), or (None, why, None)."""
    T = np.eye(4)
    for (voxel, dist), s, t in zip(SCALES, pyr_s, pyr_t):
        T = reg.registration_icp(s, t, dist, T, reg.TransformationEstimationPointToPlane(),
                                 reg.ICPConvergenceCriteria(1e-6, 1e-6, 30)).transformation
    tilt = np.degrees(np.arccos(np.clip(T[1, 1], -1, 1)))
    pts = np.asarray(src.points)
    near = gaps(src, tgt, T) < NEAR_DIST
    if near.sum() < MIN_INLIERS:
        return None, "no overlap after ICP", None
    c = pts[near].mean(0)
    Tp = yaw_only(T, c)
    d = gaps(src, tgt, Tp)
    near = d < NEAR_DIST
    inl = d < INLIER_DIST
    q = {"inliers": int(inl.sum()), "ratio": float(inl.sum() / max(near.sum(), 1)),
         "rmse": float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else np.inf,
         "shift": shift_at(Tp, c), "yaw": float(np.degrees(yaw_of(Tp[:3, :3]))), "tilt": float(tilt)}
    if (q["inliers"] < MIN_INLIERS or q["ratio"] < MIN_INLIER_RATIO or q["rmse"] > MAX_RMSE
            or q["shift"] > MAX_SHIFT or abs(q["yaw"]) > MAX_YAW or tilt > MAX_TILT):
        return None, q, None
    return Tp, q, c


def frame_poses(frames, centres, theta, trans):
    """Interpolate the per-fragment (yaw, translation) corrections to every frame and apply them."""
    n = np.arange(len(frames))
    th = np.interp(n, centres, theta)
    tr = np.stack([np.interp(n, centres, trans[:, k]) for k in range(3)], 1)
    out = np.empty((len(frames), 4, 4))
    for k, f in enumerate(frames):
        C = rot_y(th[k])
        out[k] = f["T"]
        out[k, :3, :3] = C @ f["T"][:3, :3]
        out[k, :3, 3] = C @ f["T"][:3, 3] + tr[k]
    return out


def optimize(walk, out, frag_frames=FRAG_FRAMES, log=print, frame_ids=None):
    """frame_ids: optimise only these frames (a clean segment from split_walk); poses.npz then lists only them."""
    t0 = time.time()
    walk, out = Path(walk), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta, frames = fuse.read_walk(walk)
    if frame_ids is not None:
        keep = set(frame_ids)
        frames = [f for f in frames if f["i"] in keep]
    recorded = np.array([f["T"] for f in frames])
    snaps = detect_snaps(frames)
    for s in snaps:
        log(f"snap at t={s['t']:.1f} s: {s['frames']} frame(s), largest step {s['jump_m']:.2f} m, {s['rot_deg']:.1f} deg")
    remove_snaps(frames, snaps)  # frames' T are snap-free from here on
    groups = fragment_groups(frames, frag_frames, {s["n"] for s in snaps})

    clouds, pyrs = [], []
    for g in groups:
        clouds.append(build_fragment(walk, meta, frames, g))
        pyrs.append(pyramid(clouds[-1]))
    pts = [np.asarray(c.points) for c in clouds]
    cen = [p.mean(0) for p in pts]
    cams = [np.array([frames[n]["T"][:3, 3] for n in g]) for g in groups]
    mid = [cam[len(cam) // 2] for cam in cams]
    walked = np.concatenate([[0], np.cumsum([np.linalg.norm(mid[k + 1] - mid[k]) for k in range(len(mid) - 1)])])
    log(f"{len(groups)} fragments built ({time.time() - t0:.0f} s)")

    pg = reg.PoseGraph()
    for _ in groups:
        pg.nodes.append(reg.PoseGraphNode(np.eye(4)))
    for k in range(len(groups) - 1):
        gap = groups[k + 1][0] - groups[k][-1] > 1  # tracking was lost in between
        pg.edges.append(reg.PoseGraphEdge(k, k + 1, np.eye(4), np.eye(6) * (GAP_INFO if gap else ODOM_INFO), uncertain=False))
    for k, g in enumerate(groups):
        if k > 0 and any(s["n"] == g[0] for s in snaps):  # stage 1 left the poses after a snap alone: keep them so
            pg.edges.append(reg.PoseGraphEdge(0, k, np.eye(4), np.eye(6) * ANCHOR_INFO, uncertain=False))

    candidates, loops = [], []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if walked[j] - walked[i] < MIN_LOOP_PATH:
                continue
            a, b = cams[i][::5], cams[j][::5]
            if np.linalg.norm(a[:, None] - b[None], axis=2).min() > LOOP_CAM_DIST:
                continue
            if overlap_fraction(pts[i], pts[j]) < MIN_OVERLAP:
                continue
            candidates.append((i, j))
    log(f"{len(candidates)} loop candidates")
    rejected, found = [], []
    for n, (i, j) in enumerate(candidates):
        Tp, q, c = register_pair(clouds[i], clouds[j], pyrs[i], pyrs[j])
        if Tp is None:
            rejected.append({"pair": [i, j], "why": q})
        else:
            found.append((i, j, Tp, q, c))
        if n % 20 == 19:
            log(f"  {n + 1}/{len(candidates)} ICP done, {len(found)} passed ({time.time() - t0:.0f} s)")

    def corroborated(i, j, Tp, q, c):
        """Another loop between nearby places that asks for about the same correction."""
        v = Tp[:3, :3] @ c + Tp[:3, 3]
        return any(
            (i2, j2) != (i, j) and abs(walked[i2] - walked[i]) <= GUARD_PATH and abs(walked[j2] - walked[j]) <= GUARD_PATH
            and np.linalg.norm(v - (T2[:3, :3] @ c + T2[:3, 3])) <= GUARD_AGREE_SHIFT and abs(q["yaw"] - q2["yaw"]) <= GUARD_AGREE_YAW
            for i2, j2, T2, q2, _ in found)

    for i, j, Tp, q, c in found:
        if (q["shift"] > GUARD_SHIFT or abs(q["yaw"]) > GUARD_YAW) and not corroborated(i, j, Tp, q, c):
            rejected.append({"pair": [i, j], "why": "large correction, no agreeing loop nearby", "quality": q})
            continue
        info = reg.get_information_matrix_from_point_clouds(clouds[i], clouds[j], INLIER_DIST, Tp)
        pg.edges.append(reg.PoseGraphEdge(i, j, Tp, info, uncertain=True))
        loops.append({"pair": [i, j], **q})
    log(f"{len(loops)} loops accepted, {len(found) - len(loops)} dropped by the corroboration guard ({time.time() - t0:.0f} s)")

    if loops:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(), reg.GlobalOptimizationConvergenceCriteria(),
                                reg.GlobalOptimizationOption(max_correspondence_distance=INLIER_DIST, edge_prune_threshold=0.25,
                                                             preference_loop_closure=2.0, reference_node=0))
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    kept = {(e.source_node_id, e.target_node_id) for e in pg.edges if e.uncertain}

    C = [yaw_only(np.array(nd.pose), cen[k]) for k, nd in enumerate(pg.nodes)]
    theta = np.array([yaw_of(c[:3, :3]) for c in C])
    trans = np.array([c[:3, 3] for c in C])
    centres = [np.mean(g) for g in groups]
    poses = frame_poses(frames, centres, theta, trans)
    np.savez(out / "poses.npz", i=np.array([f["i"] for f in frames]), T=poses)

    for lp in loops:
        i, j = lp["pair"]
        lp["kept"] = (i, j) in kept
        before, after = gaps(clouds[i], clouds[j], np.eye(4)), gaps(clouds[i], clouds[j], np.linalg.inv(C[j]) @ C[i])
        both = (before < NEAR_DIST) | (after < NEAR_DIST)  # same points for both, so the comparison is fair
        for name, d in (("before", before), ("after", after)):
            lp[f"gap_{name}_median_m"] = float(np.median(d[both]))
            lp[f"within_{int(INLIER_DIST * 100)}cm_{name}"] = float((d[both] < INLIER_DIST).mean())

    for s in snaps:
        s["corrected_jump_m"] = float(np.linalg.norm(poses[s["n"], :3, 3] - poses[s["n"] - 1, :3, 3]))
    n_kept = sum(lp["kept"] for lp in loops)
    move = np.array([shift_at(C[k], cen[k]) for k in range(len(C))])
    yaw = np.degrees(np.abs(theta))
    ok = [n for n in range(1, len(frames)) if frames[n]["tracking"] == frames[n - 1]["tracking"] == "normal"]
    step = lambda T: np.linalg.norm(T[ok, :3, 3] - T[[n - 1 for n in ok], :3, 3], axis=1)  # noqa: E731
    raw = step(recorded)
    total = np.linalg.norm(poses[:, :3, 3] - recorded[:, :3, 3], axis=1)
    stats = {
        "fragments": len(groups), "frames": len(frames), "candidates": len(candidates), "snaps": snaps,
        "max_frame_to_frame_step_m": {"raw": float(raw.max()), "corrected": float(step(poses).max())},
        "total_position_change_m": {"max": float(total.max()), "median": float(np.median(total))},
        "accepted_loops": len(loops), "pruned_loops": len(loops) - n_kept, "kept_loops": n_kept,
        "graph_correction_translation_m": {"max": float(move.max()), "median": float(np.median(move))},
        "graph_correction_yaw_deg": {"max": float(yaw.max()), "median": float(np.median(yaw))},
        "loops": loops, "rejected": rejected, "seconds": round(time.time() - t0, 1),
    }
    (out / "optimize.json").write_text(json.dumps(stats, indent=1))
    log(f"pose graph correction translation max {move.max():.2f} m, median {np.median(move):.2f} m; yaw max {yaw.max():.2f} deg; "
        f"total position change max {total.max():.2f} m; "
        f"{n_kept}/{len(loops)} loops kept; max frame-to-frame step {raw.max():.2f} -> {step(poses).max():.2f} m "
        f"({time.time() - t0:.0f} s)")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("walk_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--jobs", type=int, default=cores.DEFAULT_JOBS, help="cores for Open3D (default %(default)s); merge.py also runs walks in parallel")
    a = ap.parse_args()
    optimize(a.walk_dir, a.out_dir)
