"""Merge several walks into one model in the first walk's frame.

usage: merge.py <out_dir> <walk_dir> [<walk_dir> ...] [--jobs 3] [--no-visual] [--cache DIR | --no-cache] [--stride 1]

Bad tracking is cut out first (optimize.find_failures: ARKit sliding for seconds then relocalising, jumps too
big to ramp, long tracking loss; each padded by a second). A walk with such stretches is split into its clean
segments (pieces under 10 s or 5 m dropped), and every segment is handled below as a walk of its own, so one
walk merged with itself is just this: split, register the pieces to each other, joint graph. A segment that
cannot be registered (typically: it walked a different part of the plot, so there is nothing to match) is placed in
the frame of its neighbouring segment if ARKit's pose is continuous across the cut (the world survives a tracking
failure: 2.8 m and 5.9 m in 7 and 16 s on the first real walks), and is flagged as unverified; if it is not
continuous either it is excluded from the model, since a GPS placement alone (+-3 m) would double what it overlaps.
Both are reported in merge.json. A whole walk that fails keeps its GPS placement as before.

Every walk (or segment) is optimised on its own (optimize.py: snaps, drift loops). Then, in order, each is
registered to the union of the ones before it (a failed one is retried once others have been added):
  1. initial guess from GPS (translation only, yaw 0: each walk's frame is gravity and compass aligned);
  2. coarse search for yaw (+-20 deg) and translation (+-12 m) by FFT cross-correlation of top-down "structure"
     rasters (vertical extent per 25 cm cell, so bare slopes do not vote), then the best candidates are ranked by a
     point-to-plane ICP and the winner is refined at 3 scales; the result is projected to yaw + translation (no tilt);
  3. validated (inliers, inlier ratio, rmse, overlap, plausibility). A walk that fails keeps its GPS placement,
     is left out of everything below and gets a warning in merge.json and on stderr.
Then a joint pose graph over all registered walks' ~100-frame fragments: consecutive fragments of a walk are tied
by identity edges (the walk is already optimised), overlapping fragments of different walks by yaw-only ICP loop
closures (same validation and corroboration guard as optimize.py), so overlap anywhere tightens the merge and not
just one rigid transform per walk. The frames of all walks are fused together and georeferenced from all GPS fixes.

Writes to out_dir: the process.py outputs (points.ply, mesh.ply, ground.*, georef.*, previews, viewer.html),
walk<k>/ (optimize.json, poses.npz from the walk's own optimisation, poses_merged.npz in the first walk's frame)
and merge.json.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # python -I drops the script dir from sys.path

import cache  # noqa: E402
import cores  # noqa: E402

if __name__ == "__main__":
    cores.setup()  # caps the threads Open3D uses; must happen before it is imported

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

import fuse  # noqa: E402
import georef  # noqa: E402
import optimize  # noqa: E402
from optimize import INLIER_DIST, MIN_INLIERS, NEAR_DIST, gaps, reg, rot_y, shift_at, yaw_of, yaw_only  # noqa: E402

GRID = 0.25  # raster cell for the coarse search (m)
STRUCT_MIN = 0.15  # vertical extent per cell that counts as ground relief, not structure (m)
SEARCH = 25.0  # translation search radius around the GPS guess (m): GPS placements of two walks can be 15 m apart
VISUAL_REGISTER_MIN = 3  # visual registration of a piece needs this many constraints that agree on one placement
VISUAL_REGISTER_RADIUS, VISUAL_REGISTER_PAIRS = 18.0, 1500  # m, keyframe pairs: GPS can put a piece 15 m off, so look that far
RETRY_VISUAL_RADIUS = 8.0  # m: the search of a retried piece (second chance after the first pass) starts this close
VISUAL_PAIR_RADIUS = 20.0  # m: pieces whose placed camera paths come closer than this get visual loops between them
VISUAL_MAX_CORRECTION = 5.0  # m: a visual edge may not ask a fragment to move further than this from its placement
YAW_RANGE, YAW_STEP = 20.0, 1.0  # degrees
BLUR = 1.5  # cells
N_CANDIDATES = 4  # coarse peaks ranked by ICP
MIN_WALK_OVERLAP = 0.1  # fraction of the smaller cloud lying within 30 cm of the other, after registration
MIN_WALK_RATIO = 0.3  # inliers / points with a neighbour within NEAR_DIST (whole walks carry residual drift)
MAX_WALK_RMSE = 0.07  # of the inliers (< 10 cm); looser than for fragments: a whole walk carries residual drift
MAX_GPS_ERR = 15.0  # the registered placement may not be further than this from the GPS one (m)
OVERLAP_DIST = 0.3
CONTINUITY_SPEED, CONTINUITY_MARGIN = 2.0, 3.0 # m/s and m: how far the camera may be across a cut and still count as continuous


def make_G(yaw, t):
    G = np.eye(4)
    G[:3, :3], G[:3, 3] = rot_y(yaw), t
    return G


def place(pcd, T):
    out = o3d.geometry.PointCloud(pcd)
    out.transform(T)
    return out


def join(clouds):
    acc = o3d.geometry.PointCloud()
    for c in clouds:
        acc += c
    return acc.voxel_down_sample(optimize.VOXEL)


def raster(pts, x0, z0, shape):
    """Per cell: structure (vertical extent beyond ground relief, 0..1) and lowest Y (NaN = empty). Arrays are [z, x]."""
    h, w = shape
    ix = np.floor((pts[:, 0] - x0) / GRID).astype(int)
    iz = np.floor((pts[:, 2] - z0) / GRID).astype(int)
    ok = (ix >= 0) & (ix < w) & (iz >= 0) & (iz < h)
    flat, y = (iz * w + ix)[ok], pts[ok, 1]
    order = np.argsort(flat, kind="stable")
    flat, y = flat[order], y[order]
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    lo, hi = np.minimum.reduceat(y, starts), np.maximum.reduceat(y, starts)
    struct, low = np.zeros(h * w), np.full(h * w, np.nan)
    struct[flat[starts]] = np.clip(hi - lo - STRUCT_MIN, 0, 1)
    low[flat[starts]] = lo
    return struct.reshape(h, w), low.reshape(h, w)


def coarse_candidates(src, ref, tg):
    """Yaw + translation candidates (best first) placing src (own frame) on ref, starting from the translation tg.
    Each is {score, yaw_deg, lag_m, G}; [] with a reason if the clouds cannot overlap within the search radius."""
    sp = np.asarray(src.voxel_down_sample(0.1).points) + tg
    rp = np.asarray(ref.voxel_down_sample(0.1).points)
    gap = np.maximum(sp.min(0)[[0, 2]] - rp.max(0)[[0, 2]], rp.min(0)[[0, 2]] - sp.max(0)[[0, 2]]).max()
    if gap > SEARCH:
        return [], f"GPS places the walks {gap:.0f} m apart (search radius {SEARCH:.0f} m)"
    lo = np.minimum(sp.min(0), rp.min(0)) - SEARCH - 2
    hi = np.maximum(sp.max(0), rp.max(0)) + SEARCH + 2
    x0, z0 = lo[0], lo[2]
    shape = (int(np.ceil((hi[2] - z0) / GRID)), int(np.ceil((hi[0] - x0) / GRID)))
    filt = np.exp(-2 * np.pi**2 * BLUR**2 * (np.fft.fftfreq(shape[0])[:, None] ** 2 + np.fft.rfftfreq(shape[1])[None, :] ** 2))
    ref_s, ref_low = raster(rp, x0, z0, shape)
    F_ref = np.fft.rfft2(ref_s) * filt
    L = int(SEARCH / GRID)
    zi, xi = np.arange(-L, L + 1) % shape[0], np.arange(-L, L + 1) % shape[1]
    c = sp.mean(0)
    found = []
    for yaw in np.radians(np.arange(-YAW_RANGE, YAW_RANGE + 1e-6, YAW_STEP)):
        q = (sp - c) @ rot_y(yaw).T + c
        s_img, _ = raster(q, x0, z0, shape)
        corr = np.fft.irfft2(F_ref * np.conj(np.fft.rfft2(s_img) * filt), s=shape)[np.ix_(zi, xi)]
        norm = min(np.linalg.norm(ref_s), np.linalg.norm(s_img)) ** 2 + 1e-9
        k = np.unravel_index(corr.argmax(), corr.shape)
        found.append((float(corr[k] / norm), yaw, np.array([(k[1] - L) * GRID, 0.0, (k[0] - L) * GRID])))
    found.sort(key=lambda f: -f[0])
    picked = []
    for score, yaw, lag in found:
        if all(abs(yaw - p[1]) > np.radians(5) or np.linalg.norm(lag - p[2]) > 2.0 for p in picked):
            picked.append((score, yaw, lag))
        if len(picked) == N_CANDIDATES:
            break
    out = []
    for score, yaw, lag in picked:
        R = rot_y(yaw)
        q = (sp - c) @ R.T + c + lag
        _, low = raster(q, x0, z0, shape)
        both = ~np.isnan(low) & ~np.isnan(ref_low)
        if both.sum() < 20:
            continue
        lag[1] = float(np.median(ref_low[both] - low[both]))
        out.append({"score": score, "yaw_deg": float(np.degrees(yaw)), "lag_m": lag.tolist(),
                    "G": make_G(yaw, R @ (tg - c) + c + lag)})
    return out, None


def quality(src, ref, T, tg):
    """Yaw-only projection of T and its metrics against ref (tilt dropped as in optimize.register_pair), or
    (None, why, None). Shift is measured against the GPS placement tg."""
    pts = np.asarray(src.points)
    tilt = np.degrees(np.arccos(np.clip(T[1, 1], -1, 1)))
    near = gaps(src, ref, T) < NEAR_DIST
    if near.sum() < MIN_INLIERS:
        return None, {"why": "no overlap after ICP", "points_near": int(near.sum())}, None
    c = pts[near].mean(0)
    Tp = yaw_only(T, c)
    d = gaps(src, ref, Tp)
    near, inl = d < NEAR_DIST, d < INLIER_DIST
    back = gaps(ref, src, np.linalg.inv(Tp))
    q = {"inliers": int(inl.sum()), "ratio": float(inl.sum() / max(near.sum(), 1)),
         "rmse": float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else np.inf,
         "overlap": float(max((d < OVERLAP_DIST).mean(), (back < OVERLAP_DIST).mean())),
         "gps_error_m": float(np.linalg.norm(Tp[:3, :3] @ c + Tp[:3, 3] - (c + tg))),
         "yaw_deg": float(np.degrees(yaw_of(Tp[:3, :3]))), "tilt_deg": float(tilt)}
    bad = [name for name, ok in (
        ("inliers", q["inliers"] >= MIN_INLIERS), ("inlier ratio", q["ratio"] >= MIN_WALK_RATIO),
        ("rmse", q["rmse"] <= MAX_WALK_RMSE), ("overlap", q["overlap"] >= MIN_WALK_OVERLAP),
        ("distance from GPS placement", q["gps_error_m"] <= MAX_GPS_ERR), ("yaw", abs(q["yaw_deg"]) <= 2 * YAW_RANGE),
        ("tilt", tilt <= optimize.MAX_TILT)) if not ok]
    if bad:
        q["why"] = "failed: " + ", ".join(bad)
        return None, q, None
    return Tp, q, c


def register_walk(src, ref, tg, log):
    """src: walk cloud in its own frame, ref: the merged cloud so far, tg: GPS translation. Returns (T or None, report)."""
    cands, why = coarse_candidates(src, ref, tg)
    report = {"candidates": [{k: v for k, v in c.items() if k != "G"} for c in cands]}
    if not cands:
        report["why"] = why or "no coarse candidate with overlapping ground"
        return None, report
    pyr_s, pyr_t = optimize.pyramid(src), optimize.pyramid(ref)
    icp = reg.ICPConvergenceCriteria(1e-6, 1e-6, 30)
    best = None
    for c in cands:  # rank the coarse peaks by how much of the walk lies within 20 cm after a coarse ICP
        T = reg.registration_icp(pyr_s[0], pyr_t[0], optimize.SCALES[0][1], c["G"],
                                 reg.TransformationEstimationPointToPlane(), icp).transformation
        c["icp_fit"] = float((gaps(pyr_s[0], pyr_t[0], T) < 0.2).mean())
        log(f"  candidate yaw {c['yaw_deg']:+.0f} deg, shift {np.round(c['lag_m'], 1).tolist()} m: correlation {c['score']:.2f}, "
            f"ICP fit {c['icp_fit']:.2f}")
        if best is None or c["icp_fit"] > best[0]:
            best = (c["icp_fit"], T)
    T = best[1]
    for (voxel, dist), s, t in list(zip(optimize.SCALES, pyr_s, pyr_t))[1:]:
        T = reg.registration_icp(s, t, dist, T, reg.TransformationEstimationPointToPlane(), icp).transformation
    Tp, q, _ = quality(src, ref, T, tg)
    report["registration" if Tp is not None else "attempt"] = q
    return Tp, report


def fragment_data(walk, poses, frag_frames):
    """Frames with the walk's own optimised poses, their fragments, and fragment geometry in the walk's frame."""
    meta, frames = fuse.read_walk(walk)
    z = np.load(poses)
    corrected = dict(zip(z["i"].tolist(), z["T"]))
    frames = [f for f in frames if f["i"] in corrected]  # a segment's poses list only its own frames
    for f in frames:
        f["T"] = corrected[f["i"]]
    groups = optimize.fragment_groups(frames, frag_frames)
    clouds = [optimize.build_fragment(walk, meta, frames, g) for g in groups]
    cams = [np.array([frames[n]["T"][:3, 3] for n in g]) for g in groups]
    mid = [c[len(c) // 2] for c in cams]
    walked = np.concatenate([[0], np.cumsum([np.linalg.norm(mid[k + 1] - mid[k]) for k in range(len(mid) - 1)])])
    return {"frames": frames, "groups": groups, "clouds": clouds, "cams": cams, "walked": walked,
            "cen": [np.asarray(c.points).mean(0) for c in clouds]}


def cloud_of(points, normals):
    pcd = o3d.geometry.PointCloud()
    pcd.points, pcd.normals = o3d.utility.Vector3dVector(points), o3d.utility.Vector3dVector(normals)
    return pcd


def unit_job(args):
    """Worker: optimise one unit on its own, then build its fragments from the result. Returns (log lines, fragment
    data with each cloud as a (points, normals) pair: Open3D clouds cannot be pickled)."""
    walk, out, frag_frames, ids, visual, jobs, cache_dir = args
    entry = None
    if cache_dir:
        code = [optimize, fuse, fragment_data]
        if visual:
            import visual as vis
            cache.install(cache_dir)  # photo features and visual pair results are cached too (this runs in a worker process)
            code.append(vis)
        entry = cache.unit_dir(cache_dir, walk, ids, frag_frames, visual, cache.source(*code))
        hit = cache.load_unit(entry, out)
        if hit:
            return hit[0] + ["(this unit came from the cache)"], hit[1]
    lines = []
    optimize.optimize(walk, out, frag_frames, log=lines.append, frame_ids=ids, visual=visual, jobs=jobs)
    d = fragment_data(walk, Path(out) / "poses.npz", frag_frames)
    # float32 is what the cache stores and plenty for 5 cm fragments (8 um at 100 m); a cold run rounds the same way
    d["clouds"] = [(np.asarray(c.points, np.float32), np.asarray(c.normals, np.float32)) for c in d["clouds"]]
    if entry:
        cache.store_unit(entry, out, lines, d)
    return lines, d


def consistency(data, transforms, k, earlier):
    """Nearest-neighbour gap from walk k's points to the earlier walks' points, where they are within NEAR_DIST,
    with every fragment moved by transforms[walk][fragment]."""
    src = join([place(c, transforms[k][f]) for f, c in enumerate(data[k]["clouds"])])
    tgt = join([place(c, transforms[w][f]) for w in earlier for f, c in enumerate(data[w]["clouds"])])
    d = np.asarray(src.compute_point_cloud_distance(tgt))
    near = d < NEAR_DIST
    return {"median_gap_m": float(np.median(d[near])) if near.any() else None, "within_1m_fraction": float(near.mean()),
            "within_10cm_of_overlap": float((d[near] < INLIER_DIST).mean()) if near.any() else None}


def continuity(k, anchored, units, data):
    """(sibling, distance, gap) if unit k, a segment of a split walk, can be placed in the frame of an adjacent placed
    segment of the same walk: the camera pose just before / after the cut is within what walking covers in the gap
    (ARKit relocalises into the same world, so the frame survives a tracking failure). None otherwise."""
    for s in (k - 1, k + 1):
        if s in anchored and 0 <= s < len(units) and units[s]["split"] and units[s]["walk"] == units[k]["walk"]:
            a, b = (data[s]["frames"], data[k]["frames"]) if s < k else (data[k]["frames"], data[s]["frames"])
            gap = b[0]["t"] - a[-1]["t"]
            dist = float(np.linalg.norm(b[0]["T"][:3, 3] - a[-1]["T"][:3, 3]))
            if dist <= CONTINUITY_SPEED * gap + CONTINUITY_MARGIN:
                return s, dist, float(gap)
    return None


def visual_register(k, registered, walks, own, G, jobs, log, radius=VISUAL_REGISTER_RADIUS):
    """Place unit k by photo features when ICP could not (too little structure to match, or further off than the search
    radius): visual constraints between k and every registered unit j each imply a placement of k's world in the
    reference frame (G_j @ H); the placement most of them agree on, if at least VISUAL_REGISTER_MIN do, wins. The first
    guess G[k] (GPS or neighbour) only chooses which keyframes to try; visual.visual_loops re-places k from the
    constraints it finds and tries again. Returns (G_k or None, report)."""
    import visual
    kfs_k = visual.keyframes(walks[k], own[k])
    at = np.mean([f["T"][:3, 3] for f in kfs_k], axis=0)  # the unit's own centre: the projection to yaw keeps it in place
    report, guess = {"rounds": []}, G[k]
    for rnd in range(2):
        Gs, inl, against = [], [], []
        for j in registered:
            cons, summ = visual.visual_loops(walks[j], walks[k], poses_a=own[j], poses_b=own[k], init=np.linalg.inv(G[j]) @ guess,
                                             max_pairs=VISUAL_REGISTER_PAIRS, workers=max(1, jobs), log=log,
                                             radius=radius, max_angle=90.0)  # GPS placements are metres off
            if cons:
                Gs += [G[j] @ H for H in visual.world_alignments(cons, visual.keyframes(walks[j], own[j]), kfs_k)]
                inl += [c[6]["n_inliers"] for c in cons]
            against.append({"unit": j, "pairs": summ["pairs"], "constraints": len(cons), "rejected": summ["rejected"]})
        report["rounds"].append({"first_guess_m": guess[:3, 3].tolist(), "against": against, "constraints": len(Gs)})
        if len(Gs) >= VISUAL_REGISTER_MIN or not Gs:
            break
        guess = Gs[int(np.argmax(inl))]  # one or two constraints: not enough to place the piece, but they say where to look
    report["constraints"] = len(Gs)
    if len(Gs) < VISUAL_REGISTER_MIN:
        report["why"] = f"only {len(Gs)} visual constraints (need {VISUAL_REGISTER_MIN})"
        return None, report
    Gs = np.array(Gs)
    best, mask = optimize.agreeing(Gs, at)
    report["agreeing"] = int(mask.sum())
    report["placements"] = [{"at_m": (g[:3, :3] @ at + g[:3, 3]).round(2).tolist(), "yaw_deg": round(float(np.degrees(yaw_of(g[:3, :3]))), 2)} for g in Gs]
    if mask.sum() < VISUAL_REGISTER_MIN:
        report["why"] = f"{len(Gs)} visual constraints but only {int(mask.sum())} agree on a placement (need {VISUAL_REGISTER_MIN})"
        return None, report
    where = np.mean([g[:3, :3] @ at + g[:3, 3] for g in Gs[mask]], axis=0)  # mean of the agreeing placements
    yaw = float(np.degrees(np.arctan2(np.mean([np.sin(yaw_of(g[:3, :3])) for g in Gs[mask]]),
                                      np.mean([np.cos(yaw_of(g[:3, :3])) for g in Gs[mask]]))))
    R = rot_y(np.radians(yaw))
    Gk = make_G(np.radians(yaw), where - R @ at)
    report.update(yaw_deg=yaw, translation_m=Gk[:3, 3].tolist(),
                  spread_m=float(np.max([np.linalg.norm(g[:3, :3] @ at + g[:3, 3] - where) for g in Gs[mask]])),
                  moved_from_first_guess_m=shift_at(Gk @ np.linalg.inv(G[k]), G[k][:3, :3] @ at + G[k][:3, 3]), search_rounds=rnd + 1)
    return Gk, report


def split_inputs(walks, log):
    """What gets merged: each walk whole, or, if it has bad-tracking stretches (optimize.find_failures), its clean
    segments, each handled as a walk of its own. Returns (units, plans): units [{walk, ids (None = whole walk),
    split, segment, label, info}], plans = per input walk report."""
    units, plans = [], []
    for w in walks:
        _, frames = fuse.read_walk(w)
        segments, removed, dropped = optimize.split_walk(frames)
        strip = lambda d: {k: v for k, v in d.items() if k != "ids"}  # noqa: E731
        plan = {"walk": w.name, "frames": len(frames), "removed_intervals": removed,
                "removed_seconds": float(sum(r["seconds"] for r in removed)),
                "segments_kept": [strip(s) for s in segments], "segments_dropped": [strip(s) for s in dropped]}
        for r in removed:
            log(f"{w.name}: bad tracking removed t={r['t0']:.1f}-{r['t1']:.1f} s ({r['seconds']:.1f} s; {', '.join(r['reasons'])}; "
                f"largest step {r['max_step_m']:.2f} m)")
        for s in dropped:
            log(f"{w.name}: dropped piece t={s['t0']:.1f}-{s['t1']:.1f} s: {s['why']}")
        for s, seg in enumerate(segments):
            split = bool(removed)
            label = f"{w.name} segment {s} (t={seg['t0']:.0f}-{seg['t1']:.0f} s)" if split else w.name
            units.append({"walk": w, "ids": seg["ids"] if split else None, "split": split, "segment": s, "label": label})
        if not segments:
            log(f"WARNING: {w.name} has no clean stretch left, left out")
        plans.append(plan)
    return units, plans


def merge(out, walks, frag_frames=optimize.FRAG_FRAMES, log=print, jobs=1, visual=False, cache_dir=None, stride=1):
    """visual: use photo-feature constraints (visual.py): to register a piece ICP could not place, and as loop edges in
    the joint graph (optimize.py adds them within each walk too). cache_dir: reuse / store what depends on one walk only
    (see cache.py). stride: fuse every stride-th depth frame (see fuse.py)."""
    t0 = time.time()
    if cache_dir and visual:
        cache.install(cache_dir)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    warnings = []
    units, plans = split_inputs([Path(w) for w in walks], log)
    walks = [u["walk"] for u in units]  # one entry per unit: the same walk appears once per clean segment
    n = len(units)
    if not n:
        raise SystemExit("no walk has a clean stretch of tracking left to merge")

    # every unit is optimised on its own and its fragments built: independent, so one worker process each
    each = max(1, jobs // min(jobs, n))  # Open3D / SIFT threads per worker: the workers share the jobs budget
    done = cores.pmap(unit_job, [(u["walk"], out / f"walk{k}", frag_frames, u["ids"], visual, each, cache_dir) for k, u in enumerate(units)], jobs)
    from_cache = [k for k, (_, d) in enumerate(done) if d.pop("from_cache", False)]
    for k, (u, (lines, _)) in enumerate(zip(units, done)):
        log(f"walk {k}: optimising on its own" if not u["split"] else f"walk {k}: optimising {u['label']} on its own")
        for line in lines:
            log(f"  [{k}] {line}")
    if cache_dir:
        log(f"cache {cache_dir}: {len(from_cache)} of {n} units came from it ({time.time() - t0:.0f} s)")
    own = [out / f"walk{k}" / "poses.npz" for k in range(n)]

    fits = [georef.fit_translation(w, p) for w, p in zip(walks, own)]  # (E0, N0, dz, ...) per walk
    gps_t = [np.array([f[0] - fits[0][0], f[2] - fits[0][2], -(f[1] - fits[0][1])]) for f in fits]  # walk k -> walk 0
    data = []
    for _, d in done:
        d["clouds"] = [cloud_of(p, nrm) for p, nrm in d["clouds"]]
        data.append(d)
    log(f"fragments: {[len(d['clouds']) for d in data]} ({time.time() - t0:.0f} s)")

    # ---- walk-level registration
    G = [np.eye(4)] + [make_G(0, t) for t in gps_t[1:]]
    G_gps = [g.copy() for g in G]
    reports = [{"method": "reference"}] + [{} for _ in range(n - 1)]
    registered, excluded, anchored = [0], set(), {0}  # anchored: registered or placed some other way (GPS, continuity)
    ref = join(data[0]["clouds"])
    pending, attempts = list(range(1, n)), {}
    while pending:  # a unit that fails is retried once something else got registered (the reference has grown)
        progressed, still = False, []
        for k in pending:
            log(f"walk {k}: registering to walk(s) {registered}")
            src = join(data[k]["clouds"])
            guesses = [("GPS", gps_t[k])]
            siblings = [s for s in registered if units[k]["split"] and units[s]["walk"] == units[k]["walk"]]
            if siblings:  # same walk, same ARKit world after relocalising: its frame is the better first guess than GPS
                guesses.insert(0, ("sibling segment's placement", G[siblings[-1]][:3, 3].copy()))
            attempts[k] = []
            for name, tg in guesses:
                T, rep = register_walk(src, ref, tg, log)
                attempts[k].append({"initial_guess": name, **rep})
                if T is not None:
                    break
            if T is None:
                still.append(k)
                continue
            G[k] = T
            reports[k] = {"method": "registered", "initial_guess": name, **rep}
            registered.append(k)
            anchored.add(k)
            progressed = True
            ref = join([ref, place(src, T)])
            q = rep["registration"]
            log(f"  registered ({name}): yaw {q['yaw_deg']:+.2f} deg, {q['gps_error_m']:.2f} m from GPS, ratio {q['ratio']:.2f}, "
                f"rmse {q['rmse'] * 100:.1f} cm, overlap {q['overlap']:.2f} ({time.time() - t0:.0f} s)")
        if not progressed and visual:  # ICP found nothing for anyone left: try the photos
            for k in list(still):
                Gk, rep = visual_register(k, registered, walks, own, G, jobs, lambda s, k=k: log(f"  [{k}] {s}"))
                attempts[k].append({"initial_guess": "visual", **rep})
                if Gk is None:
                    log(f"  visual registration of walk {k} failed: {rep['why']}")
                    continue
                G[k] = Gk
                registered.append(k)
                anchored.add(k)
                still.remove(k)
                progressed = True
                reports[k] = {"method": "visual", "visual": rep}
                ref = join([ref, place(join(data[k]["clouds"]), Gk)])
                log(f"  registered visually: yaw {rep['yaw_deg']:+.2f} deg, {rep['agreeing']} of {rep['constraints']} constraints agree "
                    f"(spread {rep['spread_m']:.2f} m), {rep['moved_from_first_guess_m']:.1f} m from its first guess ({time.time() - t0:.0f} s)")
        if not progressed:  # a piece of a walk next to a registered piece of it: the walk's own frame survived the cut, and
            for k in list(still):  # being placed there it becomes a reference for the photos of the others
                cont = continuity(k, set(registered), units, data) if units[k]["split"] else None
                if cont:
                    s, dist, gap = cont
                    last = attempts[k][-1]
                    G[k] = G[s].copy()
                    registered.append(k)
                    anchored.add(k)
                    still.remove(k)
                    progressed = True
                    msg = (f"{units[k]['label']} could not be registered ({last.get('why') or last['attempt']['why']}); placed in the frame of "
                           f"{units[s]['label']} instead: ARKit's pose is continuous across the cut ({dist:.1f} m in {gap:.1f} s), "
                           f"not verified geometrically")
                    reports[k] = {"method": "continuity_placement", "sibling": s, "continuity_distance_m": dist, "gap_s": gap,
                                  "attempts": attempts[k], "warnings": [msg]}
                    warnings.append(msg)
                    log("WARNING: " + msg)
        pending = still if progressed else []
        if not progressed:
            for k in still:
                last = attempts[k][-1]
                why = last.get("why") or last["attempt"]["why"]
                cont = continuity(k, anchored, units, data) if units[k]["split"] else None
                if cont:  # no overlap to register on, but the walk's own frame survived the cut: place it there, unverified
                    s, dist, gap = cont
                    G[k] = G[s].copy()
                    anchored.add(k)
                    if s in registered:  # a GPS-placed sibling keeps its pieces out of the joint graph with it
                        registered.append(k)
                    msg = (f"{units[k]['label']} could not be registered ({why}); placed in the frame of {units[s]['label']} "
                           f"instead: ARKit's pose is continuous across the cut ({dist:.1f} m in {gap:.1f} s), not verified geometrically"
                           + ("" if s in registered else " (that segment itself keeps its GPS placement)"))
                    reports[k] = {"method": "continuity_placement", "sibling": s, "continuity_distance_m": dist, "gap_s": gap,
                                  "attempts": attempts[k], "warnings": [msg]}
                elif units[k]["split"] and any(units[s]["walk"] == units[k]["walk"] for s in anchored):
                    # another piece of this walk is placed but this one is not continuous with it: GPS alone could double what it overlaps
                    msg = f"{units[k]['label']} could not be registered ({why}) and is not continuous with its neighbours: excluded from the model"
                    excluded.add(k)
                    reports[k] = {"method": "excluded", "attempts": attempts[k], "warnings": [msg]}
                else:  # a whole walk, or the first piece of a split one to be placed: its GPS placement, as ever
                    anchored.add(k)
                    msg = (f"{units[k]['label'] if units[k]['split'] else f'walk {k}'} could not be registered ({why}): "
                           f"kept its GPS placement (~1-5 m, yaw 0), left out of the joint graph")
                    reports[k] = {"method": "gps_fallback", **last, "warnings": [msg]}
                warnings.append(msg)
                log("WARNING: " + msg)

    # ---- second chance: the first pass registers in order, against what was registered so far. A piece that was placed
    # by continuity / GPS or excluded then may overlap pieces registered after it: retry it against every piece that has
    # real evidence (ICP, then photos), until nothing changes.
    weak_methods = ("continuity_placement", "gps_fallback", "excluded")
    changed, retries = True, {}
    while changed:
        changed = False
        for k in range(1, n):
            if reports[k]["method"] not in weak_methods:
                continue
            strong = [s for s in range(n) if reports[s]["method"] not in weak_methods]
            log(f"retrying {units[k]['label'] if units[k]['split'] else f'walk {k}'} ({reports[k]['method']}) against {strong}")
            ref_k = join([place(join(data[s]["clouds"]), G[s]) for s in strong])
            src = join(data[k]["clouds"])
            old = reports[k]
            T, rep, name = None, {}, "GPS"
            for name, tg in [("current placement", G[k][:3, 3].copy()), ("GPS", gps_t[k])]:
                T, rep = register_walk(src, ref_k, tg, log)
                if T is not None:
                    break
            if T is not None:
                new = {"method": "registered", "initial_guess": name, **rep}
            elif visual:
                # the piece already sits near where it belongs (continuity), so first look close by, where the pairs are
                # dense enough to be all tried; then as wide as for a piece with only a GPS guess
                for radius in (RETRY_VISUAL_RADIUS, VISUAL_REGISTER_RADIUS):
                    Gk, vrep = visual_register(k, strong, walks, own, G, jobs, lambda s, k=k: log(f"  [{k}] {s}"), radius=radius)
                    if Gk is not None:
                        T, new = Gk, {"method": "visual", "visual": vrep}
                        break
                    log(f"  retry by photos within {radius:.0f} m failed: {vrep['why']}" + (f"; placements {vrep['placements']}" if "placements" in vrep else ""))
                    retries.setdefault(k, []).append({"radius_m": radius, **vrep})
            if T is None:
                if k in retries:
                    reports[k]["retries"] = retries[k]
                continue
            for m in old.get("warnings", []):
                if m in warnings:
                    warnings.remove(m)
            G[k] = T
            new["retried_after"] = old["method"]
            reports[k] = new
            excluded.discard(k)
            anchored.add(k)
            if k not in registered:
                registered.append(k)
            changed = True
            log(f"  {units[k]['label']} now placed by {new['method']} (was {old['method']})")
            for j in range(1, n):  # pieces placed in this one's frame by continuity follow it
                if reports[j]["method"] == "continuity_placement" and reports[j]["sibling"] == k:
                    G[j] = G[k].copy()

    # ---- joint pose graph over the registered walks' fragments
    nodes =[(k, f) for k in registered for f in range(len(data[k]["clouds"]))]
    node_of = {kf: i for i, kf in enumerate(nodes)}
    placed = [place(data[k]["clouds"][f], G[k]) for k, f in nodes]
    pyrs = [optimize.pyramid(p) for p in placed]
    pts = [np.asarray(p.points) for p in placed]
    cams = [data[k]["cams"][f] @ G[k][:3, :3].T + G[k][:3, 3] for k, f in nodes]
    walked = [data[k]["walked"][f] for k, f in nodes]

    pg = reg.PoseGraph()
    for k, f in nodes:
        pg.nodes.append(reg.PoseGraphNode(G[k]))
    for k in registered:
        g = data[k]["groups"]
        for f in range(len(g) - 1):
            gap = g[f + 1][0] - g[f][-1] > 1  # tracking was lost in between
            pg.edges.append(reg.PoseGraphEdge(node_of[(k, f)], node_of[(k, f + 1)], np.eye(4),
                                              np.eye(6) * (optimize.GAP_INFO if gap else optimize.ODOM_INFO), uncertain=False))

    candidates = []
    for ia, a in enumerate(registered):
        for b in registered[ia + 1:]:
            for fi in range(len(data[a]["clouds"])):
                for fj in range(len(data[b]["clouds"])):
                    i, j = node_of[(a, fi)], node_of[(b, fj)]
                    if np.linalg.norm(cams[i][::5, None] - cams[j][None, ::5], axis=2).min() > optimize.LOOP_CAM_DIST:
                        continue
                    if optimize.overlap_fraction(pts[i], pts[j]) < optimize.MIN_OVERLAP:
                        continue
                    candidates.append((i, j))
    log(f"{len(candidates)} inter-walk loop candidates")
    found, rejected = [], []
    for m, (i, j) in enumerate(candidates):
        R, q, c = optimize.register_pair(placed[i], placed[j], pyrs[i], pyrs[j])
        if R is None:
            rejected.append({"walks": [nodes[i][0], nodes[j][0]], "fragments": [nodes[i][1], nodes[j][1]], "why": q})
        else:
            found.append((i, j, R, q, c))
        if m % 20 == 19:
            log(f"  {m + 1}/{len(candidates)} ICP done, {len(found)} passed ({time.time() - t0:.0f} s)")

    def corroborated(i, j, R, q, c):
        """Another loop between the same two walks, near these places in both, asking for about the same correction."""
        v = R[:3, :3] @ c + R[:3, 3]
        return any(
            (i2, j2) != (i, j) and nodes[i2][0] == nodes[i][0] and nodes[j2][0] == nodes[j][0]
            and abs(walked[i2] - walked[i]) <= optimize.GUARD_PATH and abs(walked[j2] - walked[j]) <= optimize.GUARD_PATH
            and np.linalg.norm(v - (R2[:3, :3] @ c + R2[:3, 3])) <= optimize.GUARD_AGREE_SHIFT
            and abs(q["yaw"] - q2["yaw"]) <= optimize.GUARD_AGREE_YAW
            for i2, j2, R2, q2, _ in found)

    loops = []
    for i, j, R, q, c in found:
        wa, wb = nodes[i][0], nodes[j][0]
        if (q["shift"] > optimize.GUARD_SHIFT or abs(q["yaw"]) > optimize.GUARD_YAW) and not corroborated(i, j, R, q, c):
            rejected.append({"walks": [wa, wb], "fragments": [nodes[i][1], nodes[j][1]],
                             "why": "large correction, no agreeing loop nearby", "quality": q})
            continue
        info = reg.get_information_matrix_from_point_clouds(placed[i], placed[j], INLIER_DIST, R)
        edge = np.linalg.inv(G[wb]) @ R @ G[wa]  # fragment i's own frame -> fragment j's own frame
        pg.edges.append(reg.PoseGraphEdge(i, j, edge, info, uncertain=True))
        loops.append({"walks": [wa, wb], "fragments": [nodes[i][1], nodes[j][1]], **q})
    log(f"{len(loops)} inter-walk loops accepted, {len(found) - len(loops)} dropped by the corroboration guard ({time.time() - t0:.0f} s)")

    visual_graph = {}
    if visual and len(registered) > 1:  # photo-feature loops between the pieces whose paths come close once placed
        import visual as vis
        kfs = {k: vis.keyframes(walks[k], own[k]) for k in registered}
        frag_of = {k: {data[k]["frames"][m]["i"]: f for f, g in enumerate(data[k]["groups"]) for m in g} for k in registered}
        cam = {k: np.array([fr["T"][:3, 3] for fr in data[k]["frames"]]) @ G[k][:3, :3].T + G[k][:3, 3] for k in registered}
        for ia, a in enumerate(registered):
            for b in registered[ia + 1:]:
                if np.linalg.norm(cam[a][::10, None] - cam[b][None, ::10], axis=2).min() > VISUAL_PAIR_RADIUS:
                    continue
                cons, summ = vis.visual_loops(walks[a], walks[b], poses_a=own[a], poses_b=own[b], init=np.linalg.inv(G[a]) @ G[b],
                                              max_pairs=optimize.VISUAL_MAX_PAIRS, workers=max(1, jobs), log=lambda s, a=a, b=b: log(f"  [{a}-{b}] {s}"))
                added = 0
                for fa, fb, T, info, n_ok, n_all, _ in optimize.visual_edges(cons, kfs[a], kfs[b], frag_of[a], frag_of[b]):
                    X = G[b] @ T  # where the edge would put fragment fa of walk a
                    moved = float(np.linalg.norm(X[:3, :3] @ data[a]["cen"][fa] + X[:3, 3] - (G[a][:3, :3] @ data[a]["cen"][fa] + G[a][:3, 3])))
                    if moved > VISUAL_MAX_CORRECTION:
                        rejected.append({"walks": [a, b], "fragments": [fa, fb], "why": "visual: implausible correction", "moved_m": moved})
                        continue
                    pg.edges.append(reg.PoseGraphEdge(node_of[(a, fa)], node_of[(b, fb)], T, info, uncertain=True))
                    loops.append({"walks": [a, b], "fragments": [fa, fb], "kind": "visual", "constraints": n_ok, "of": n_all, "moved_m": moved})
                    added += 1
                visual_graph[f"{a}-{b}"] = {"pairs": summ["pairs"], "constraints": len(cons), "edges_added": added}
                log(f"visual: walks {a}-{b}: {len(cons)} constraints, {added} fragment-pair edges added ({time.time() - t0:.0f} s)")

    # Open3D skips the whole optimisation ("graph is not connected", only visible at debug verbosity) when any piece has no
    # edge to the rest, e.g. a piece placed by continuity that no loop reaches. Tie each such group to node 0 with a nearly
    # free "stay where you are" edge, so the loops of everything else still get used.
    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in pg.edges:
        parent[find(e.source_node_id)] = find(e.target_node_id)
    tied = 0
    for i in range(len(nodes)):
        if find(i) != find(0):
            pg.edges.append(reg.PoseGraphEdge(0, i, np.linalg.inv(G[nodes[i][0]]) @ G[nodes[0][0]], np.eye(6) * optimize.GAP_INFO, uncertain=False))
            parent[find(i)] = find(0)
            tied += 1
    if tied:
        log(f"{tied} piece(s) without any edge to the rest were tied to the reference weakly (at their placement)")

    if loops:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(), reg.GlobalOptimizationConvergenceCriteria(),
                                reg.GlobalOptimizationOption(max_correspondence_distance=INLIER_DIST, edge_prune_threshold=0.25,
                                                             preference_loop_closure=2.0, reference_node=0))
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    kept = {(e.source_node_id, e.target_node_id) for e in pg.edges if e.uncertain}

    # ---- final per-fragment transforms, per-frame poses
    C = {k: [G[k]] * len(data[k]["clouds"]) for k in range(n)}  # fallback walks stay at their GPS placement
    for i, (k, f) in enumerate(nodes):
        C[k][f] = yaw_only(np.array(pg.nodes[i].pose), data[k]["cen"][f])
    merged = []
    for k, w in enumerate(walks):
        d = data[k]
        centres = [np.mean(g) for g in d["groups"]]
        theta = np.array([yaw_of(c[:3, :3]) for c in C[k]])
        trans = np.array([c[:3, 3] for c in C[k]])
        poses = optimize.frame_poses(d["frames"], centres, theta, trans)
        merged.append(out / f"walk{k}" / "poses_merged.npz")
        np.savez(merged[-1], i=np.array([f["i"] for f in d["frames"]]), T=poses)

    for lp in loops:
        (a, b), (fi, fj) = lp["walks"], lp["fragments"]
        i, j = node_of[(a, fi)], node_of[(b, fj)]
        lp["kept"] = (i, j) in kept
        for name, Ta, Tb in (("before", G[a], G[b]), ("after", C[a][fi], C[b][fj])):
            dd = gaps(data[a]["clouds"][fi], data[b]["clouds"][fj], np.linalg.inv(Tb) @ Ta)
            lp[f"within_{int(INLIER_DIST * 100)}cm_{name}"] = float((dd[dd < NEAR_DIST] < INLIER_DIST).mean()) if (dd < NEAR_DIST).any() else 0.0
            lp[f"gap_{name}_median_m"] = float(np.median(dd[dd < NEAR_DIST])) if (dd < NEAR_DIST).any() else None

    gps_only = {k: [G_gps[k]] * len(data[k]["clouds"]) for k in range(n)}
    rigid = {k: [G[k]] * len(data[k]["clouds"]) for k in range(n)}
    for k in registered[1:]:
        reports[k]["consistency_with_earlier_walks"] = {
            "gps_placement": consistency(data, gps_only, k, [w for w in registered if w < k]),
            "rigid_walk_transform": consistency(data, rigid, k, [w for w in registered if w < k]),
            "joint_graph": consistency(data, C, k, [w for w in registered if w < k])}
    for k in range(n):
        reports[k].update({
            "walk": walks[k].name, "unit": units[k]["label"], "segment": units[k]["segment"] if units[k]["split"] else None,
            "frames": len(data[k]["frames"]), "seconds": data[k]["frames"][-1]["t"] - data[k]["frames"][0]["t"],
            "yaw_deg": float(np.degrees(yaw_of(G[k][:3, :3]))), "translation_m": G[k][:3, 3].tolist(),
            "gps_translation_m": gps_t[k].tolist(),
            "joint_graph_deviation_from_rigid": {
                "max_shift_m": float(max(shift_at(np.linalg.inv(G[k]) @ c, data[k]["cen"][f]) for f, c in enumerate(C[k]))),
                "max_yaw_deg": float(max(abs(np.degrees(yaw_of((np.linalg.inv(G[k]) @ c)[:3, :3]))) for c in C[k]))}})

    stats = {"frame": "walk 0's ARKit frame; transforms map each walk's own frame into it (yaw about +Y, then translation)",
             "cache": {"dir": str(cache_dir) if cache_dir else None, "units_from_cache": from_cache},
             "fuse_stride": stride, "visual": visual, "visual_graph": visual_graph, "pieces_tied_weakly_to_the_reference": tied, "input_walks": plans, "walks": reports, "excluded": [units[k]["label"] for k in sorted(excluded)],
             "inter_walk_loops": loops, "rejected_loops": rejected, "candidates": len(candidates),
             "kept_loops": sum(lp["kept"] for lp in loops), "pruned_loops": sum(not lp["kept"] for lp in loops),
             "warnings": warnings}
    log(f"{stats['kept_loops']}/{len(loops)} inter-walk loops kept after pruning ({time.time() - t0:.0f} s)")

    used = [k for k in range(n) if k not in excluded]
    fuse.fuse_many([(walks[k], merged[k]) for k in used], out, jobs=jobs, stride=stride)
    georef.georef(out, [walks[k] for k in used], [merged[k] for k in used])
    subprocess.run([sys.executable, "-I", str(HERE / "preview.py"), str(out)], check=True)
    shutil.copy(HERE / "viewer.html", out / "viewer.html")
    stats["seconds"] = round(time.time() - t0, 1)
    (out / "merge.json").write_text(json.dumps(stats, indent=1))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("walk_dirs", nargs="+")
    ap.add_argument("--jobs", type=int, default=cores.DEFAULT_JOBS, help="cores to use (default %(default)s)")
    ap.add_argument("--no-visual", action="store_true", help="no photo-feature constraints (registration fallback, loop edges)")
    ap.add_argument("--cache", help="per-walk cache directory (default: cache/ next to out_dir)")
    ap.add_argument("--no-cache", action="store_true", help="recompute everything, store nothing")
    ap.add_argument("--stride", type=int, default=1, help="fuse every N-th depth frame (default %(default)s)")
    a = ap.parse_args()
    cache_dir = None if a.no_cache else Path(a.cache) if a.cache else Path(a.out_dir).resolve().parent / "cache"
    s = merge(a.out_dir, a.walk_dirs, jobs=a.jobs, visual=not a.no_visual, cache_dir=cache_dir, stride=a.stride)
    print(f"\nDone: {a.out_dir}")
    for w in s["warnings"]:
        print("WARNING:", w, file=sys.stderr)
    print(f"View in 3D:  python3 -m http.server -d {a.out_dir} 8000   then open http://localhost:8000/viewer.html")
