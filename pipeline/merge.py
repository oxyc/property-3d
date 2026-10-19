"""Merge several walks into one model in the first walk's frame.

usage: merge.py <out_dir> <walk_dir> <walk_dir> [...]

Every walk is optimised on its own first (optimize.py: snaps, drift loops). Then, in order, each walk is
registered to the union of the walks before it:
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

import numpy as np
import open3d as o3d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # python -I drops the script dir from sys.path

import fuse  # noqa: E402
import georef  # noqa: E402
import optimize  # noqa: E402
from optimize import INLIER_DIST, MIN_INLIERS, NEAR_DIST, gaps, reg, rot_y, shift_at, yaw_of, yaw_only  # noqa: E402

GRID = 0.25  # raster cell for the coarse search (m)
STRUCT_MIN = 0.15  # vertical extent per cell that counts as ground relief, not structure (m)
SEARCH = 12.0  # translation search radius around the GPS guess (m)
YAW_RANGE, YAW_STEP = 20.0, 1.0  # degrees
BLUR = 1.5  # cells
N_CANDIDATES = 4  # coarse peaks ranked by ICP
MIN_WALK_OVERLAP = 0.1  # fraction of the smaller cloud lying within 30 cm of the other, after registration
MIN_WALK_RATIO = 0.3  # inliers / points with a neighbour within NEAR_DIST (whole walks carry residual drift)
MAX_WALK_RMSE = 0.07  # of the inliers (< 10 cm); looser than for fragments: a whole walk carries residual drift
MAX_GPS_ERR = 15.0  # the registered placement may not be further than this from the GPS one (m)
OVERLAP_DIST = 0.3


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
    for f in frames:
        f["T"] = corrected[f["i"]]
    groups = optimize.fragment_groups(frames, frag_frames)
    clouds = [optimize.build_fragment(walk, meta, frames, g) for g in groups]
    cams = [np.array([frames[n]["T"][:3, 3] for n in g]) for g in groups]
    mid = [c[len(c) // 2] for c in cams]
    walked = np.concatenate([[0], np.cumsum([np.linalg.norm(mid[k + 1] - mid[k]) for k in range(len(mid) - 1)])])
    return {"frames": frames, "groups": groups, "clouds": clouds, "cams": cams, "walked": walked,
            "cen": [np.asarray(c.points).mean(0) for c in clouds]}


def consistency(data, transforms, k, earlier):
    """Nearest-neighbour gap from walk k's points to the earlier walks' points, where they are within NEAR_DIST,
    with every fragment moved by transforms[walk][fragment]."""
    src = join([place(c, transforms[k][f]) for f, c in enumerate(data[k]["clouds"])])
    tgt = join([place(c, transforms[w][f]) for w in earlier for f, c in enumerate(data[w]["clouds"])])
    d = np.asarray(src.compute_point_cloud_distance(tgt))
    near = d < NEAR_DIST
    return {"median_gap_m": float(np.median(d[near])) if near.any() else None, "within_1m_fraction": float(near.mean()),
            "within_10cm_of_overlap": float((d[near] < INLIER_DIST).mean()) if near.any() else None}


def merge(out, walks, frag_frames=optimize.FRAG_FRAMES, log=print):
    t0 = time.time()
    out, walks = Path(out), [Path(w) for w in walks]
    out.mkdir(parents=True, exist_ok=True)
    n = len(walks)
    warnings = []

    for k, w in enumerate(walks):
        log(f"walk {k}: optimising on its own")
        optimize.optimize(w, out / f"walk{k}", frag_frames, log=lambda s, k=k: log(f"  [{k}] {s}"))
    own = [out / f"walk{k}" / "poses.npz" for k in range(n)]

    fits = [georef.fit_translation(w, p) for w, p in zip(walks, own)]  # (E0, N0, dz, ...) per walk
    gps_t = [np.array([f[0] - fits[0][0], f[2] - fits[0][2], -(f[1] - fits[0][1])]) for f in fits]  # walk k -> walk 0
    data = [fragment_data(w, p, frag_frames) for w, p in zip(walks, own)]
    log(f"fragments: {[len(d['clouds']) for d in data]} ({time.time() - t0:.0f} s)")

    # ---- walk-level registration
    G = [np.eye(4)] + [make_G(0, t) for t in gps_t[1:]]
    G_gps = [g.copy() for g in G]
    reports = [{"method": "reference"}] + [{} for _ in range(n - 1)]
    registered = [0]
    ref = join(data[0]["clouds"])
    for k in range(1, n):
        log(f"walk {k}: registering to walk(s) {registered}")
        src = join(data[k]["clouds"])
        T, rep = register_walk(src, ref, gps_t[k], log)
        if T is None:
            msg = f"walk {k} could not be registered ({rep.get('why') or rep['attempt']['why']}): kept its GPS placement (~1-5 m, yaw 0), left out of the joint graph"
            warnings.append(msg)
            reports[k] = {"method": "gps_fallback", **rep, "warnings": [msg]}
            log("WARNING: " + msg)
            continue
        G[k] = T
        reports[k] = {"method": "registered", **rep}
        registered.append(k)
        ref = join([ref, place(src, T)])
        q = rep["registration"]
        log(f"  registered: yaw {q['yaw_deg']:+.2f} deg, {q['gps_error_m']:.2f} m from GPS, ratio {q['ratio']:.2f}, "
            f"rmse {q['rmse'] * 100:.1f} cm, overlap {q['overlap']:.2f} ({time.time() - t0:.0f} s)")

    # ---- joint pose graph over the registered walks' fragments
    nodes = [(k, f) for k in registered for f in range(len(data[k]["clouds"]))]
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
            "walk": walks[k].name,
            "yaw_deg": float(np.degrees(yaw_of(G[k][:3, :3]))), "translation_m": G[k][:3, 3].tolist(),
            "gps_translation_m": gps_t[k].tolist(),
            "joint_graph_deviation_from_rigid": {
                "max_shift_m": float(max(shift_at(np.linalg.inv(G[k]) @ c, data[k]["cen"][f]) for f, c in enumerate(C[k]))),
                "max_yaw_deg": float(max(abs(np.degrees(yaw_of((np.linalg.inv(G[k]) @ c)[:3, :3]))) for c in C[k]))}})

    stats = {"frame": "walk 0's ARKit frame; transforms map each walk's own frame into it (yaw about +Y, then translation)",
             "walks": reports, "inter_walk_loops": loops, "rejected_loops": rejected, "candidates": len(candidates),
             "kept_loops": sum(lp["kept"] for lp in loops), "pruned_loops": sum(not lp["kept"] for lp in loops),
             "warnings": warnings}
    log(f"{stats['kept_loops']}/{len(loops)} inter-walk loops kept after pruning ({time.time() - t0:.0f} s)")

    fuse.fuse_many(list(zip(walks, merged)), out)
    georef.georef(out, walks, merged)
    subprocess.run([sys.executable, "-I", str(HERE / "preview.py"), str(out)], check=True)
    shutil.copy(HERE / "viewer.html", out / "viewer.html")
    stats["seconds"] = round(time.time() - t0, 1)
    (out / "merge.json").write_text(json.dumps(stats, indent=1))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("walk_dirs", nargs="+")
    a = ap.parse_args()
    if len(a.walk_dirs) < 2:
        ap.error("need at least two walks")
    s = merge(a.out_dir, a.walk_dirs)
    print(f"\nDone: {a.out_dir}")
    for w in s["warnings"]:
        print("WARNING:", w, file=sys.stderr)
    print(f"View in 3D:  python3 -m http.server -d {a.out_dir} 8000   then open http://localhost:8000/viewer.html")
