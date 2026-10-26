"""End-to-end check: synth -> fuse -> georef against the known synthetic scene.

Run: uv run --project . python test_pipeline.py     (or pytest)

Wrong camera/axis conventions (column-major, Y/Z flips, depth->RGB scaling, east/north
mapping) show up here as errors of decimetres to metres, far above the thresholds.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent))  # not on sys.path when run with -I
import fuse
import georef
import merge
import optimize
import synth

GPS_SIGMA = 4.0  # mean synthetic h_acc, metres


def pct(a, q):
    return float(np.percentile(a, q))


def test_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        walk, out = Path(tmp) / "walk", Path(tmp) / "out"
        synth.generate(walk)
        fuse.fuse(walk, out)
        info = georef.georef(out, walk)

        # --- geometry: points and mesh lie on the true surfaces
        pts = np.asarray(o3d.io.read_point_cloud(str(out / "points.ply")).points)
        d = np.abs(synth.scene_sdf(pts))
        print(f"points: n={len(pts)}  |dist to scene| median {pct(d, 50)*100:.2f} cm, p90 {pct(d, 90)*100:.2f} cm, p99 {pct(d, 99)*100:.2f} cm")
        assert len(pts) > 100_000
        assert pct(d, 50) < 0.015 and pct(d, 90) < 0.04 and pct(d, 99) < 0.15

        verts = np.asarray(o3d.io.read_triangle_mesh(str(out / "mesh.ply")).vertices)
        dm = np.abs(synth.scene_sdf(verts))
        print(f"mesh:   n={len(verts)}  |dist to scene| median {pct(dm, 50)*100:.2f} cm, p90 {pct(dm, 90)*100:.2f} cm")
        assert len(verts) > 50_000 and pct(dm, 50) < 0.02 and pct(dm, 90) < 0.05

        # --- colour: coloured points match the albedo of the surface they sit on
        pcd = o3d.io.read_point_cloud(str(out / "points.ply"))
        col = np.asarray(pcd.colors) * 255
        coloured = np.abs(col - 127.5).max(1) > 3  # not the grey fallback
        near = coloured & (np.abs(synth.scene_sdf(pts)) < 0.1)
        ids = synth.scene_sdf_all(pts[near]).argmin(0)
        err = np.abs(col[near] - synth.albedo(pts[near], ids)).max(1)
        print(f"colour: {coloured.mean()*100:.0f}% of points coloured, median max-channel error {pct(err, 50):.1f}/255, p75 {pct(err, 75):.1f}")
        assert coloured.mean() > 0.5 and pct(err, 50) < 12

        # --- ground grid: lowest point per cell vs true ground height
        g = np.load(out / "ground.npz")
        h, c = g["height"], float(g["cell"])
        rows, cols = np.nonzero(~np.isnan(h))
        east, north = g["west"] + (cols + 0.5) * c, g["north"] - (rows + 0.5) * c
        e = h[rows, cols] - synth.ground_height(east, -north)
        print(f"ground: {len(e)} cells, error median {np.median(e)*100:.1f} cm, p10 {pct(e, 10)*100:.1f} cm, "
              f"{(np.abs(e) < 0.1).mean()*100:.0f}% within 10 cm (rest = cells holding a box/trunk)")
        assert len(e) > 3000 and abs(np.median(e)) < 0.05 and (np.abs(e) < 0.1).mean() > 0.85

        # --- georef: estimated origin vs truth, and georeferenced points vs independently mapped points
        E0, N0 = synth.ORIGIN_EN
        err_en = np.hypot(info["E0"] - E0, info["N0"] - N0)
        print(f"georef: translation error {err_en:.2f} m (GPS noise ~{GPS_SIGMA} m per fix, {info['n_fixes']} fixes, "
              f"weighted residual rms {info['weighted_rms_residual_m']:.2f} m)")
        assert err_en < 2.0 < GPS_SIGMA

        gp = np.asarray(o3d.io.read_point_cloud(str(out / "georef_points.ply")).points) + info["offset"]
        truth = np.stack([E0 + pts[:, 0], N0 - pts[:, 2]], 1)  # east = X, north = -Z
        diff = np.abs(gp[:, :2] - truth).max()
        print(f"georef: max |georeferenced - true| over all points {diff:.2f} m (equals the translation error)")
        assert diff < 2.0 + 1e-3 and np.ptp(gp[:, 0]) < 30

        import rasterio
        with rasterio.open(out / "ground.tif") as r:
            assert r.crs.to_epsg() == 5382 and abs(r.bounds.left - (info["E0"] + float(g["west"]))) < 1e-6


def scene_error(walk, out, poses=None):
    """Fuse and return (median, p90) of |distance to the true scene| over the fused points, metres."""
    fuse.fuse(walk, out, poses=poses)
    d = np.abs(synth.scene_sdf(np.asarray(o3d.io.read_point_cloud(str(out / "points.ply")).points)))
    return pct(d, 50), pct(d, 90)


def true_pose_walk(drift_walk, dst, start=0):
    """Same depth/conf/rgb as the drifted walk, but frames.jsonl carries the true transforms from frame `start` on."""
    dst.mkdir()
    for sub in ("depth", "conf", "rgb"):
        (dst / sub).symlink_to(drift_walk / sub)
    (dst / "meta.json").write_text((drift_walk / "meta.json").read_text())
    true = json.loads((drift_walk / "truth.json").read_text())["true_transforms"]
    frames = [json.loads(l) for l in (drift_walk / "frames.jsonl").read_text().splitlines()]
    for f, T in list(zip(frames, true))[start:]:
        f["transform"] = T
    (dst / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    return dst


def pose_error(poses, drift_walk, recorded=None):
    """Camera position error (m) of poses.npz (or the poses recorded in `recorded`, default drift_walk, if None)
    against the true ones."""
    true = np.array(json.loads((drift_walk / "truth.json").read_text())["true_transforms"])[:, 12:15]
    if poses is None:
        got = np.array([json.loads(l)["transform"][12:15] for l in ((recorded or drift_walk) / "frames.jsonl").read_text().splitlines()])
    else:
        got = np.load(poses)["T"][:, :3, 3]
    e = np.linalg.norm(got - true, axis=1)
    return float(e.mean()), float(e.max())


def test_drift():
    """Two loops of the same circle with accumulating drift: loop closure must pull the map back to the scene."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        walk = tmp / "drift"
        synth.generate(walk, 440, drift=True)
        clean = true_pose_walk(walk, tmp / "clean")
        frag = 40  # frames; a fragment is ~5 m of the 28 m circle

        stats = optimize.optimize(walk, tmp / "opt", frag, log=lambda *_: None)
        raw = scene_error(walk, tmp / "f_raw")
        opt = scene_error(walk, tmp / "f_opt", tmp / "opt" / "poses.npz")
        pe_raw, pe_opt = pose_error(None, walk), pose_error(tmp / "opt" / "poses.npz", walk)
        print(f"drift:  {stats['fragments']} fragments, {stats['candidates']} candidates, {stats['accepted_loops']} accepted, "
              f"{stats['pruned_loops']} pruned")
        print(f"drift:  camera position error mean/max raw {pe_raw[0]:.2f}/{pe_raw[1]:.2f} m -> optimised {pe_opt[0]:.2f}/{pe_opt[1]:.2f} m")
        print(f"drift:  |dist to scene| median/p90 raw {raw[0]*100:.2f}/{raw[1]*100:.2f} cm -> optimised {opt[0]*100:.2f}/{opt[1]*100:.2f} cm")
        assert stats["accepted_loops"] >= 2 and stats["accepted_loops"] > stats["pruned_loops"]
        assert pe_opt[0] < 0.25 * pe_raw[0]
        assert opt[0] < 0.5 * raw[0] and opt[1] < 0.5 * raw[1]

        stats = optimize.optimize(clean, tmp / "opt_clean", frag, log=lambda *_: None)
        c_raw = scene_error(clean, tmp / "c_raw")
        c_opt = scene_error(clean, tmp / "c_opt", tmp / "opt_clean" / "poses.npz")
        pe = pose_error(tmp / "opt_clean" / "poses.npz", walk)
        print(f"clean:  {stats['accepted_loops']} loops accepted, max pose change {pe[1]*100:.1f} cm; "
              f"|dist to scene| median/p90 raw {c_raw[0]*100:.2f}/{c_raw[1]*100:.2f} cm -> optimised {c_opt[0]*100:.2f}/{c_opt[1]*100:.2f} cm")
        assert c_opt[0] < c_raw[0] + 0.003 and c_opt[1] < c_raw[1] + 0.005 and pe[1] < 0.05


def max_step(poses):
    return float(np.linalg.norm(np.diff(np.load(poses)["T"][:, :3, 3], axis=0), axis=1).max())


def test_snap():
    """Drifted poses that snap back to the truth at frame 360 (ARKit relocalising): the jump must be gone, the earlier
    drift ramped out and the poses after the snap left alone."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        walk = tmp / "drift"
        synth.generate(walk, 440, drift=True)
        snapped = true_pose_walk(walk, tmp / "snapped", start=360)

        stats = optimize.optimize(snapped, tmp / "opt", 40, log=lambda *_: None)
        poses = tmp / "opt" / "poses.npz"
        recorded = np.array([json.loads(l)["transform"][12:15] for l in (snapped / "frames.jsonl").read_text().splitlines()])
        raw_jump = float(np.linalg.norm(np.diff(recorded, axis=0), axis=1).max())
        pe_raw, pe_opt = pose_error(None, walk, snapped), pose_error(poses, walk)
        raw = scene_error(snapped, tmp / "f_raw")
        opt = scene_error(snapped, tmp / "f_opt", poses)
        print(f"snap:   {len(stats['snaps'])} snap at t={stats['snaps'][0]['t']:.1f} s, max frame-to-frame step raw {raw_jump:.2f} m -> "
              f"optimised {max_step(poses):.2f} m (walking is {synth.SPEED / synth.HZ:.2f} m per frame)")
        print(f"snap:   camera position error mean/max raw {pe_raw[0]:.2f}/{pe_raw[1]:.2f} m -> optimised {pe_opt[0]:.2f}/{pe_opt[1]:.2f} m; "
              f"|dist to scene| median/p90 raw {raw[0]*100:.2f}/{raw[1]*100:.2f} cm -> optimised {opt[0]*100:.2f}/{opt[1]*100:.2f} cm")
        assert len(stats["snaps"]) == 1 and raw_jump > 0.4 and max_step(poses) < 0.2
        assert pe_opt[0] < 0.3 * pe_raw[0] and opt[0] < 0.5 * raw[0] and opt[1] < 0.5 * raw[1]


def merged_scene_error(out):
    d = np.abs(synth.scene_sdf(np.asarray(o3d.io.read_point_cloud(str(out / "points.ply")).points)))
    return pct(d, 50), pct(d, 90)


def test_merge():
    """Two walks of the same scene from different start points, ARKit origins and 4 degrees of compass error, GPS noise
    as usual: the merge must recover the relative placement and reconstruct about as well as one walk. Walks that do not
    overlap must come back as a GPS-placement fallback with a warning, not as a confident merge."""
    quiet = lambda *_: None  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a, b = tmp / "a", tmp / "b"
        synth.generate(a, 240, seed=0)
        synth.generate(b, 240, seed=1, phase=2.0, yaw_deg=4.0)
        frag = 40

        optimize.optimize(a, tmp / "opt_a", frag, log=quiet)
        single = scene_error(a, tmp / "f_a", tmp / "opt_a" / "poses.npz")
        stats = merge.merge(tmp / "m", [a, b], frag, log=quiet)
        both = merged_scene_error(tmp / "m")

        to_scene = np.array(json.loads((b / "truth.json").read_text())["to_scene"]).reshape(4, 4, order="F")
        rec = np.array([json.loads(l)["transform"][12:15] for l in (b / "frames.jsonl").read_text().splitlines()])
        got = np.load(tmp / "m" / "walk1" / "poses_merged.npz")["T"][:, :3, 3]
        pose_err = np.linalg.norm(got - (rec @ to_scene[:3, :3].T + to_scene[:3, 3]), axis=1)
        w = stats["walks"][1]
        want_yaw = float(np.degrees(np.arctan2(to_scene[0, 2], to_scene[0, 0])))
        gps_err = np.linalg.norm(np.array(w["gps_translation_m"]) - to_scene[:3, 3])
        cons = w["consistency_with_earlier_walks"]
        print(f"merge:  walk b recovered yaw {w['yaw_deg']:+.2f} deg (true {want_yaw:+.2f}), GPS-only placement was {gps_err:.1f} m off; "
              f"camera position error mean/max {pose_err.mean() * 100:.1f}/{pose_err.max() * 100:.1f} cm")
        print(f"merge:  {stats['candidates']} loop candidates, {len(stats['inter_walk_loops'])} accepted, {stats['kept_loops']} kept")
        print(f"merge:  median gap between the walks in their overlap: GPS {cons['gps_placement']['median_gap_m'] * 100:.1f} cm -> "
              f"rigid {cons['rigid_walk_transform']['median_gap_m'] * 100:.1f} cm -> joint {cons['joint_graph']['median_gap_m'] * 100:.1f} cm")
        print(f"merge:  |dist to scene| median/p90 one walk {single[0] * 100:.2f}/{single[1] * 100:.2f} cm -> merged {both[0] * 100:.2f}/{both[1] * 100:.2f} cm")
        assert w["method"] == "registered" and not stats["warnings"]
        assert abs(w["yaw_deg"] - want_yaw) < 0.5 and pose_err.max() < 0.1
        assert both[0] < single[0] + 0.005 and both[1] < single[1] + 0.01

        # a walk 40 m away (ground only, outside the search radius) and one 18 m away (search runs, nothing to match)
        for name, shift, frames in (("far", (40.0, 0.0), 100), ("near", (18.0, 0.0), 100)):
            c = tmp / name
            synth.generate(c, frames, seed=2, shift=shift, yaw_deg=3.0)
            s = merge.merge(tmp / f"m_{name}", [a, c], frag, log=quiet)
            w = s["walks"][1]
            print(f"merge:  non-overlapping walk ({name}): {w['method']}, {w.get('why') or w['attempt']['why']}; warnings: {len(s['warnings'])}")
            assert w["method"] == "gps_fallback" and len(s["warnings"]) == 1 and not s["inter_walk_loops"]
            assert np.allclose(w["translation_m"], w["gps_translation_m"]) and w["yaw_deg"] == 0.0


def slide_walk(drift_walk, dst, start=200, length=100):
    """The drifted walk with a bad-tracking stretch injected into the recorded poses (depth is untouched): for `length`
    frames from `start` the pose slides away with steps growing from 0.3 to 0.9 m per frame (3-9 m/s, ~60 m in all) while
    turning up to 32 degrees, then relocalises, i.e. snaps back to where the recorded walk was."""
    dst.mkdir()
    for sub in ("depth", "conf", "rgb"):
        (dst / sub).symlink_to(drift_walk / sub)
    for name in ("meta.json", "gps.jsonl", "truth.json"):
        (dst / name).write_text((drift_walk / name).read_text())
    frames = [json.loads(l) for l in (drift_walk / "frames.jsonl").read_text().splitlines()]
    away = np.array([0.8, 0.0, 0.6])
    for k in range(length):
        d = sum(0.3 + 0.6 * j / length for j in range(k + 1))
        a = np.radians(32.0) * k / length
        M = np.array(frames[start + k]["transform"]).reshape(4, 4, order="F")
        M[:3, :3], M[:3, 3] = optimize.rot_y(a) @ M[:3, :3], optimize.rot_y(a) @ M[:3, 3] + d * away
        frames[start + k]["transform"] = M.flatten(order="F").tolist()
    (dst / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    return dst


def test_slide():
    """A drifted two-loop walk with a ~10 s slide in the middle (see slide_walk). The bad stretch must be found and cut
    out, the two clean segments registered to each other, and the result be far better than the uncorrected drift (and
    free of the slide): camera positions and the surface they reconstruct."""
    quiet = lambda *_: None  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        walk = tmp / "drift"
        synth.generate(walk, 440, drift=True)
        slid = slide_walk(walk, tmp / "slid")
        recorded = np.array([json.loads(l)["transform"][12:15] for l in (slid / "frames.jsonl").read_text().splitlines()])
        raw_step = float(np.linalg.norm(np.diff(recorded, axis=0), axis=1).max())

        stats = merge.merge(tmp / "m", [slid], 40, log=quiet)
        plan = stats["input_walks"][0]
        (cut,) = plan["removed_intervals"]
        segs = stats["walks"]
        true = np.array(json.loads((walk / "truth.json").read_text())["true_transforms"])[:, 12:15]
        ids, got = [], []
        for k in range(len(segs)):
            z = np.load(tmp / "m" / f"walk{k}" / "poses_merged.npz")
            ids += z["i"].tolist()
            got.append(z["T"][:, :3, 3])
            assert max(np.linalg.norm(np.diff(z["T"][:, :3, 3], axis=0), axis=1)) < 0.3  # no slide, no jump left
        ids, got = np.array(ids), np.vstack(got)
        err = np.linalg.norm(got - true[ids], axis=1)
        pts = np.asarray(o3d.io.read_point_cloud(str(tmp / "m" / "points.ply")).points)
        d = np.abs(synth.scene_sdf(pts))
        print(f"slide:  recorded max step {raw_step:.1f} m; removed t={cut['t0']:.1f}-{cut['t1']:.1f} s ({cut['seconds']:.1f} s: "
              f"{', '.join(cut['reasons'])}); injected at t=20.0-30.0 s")
        print(f"slide:  segments kept {[(round(s['t0']), round(s['t1'])) for s in plan['segments_kept']]} s, "
              f"{[w['method'] for w in segs]}, {len(ids)} of 440 frames used")
        print(f"slide:  camera position error mean/max {err.mean() * 100:.1f}/{err.max() * 100:.1f} cm, |dist to scene| "
              f"median/p90 {pct(d, 50) * 100:.2f}/{pct(d, 90) * 100:.2f} cm")
        print("slide:  for reference, the same walk without the slide: raw drift 36/74 cm and 14.2/26.3 cm, optimised 6/9 cm and 2.70/5.53 cm. "
              "Cutting out 12 s costs the loop closure with the start (the walk's only absolute anchor), so what is left of the drift is "
              "corrected relative to the segments, not to the truth: better than uncorrected, not as good as the whole walk")
        assert raw_step > 0.5 and len(plan["removed_intervals"]) == 1
        assert 17 < cut["t0"] < 21 and 30 < cut["t1"] < 33 and "sustained speed" in cut["reasons"]
        assert len(plan["segments_kept"]) == 2 and all(w["method"] in ("reference", "registered") for w in segs)
        assert not stats["warnings"] and not stats["excluded"]
        assert not (set(ids) & set(range(205, 296)))  # nothing from inside the slide is fused
        assert err.mean() < 0.3 and err.max() < 0.5  # uncorrected: 0.36 / 0.74 m, and tens of metres with the slide in
        assert pct(d, 50) < 0.10 and pct(d, 90) < 0.16  # uncorrected: 14.2 / 26.3 cm


def gps_lines(path, positions, bias_en, rng):
    """gps.jsonl for a walk whose camera is at `positions` (in the A frame: X east, Z south, origin = ORIGIN_EN) at frame times
    i * 30 s, shifted by bias_en metres (east, north), with a little noise."""
    from pyproj import Transformer
    to_ll = Transformer.from_crs("EPSG:5382", "EPSG:4326", always_xy=True)
    rows = []
    for k, p in enumerate(positions):
        lon, lat = to_ll.transform(synth.ORIGIN_EN[0] + p[0] + bias_en[0] + rng.normal() * 0.3, synth.ORIGIN_EN[1] - p[2] + bias_en[1] + rng.normal() * 0.3)
        rows.append({"t": 30.0 * k, "lat": lat, "lon": lon, "alt": synth.ORIGIN_ALT + p[1], "h_acc": 3.0, "v_acc": 3.0})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_visual_registration():
    """Two walks over the same textured ground. B has its own ARKit frame (turned 6 degrees, 4 m / 5.5 m off A's) and its GPS
    is 6 m out; ICP finds nothing to register on (as on bare ground: here switched off), so B must be placed by photo
    features alone (visual.py, via merge.py)."""
    import test_visual as tv
    scene = tv.Scene(3)
    rng = np.random.default_rng(0)
    # both walk forward over the same stretch of ground (looking towards -z), B 0.4 m aside and aimed a little differently
    cams_a = [tv.lookat(np.array([-2.4 + 0.7 * k, 1.7 + 0.05 * k, 1.0 - 0.1 * k]), [0.3, 0.0, -3.0 + 0.1 * k]) for k in range(8)]
    true_b = [tv.lookat(np.array([-2.0 + 0.7 * k, 1.8, 0.7 - 0.1 * k]), [0.0, 0.0, -3.2]) for k in range(8)]
    H = np.eye(4)  # B's own world -> A's: p_A = H p_B
    H[:3, :3], H[:3, 3] = optimize.rot_y(np.radians(6.0)), [4.0, 0.0, 5.5]
    cams_b = [np.linalg.inv(H) @ T for T in true_b]  # what B recorded, in its own frame

    quiet = lambda *_: None  # noqa: E731
    real_register = merge.register_walk
    merge.register_walk = lambda *a, **k: (None, {"why": "test: ICP switched off"})
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a, b = tmp / "a", tmp / "b"
            tv.write_walk(a, scene, cams_a, seed=1)
            tv.write_walk(b, scene, true_b, seed=2)  # rendered from where B really was ...
            lines = [json.loads(l) for l in (b / "frames.jsonl").read_text().splitlines()]
            for line, T in zip(lines, cams_b):  # ... but it recorded the poses in its own ARKit frame
                line["transform"] = T.flatten(order="F").tolist()
            (b / "frames.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
            for w, T in ((a, cams_a), (b, true_b)):  # the GPS sees the true positions (in A's frame); B's is 6 m east of the truth
                gps_lines(w / "gps.jsonl", [t[:3, 3] for t in T], (0.0, 0.0) if w is a else (6.0, 0.0), rng)
            stats = merge.merge(tmp / "m", [a, b], 8, log=quiet, visual=True)
            w = stats["walks"][1]
            v = w.get("visual", {})
            yaw, shift = w["yaw_deg"], np.array(w["translation_m"])
            c = np.mean([T[:3, 3] for T in cams_b], axis=0)  # compare where B's own centre lands
            err = np.linalg.norm((optimize.rot_y(np.radians(yaw)) @ c + shift) - (H[:3, :3] @ c + H[:3, 3]))
            gps_err = np.linalg.norm(np.array(w["gps_translation_m"]) - H[:3, 3])
            print(f"visual: method {w['method']}; yaw {yaw:+.2f} deg (true +6.00); B's centre lands {err * 100:.1f} cm from the truth "
                  f"(its GPS placement was {gps_err:.1f} m off); {v.get('agreeing')} of {v.get('constraints')} constraints agree")
            if w["method"] != "visual":
                print("visual attempt:", {k: w.get(k) for k in ("initial_guess", "why", "against", "constraints", "agreeing", "spread_m")})
            assert w["method"] == "visual" and v["agreeing"] >= merge.VISUAL_REGISTER_MIN
            assert abs(yaw - 6.0) < 0.5 and err < 0.15 and gps_err > 5.0
    finally:
        merge.register_walk = real_register


def test_jobs():
    """--jobs only changes how long things take: fusing with worker threads and a mesh process, and merging with the
    walks optimised in parallel worker processes, must give exactly the serial result."""
    quiet = lambda *_: None  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a, b = tmp / "a", tmp / "b"
        synth.generate(a, 240, seed=0)
        synth.generate(b, 240, seed=1, phase=2.0, yaw_deg=4.0)

        for jobs in (1, 3):
            fuse.fuse(a, tmp / f"f{jobs}", jobs=jobs)
            merge.merge(tmp / f"m{jobs}", [a, b], 40, log=quiet, jobs=jobs)
        # fuse: exactly the same. merge: Open3D's multi-threaded ICP sums in a different order with a different thread
        # count (and from run to run), so the poses agree to a fraction of a millimetre rather than bit for bit
        synth_voxel = 0.02  # the fused cloud's voxel: no point may move more than that
        for name in ("f", "m"):
            p1, p3 = (o3d.io.read_point_cloud(str(tmp / f"{name}{j}" / "points.ply")) for j in (1, 3))
            m1, m3 = (o3d.io.read_triangle_mesh(str(tmp / f"{name}{j}" / "mesh.ply")) for j in (1, 3))
            n1, n3 = len(p1.points), len(p3.points)
            if n1 == n3:
                diff = np.abs(np.asarray(p1.points) - np.asarray(p3.points)).max(1)
                dp, off = float(diff.max()), float((diff > 1e-3).mean())  # a point near a voxel edge can land in the next voxel
                dc = float((np.abs(np.asarray(p1.colors) - np.asarray(p3.colors)).max(1) > 1e-3).mean())  # a near-tie flips a keyframe
            else:
                dp = off = dc = float("inf")
            print(f"jobs:   {'fuse' if name == 'f' else 'merge'} jobs=1 vs jobs=3: {n1} / {n3} points, {len(m1.triangles)} / {len(m3.triangles)} triangles, "
                  f"largest point difference {dp:.2e} m ({off:.4%} of points more than 1 mm), share of points with another colour {dc:.4%}")
            assert n1 == n3 and len(m1.triangles) == len(m3.triangles)
            assert dp <= (0 if name == "f" else synth_voxel) and off <= (0 if name == "f" else 1e-3) and dc <= (0 if name == "f" else 1e-3)
        for k in (0, 1):
            t1, t3 = (np.load(tmp / f"m{j}" / f"walk{k}" / "poses_merged.npz")["T"] for j in (1, 3))
            d = float(np.abs(t1 - t3).max())
            print(f"jobs:   walk {k} merged poses differ by at most {d:.2e}")
            assert d < 1e-3


GHOST = (0.2, -2.6, (0.3, 0.3, 0.3), 0.3)  # a box (a bag, a dog): cx, cz, half extents, yaw; only there during the first pass
# (low, so that the ground behind it is within the 5 m sensor range from where it is seen: free space can only be
# confirmed where the camera sees something farther away)
POLE = (-1.2, -2.2, 0.015, 2.0)  # cx, cz, radius, height: a 3 cm pole, there all the time


def ghost_sdf(p):
    return synth._box_sdf(p, *GHOST)


def test_consensus():
    """Three passes over the same scene. In the first only, a box (a bag left and removed) stands in the scene, and in every
    frame extra flying pixels appear. The consensus filter must remove most of that, keep the real scene (including a
    3 cm pole seen from many sides) and not make the reconstruction worse. Three passes because with two, one saying
    "there" and one saying "not there" is a tie nothing can settle."""
    orig = (synth.simulate_depth, synth.scene_sdf, list(synth.CYLINDERS), list(synth.ALBEDO))
    try:
        synth.CYLINDERS.append(POLE)
        synth.ALBEDO.append((200, 200, 200))
        with_pole = synth.scene_sdf
        calls = [0]

        def simulate_depth(rng, R, pos):
            first_pass = calls[0] < 220  # a loop is 217 frames
            calls[0] += 1
            synth.scene_sdf = (lambda p: np.minimum(with_pole(p), ghost_sdf(p))) if first_pass else with_pole
            return orig[0](rng, R, pos)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            walk = tmp / "walk"
            synth.simulate_depth = simulate_depth
            synth.generate(walk, 660)
            synth.simulate_depth, synth.scene_sdf = orig[0], with_pole

            rng = np.random.default_rng(5)  # extra flying pixels, confidence 1, in every frame
            for i in range(660):
                depth, conf = fuse.read_depth(walk, json.loads((walk / "meta.json").read_text()), i)
                fly = rng.random(depth.shape) < 0.003
                depth[fly], conf[fly] = rng.uniform(0.5, 4.0, fly.sum()), 1
                depth.astype("<f2").tofile(walk / "depth" / f"{i:06d}.bin")
                conf.tofile(walk / "conf" / f"{i:06d}.bin")

            res = {}
            for name, on in (("off", False), ("on", True)):
                fuse.fuse(walk, tmp / name, consensus=on)
                p = np.asarray(o3d.io.read_point_cloud(str(tmp / name / "points.ply")).points)
                scene, ghost = np.abs(synth.scene_sdf(p)), ghost_sdf(p)
                res[name] = {"all": scene, "true": (scene < 0.03).sum(), "ghost": ((ghost < 0.04) & (scene > 0.08)).sum(),
                             "pole": (np.abs(synth._cyl_sdf(p, *POLE)) < 0.02).sum(), "floaters": ((scene > 0.1) & (ghost > 0.1)).sum()}
            off, on = res["off"], res["on"]
            print(f"consensus: ghost box points {off['ghost']} -> {on['ghost']}, floating points {off['floaters']} -> {on['floaters']}, "
                  f"true-surface points {off['true']} -> {on['true']} ({on['true'] / off['true']:.1%} kept), pole points {off['pole']} -> {on['pole']}")
            print(f"consensus: |dist to scene| median/p90 {pct(off['all'], 50) * 100:.2f}/{pct(off['all'], 90) * 100:.2f} cm -> "
                  f"{pct(on['all'], 50) * 100:.2f}/{pct(on['all'], 90) * 100:.2f} cm")
            assert off["ghost"] > 500 and off["pole"] > 100 and off["floaters"] > 1000  # the test is meaningful
            assert on["ghost"] <= 0.1 * off["ghost"] and on["floaters"] <= 0.15 * off["floaters"]  # floaters: some only have judges within one pass
            assert on["true"] >= 0.98 * off["true"] and on["pole"] >= 0.9 * off["pole"]
            assert pct(on["all"], 50) <= pct(off["all"], 50) and pct(on["all"], 90) <= pct(off["all"], 90)
    finally:
        synth.simulate_depth, synth.scene_sdf = orig[0], orig[1]
        synth.CYLINDERS[:], synth.ALBEDO[:] = orig[2], orig[3]


if __name__ == "__main__":
    for test in sys.argv[1:] or ["test_pipeline", "test_drift", "test_snap", "test_merge", "test_slide", "test_visual_registration", "test_jobs", "test_consensus"]:  # optionally just some of them
        globals()[test]()
    print("OK")
