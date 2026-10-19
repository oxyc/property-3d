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


if __name__ == "__main__":
    for test in sys.argv[1:] or ["test_pipeline", "test_drift", "test_snap", "test_merge"]:  # optionally just some of them
        globals()[test]()
    print("OK")
