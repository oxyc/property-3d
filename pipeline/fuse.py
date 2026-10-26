"""Fuse a walk (FORMAT.md) into a point cloud, a TSDF mesh and a ground height grid.

usage: fuse.py <walk_dir> <out_dir> [--voxel 0.02] [--poses poses.npz] [--consensus] [--jobs 3] [--stride 1]

Outputs in out_dir (all in the ARKit world frame: X east, Y up, Z south, metres):
  points.ply   voxel-downsampled, outlier-filtered, coloured from the closest unoccluded RGB keyframe.
               Experimental --consensus drops points most frames saw straight through (people, flying pixels).
               Off by default: on a real garden walk it also removed real structure next to walls and edges
               (ARKit's smoothed depth makes silhouettes look see-through). The TSDF only updates a thin band
               around surfaces, so it cannot hold these free-space votes, and mesh.ply is unaffected.
  mesh.ply     TSDF mesh (depth only, 3 cm voxels)
  ground.npz   lowest-point height grid (10 cm cells): height[row, col] in Y, NaN = empty.
               Row 0 is the northernmost row. west/north = local east and north (= -Z) of the
               grid's top-left corner, cell = cell size. Same layout as a north-up GeoTIFF.
  ground.png   preview
"""
import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script dir from sys.path
import cores  # noqa: E402

if __name__ == "__main__":
    cores.setup()  # caps the threads Open3D uses; must happen before it is imported

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
from PIL import Image  # noqa: E402

MIN_DEPTH, MAX_DEPTH = 0.2, 4.5
CHUNK = 10  # frames voxelised together before being folded into the cloud, bounds memory
GROUND_CELL = 0.1
TSDF_VOXEL = 0.03
CONSENSUS_FRAMES = 1500  # frames per walk that vote (every k-th), bounds the time on long walks
CONSENSUS_MIN_VOTES = 3  # a point needs this many "seen through" votes before it can be dropped
CONSENSUS_FRACTION = 0.4  # ... and be this share of all votes (a shade under a majority: passes see a spot from unequal numbers of frames)
CONSENSUS_MIN_SPAN = 30.0  # ... and the "seen through" votes must span this many seconds: independent passes, not one stretch
CONSENSUS_MARGIN_VOXELS, CONSENSUS_MARGIN_RANGE = 2, 0.015  # depth margin: max(2 voxels, 1.5% of range)


def read_walk(walk):
    meta = json.loads((walk / "meta.json").read_text())
    frames = [json.loads(line) for line in (walk / "frames.jsonl").read_text().splitlines() if line]
    # an unfinished walk (phone out of space, app killed) can list frames whose files never got written
    n = len(frames)
    px = meta["depth_w"] * meta["depth_h"]
    sizes = {"depth": px * (2 if meta.get("depth_dtype") == "f16" else 4), "conf": px}  # the last file may be cut short

    def complete(f, sub):
        p = walk / sub / f"{f['i']:06d}.bin"
        return p.exists() and p.stat().st_size == sizes[sub]

    frames = [f for f in frames if complete(f, "depth") and complete(f, "conf")]
    if len(frames) < n:
        print(f"{walk.name}: skipped {n - len(frames)} frames with missing or incomplete depth files")
    for f in frames:
        if f.get("rgb") and not (walk / "rgb" / f"{f['i']:06d}.jpg").exists():
            f["rgb"] = False
        # column-major 16 floats: reshape in Fortran order to get the usual 4x4 camera-to-world matrix
        f["T"] = np.array(f["transform"]).reshape(4, 4, order="F")
    return meta, frames


def read_depth(walk, meta, i):
    shape = (meta["depth_h"], meta["depth_w"])
    dtype = "<f2" if meta.get("depth_dtype") == "f16" else "<f4"  # format v2 writes float16
    depth = np.fromfile(walk / "depth" / f"{i:06d}.bin", dtype).reshape(shape).astype(np.float32)
    conf = np.fromfile(walk / "conf" / f"{i:06d}.bin", np.uint8).reshape(shape)
    return depth, conf


def unproject(depth, conf, f, meta):
    """Valid depth pixels -> world points. Camera space is ARKit's: +X right, +Y up, -Z forward."""
    fx, fy, cx, cy = f["intrinsics"]  # in RGB pixels
    ok = (conf >= 1) & (depth >= MIN_DEPTH) & (depth <= MAX_DEPTH)
    v, u = np.nonzero(ok)
    d = depth[ok].astype(np.float64)
    u_rgb, v_rgb = u * meta["rgb_w"] / meta["depth_w"], v * meta["rgb_h"] / meta["depth_h"]
    cam = np.stack([(u_rgb - cx) / fx * d, -(v_rgb - cy) / fy * d, -d], 1)
    T = f["T"]
    return cam @ T[:3, :3].T + T[:3, 3]


def project_into_keyframe(pts, kf, kf_img, kf_depth, kf_conf, meta):
    """Project world points into an RGB keyframe. Returns (ok, z, colours): ok is False for
    points outside its view or occluded there (keyframe depth disagrees by > 10 cm)."""
    fx, fy, cx, cy = kf["intrinsics"]
    T = kf["T"]
    cam = (pts - T[:3, 3]) @ T[:3, :3]  # world -> camera: R^T (p - t)
    z = -cam[:, 2]
    zs = np.maximum(z, 1e-6)
    u, v = fx * cam[:, 0] / zs + cx, cy - fy * cam[:, 1] / zs
    h, w = kf_img.shape[:2]
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (z > 0.1) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ud = np.clip(np.round(u * meta["depth_w"] / w).astype(int), 0, meta["depth_w"] - 1)
    vd = np.clip(np.round(v * meta["depth_h"] / h).astype(int), 0, meta["depth_h"] - 1)
    ok &= (kf_conf[vd, ud] >= 1) & (np.abs(kf_depth[vd, ud] - z) < 0.1)
    col = np.zeros((len(pts), 3))
    col[ok] = kf_img[vi[ok], ui[ok]] / 255.0
    return ok, z, col


CELL = 3.0  # m: coarse grid for finding the points around a camera
CELL_OFF = 1 << 20


def cell_keys(cells):
    """Integer cell coordinates (n, 3) -> one int64 key each (21 bits per axis)."""
    c = cells.astype(np.int64) + CELL_OFF
    return (c[..., 0] << 42) | (c[..., 1] << 21) | c[..., 2]


def grid_index(pts):
    """(order, sorted keys): the points sorted by their CELL-sized grid cell, to look up the ones near a camera."""
    key = cell_keys(np.floor(pts / CELL))
    order = np.argsort(key, kind="stable")
    return order, key[order]


def points_near(index, pts, centre, radius):
    """Indices of the points within radius of centre (what a KD-tree radius search returns, without building a tree)."""
    order, skeys = index
    lo, hi = np.floor((centre - radius) / CELL).astype(int), np.floor((centre + radius) / CELL).astype(int)
    axes = np.meshgrid(*[np.arange(lo[d], hi[d] + 1) for d in range(3)], indexing="ij")
    keys = cell_keys(np.stack(axes, -1).reshape(-1, 3))
    first, last = np.searchsorted(skeys, keys, "left"), np.searchsorted(skeys, keys, "right")
    idx = np.concatenate([order[a:b] for a, b in zip(first, last) if b > a] or [np.empty(0, np.int64)])
    return idx[((pts[idx] - centre) ** 2).sum(1) <= radius * radius]


def colour_points(pcd, sources, jobs=1):
    """Colour each final point from the keyframe that saw it closest and unoccluded.
    Done after downsampling so grey fallbacks never get averaged into real colours.
    sources: [(walk, meta, keyframes)], one per walk. The keyframes are decoded and projected in `jobs` threads
    (JPEG decoding and numpy release the GIL); their results are applied in keyframe order, so the colours do not
    depend on jobs."""
    pts = np.asarray(pcd.points)
    cols = np.full((len(pts), 3), 0.5)
    best = np.full(len(pts), np.inf)
    index = grid_index(pts)

    def project(job):
        walk, meta, kf = job
        idx = points_near(index, pts, kf["T"][:3, 3], MAX_DEPTH)
        if len(idx) == 0:
            return None
        kd, kc = read_depth(walk, meta, kf["i"])
        try:
            img = np.asarray(Image.open(walk / "rgb" / f"{kf['i']:06d}.jpg").convert("RGB"))
        except OSError:  # the last image of an unfinished walk can be cut short
            return None
        ok, z, col = project_into_keyframe(pts[idx], kf, img, kd, kc, meta)
        return idx[ok], z[ok], col[ok]

    todo = [(walk, meta, kf) for walk, meta, kfs in sources for kf in kfs]
    with ThreadPoolExecutor(max(1, jobs)) as pool:
        for k in range(0, len(todo), 4 * max(1, jobs)):  # a few keyframes ahead, bounded memory
            for res in pool.map(project, todo[k:k + 4 * max(1, jobs)]):
                if res is not None:
                    idx, z, col = res
                    better = z < best[idx]
                    best[idx[better]] = z[better]
                    cols[idx[better]] = col[better]
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return np.isfinite(best).mean()


def frame_votes(pts, f, depth, conf, meta):
    """Per point: (free, support) for one frame. Free = the frame saw through the point (a surface well behind
    it), support = the frame saw a surface at it. Neither if the point is outside the view, occluded, or the
    depth around it is missing. The 3x3 pixel neighbourhood keeps thin things alive: a pole counts as seen if
    any neighbouring pixel hits it, and as seen through only if every neighbour is valid and farther away."""
    fx, fy, cx, cy = f["intrinsics"]
    T = f["T"]
    cam = (pts - T[:3, 3]) @ T[:3, :3]  # world -> camera: R^T (p - t)
    z = -cam[:, 2]
    zs = np.maximum(z, 1e-6)
    dw, dh = meta["depth_w"], meta["depth_h"]
    u = np.round((fx * cam[:, 0] / zs + cx) * dw / meta["rgb_w"]).astype(int)
    v = np.round((cy - fy * cam[:, 1] / zs) * dh / meta["rgb_h"]).astype(int)
    inside = (z > MIN_DEPTH) & (u >= 1) & (u < dw - 1) & (v >= 1) & (v < dh - 1)
    free, support = np.zeros(len(pts), bool), np.zeros(len(pts), bool)
    u, v, z = u[inside], v[inside], z[inside]
    D = np.where((conf >= 1) & (depth >= MIN_DEPTH), depth, np.nan)
    nb = np.stack([D[v + dy, u + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    margin = np.maximum(CONSENSUS_MARGIN_VOXELS * TSDF_VOXEL, CONSENSUS_MARGIN_RANGE * z)
    free[inside] = ~np.isnan(nb).any(0) & (z < nb.min(0) - margin)
    support[inside] = np.nan_to_num(np.abs(nb - z), nan=np.inf).min(0) <= margin
    return free, support


def consensus_filter(acc, sources):
    """Drop points that a large share of the frames able to judge them saw straight through ("2 of 3 passes say it wasn't
    there"): transient things (people, swaying leaves) and flying pixels. Two guards, because ARKit's smoothed depth
    maps are unreliable at thin things and silhouettes: the "seen through" votes must come from at least
    CONSENSUS_MIN_SPAN seconds apart (a revisit, not neighbouring frames of one pass), and points too rarely observed
    to judge are kept: no evidence is not evidence of absence. sources: [(walk, meta, tracking-normal frames)]."""
    pts = np.asarray(acc.points)
    free, support = np.zeros(len(pts), np.int32), np.zeros(len(pts), np.int32)
    first, last = np.full(len(pts), np.inf), np.full(len(pts), -np.inf)  # times of the first / last "seen through" vote
    tree = o3d.geometry.KDTreeFlann(acc)
    for k, (walk, meta, frames) in enumerate(sources):
        for f in frames[::max(1, -(-len(frames) // CONSENSUS_FRAMES))]:
            _, idx, _ = tree.search_radius_vector_3d(f["T"][:3, 3], MAX_DEPTH + 0.5)
            idx = np.asarray(idx)
            if len(idx) == 0:
                continue
            depth, conf = read_depth(walk, meta, f["i"])
            fr, su = frame_votes(pts[idx], f, depth, conf, meta)
            free[idx] += fr
            support[idx] += su
            seen_through, t = idx[fr], f["t"] + k * 1e6  # separate walks are always far apart in time
            first[seen_through] = np.minimum(first[seen_through], t)
            last[seen_through] = np.maximum(last[seen_through], t)
    leaning = (free >= CONSENSUS_MIN_VOTES) & (free >= CONSENSUS_FRACTION * (free + support))
    drop = leaning & (last - first >= CONSENSUS_MIN_SPAN)
    stats = {"points": len(pts), "removed": int(drop.sum()), "removed_never_supported": int((drop & (support == 0)).sum()),
             "removed_outvoted": int((drop & (support > 0)).sum()),
             "kept_one_pass_only": int((leaning & ~drop).sum()),
             "kept_unjudged": int(((free + support < CONSENSUS_MIN_VOTES) & ~drop).sum())}
    return acc.select_by_index(np.nonzero(~drop)[0]), stats


def to_cloud(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def tsdf_extrinsic(T):
    """World -> camera for Open3D, whose camera is x right, y down, z forward (flip ARKit's Y and Z)."""
    return np.linalg.inv(T @ np.diag([1, -1, -1, 1]))


def ground_grid(pts):
    """Lowest point per 10 cm cell, north-up layout."""
    east, north, up = pts[:, 0], -pts[:, 2], pts[:, 1]
    west, top = np.floor(east.min() / GROUND_CELL) * GROUND_CELL, np.ceil(north.max() / GROUND_CELL) * GROUND_CELL
    col = ((east - west) / GROUND_CELL).astype(int)
    row = ((top - north) / GROUND_CELL).astype(int)
    shape = (row.max() + 1, col.max() + 1)
    height = np.full(shape, np.inf)
    np.minimum.at(height, (row, col), up)
    height[np.isinf(height)] = np.nan
    return height.astype(np.float32), west, top


def save_preview(height, path):
    img = np.zeros(height.shape, np.uint8)
    ok = ~np.isnan(height)
    lo, hi = height[ok].min(), height[ok].max()
    img[ok] = 40 + 215 * (height[ok] - lo) / max(hi - lo, 1e-6)
    Image.fromarray(img).resize((img.shape[1] * 4, img.shape[0] * 4), Image.NEAREST).save(path)


def fuse(walk, out, voxel=0.02, poses=None, consensus=False, jobs=1, stride=1):
    """poses: optional poses.npz from optimize.py; its transforms replace the recorded ones."""
    return fuse_many([(walk, poses)], out, voxel, consensus, jobs, stride)


def load_frames(walk, poses):
    """(meta, frames) of a walk; with poses (a poses.npz from optimize.py) only the frames it lists, at its poses
    (the frames it leaves out were dropped as bad tracking)."""
    walk = Path(walk)
    meta, frames = read_walk(walk)
    if poses is not None:
        z = np.load(poses)
        corrected = dict(zip(z["i"].tolist(), z["T"]))
        frames = [f for f in frames if f["i"] in corrected]
        for f in frames:
            f["T"] = corrected[f["i"]]
    return meta, frames


def build_mesh(walks, out, stride=1):
    """TSDF mesh (depth only) of [(walk_dir, poses.npz or None)] -> out/mesh.ply, from every stride-th frame. Returns
    (triangles, seconds). Reads the walks from disk itself, so it can run in a separate process."""
    t0 = time.perf_counter()
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        TSDF_VOXEL, 4 * TSDF_VOXEL, o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    for walk, poses in walks:
        walk = Path(walk)
        meta, frames = load_frames(walk, poses)
        dw, dh = meta["depth_w"], meta["depth_h"]
        sx, sy = dw / meta["rgb_w"], dh / meta["rgb_h"]
        dummy_rgb = o3d.geometry.Image(np.zeros((dh, dw, 3), np.uint8))
        for f in frames[::stride]:
            if f["tracking"] != "normal":
                continue
            depth, conf = read_depth(walk, meta, f["i"])
            fx, fy, cx, cy = f["intrinsics"]
            masked = np.where((conf >= 1) & (depth >= MIN_DEPTH), depth, 0).astype(np.float32)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                dummy_rgb, o3d.geometry.Image(masked), depth_scale=1.0, depth_trunc=MAX_DEPTH,
                convert_rgb_to_intensity=False)
            intr = o3d.camera.PinholeCameraIntrinsic(dw, dh, fx * sx, fy * sy, cx * sx, cy * sy)
            tsdf.integrate(rgbd, intr, tsdf_extrinsic(f["T"]))
    mesh = tsdf.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(Path(out) / "mesh.ply"), mesh)
    return len(mesh.triangles), time.perf_counter() - t0


def voxel_sums(pts, voxel):
    """(keys, coordinate sums, counts) per occupied voxel of a fixed global grid, keys sorted."""
    return reduce_cells(cell_keys(np.floor(pts / voxel)), pts, np.ones(len(pts)))


def reduce_cells(keys, sums, counts):
    """Add up entries with the same voxel key. Centroids come out exact (sum / count) whatever the chunking, so
    the fused cloud does not depend on chunk sizes or on how the work was split."""
    uk, inv = np.unique(keys, return_inverse=True)
    inv = inv.ravel()
    s = np.stack([np.bincount(inv, weights=sums[:, d], minlength=len(uk)) for d in range(3)], 1)
    return uk, s, np.bincount(inv, weights=counts, minlength=len(uk))


def fuse_many(walks, out, voxel=0.02, consensus=False, jobs=1, stride=1):
    """walks: [(walk_dir, poses.npz or None)], all in one common world frame through their poses.
    Frame indices only need to be unique within a walk. consensus: drop points the other frames saw through
    (see consensus_filter). jobs > 1: the mesh is built in its own process while the points are fused, and the
    colouring uses that many threads. stride: fuse (points and mesh) only every stride-th frame; colours still come
    from all keyframes. Returns the consensus stats ({} if off)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    specs = [(str(w), None if p is None else str(p)) for w, p in walks]
    mesh_job = None
    if jobs > 1:  # the TSDF mesh needs nothing from the points: build it in another process, meanwhile
        pool = multiprocessing.get_context("spawn").Pool(1)
        mesh_job = pool.apply_async(build_mesh, (specs, str(out), stride))
    sources, voters, parts = [], [], []
    spent = {}  # seconds per stage, printed at the end

    def lap(stage, since):
        spent[stage] = spent.get(stage, 0.0) + time.perf_counter() - since
        return time.perf_counter()

    for walk, poses in walks:
        walk = Path(walk)
        meta, frames = load_frames(walk, poses)
        sources.append((walk, meta, [f for f in frames if f["rgb"]]))
        voters.append((walk, meta, [f for f in frames if f["tracking"] == "normal"]))
        pend = []
        used = frames[::stride]
        for n, f in enumerate(used):
            if f["tracking"] != "normal":
                continue
            t = time.perf_counter()
            depth, conf = read_depth(walk, meta, f["i"])
            t = lap("read depth", t)
            pend.append(unproject(depth, conf, f, meta))
            t = lap("unproject", t)
            if len(pend) == CHUNK or n == len(used) - 1:
                parts.append(voxel_sums(np.vstack(pend), voxel))
                pend = []
                if sum(len(p[0]) for p in parts) >= max(2_000_000, 1.5 * len(parts[0][0])):  # fold geometrically: linear in all
                    parts = [reduce_cells(*[np.concatenate(x) for x in zip(*parts)])]
                lap("merge into cloud (voxel grid)", t)
        if pend:  # tracking-skipped tail
            parts.append(voxel_sums(np.vstack(pend), voxel))
    t = time.perf_counter()
    keys, sums, counts = reduce_cells(*[np.concatenate(x) for x in zip(*parts)])
    acc = to_cloud(sums / counts[:, None])
    del parts, keys, sums, counts
    t = lap("merge into cloud (voxel grid)", t)
    stats = {}
    if consensus:
        acc, stats = consensus_filter(acc, voters)
        t = lap("consensus", t)
        print(f"consensus: {stats['removed']} of {stats['points']} points removed ({stats['removed'] / stats['points']:.1%}): "
              f"{stats['removed_never_supported']} seen through and never seen there, {stats['removed_outvoted']} outvoted; "
              f"{stats['kept_one_pass_only']} kept although leaning that way (seen through within one pass only), "
              f"{stats['kept_unjudged']} kept unjudged (too few observations)")
    n_before = len(acc.points)
    t = time.perf_counter()
    acc, _ = acc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    t = lap("statistical outlier removal", t)
    coloured = colour_points(acc, sources, jobs)
    t = lap("colour points", t)
    o3d.io.write_point_cloud(str(out / "points.ply"), acc)
    t = lap("write points.ply", t)

    height, west, north = ground_grid(np.asarray(acc.points))
    np.savez(out / "ground.npz", height=height, west=west, north=north, cell=GROUND_CELL)
    save_preview(height, out / "ground.png")
    t = lap("ground grid", t)
    if mesh_job is not None:
        triangles, mesh_seconds = mesh_job.get()
        pool.close()
        lap("wait for the mesh process", t)
    else:
        triangles, mesh_seconds = build_mesh(specs, str(out), stride)
    spent["mesh (own process)" if mesh_job is not None else "mesh"] = mesh_seconds
    print(f"points: {n_before} -> {len(acc.points)} after outlier removal, {coloured:.0%} coloured; "
          f"mesh: {triangles} triangles; ground grid {height.shape[1]}x{height.shape[0]} cells")
    print("fuse stages (s): " + ", ".join(f"{k} {v:.0f}" for k, v in spent.items()))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("walk_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--poses", help="poses.npz from optimize.py")
    ap.add_argument("--consensus", action="store_true", help="experimental: drop points other frames saw straight through")
    ap.add_argument("--jobs", type=int, default=cores.DEFAULT_JOBS, help="cores to use (default %(default)s)")
    ap.add_argument("--stride", type=int, default=1, help="fuse every N-th depth frame (default %(default)s)")
    a = ap.parse_args()
    fuse(a.walk_dir, a.out_dir, a.voxel, a.poses, a.consensus, a.jobs, a.stride)
