"""Fuse a walk (FORMAT.md) into a point cloud, a TSDF mesh and a ground height grid.

usage: fuse.py <walk_dir> <out_dir> [--voxel 0.02] [--poses poses.npz]

Outputs in out_dir (all in the ARKit world frame: X east, Y up, Z south, metres):
  points.ply   voxel-downsampled, outlier-filtered, coloured from the closest unoccluded RGB keyframe
  mesh.ply     TSDF mesh (depth only, 3 cm voxels)
  ground.npz   lowest-point height grid (10 cm cells): height[row, col] in Y, NaN = empty.
               Row 0 is the northernmost row. west/north = local east and north (= -Z) of the
               grid's top-left corner, cell = cell size. Same layout as a north-up GeoTIFF.
  ground.png   preview
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

MIN_DEPTH, MAX_DEPTH = 0.2, 4.5
CHUNK = 10  # frames merged between voxel-downsamples, bounds memory
GROUND_CELL = 0.1
TSDF_VOXEL = 0.03


def read_walk(walk):
    meta = json.loads((walk / "meta.json").read_text())
    frames = [json.loads(line) for line in (walk / "frames.jsonl").read_text().splitlines() if line]
    for f in frames:
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


def colour_points(pcd, walk, meta, kfs):
    """Colour each final point from the keyframe that saw it closest and unoccluded.
    Done after downsampling so grey fallbacks never get averaged into real colours."""
    pts = np.asarray(pcd.points)
    cols = np.full((len(pts), 3), 0.5)
    best = np.full(len(pts), np.inf)
    tree = o3d.geometry.KDTreeFlann(pcd)
    for kf in kfs:
        _, idx, _ = tree.search_radius_vector_3d(kf["T"][:3, 3], MAX_DEPTH)
        idx = np.asarray(idx)
        if len(idx) == 0:
            continue
        kd, kc = read_depth(walk, meta, kf["i"])
        img = np.asarray(Image.open(walk / "rgb" / f"{kf['i']:06d}.jpg").convert("RGB"))
        ok, z, col = project_into_keyframe(pts[idx], kf, img, kd, kc, meta)
        better = ok & (z < best[idx])
        best[idx[better]] = z[better]
        cols[idx[better]] = col[better]
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return np.isfinite(best).mean()


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


def fuse(walk, out, voxel=0.02, poses=None):
    """poses: optional poses.npz from optimize.py; its transforms replace the recorded ones."""
    walk, out = Path(walk), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta, frames = read_walk(walk)
    if poses is not None:
        z = np.load(poses)
        corrected = dict(zip(z["i"].tolist(), z["T"]))
        for f in frames:
            f["T"] = corrected[f["i"]]
    kfs = [f for f in frames if f["rgb"]]

    dw, dh = meta["depth_w"], meta["depth_h"]
    sx, sy = dw / meta["rgb_w"], dh / meta["rgb_h"]
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        TSDF_VOXEL, 4 * TSDF_VOXEL, o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    dummy_rgb = o3d.geometry.Image(np.zeros((dh, dw, 3), np.uint8))

    acc = o3d.geometry.PointCloud()
    pend = []
    for n, f in enumerate(frames):
        if f["tracking"] != "normal":
            continue
        depth, conf = read_depth(walk, meta, f["i"])
        pend.append(unproject(depth, conf, f, meta))
        if len(pend) == CHUNK or n == len(frames) - 1:
            acc = (acc + to_cloud(np.vstack(pend))).voxel_down_sample(voxel)
            pend = []

        fx, fy, cx, cy = f["intrinsics"]
        masked = np.where((conf >= 1) & (depth >= MIN_DEPTH), depth, 0).astype(np.float32)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            dummy_rgb, o3d.geometry.Image(masked), depth_scale=1.0, depth_trunc=MAX_DEPTH,
            convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(dw, dh, fx * sx, fy * sy, cx * sx, cy * sy)
        tsdf.integrate(rgbd, intr, tsdf_extrinsic(f["T"]))

    if pend:  # tracking-skipped tail
        acc = (acc + to_cloud(np.vstack(pend))).voxel_down_sample(voxel)
    n_before = len(acc.points)
    acc, _ = acc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    coloured = colour_points(acc, walk, meta, kfs)
    o3d.io.write_point_cloud(str(out / "points.ply"), acc)

    mesh = tsdf.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)

    height, west, north = ground_grid(np.asarray(acc.points))
    np.savez(out / "ground.npz", height=height, west=west, north=north, cell=GROUND_CELL)
    save_preview(height, out / "ground.png")
    print(f"points: {n_before} -> {len(acc.points)} after outlier removal, {coloured:.0%} coloured; "
          f"mesh: {len(mesh.triangles)} triangles; ground grid {height.shape[1]}x{height.shape[0]} cells")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("walk_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--poses", help="poses.npz from optimize.py")
    a = ap.parse_args()
    fuse(a.walk_dir, a.out_dir, a.voxel, a.poses)
