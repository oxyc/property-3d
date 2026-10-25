"""Rough georeferencing of a fused walk from gps.jsonl into EPSG:5382.

usage: georef.py <out_dir> <walk_dir>      (out_dir = fuse.py output)

ACCURACY: expect only ~3-5 m. Phone GPS is that good at best, errors are correlated in
time, and a single ~30 m walk averages down little of it. This is only good enough to
place the model on the aerial data; align to the ortho/DSM features for anything finer.

Model: ARKit world is already metric and gravity/heading aligned, so scale = 1 and no
rotation is fitted (a ~10 m walk with 3-5 m GPS noise cannot constrain yaw). With
east = X and north = -Z in ARKit space:
    E = E0 + X,   N = N0 - Z,   Z_up = Y + dz
(E0, N0) is the weighted mean of (GPS - ARKit position), weights 1/h_acc^2; dz likewise
from alt and v_acc (GPS altitude is worse than horizontal; treat dz as +-5 m).

Writes to out_dir:
  georef.json   translation, residuals, and `offset`: georef_points.ply stores
                coordinates minus this offset (add it back), so float32 keeps mm precision.
  georef_points.ply
  ground.tif    the ground height grid as a GeoTIFF, if ground.npz exists
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script dir from sys.path
import fuse  # noqa: E402

CRS = "EPSG:5382"


def gps_offsets(walk, poses=None):
    """Per GPS fix: (E - X, N + Z, alt - Y, h_acc, v_acc) with the ARKit position X, Y, Z interpolated to the fix."""
    walk = Path(walk)
    _, frames = fuse.read_walk(walk)  # same frame list (and order) as optimize.py's poses
    ft = np.array([f["t"] for f in frames])
    xyz = np.array([f["transform"][12:15] for f in frames])  # column-major: last column = translation
    if poses is not None:  # optimize.py output: the fused points use these positions, for the frames it lists
        z = np.load(poses)
        at = dict(zip(z["i"].tolist(), z["T"][:, :3, 3]))
        keep = [k for k, f in enumerate(frames) if f["i"] in at]
        ft, xyz = ft[keep], np.array([at[frames[k]["i"]] for k in keep])
    gps = [json.loads(l) for l in (walk / "gps.jsonl").read_text().splitlines() if l]
    gps = [g for g in gps if ft[0] <= g["t"] <= ft[-1]]

    to_en = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    e, n = to_en.transform([g["lon"] for g in gps], [g["lat"] for g in gps])
    gt = np.array([g["t"] for g in gps])
    arkit = np.stack([np.interp(gt, ft, xyz[:, k]) for k in range(3)], 1)
    return np.stack([e - arkit[:, 0], n - (-arkit[:, 2]), np.array([g["alt"] for g in gps]) - arkit[:, 1],
                     [g["h_acc"] for g in gps], [g["v_acc"] for g in gps]], 1)


def fit_translation(walk, poses=None):
    """walk/poses may be lists (one entry each, all poses in one world frame): their fixes are fitted together."""
    walks = walk if isinstance(walk, (list, tuple)) else [walk]
    poses = poses if isinstance(poses, (list, tuple)) else [poses] * len(walks)
    off = np.vstack([gps_offsets(w, p) for w, p in zip(walks, poses)])
    dE, dN, dAlt, h_acc, v_acc = off.T
    w = 1 / h_acc**2
    E0, N0 = np.average(dE, weights=w), np.average(dN, weights=w)
    dz = np.average(dAlt, weights=1 / v_acc**2)
    res = np.hypot(dE - E0, dN - N0)
    return E0, N0, dz, len(off), float(np.sqrt(np.average(res**2, weights=w)))


def georef(out, walk, poses=None):
    out = Path(out)
    E0, N0, dz, n, rms = fit_translation(walk, poses)
    offset = [float(np.floor(E0 / 100) * 100), float(np.floor(N0 / 100) * 100), 0.0]
    info = {"crs": CRS, "E0": E0, "N0": N0, "dz": dz, "offset": offset, "n_fixes": n,
            "weighted_rms_residual_m": rms, "model": "E = E0 + X, N = N0 - Z, Z_up = Y + dz; scale 1, no rotation",
            "ply_convention": "georef_points.ply coordinates + offset = (E, N, Z_up)"}
    (out / "georef.json").write_text(json.dumps(info, indent=2))

    pcd = o3d.io.read_point_cloud(str(out / "points.ply"))
    p = np.asarray(pcd.points)
    local = np.stack([E0 + p[:, 0], N0 - p[:, 2], p[:, 1] + dz], 1) - offset
    pcd.points = o3d.utility.Vector3dVector(local)
    o3d.io.write_point_cloud(str(out / "georef_points.ply"), pcd)

    if (out / "ground.npz").exists():
        g = np.load(out / "ground.npz")
        h = np.where(np.isnan(g["height"]), -32767, g["height"] + dz).astype(np.float32)
        cell = float(g["cell"])
        with rasterio.open(out / "ground.tif", "w", driver="GTiff", height=h.shape[0], width=h.shape[1], count=1,
                           dtype="float32", crs=CRS, nodata=-32767,
                           transform=from_origin(E0 + float(g["west"]), N0 + float(g["north"]), cell, cell)) as dst:
            dst.write(h, 1)
    print(f"georef from {n} fixes: E0={E0:.2f} N0={N0:.2f} dz={dz:.2f}, weighted rms residual {rms:.2f} m")
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("walk_dir")
    a = ap.parse_args()
    georef(a.out_dir, a.walk_dir)
