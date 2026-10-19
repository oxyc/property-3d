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
from pathlib import Path

import numpy as np
import open3d as o3d
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

CRS = "EPSG:5382"


def fit_translation(walk, poses=None):
    walk = Path(walk)
    frames = [json.loads(l) for l in (walk / "frames.jsonl").read_text().splitlines() if l]
    ft = np.array([f["t"] for f in frames])
    xyz = np.array([f["transform"][12:15] for f in frames])  # column-major: last column = translation
    if poses is not None:  # optimize.py output, same frame order: the fused points use these positions
        xyz = np.load(poses)["T"][:, :3, 3]
    gps = [json.loads(l) for l in (walk / "gps.jsonl").read_text().splitlines() if l]
    gps = [g for g in gps if ft[0] <= g["t"] <= ft[-1]]

    to_en = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    e, n = to_en.transform([g["lon"] for g in gps], [g["lat"] for g in gps])
    gt = np.array([g["t"] for g in gps])
    arkit = np.stack([np.interp(gt, ft, xyz[:, k]) for k in range(3)], 1)

    w = 1 / np.array([g["h_acc"] for g in gps]) ** 2
    dE, dN = e - arkit[:, 0], n - (-arkit[:, 2])
    E0, N0 = np.average(dE, weights=w), np.average(dN, weights=w)
    wv = 1 / np.array([g["v_acc"] for g in gps]) ** 2
    dz = np.average(np.array([g["alt"] for g in gps]) - arkit[:, 1], weights=wv)
    res = np.hypot(dE - E0, dN - N0)
    return E0, N0, dz, len(gps), float(np.sqrt(np.average(res**2, weights=w)))


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
