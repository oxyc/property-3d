"""Add a processed walk to the static site (GitHub Pages) as a web-sized point cloud.

usage: publish.py <out_dir> <name> <title> [--max-points 3000000] [--site ../site]

Only points.ply (local ARKit frame, metres from the walk start) is published; nothing
georeferenced (georef*.ply / ground.tif / georef.json) leaves the machine. Points are
voxel-thinned to at most --max-points and written as float32 + uint8 RGB binary PLY
(15 bytes/point, so 3M points ≈ 45 MB, under GitHub's 100 MB file limit).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d

HERE = Path(__file__).resolve().parent


def thin(pcd, max_points):
    """Smallest voxel size (1 cm steps up) that brings the cloud under max_points."""
    voxel = 0.01
    while len(pcd.points) > max_points:
        voxel += 0.01
        pcd = pcd.voxel_down_sample(voxel)
    return pcd


def write_ply(path, pts, cols):
    rec = np.empty(len(pts), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = pts.T.astype(np.float32)
    rec["red"], rec["green"], rec["blue"] = (np.clip(cols, 0, 1) * 255).round().astype(np.uint8).T
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("name")
ap.add_argument("title")
ap.add_argument("--max-points", type=int, default=3_000_000)
ap.add_argument("--site", default=str(HERE.parent / "site"))
a = ap.parse_args()

site = Path(a.site)
dest = site / "models" / a.name
dest.mkdir(parents=True, exist_ok=True)

pcd = thin(o3d.io.read_point_cloud(str(Path(a.out_dir) / "points.ply")), a.max_points)
write_ply(dest / "points.ply", np.asarray(pcd.points), np.asarray(pcd.colors))
size = (dest / "points.ply").stat().st_size

index_path = site / "models" / "index.json"
index = json.loads(index_path.read_text()) if index_path.exists() else []
index = [m for m in index if m["name"] != a.name] + [
    {"name": a.name, "title": a.title, "points": len(pcd.points), "bytes": size}]
index_path.write_text(json.dumps(index, indent=1) + "\n")
shutil.copy(HERE / "viewer.html", site / "index.html")
print(f"published {a.name}: {len(pcd.points):,} points, {size / 1e6:.1f} MB -> {dest}")
