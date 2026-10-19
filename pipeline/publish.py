"""Add a processed walk to the static site (GitHub Pages) as a web-sized point cloud.

usage: publish.py <out_dir> <name> <title> [--max-points 3000000] [--max-faces 500000] [--site ../site]

Only points.ply and mesh.ply (local ARKit frame, metres from the walk start) are published;
nothing georeferenced (georef*.ply / ground.tif / georef.json) leaves the machine. Points are
voxel-thinned to at most --max-points and written as float32 + uint8 RGB binary PLY
(15 bytes/point, so 3M points ≈ 45 MB, under GitHub's 100 MB file limit).

If out_dir/mesh.ply exists it is also published as a coloured surface: tiny disconnected
clusters are dropped, the mesh is quadric-decimated to --max-faces, and every vertex takes
the colour of the nearby published points. Binary PLY: float32 xyz + uint8 rgb, faces as
uint8 count + int32 indices (~13 bytes/face + ~15 bytes/vertex).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
from pyproj import Transformer
from scipy.spatial import cKDTree

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


def colour_from_points(verts, pts, cols, radius=0.05, k=4):
    """Mean colour of the up-to-k points within `radius` of each vertex, else the nearest point's."""
    tree = cKDTree(pts)
    d, i = tree.query(verts, k=k, distance_upper_bound=radius)
    ok = np.isfinite(d)
    padded = np.vstack([cols, np.zeros((1, 3))])  # index len(pts) marks "no neighbour"
    out = (padded[i] * ok[..., None]).sum(1) / np.maximum(ok.sum(1), 1)[:, None]
    miss = ~ok[:, 0]
    out[miss] = cols[tree.query(verts[miss])[1]]
    return out


def write_mesh_ply(path, verts, cols, faces):
    vrec = np.empty(len(verts), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                 ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vrec["x"], vrec["y"], vrec["z"] = verts.T.astype(np.float32)
    vrec["red"], vrec["green"], vrec["blue"] = (np.clip(cols, 0, 1) * 255).round().astype(np.uint8).T
    frec = np.empty(len(faces), [("n", "u1"), ("v", "<i4", 3)])
    frec["n"], frec["v"] = 3, faces
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(verts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(vrec.tobytes())
        f.write(frec.tobytes())


def publish_mesh(src, dest, pcd, max_faces, min_cluster):
    """Clean, decimate and colour src; write it to dest. Returns the face count."""
    mesh = o3d.io.read_triangle_mesh(str(src))
    cluster, sizes, _ = mesh.cluster_connected_triangles()
    cluster, sizes = np.asarray(cluster), np.asarray(sizes)
    mesh.remove_triangles_by_mask(sizes[cluster] < min_cluster)
    mesh.remove_unreferenced_vertices()
    if max_faces and len(mesh.triangles) > max_faces:
        mesh = mesh.simplify_quadric_decimation(max_faces)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    verts = np.asarray(mesh.vertices)
    cols = colour_from_points(verts, np.asarray(pcd.points), np.asarray(pcd.colors))
    write_mesh_ply(dest, verts, cols, np.asarray(mesh.triangles))
    return len(mesh.triangles)


ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("name")
ap.add_argument("title")
ap.add_argument("--max-points", type=int, default=3_000_000)
ap.add_argument("--max-faces", type=int, default=500_000)
ap.add_argument("--min-cluster", type=int, default=500, help="drop mesh clusters with fewer triangles")
ap.add_argument("--max-full-points", type=int, default=5_900_000, help="points_full.ply cap (15 B/point, GitHub's limit is 100 MB)")
ap.add_argument("--max-full-faces", type=int, default=4_000_000, help="mesh_full.ply is decimated only above this")
ap.add_argument("--no-full", action="store_true", help="skip the full-resolution sources")
ap.add_argument("--utc", type=float, help="the site's UTC offset in hours for the viewer's sun (e.g. -3 for Uruguay)")
ap.add_argument("--site", default=str(HERE.parent / "site"))
a = ap.parse_args()
FULL_LIMIT = 90_000_000  # bytes per file, GitHub rejects 100 MB

site = Path(a.site)
dest = site / "models" / a.name
dest.mkdir(parents=True, exist_ok=True)

raw = o3d.io.read_point_cloud(str(Path(a.out_dir) / "points.ply"))
pcd = thin(raw, a.max_points)
write_ply(dest / "points.ply", np.asarray(pcd.points), np.asarray(pcd.colors))
size = (dest / "points.ply").stat().st_size
entry = {"name": a.name, "title": a.title, "points": len(pcd.points), "bytes": size}

# The viewer's sun needs a latitude/longitude. PRIVACY: round to 0.5 degrees (tens of km). That is plenty for the
# sun's path but useless for locating the property; never publish anything more precise.
georef = Path(a.out_dir) / "georef.json"
if georef.exists():
    g = json.loads(georef.read_text())
    lon, lat = Transformer.from_crs(g["crs"], "EPSG:4326", always_xy=True).transform(g["E0"], g["N0"])
    entry["sun_location"] = {"lat": round(lat * 2) / 2, "lon": round(lon * 2) / 2}
    if a.utc is not None:
        entry["sun_location"]["utc"] = a.utc

mesh_src = Path(a.out_dir) / "mesh.ply"
for f in ("mesh.ply", "points_full.ply", "mesh_full.ply"):
    (dest / f).unlink(missing_ok=True)
if mesh_src.exists():
    faces = publish_mesh(mesh_src, dest / "mesh.ply", pcd, a.max_faces, a.min_cluster)
    entry |= {"mesh": True, "mesh_bytes": (dest / "mesh.ply").stat().st_size}
    print(f"mesh: {faces:,} faces, {entry['mesh_bytes'] / 1e6:.1f} MB")

# Full-resolution sources for fine exports; the viewer fetches them only when asked for more detail than the defaults.
if not a.no_full:
    full = thin(raw, a.max_full_points)
    if len(full.points) > len(pcd.points):
        write_ply(dest / "points_full.ply", np.asarray(full.points), np.asarray(full.colors))
        n = (dest / "points_full.ply").stat().st_size
        if n > FULL_LIMIT:
            (dest / "points_full.ply").unlink()
            print(f"WARNING: points_full.ply would be {n / 1e6:.0f} MB, skipped")
        else:
            entry |= {"points_full": True, "points_full_bytes": n, "points_full_count": len(full.points)}
            print(f"points_full: {len(full.points):,} points, {n / 1e6:.1f} MB")
    if mesh_src.exists():
        faces = publish_mesh(mesh_src, dest / "mesh_full.ply", full, a.max_full_faces, a.min_cluster)
        n = (dest / "mesh_full.ply").stat().st_size
        if n > FULL_LIMIT:
            (dest / "mesh_full.ply").unlink()
            print(f"WARNING: mesh_full.ply would be {n / 1e6:.0f} MB, skipped (lower --max-full-faces)")
        else:
            entry |= {"mesh_full": True, "mesh_full_bytes": n, "mesh_full_faces": faces}
            print(f"mesh_full: {faces:,} faces, {n / 1e6:.1f} MB")

# Photo-textured mesh from texture.py (unlit GLB with an embedded JPEG atlas): a full and a smaller one for phones.
for src_name, dst_name, key in (("model.glb", "photo.glb", "photo_bytes"), ("model_mobile.glb", "photo_mobile.glb", "photo_mobile_bytes")):
    (dest / dst_name).unlink(missing_ok=True)
    src = Path(a.out_dir) / "textured" / src_name
    if not src.exists():
        continue
    if src.stat().st_size > FULL_LIMIT:
        print(f"WARNING: {src_name} is {src.stat().st_size / 1e6:.0f} MB, skipped")
        continue
    shutil.copy(src, dest / dst_name)
    entry |= {"photo": True, key: src.stat().st_size}
    print(f"{dst_name}: {src.stat().st_size / 1e6:.1f} MB")

index_path = site / "models" / "index.json"
index = json.loads(index_path.read_text()) if index_path.exists() else []
index = [m for m in index if m["name"] != a.name] + [entry]
index_path.write_text(json.dumps(index, indent=1) + "\n")
shutil.copy(HERE / "viewer.html", site / "index.html")
print(f"published {a.name}: {len(pcd.points):,} points, {size / 1e6:.1f} MB -> {dest}")
