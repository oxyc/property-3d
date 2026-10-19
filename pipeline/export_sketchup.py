"""Export a fused walk for SketchUp: textured terrain, decimated scene mesh and a point cloud.

usage: export_sketchup.py <out_dir> [--terrain-cell 0.1] [--max-faces 150000]

Decimating the scanned scene much below ~150k faces breaks it up (holes, blobs), so keep the
default; SketchUp Free imports only the .stl files (choose metres), paid plans the .dae too.

Reads points.ply, mesh.ply and ground.npz from out_dir (fuse.py output) and writes out_dir/sketchup/:
  terrain.dae + terrain.png  COLLADA terrain from the ground grid, textured with a top-down orthophoto
  scene.dae                  TSDF mesh decimated to --max-faces, one plain material
  terrain.kmz                terrain.dae + terrain.png in one file (texture survives single-file upload), geolocated
  terrain.stl, scene.stl     the same geometry without texture, for SketchUp plans that don't import DAE
  points.las                 LAS 1.4 with RGB, for Scan Essentials

SketchUp is Z-up, so all of it is converted from the world frame (X east, Y up, Z south) to
x = east, y = north (= -Z), z = up, in metres (<unit meter="1">). The origin stays the walk start,
so the files line up with each other.
"""
import argparse
import json
import time
import warnings
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
from PIL import Image
from pyproj import Transformer
from scipy import ndimage

TERRAIN_MAX_FACES = 100_000
OUTLIER = 0.4  # m, a cell this far from the median of its neighbours is dropped
BUMP, BUMP_WINDOW = 0.25, 1.5  # m: cells this far above the local 1.5 m median are vegetation/clutter, not ground
MAX_SLOPE = 1.0  # rise/run (45°); steeper cells are walls or hedges at the edge, not ground
MAX_HOLE = 4.0  # m^2, interior holes up to this (e.g. under a removed bush) are filled; bigger stay empty
GROUND_BAND = (-0.15, 0.3)  # m around the terrain: only these points colour the orthophoto
ORTHO_PX, ORTHO_MAX = 0.03, 8192  # m per pixel, pixels per side
ORTHO_FILL = 1.0  # m, empty ortho pixels take the colour of the nearest ground point up to this far
NS = "{http://www.collada.org/2005/11/COLLADASchema}"


def to_su(p):
    """World (X east, Y up, Z south) -> SketchUp (east, north, up)."""
    p = np.asarray(p)
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1)


# ---------------------------------------------------------------- terrain


def median_resample(g, cell):
    """ground.npz -> grid of per-cell medians at `cell` metres, same north-up layout."""
    h, c0 = g["height"], float(g["cell"])
    rows, cols = np.nonzero(~np.isnan(h))
    nr, nc = int(np.ceil(h.shape[0] * c0 / cell - 1e-9)), int(np.ceil(h.shape[1] * c0 / cell - 1e-9))
    r = np.minimum(((rows + 0.5) * c0 / cell).astype(int), nr - 1)
    c = np.minimum(((cols + 0.5) * c0 / cell).astype(int), nc - 1)
    ids, v = r * nc + c, h[rows, cols]
    order = np.lexsort((v, ids))
    ids, v = ids[order], v[order]
    cells, start, n = np.unique(ids, return_index=True, return_counts=True)
    out = np.full(nr * nc, np.nan)
    out[cells] = (v[start + (n - 1) // 2] + v[start + n // 2]) / 2
    return out.reshape(nr, nc)


def neighbours(a):
    """The 8 shifted copies of a (NaN outside the grid), shape (8, rows, cols)."""
    p = np.pad(a, 1, constant_values=np.nan)
    return np.stack([p[1 + dr:p.shape[0] - 1 + dr, 1 + dc:p.shape[1] - 1 + dc]
                     for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc])


def blur(a, valid):
    """3x3 binomial blur that only averages valid cells."""
    k = np.outer([1, 2, 1], [1, 2, 1]) / 16
    num = ndimage.convolve(np.where(valid, a, 0.0), k, mode="constant")
    den = ndimage.convolve(valid.astype(float), k, mode="constant")
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def clean_grid(a, cell):
    """Drop isolated spikes, fill small holes, smooth lightly. NaN stays where nothing was scanned."""
    nb = neighbours(a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # all-NaN neighbourhoods
        med = np.nanmedian(nb, axis=0)
    spike = ~np.isnan(a) & ((np.isfinite(nb).sum(0) < 3) | (np.abs(a - med) > OUTLIER))
    a = np.where(spike, np.nan, a)

    # Lowest points under dense bushes and along walls are not ground: drop bumps above the local
    # median surface and steep cells, a few passes so edge ramps erode away.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            local = ndimage.generic_filter(a, np.nanmedian, size=max(3, int(BUMP_WINDOW / cell) | 1), mode="constant", cval=np.nan)
            slope = np.nanmax(np.abs(neighbours(a) - a), axis=0) / cell
            a = np.where((a - local > BUMP) | (slope > MAX_SLOPE), np.nan, a)

    lab, n = ndimage.label(np.isnan(np.pad(a, 1, constant_values=np.nan)))  # padding joins the outside into one label
    size = np.bincount(lab.ravel(), minlength=n + 1)
    small = (size <= MAX_HOLE / cell**2) & (np.arange(n + 1) != lab[0, 0]) & (np.arange(n + 1) > 0)
    hole = small[lab][1:-1, 1:-1]
    for _ in range(int(np.sqrt(MAX_HOLE) / cell) + 2):
        todo = hole & np.isnan(a)
        if not todo.any():
            break
        a = np.where(todo, blur(a, ~np.isnan(a)), a)

    ok = ~np.isnan(a)
    return np.where(ok, blur(a, ok), np.nan)


def crop(a, west, north, cell):
    ok = ~np.isnan(a)
    rows, cols = np.nonzero(ok.any(1))[0], np.nonzero(ok.any(0))[0]
    return a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], west + cols[0] * cell, north - rows[0] * cell


def terrain_grid(g, cell, max_faces=TERRAIN_MAX_FACES):
    """Returns (height grid, west, north, cell); the cell grows by 25% steps until it fits max_faces."""
    while True:
        a = clean_grid(median_resample(g, cell), cell)
        a, west, north = crop(a, float(g["west"]), float(g["north"]), cell)
        ok = ~np.isnan(a)
        quads = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
        if 2 * quads.sum() <= max_faces:
            return a, west, north, cell
        print(f"terrain: {2 * quads.sum()} faces at {cell:.3f} m cells, coarsening")
        cell *= 1.25


def triangulate(a, west, north, cell):
    """Grid -> SketchUp-frame vertices (n, 3), uvs (n, 2) over the grid's extent, CCW-from-above triangles (m, 3)."""
    ok = ~np.isnan(a)
    quads = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    idx = np.arange(a.size).reshape(a.shape)
    nw, ne, sw, se = (idx[:-1, :-1][quads], idx[:-1, 1:][quads], idx[1:, :-1][quads], idx[1:, 1:][quads])
    tris = np.concatenate([np.stack([nw, sw, se], 1), np.stack([nw, se, ne], 1)])
    used, tris = np.unique(tris, return_inverse=True)
    tris = tris.reshape(-1, 3)
    rows, cols = np.divmod(used, a.shape[1])
    x, y = west + (cols + 0.5) * cell, north - (rows + 0.5) * cell
    xmin, ymin, w, h = west, north - a.shape[0] * cell, a.shape[1] * cell, a.shape[0] * cell
    return np.stack([x, y, a[rows, cols]], 1), np.stack([(x - xmin) / w, (y - ymin) / h], 1), tris


def orthophoto(pts, cols, west, north, w, h, ground=None, fill=ORTHO_FILL):
    """Top-down image over [west, west+w] x [north-h, north]: the highest coloured point in each pixel.
    With ground=(grid, cell) only points within GROUND_BAND of the terrain count, so bushes and walls
    aren't painted onto it."""
    px = max(ORTHO_PX, max(w, h) / ORTHO_MAX)
    W, H = int(np.ceil(w / px)), int(np.ceil(h / px))
    keep = np.abs(cols * 255 - 127.5).max(1) > 1.5  # fuse.py colours uncovered points grey
    if ground is not None:
        a, cell = ground
        gc = np.clip(((pts[:, 0] - west) / cell).astype(int), 0, a.shape[1] - 1)
        gr = np.clip(((north - pts[:, 1]) / cell).astype(int), 0, a.shape[0] - 1)
        dz = pts[:, 2] - a[gr, gc]
        keep &= (dz >= GROUND_BAND[0]) & (dz <= GROUND_BAND[1])  # NaN terrain -> False
    p, c = pts[keep], cols[keep]
    col, row = ((p[:, 0] - west) / px).astype(int), ((north - p[:, 1]) / px).astype(int)
    inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    p, c, col, row = p[inside], c[inside], col[inside], row[inside]
    order = np.argsort(p[:, 2])[::-1]  # highest first, np.unique keeps the first occurrence
    pix, first = np.unique((row * W + col)[order], return_index=True)
    img = np.full((H, W, 3), (110, 110, 100), np.uint8)
    empty = np.ones((H, W), bool)
    r, k = np.divmod(pix, W)
    img[r, k] = np.round(c[order[first]] * 255)
    empty[r, k] = False
    dist, (ir, ic) = ndimage.distance_transform_edt(empty, return_indices=True)
    near = empty & (dist * px <= fill)
    if ground is not None:  # only fill pixels that lie on terrain geometry
        yy, xx = np.mgrid[0:H, 0:W]
        on = ~np.isnan(a[np.clip((yy * px / cell).astype(int), 0, a.shape[0] - 1),
                         np.clip((xx * px / cell).astype(int), 0, a.shape[1] - 1)])
        near &= on
    img[near] = img[ir[near], ic[near]]
    print(f"terrain: orthophoto {W}x{H} px at {px * 100:.1f} cm, {1 - empty.mean():.0%} covered by points, "
          f"{near.sum() / empty.sum() if empty.any() else 0:.0%} of the rest filled from neighbours")
    return img


# ---------------------------------------------------------------- COLLADA


def float_source(sid, arr, params):
    n, stride = arr.shape
    text = " ".join(f"{v:.5f}" for v in arr.ravel())
    ps = "".join(f'<param name="{p}" type="float"/>' for p in params)
    return (f'<source id="{sid}"><float_array id="{sid}-arr" count="{arr.size}">{text}</float_array>'
            f'<technique_common><accessor source="#{sid}-arr" count="{n}" stride="{stride}">{ps}</accessor>'
            f'</technique_common></source>')


def write_dae(path, name, verts, tris, uvs=None, texture=None, rgb=(0.78, 0.76, 0.72)):
    """COLLADA 1.4.1 triangle mesh in metres, Z up. With uvs+texture the material is textured, else a plain colour."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if texture:
        fx_params = ('<newparam sid="surf"><surface type="2D"><init_from>tex</init_from></surface></newparam>'
                     '<newparam sid="samp"><sampler2D><source>surf</source></sampler2D></newparam>')
        diffuse = '<texture texture="samp" texcoord="UV"/>'
        images = f'<library_images><image id="tex" name="tex"><init_from>{texture}</init_from></image></library_images>'
    else:
        fx_params, diffuse, images = "", f"<color>{rgb[0]} {rgb[1]} {rgb[2]} 1</color>", ""
    uv_src = float_source("uv", uvs, ("S", "T")) if uvs is not None else ""
    uv_in = '<input semantic="TEXCOORD" source="#uv" offset="0" set="0"/>' if uvs is not None else ""
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">'
        f'<asset><contributor><authoring_tool>export_sketchup.py</authoring_tool></contributor>'
        f'<created>{now}</created><modified>{now}</modified>'
        f'<unit meter="1" name="meter"/><up_axis>Z_UP</up_axis></asset>'
        f'{images}'
        f'<library_effects><effect id="fx"><profile_COMMON>{fx_params}<technique sid="common">'
        f'<phong><diffuse>{diffuse}</diffuse></phong></technique></profile_COMMON></effect></library_effects>'
        f'<library_materials><material id="mat" name="{name}"><instance_effect url="#fx"/></material></library_materials>'
        f'<library_geometries><geometry id="geo" name="{name}"><mesh>'
        f'{float_source("pos", verts, ("X", "Y", "Z"))}{uv_src}'
        f'<vertices id="vtx"><input semantic="POSITION" source="#pos"/></vertices>'
        f'<triangles material="sym" count="{len(tris)}"><input semantic="VERTEX" source="#vtx" offset="0"/>{uv_in}'
        f'<p>{" ".join(map(str, tris.ravel().tolist()))}</p></triangles></mesh></geometry></library_geometries>'
        f'<library_visual_scenes><visual_scene id="scene"><node id="node" name="{name}">'
        f'<instance_geometry url="#geo"><bind_material><technique_common>'
        f'<instance_material symbol="sym" target="#mat"/></technique_common></bind_material></instance_geometry>'
        f'</node></visual_scene></library_visual_scenes>'
        f'<scene><instance_visual_scene url="#scene"/></scene></COLLADA>\n')
    Path(path).write_text(xml)


def check_dae(path):
    """Parse a DAE written by write_dae and sanity-check it. Returns dict with verts, tris, uvs (or None),
    texture (resolved path or None) and unit/up_axis."""
    root = ET.parse(path).getroot()
    assert root.get("version") == "1.4.1"
    unit = root.find(f"{NS}asset/{NS}unit")
    up = root.find(f"{NS}asset/{NS}up_axis").text
    assert unit.get("meter") == "1" and up == "Z_UP"
    mesh = root.find(f"{NS}library_geometries/{NS}geometry/{NS}mesh")
    sources = {}
    for s in mesh.findall(f"{NS}source"):
        arr = s.find(f"{NS}float_array")
        acc = s.find(f"{NS}technique_common/{NS}accessor")
        a = np.array(arr.text.split(), float).reshape(int(acc.get("count")), int(acc.get("stride")))
        assert a.size == int(arr.get("count"))
        sources[s.get("id")] = a
    verts = sources["pos"]
    tri = mesh.find(f"{NS}triangles")
    tris = np.array(tri.find(f"{NS}p").text.split(), int).reshape(-1, 3)
    assert len(tris) == int(tri.get("count")) and tris.min() >= 0 and tris.max() < len(verts)
    uvs = sources.get("uv")
    if uvs is not None:
        assert len(uvs) == len(verts) and uvs.min() >= 0 and uvs.max() <= 1
    img = root.find(f"{NS}library_images/{NS}image/{NS}init_from")
    texture = None
    if img is not None:
        assert not Path(img.text).is_absolute()
        texture = Path(path).parent / img.text
        assert texture.is_file()
    return dict(verts=verts, tris=tris, uvs=uvs, texture=texture, unit=unit.get("meter"), up_axis=up)


# ---------------------------------------------------------------- scene, points


def scene_mesh(src, max_faces):
    mesh = o3d.io.read_triangle_mesh(str(src))
    n0 = len(mesh.triangles)
    cluster, count, _ = mesh.cluster_connected_triangles()
    cluster, count = np.asarray(cluster), np.asarray(count)
    keep = count >= max(500, 0.002 * n0)
    mesh.remove_triangles_by_mask(~keep[cluster])
    mesh.remove_unreferenced_vertices()
    n1 = len(mesh.triangles)
    if n1 > max_faces:
        mesh = mesh.simplify_quadric_decimation(max_faces)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    print(f"scene: {n0} triangles, {(~keep).sum()} of {len(keep)} bits dropped -> {n1}, decimated to {len(mesh.triangles)}")
    return to_su(np.asarray(mesh.vertices)), np.asarray(mesh.triangles)


def write_las(path, pts, cols):
    las = laspy.create(point_format=7, file_version="1.4")
    las.header.scales = [0.001] * 3
    las.header.offsets = pts.min(0)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    rgb = np.round(cols * 255).astype(np.uint16) * 257
    las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    las.write(str(path))


def write_kmz(path, dae, texture, georef):
    """KMZ = zip of doc.kml + the DAE + its texture, so the texture travels with the model (SketchUp
    for Web uploads a single file). Placed at the walk start's lat/lon when georef.json exists.
    DAE axes (x east, y north, z up) are KML model axes, so heading 0."""
    lon = lat = 0.0
    if georef.exists():
        g = json.loads(georef.read_text())
        lon, lat = Transformer.from_crs(g["crs"], "EPSG:4326", always_xy=True).transform(g["E0"], g["N0"])
    kml = ('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Placemark>'
           f'<name>{dae.stem}</name><Model><altitudeMode>relativeToGround</altitudeMode>'
           f'<Location><longitude>{lon:.8f}</longitude><latitude>{lat:.8f}</latitude><altitude>0</altitude></Location>'
           '<Orientation><heading>0</heading><tilt>0</tilt><roll>0</roll></Orientation><Scale><x>1</x><y>1</y><z>1</z></Scale>'
           f'<Link><href>models/{dae.name}</href></Link></Model></Placemark></kml>\n')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)
        z.write(dae, f"models/{dae.name}")
        z.write(texture, f"models/{texture.name}")  # DAE references it relatively


def write_stl(path, verts, tris):
    """Binary STL in metres (STL has no unit field: pick metres when importing)."""
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris))
    mesh.compute_triangle_normals()
    o3d.io.write_triangle_mesh(str(path), mesh)


def export(out, terrain_cell=0.1, max_faces=150_000):
    out = Path(out)
    dst = out / "sketchup"
    dst.mkdir(exist_ok=True)
    t0 = time.time()

    pcd = o3d.io.read_point_cloud(str(out / "points.ply"))
    pts, cols = to_su(np.asarray(pcd.points)), np.asarray(pcd.colors)
    g = np.load(out / "ground.npz")
    a, west, north, cell = terrain_grid(g, terrain_cell)
    verts, uvs, tris = triangulate(a, west, north, cell)
    Image.fromarray(orthophoto(pts, cols, west, north, a.shape[1] * cell, a.shape[0] * cell, (a, cell))).save(dst / "terrain.png")
    write_dae(dst / "terrain.dae", "terrain", verts, tris, uvs, "terrain.png")
    write_stl(dst / "terrain.stl", verts, tris)
    write_kmz(dst / "terrain.kmz", dst / "terrain.dae", dst / "terrain.png", out / "georef.json")
    print(f"terrain: {len(tris)} faces, {len(verts)} vertices, {cell:.3f} m cells, {a.shape[1] * cell:.0f} x {a.shape[0] * cell:.0f} m")

    sv, st = scene_mesh(out / "mesh.ply", max_faces)
    write_dae(dst / "scene.dae", "scene", sv, st)
    write_stl(dst / "scene.stl", sv, st)

    write_las(dst / "points.las", pts, cols)

    for name in ("terrain.dae", "scene.dae"):
        check_dae(dst / name)
    print(f"done in {time.time() - t0:.0f} s -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--terrain-cell", type=float, default=0.1)
    ap.add_argument("--max-faces", type=int, default=150_000)
    a = ap.parse_args()
    export(a.out_dir, a.terrain_cell, a.max_faces)
