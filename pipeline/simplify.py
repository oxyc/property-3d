"""Light CAD-like model of a fused walk: simplified terrain, planar structures, trees and shrubs.

usage: simplify.py <out_dir> [--min-height 0.3] [--min-area 0.4] [--terrain-tol 0.04]

Reads points.ply and ground.npz from out_dir (fuse.py output) and writes out_dir/simple/:
  simple.dae + terrain.png  COLLADA, Z up, metres; one group per element type (terrain, walls, surfaces,
                            trees, shrubs), flat materials, the terrain textured with the orthophoto
  simple.stl                the same geometry without materials, for SketchUp Free
  simple.glb                the same in the viewer's world frame (X east, Y up, Z south), for the web
  elements.json             every plane / tree / shrub with its dimensions (SketchUp frame, metres)

Frame is SketchUp's (east, north, up) with the walk start as origin, like export_sketchup.py.

  terrain   ground grid cleaned as in export_sketchup, quadric-decimated until the surface stays within
            --terrain-tol of the grid (99th percentile), textured with the orthophoto
  planes    points more than --min-height above the ground, voxel-thinned; locally flat points with
            near-vertical or near-horizontal normals are RANSAC-segmented plane by plane, split into
            connected pieces, snapped to exactly vertical/horizontal and drawn as a thin slab with a
            rectangular (or convex, if the piece is not rectangular) outline
  trees     what is left is split at canopy-height peaks; each group gets a trunk cylinder if a vertical
            circle fits its low points, and a low-poly ellipsoid crown; groups without trunk are shrubs
  clutter   green-less groups without trunk, and tiny ones, are left out and counted
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull, cKDTree

import export_sketchup as ex

MIN_HEIGHT = 0.3  # m above the ground for a point to count as structure or vegetation
MIN_AREA = 0.4  # m^2 of occupied surface for a plane piece
VOXEL = 0.05
PLANE_TOL = 0.03  # m, RANSAC inlier distance
ANGLE_TOL = np.radians(8)  # normals this close to vertical/horizontal count; fitted planes are snapped
FLAT = 0.05  # surface variation (smallest eigenvalue share) below which a neighbourhood counts as flat
SLAB = 0.06  # m, thickness of drawn planes
TERRAIN_TOL = 0.04  # m
TERRAIN_MAX_FACES = 25_000
RECT_FILL = 0.5  # occupied share of the bounding rectangle above which a plane is drawn as a rectangle
MAX_CROWN = 4.0  # m, longest side of a crown ellipsoid
SCATTER = 0.1  # median surface variation of a group that counts as foliage
PEAK_SMOOTH, PEAK_RADIUS = 0.5, 1.0  # m, canopy-height smoothing and minimum distance between tree tops
MATERIALS = {  # name -> rgb, None = textured
    "ground": None, "wall": (0.88, 0.88, 0.85), "surface": (0.7, 0.7, 0.68),
    "trunk": (0.42, 0.29, 0.18), "crown": (0.25, 0.5, 0.22), "shrub": (0.35, 0.58, 0.25)}


def to_world(p):
    """SketchUp (east, north, up) -> world (X east, Y up, Z south)."""
    return np.stack([p[:, 0], p[:, 2], -p[:, 1]], 1)


# ---------------------------------------------------------------- terrain


def ground_lookup(a, west, north, cell):
    """Function (x, y) -> ground height, nearest cell, holes filled from the nearest scanned cell."""
    _, (ir, ic) = ndimage.distance_transform_edt(np.isnan(a), return_indices=True)
    filled = a[ir, ic]

    def at(x, y):
        c = np.clip(np.floor((x - west) / cell).astype(int), 0, a.shape[1] - 1)
        r = np.clip(np.floor((north - y) / cell).astype(int), 0, a.shape[0] - 1)
        return filled[r, c]
    return at


def decimate(verts, tris, target):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris))
    m = m.simplify_quadric_decimation(target)
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    v, t = np.asarray(m.vertices), np.asarray(m.triangles)
    n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    t = np.where((n[:, 2] < 0)[:, None], t[:, ::-1], t)  # height field: keep facing up
    return v, t


def terrain_error(verts, tris, ref):
    """Vertical distance from the reference vertices to the decimated surface (0 outside it)."""
    z = LinearNDInterpolator(verts[:, :2], verts[:, 2])(ref[:, :2])  # Delaunay of the kept vertices, close to the mesh
    return np.nan_to_num(np.abs(z - ref[:, 2]))


def simplify_terrain(verts, tris, tol, max_faces):
    """Fewest faces (log search) whose 99th-percentile vertical error against the full grid mesh is <= tol."""
    lo, hi = 200, min(max_faces, len(tris))
    best = decimate(verts, tris, hi)
    while hi > lo * 1.15:
        mid = int(np.sqrt(lo * hi))
        v, t = decimate(verts, tris, mid)
        if np.percentile(terrain_error(v, t, verts), 99) <= tol:
            hi, best = mid, (v, t)
        else:
            lo = mid
    return best


# ---------------------------------------------------------------- planes


def local_shape(pts, k=12, chunk=200_000):
    """Per point: unit normal of the k nearest neighbours' best-fit plane and surface variation
    (smallest covariance eigenvalue / trace; ~0 flat, 1/3 scattered)."""
    idx = cKDTree(pts).query(pts, k, workers=-1)[1]
    normals, var = np.empty((len(pts), 3)), np.empty(len(pts))
    for i in range(0, len(pts), chunk):
        nb = pts[idx[i:i + chunk]]
        nb = nb - nb.mean(1, keepdims=True)
        w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))
        normals[i:i + chunk], var[i:i + chunk] = v[:, :, 0], w[:, 0] / np.maximum(w.sum(1), 1e-12)
    return normals, var


def green(cols):
    return ((cols[:, 1] > cols[:, 0] * 1.1) & (cols[:, 1] > cols[:, 2] * 1.1)).mean()


def simplify_polygon(poly, n_max=8):
    """Drop the vertex whose removal loses the least area until n_max remain (poly is CCW)."""
    poly = poly.copy()
    while len(poly) > n_max:
        a, b, c = np.roll(poly, 1, 0), poly, np.roll(poly, -1, 0)
        area = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        poly = np.delete(poly, area.argmin(), 0)
    return poly


def outline(st, rotate):
    """2D points -> (CCW polygon, 'rect' | 'hull', occupied area). rotate: search the rectangle's angle."""
    occupied = len(np.unique(np.floor(st / VOXEL).astype(int), axis=0)) * VOXEL**2
    best = None
    for ang in (np.radians(np.arange(0, 90, 1.0)) if rotate else [0.0]):
        c, s = np.cos(ang), np.sin(ang)
        q = st @ np.array([[c, -s], [s, c]])  # coordinates in the frame rotated by +ang
        lo, hi = q.min(0), q.max(0)
        if best is None or np.prod(hi - lo) < best[0]:
            best = (np.prod(hi - lo), ang, lo, hi)
    area, ang, lo, hi = best
    if occupied / max(area, 1e-9) >= RECT_FILL:
        corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        c, s = np.cos(ang), np.sin(ang)
        return corners @ np.array([[c, -s], [s, c]]).T, "rect", occupied
    hull = ConvexHull(st)
    return simplify_polygon(st[hull.vertices]), "hull", occupied


def basis(kind, normal):
    """Snapped normal and in-plane axes (u, v) with u x v = n."""
    if kind == "horizontal":
        return np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    n = np.array([normal[0], normal[1], 0.0])
    n /= np.linalg.norm(n)
    return n, np.array([-n[1], n[0], 0.0]), np.array([0, 0, 1.0])


def ransac_plane(pts, normals, kind, rng, tries=600):
    """Best plane of the given kind through pts: each hypothesis is one point with its own normal (snapped
    vertical/horizontal), inliers lie within PLANE_TOL and have a normal within ANGLE_TOL. Needs the local
    normals, but unlike 3-point RANSAC it never fits planes of the wrong orientation. Returns (normal, inlier indices)."""
    seeds = rng.choice(len(pts), min(tries, len(pts)), replace=False)
    hyp = np.tile([0, 0, 1.0], (len(seeds), 1)) if kind == "horizontal" else normals[seeds] * [1, 1, 0]
    hyp /= np.linalg.norm(hyp, axis=1, keepdims=True)
    best, best_n = None, 0
    for i in range(0, len(hyp), 100):
        h, s = hyp[i:i + 100], seeds[i:i + 100]
        near = np.abs(h @ pts.T - np.einsum("ij,ij->i", h, pts[s])[:, None]) < PLANE_TOL
        near &= np.abs(h @ normals.T) > np.cos(ANGLE_TOL)
        k = near.sum(1).argmax()
        if near[k].sum() > best_n:
            best, best_n = (h[k], near[k]), near[k].sum()
    n, inl = best
    if kind == "vertical":  # refit the direction on the inliers, then take the inliers of the refined plane
        xy = pts[inl, :2]
        w, v = np.linalg.eigh(np.cov(xy.T))
        n = np.array([*v[:, 0], 0.0])
        inl = (np.abs(pts @ n - np.median(pts[inl] @ n)) < PLANE_TOL) & (np.abs(normals @ n) > np.cos(ANGLE_TOL))
    return n, np.nonzero(inl)[0]


def detect_planes(pts, cols, min_area, rng):
    """Planes in the near-vertical and near-horizontal flat points of pts. Returns a list of dicts
    (kind, n, d, u, v, polygon, shape, area, n_points), the boolean mask of points explained by them and
    the surface variation of every point."""
    normals, var = local_shape(pts)
    flat = var < FLAT
    tilt = np.abs(normals[:, 2])
    pools = {"horizontal": flat & (tilt > np.cos(1.5 * ANGLE_TOL)), "vertical": flat & (tilt < np.sin(1.5 * ANGLE_TOL))}
    min_pts = int(min_area / VOXEL**2 / 2)
    planes, explained = [], np.zeros(len(pts), bool)
    for kind, mask in pools.items():
        idx = np.nonzero(mask)[0]
        for _ in range(300):
            if len(idx) < min_pts:
                break
            n0, inl = ransac_plane(pts[idx], normals[idx], kind, rng)
            inl = idx[inl]
            idx = np.setdiff1d(idx, inl)
            if len(inl) < min_pts:
                break
            labels = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[inl])).cluster_dbscan(0.15, 5))
            for lab in range(labels.max() + 1):
                piece = inl[labels == lab]
                if len(piece) < min_pts or green(cols[piece]) > 0.5:
                    continue
                n, u, v = basis(kind, n0)
                d = np.median(pts[piece] @ n)
                st = np.stack([pts[piece] @ u, pts[piece] @ v], 1)
                poly, shape, area = outline(st, kind == "horizontal")
                if area < min_area or np.ptp(st, axis=0).max() < 0.6:
                    continue
                planes.append(dict(kind=kind, n=n, d=d, u=u, v=v, polygon=poly, shape=shape, area=area, n_points=len(piece)))
    for p in planes:  # everything lying on an accepted plane inside its outline is explained
        dist = np.abs(pts @ p["n"] - p["d"])
        st = np.stack([pts @ p["u"], pts @ p["v"]], 1)
        lo, hi = p["polygon"].min(0) - 0.1, p["polygon"].max(0) + 0.1
        explained |= (dist < 2 * PLANE_TOL) & (st >= lo).all(1) & (st <= hi).all(1)
    return planes, explained, var


def slab(poly, o, u, v, n, t=SLAB):
    """Convex polygon (in the u, v frame at origin o) extruded +-t/2 along n: vertices, triangles."""
    p = o + poly[:, 0, None] * u + poly[:, 1, None] * v
    m = len(p)
    verts = np.vstack([p + n * t / 2, p - n * t / 2])
    tris = [(0, i, i + 1) for i in range(1, m - 1)] + [(m, m + i + 1, m + i) for i in range(1, m - 1)]
    for i in range(m):
        j = (i + 1) % m
        tris += [(m + i, m + j, j), (m + i, j, i)]
    return verts, np.array(tris)


# ---------------------------------------------------------------- vegetation


def fit_circle(xy):
    c, *_ = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], (xy**2).sum(1), rcond=None)
    return c[:2], np.sqrt(max(c[2] + c[0]**2 + c[1]**2, 0))


def find_trunk(xy, hg, rng, iters=400):
    """Circle (centre, radius) that many of the low points lie on, or None. RANSAC with
    r in 3-40 cm, then needs >= 40 points, a quarter of a turn covered and a vertical extent >= 0.5 m."""
    if len(xy) < 40:
        return None
    best = (0, None)
    for _ in range(iters):
        c, r = fit_circle(xy[rng.choice(len(xy), 3, replace=False)])
        if not 0.03 <= r <= 0.4:
            continue
        n = (np.abs(np.linalg.norm(xy - c, axis=1) - r) < 0.02).sum()
        if n > best[0]:
            best = (n, (c, r))
    if best[1] is None:
        return None
    c, r = best[1]
    inl = np.abs(np.linalg.norm(xy - c, axis=1) - r) < 0.02
    if inl.sum() < 40:
        return None
    c, r = fit_circle(xy[inl])
    inl = np.abs(np.linalg.norm(xy - c, axis=1) - r) < 0.025
    near = np.linalg.norm(xy - c, axis=1) < r + 0.3
    ang = np.sort(np.arctan2(*(xy[inl] - c).T[::-1]))
    gap = np.diff(np.r_[ang, ang[0] + 2 * np.pi]).max()
    if not 0.03 <= r <= 0.4 or inl.sum() < 40 or inl.sum() < 0.5 * near.sum() or 2 * np.pi - gap < np.pi / 2:
        return None
    if np.ptp(hg[inl]) < 0.5:
        return None
    return c, r


def segment_canopy(pts, hg):
    """Group label per point (0 = none): watershed of the smoothed canopy-height raster from its peaks,
    which are at least PEAK_RADIUS apart. Cells without points stay unlabelled."""
    cell = 0.1
    lo = pts[:, :2].min(0)
    ij = tuple(((pts[:, :2] - lo) / cell).astype(int).T)
    chm = np.zeros(np.max(ij, axis=1) + 1)
    np.maximum.at(chm, ij, hg)
    sm = ndimage.gaussian_filter(chm, PEAK_SMOOTH / cell)
    r = int(PEAK_RADIUS / cell)
    foot = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
    markers, n = ndimage.label((sm == ndimage.maximum_filter(sm, footprint=foot)) & (sm > 0.3))
    basins = ndimage.watershed_ift(np.round(65535 * (1 - sm / sm.max())).astype(np.uint16), markers.astype(np.int32))
    return np.where(chm[ij] > 0, basins[ij], 0)


def split_long(xy, idx):
    """Halve the point group (indices into xy) across its long axis until no piece is longer than MAX_CROWN,
    so hedges and canopy that runs along a wall become a chain of blobs instead of one huge ellipsoid."""
    if len(idx) < 80:
        return [idx]
    q = xy[idx] - xy[idx].mean(0)
    proj = q @ np.linalg.eigh(np.cov(q.T))[1][:, 1]
    lo, hi = np.percentile(proj, [3, 97])
    if hi - lo <= MAX_CROWN:
        return [idx]
    upper = proj > np.median(proj)
    return split_long(xy, idx[upper]) + split_long(xy, idx[~upper])


def ellipsoid(centre, radii, yaw, subdiv):
    m = trimesh.creation.icosphere(subdivisions=subdiv)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (m.vertices * radii) @ rot.T + centre, m.faces


def find_trunks(pts, hg, ground, rng):
    """Trunks (centre, radius, height, base z) among the points: circle fits to each connected clump of the
    0.3-1.3 m height band. Height is the highest point within 6 cm of the trunk's surface."""
    band = np.nonzero(hg < 1.3)[0]
    flat = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.c_[pts[band, :2], np.zeros(len(band))]))
    labels = np.asarray(flat.cluster_dbscan(0.1, 15))
    tree = cKDTree(pts[:, :2])
    trunks = []
    for lab in range(labels.max() + 1):
        sel = band[labels == lab]
        t = find_trunk(pts[sel, :2], hg[sel], rng) if len(sel) >= 40 else None
        if t is not None:
            c, r = t
            top = hg[tree.query_ball_point(c, r + 0.06)].max()
            trunks.append((c, r, top, ground(*c)))
    return trunks


def make_tree(pts, hg, trunks, ground):
    """One group of points and the trunks standing in it -> element dict with meshes, or None if it has
    neither a crown nor a trunk worth drawing."""
    crown_pts = np.ones(len(pts), bool)
    for c, r, *_ in trunks:
        crown_pts &= np.linalg.norm(pts[:, :2] - c, axis=1) > r + 0.08
    tall = np.percentile(hg, 99) if len(hg) else 0
    out = dict(kind="tree" if trunks or tall > 2.5 else "shrub", meshes=[])
    cz = None
    if crown_pts.sum() >= 30:
        q = pts[crown_pts]
        xy = q[:, :2] - np.median(q[:, :2], 0)
        w, v = np.linalg.eigh(np.cov(xy.T))
        yaw = np.arctan2(v[1, 1], v[0, 1])
        c, s = np.cos(yaw), np.sin(yaw)
        loc = xy @ np.array([[c, -s], [s, c]])
        lo, hi = np.percentile(loc, 3, axis=0), np.percentile(loc, 97, axis=0)
        z = np.percentile(hg[crown_pts], [3, 99])
        if (hi - lo).min() < 0.3 or z[1] - z[0] < 0.2:
            return None if not trunks else out
        mid2 = (lo + hi) / 2
        centre_xy = np.median(q[:, :2], 0) + np.array([[c, -s], [s, c]]) @ mid2
        radii = np.array([(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, (z[1] - z[0]) / 2]) * 1.1
        g = ground(*centre_xy)
        cz = g + (z[0] + z[1]) / 2
        centre = np.array([*centre_xy, cz])
        v_, t_ = ellipsoid(centre, radii, yaw, 2 if out["kind"] == "tree" else 1)
        out["meshes"].append(("crown" if out["kind"] == "tree" else "shrub", v_, t_))
        out["crown"] = dict(centre=centre.round(3).tolist(), radii=radii.round(3).tolist(), yaw_deg=round(float(np.degrees(yaw)), 1))
    for c, r, top, g in trunks:
        if cz is not None:
            top = min(top, cz - g)  # the crown hides the rest
        if top < 0.5:
            continue
        m = trimesh.creation.cylinder(radius=r, height=top, sections=8)
        out["meshes"].append(("trunk", m.vertices + [c[0], c[1], g + top / 2], m.faces))
        out.setdefault("trunks", []).append(dict(x=round(float(c[0]), 3), y=round(float(c[1]), 3), radius=round(float(r), 3),
                                                  height=round(float(top), 2), base_z=round(float(g), 3)))
    if not out["meshes"]:
        return None
    out["height"] = round(float(max([tall] + [t["height"] for t in out.get("trunks", [])])), 2)
    return out


def find_vegetation(pts, cols, var, hg, ground, rng):
    """Group the points (not on planes) into trees/shrubs. A group counts as vegetation if it has a trunk,
    is green or is scattered (median surface variation >= SCATTER). Returns (tree dicts, number of clutter groups)."""
    labels = segment_canopy(pts, hg)
    groups = []
    for lab in range(1, labels.max() + 1):
        groups += split_long(pts[:, :2], np.nonzero(labels == lab)[0])
    group_of = np.full(len(pts), -1)
    for k, gi in enumerate(groups):
        group_of[gi] = k
    trunks = find_trunks(pts, hg, ground, rng)
    owner = group_of[cKDTree(pts[:, :2]).query([t[0] for t in trunks])[1]] if trunks else []
    trees, clutter = [], 0
    for k, gi in enumerate(groups):
        mine = [t for t, o in zip(trunks, owner) if o == k]
        foliage = len(gi) >= 40 and (green(cols[gi]) >= 0.25 or np.median(var[gi]) >= SCATTER)
        t = make_tree(pts[gi] if foliage else pts[:0], hg[gi] if foliage else hg[:0], mine, ground) if foliage or mine else None
        if t is None:
            clutter += 1
            continue
        trees.append(t)
    for t, o in zip(trunks, owner):  # trunks outside every canopy group
        if o == -1 and (e := make_tree(pts[:0], hg[:0], [t], ground)) is not None:
            trees.append(e)
    return trees, clutter


# ---------------------------------------------------------------- output


def write_scene_dae(path, parts):
    """parts: [(group, name, material, verts, tris, uvs or None)] -> COLLADA with a node per group holding a node per part."""
    used = sorted({p[2] for p in parts})
    fx, mats = "", ""
    for m in used:
        if MATERIALS[m] is None:
            surf = ('<newparam sid="surf"><surface type="2D"><init_from>tex</init_from></surface></newparam>'
                    '<newparam sid="samp"><sampler2D><source>surf</source></sampler2D></newparam>')
            diffuse = '<texture texture="samp" texcoord="UV"/>'
        else:
            surf, diffuse = "", "<color>{} {} {} 1</color>".format(*MATERIALS[m])
        fx += (f'<effect id="fx-{m}"><profile_COMMON>{surf}<technique sid="common"><phong><diffuse>{diffuse}</diffuse></phong>'
               f'</technique></profile_COMMON></effect>')
        mats += f'<material id="mat-{m}" name="{m}"><instance_effect url="#fx-{m}"/></material>'
    geoms, nodes = "", {}
    for i, (group, name, mat, v, t, uv) in enumerate(parts):
        uv_src = ex.float_source(f"uv{i}", uv, ("S", "T")) if uv is not None else ""
        uv_in = f'<input semantic="TEXCOORD" source="#uv{i}" offset="0" set="0"/>' if uv is not None else ""
        geoms += (f'<geometry id="geo{i}" name="{name}"><mesh>{ex.float_source(f"pos{i}", v, ("X", "Y", "Z"))}{uv_src}'
                  f'<vertices id="vtx{i}"><input semantic="POSITION" source="#pos{i}"/></vertices>'
                  f'<triangles material="sym" count="{len(t)}"><input semantic="VERTEX" source="#vtx{i}" offset="0"/>{uv_in}'
                  f'<p>{" ".join(map(str, t.ravel().tolist()))}</p></triangles></mesh></geometry>')
        nodes.setdefault(group, []).append(
            f'<node id="n{i}" name="{name}"><instance_geometry url="#geo{i}"><bind_material><technique_common>'
            f'<instance_material symbol="sym" target="#mat-{mat}"><bind_vertex_input semantic="UV" input_semantic="TEXCOORD" input_set="0"/>'
            f'</instance_material></technique_common></bind_material></instance_geometry></node>')
    scene = "".join(f'<node id="g-{g}" name="{g}">{"".join(n)}</node>' for g, n in nodes.items())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(path).write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">'
        f'<asset><contributor><authoring_tool>simplify.py</authoring_tool></contributor><created>{now}</created>'
        f'<modified>{now}</modified><unit meter="1" name="meter"/><up_axis>Z_UP</up_axis></asset>'
        '<library_images><image id="tex" name="tex"><init_from>terrain.png</init_from></image></library_images>'
        f'<library_effects>{fx}</library_effects><library_materials>{mats}</library_materials>'
        f'<library_geometries>{geoms}</library_geometries>'
        f'<library_visual_scenes><visual_scene id="scene">{scene}</visual_scene></library_visual_scenes>'
        '<scene><instance_visual_scene url="#scene"/></scene></COLLADA>\n')


def write_glb(path, parts, texture):
    scene = trimesh.Scene()
    for group, name, mat, v, t, uv in parts:
        rgb = MATERIALS[mat]
        if rgb is None:
            material = trimesh.visual.material.PBRMaterial(baseColorTexture=texture, metallicFactor=0, roughnessFactor=1)
        else:
            material = trimesh.visual.material.PBRMaterial(baseColorFactor=[*rgb, 1], metallicFactor=0, roughnessFactor=1, doubleSided=True)
        m = trimesh.Trimesh(to_world(v), t, process=False, visual=trimesh.visual.TextureVisuals(uv=uv, material=material))
        scene.add_geometry(m, node_name=f"{group}/{name}", geom_name=name)
    scene.export(str(path))


def simplify(out, min_height=MIN_HEIGHT, min_area=MIN_AREA, terrain_tol=TERRAIN_TOL):
    out = Path(out)
    dst = out / "simple"
    dst.mkdir(exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(0)
    pcd = o3d.io.read_point_cloud(str(out / "points.ply"))
    pts, cols = ex.to_su(np.asarray(pcd.points)), np.asarray(pcd.colors)
    g = np.load(out / "ground.npz")
    a, west, north, cell = ex.terrain_grid(g, 0.1, max_faces=500_000)
    ground = ground_lookup(a, west, north, cell)
    parts = []

    full_v, uvs, full_t = ex.triangulate(a, west, north, cell)
    v, t = simplify_terrain(full_v, full_t, terrain_tol, TERRAIN_MAX_FACES)
    err = terrain_error(v, t, full_v)
    w, h = a.shape[1] * cell, a.shape[0] * cell
    uv = np.stack([(v[:, 0] - west) / w, (v[:, 1] - (north - h)) / h], 1)
    tex = Image.fromarray(ex.orthophoto(pts, cols, west, north, w, h))
    tex.save(dst / "terrain.png")
    parts.append(("terrain", "terrain", "ground", v, t, np.clip(uv, 0, 1)))
    print(f"terrain: {len(full_t)} -> {len(t)} faces, error p99 {np.percentile(err, 99) * 100:.1f} cm, max {err.max() * 100:.1f} cm")

    hg = pts[:, 2] - ground(pts[:, 0], pts[:, 1])
    up = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[hg > min_height]))
    up.colors = o3d.utility.Vector3dVector(cols[hg > min_height])
    up = up.voxel_down_sample(VOXEL)
    p, c = np.asarray(up.points), np.asarray(up.colors)
    print(f"above ground: {(hg > min_height).sum()} points -> {len(p)} after {VOXEL * 100:.0f} cm voxels")

    planes, explained, var = detect_planes(p, c, min_area, rng)
    elements = {"planes": [], "trees": []}
    for i, pl in enumerate(planes, 1):
        v, t = slab(pl["polygon"], pl["n"] * pl["d"], pl["u"], pl["v"], pl["n"])
        wall = pl["kind"] == "vertical"
        name = f"{'wall' if wall else 'surface'}_{i:02d}"
        parts.append(("walls" if wall else "surfaces", name, "wall" if wall else "surface", v, t, None))
        zs = v[:, 2]
        elements["planes"].append(dict(
            id=name, kind=pl["kind"], normal=pl["n"].round(4).tolist(), offset=round(float(pl["d"]), 3),
            shape=pl["shape"], area=round(float(pl["area"]), 2), points=pl["n_points"],
            size=np.ptp(pl["polygon"], axis=0).round(2).tolist(), z_range=[round(float(zs.min()), 2), round(float(zs.max()), 2)],
            polygon=(pl["polygon"][:, 0, None] * pl["u"] + pl["polygon"][:, 1, None] * pl["v"] + pl["n"] * pl["d"]).round(3).tolist()))
    print(f"planes: {sum(q['kind'] == 'vertical' for q in planes)} vertical, {sum(q['kind'] == 'horizontal' for q in planes)} horizontal, "
          f"{explained.sum()} of {len(p)} points explained")

    rest = ~explained
    trees, clutter = find_vegetation(p[rest], c[rest], var[rest], (p[:, 2] - ground(p[:, 0], p[:, 1]))[rest], ground, rng)
    counts = {"tree": 0, "shrub": 0}
    for t_ in trees:
        counts[t_["kind"]] += 1
        name = f"{t_['kind']}_{counts[t_['kind']]:02d}"
        for mat, v, f in t_["meshes"]:
            parts.append(("trees" if t_["kind"] == "tree" else "shrubs", f"{name}_{mat}", mat, np.asarray(v), np.asarray(f), None))
        elements["trees"].append({"id": name, **{k: x for k, x in t_.items() if k != "meshes"}})
    elements["clutter_omitted"] = clutter
    print(f"vegetation: {counts['tree']} trees, {sum(len(t_.get('trunks', [])) for t_ in trees)} trunks, "
          f"{counts['shrub']} shrubs, {clutter} clutter groups omitted")

    (dst / "elements.json").write_text(json.dumps(elements, indent=1))
    write_scene_dae(dst / "simple.dae", parts)
    off = np.cumsum([0] + [len(q[3]) for q in parts])
    ex.write_stl(dst / "simple.stl", np.vstack([q[3] for q in parts]), np.vstack([q[4] + o for q, o in zip(parts, off)]))
    write_glb(dst / "simple.glb", parts, tex)
    nfaces = sum(len(q[4]) for q in parts)
    print(f"done in {time.time() - t0:.0f} s: {nfaces} faces -> {dst}")
    return dict(faces=nfaces, parts=parts, elements=elements, terrain_error=err)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--min-height", type=float, default=MIN_HEIGHT)
    ap.add_argument("--min-area", type=float, default=MIN_AREA)
    ap.add_argument("--terrain-tol", type=float, default=TERRAIN_TOL)
    a = ap.parse_args()
    simplify(a.out_dir, a.min_height, a.min_area, a.terrain_tol)
