"""Photo-textured mesh from a walk: decimated TSDF mesh + UV atlas baked from the RGB keyframes.

usage: texture.py <walk_dir> <out_dir> [--max-faces 300000] [--atlas 8192]

Reads out_dir/mesh.ply (fuse.py) and out_dir/poses.npz (optimize.py; the recorded poses if missing)
and writes out_dir/textured/:
  model.glb                       web viewer: world frame (X east, Y up, Z south), unlit, JPEG texture <= GLB_MAX_TEX px
  model_mobile.glb                the same with a texture <= MOBILE_MAX_TEX px
  model.obj, model.mtl, texture.jpg  the same mesh at full atlas resolution
  model.dae                       SketchUp frame (x east, y north, z up), metres, uses texture.jpg

Method:
  1. clean (small islands dropped) and quadric-decimate the mesh
  2. per face and keyframe: score = cos(view angle) / distance * sharpness (Laplacian variance) * depth confidence.
     A face is visible in a keyframe if at least MIN_SAMPLES of its centroid and 3 vertices are inside the image and
     the keyframe's depth agrees with them (within max(5 cm, 3% of range), i.e. not occluded); keep the best CANDIDATES views.
     The view angle is taken on either side of the face, since TSDF normals are not reliably outward
  3. choose one view per face with a Potts model (ICM): neighbouring faces prefer the same view, so patches are big
  4. per-view, per-channel gains from colour mismatches along the seams between patches (least squares)
  5. charts = connected patches of faces sharing a view, UV = that view's image coordinates, scaled and shelf-packed
     (xatlas was tried: on this noisy mesh it makes ~1 chart per 3 faces and took 20 min for 100k faces)
  6. each texel is projected into its face's view; near a seam it is blended with the neighbouring face's view
     over SEAM_RAMP of the face, which makes the colour continuous across the seam
  7. faces no keyframe saw get a tiny chart of their own whose texels interpolate the colours of the nearby points.ply
     points at their three vertices (same in every format; no vertex colours needed, so nothing is applied twice)
  8. texels beyond the charts are filled by dilation (gutter)
Open3D's color_map_optimization is not used: it refines poses/warps for vertex colours, and optimize.py already
corrects the poses, so what is left is seam handling, which the gains and the blend cover.
"""
import argparse
import io
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))  # python -I drops the script dir from sys.path
import export_sketchup  # noqa: E402
import fuse  # noqa: E402

CANDIDATES = 6  # best views kept per face for the labelling
MIN_COS = 0.25  # faces seen more obliquely than ~75 degrees are not textured from that view
MIN_SAMPLES = 2  # of the 4 sample points (centroid + vertices) that must see the keyframe's depth
DEPTH_TOL = (0.05, 0.03)  # m, and fraction of range: allowed disagreement between mesh and keyframe depth
CONF_WEIGHT = (0.5, 0.8, 1.0)  # score multiplier for ARKit depth confidence low / medium / high
POTTS = 0.2  # cost of a neighbour having a different view; unary cost is 1 - score / best score
SEAM_RAMP = 0.35  # barycentric distance from a seam edge over which the neighbour's view is blended in
GAIN_RANGE = (0.6, 1.7)
GLB_MAX_TEX = 4096
MOBILE_MAX_TEX = 2048
POINT_RADIUS = 0.05  # m, points averaged for the colour of a vertex no keyframe saw
MINI = 4  # px, legs of the tiny right-triangle chart of an unseen face
SMALL_DIV = 4  # sharpness and seam statistics use JPEGs decoded at 1/4 size
RASTER_CELLS = 2_000_000


# ---------------------------------------------------------------- mesh


def clean_mesh(src, max_faces):
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
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    print(f"mesh: {n0} triangles, {(~keep).sum()} of {len(keep)} bits dropped -> {n1}, decimated to {len(mesh.triangles)}")
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.triangle_normals)


def adjacency(tris):
    """Faces sharing an edge (manifold edges only). Returns fa, fb, and the shared edge's vertex ids p, q (all (E,)),
    plus oa, ob: the corner of fa / fb (0..2) opposite that edge."""
    nf = len(tris)
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    face = np.tile(np.arange(nf), 3)
    opp = np.repeat([2, 0, 1], nf)
    key = np.sort(edges, 1)
    key = key[:, 0].astype(np.int64) * (tris.max() + 1) + key[:, 1]
    order = np.argsort(key, kind="stable")
    k = key[order]
    first = np.nonzero((k[:-1] == k[1:]) & np.r_[True, k[:-2] != k[1:-1]] & np.r_[k[2:] != k[1:-1], True])[0]
    i, j = order[first], order[first + 1]
    return face[i], face[j], edges[i, 0], edges[i, 1], opp[i], opp[j]


# ---------------------------------------------------------------- views


def load_views(walk, out):
    meta, frames = fuse.read_walk(walk)
    poses = Path(out) / "poses.npz"
    if poses.exists():
        z = np.load(poses)
        corrected = dict(zip(z["i"].tolist(), z["T"]))
        for f in frames:
            f["T"] = corrected[f["i"]]
    kfs = [f for f in frames if f["rgb"] and f["tracking"] == "normal"]
    print(f"views: {len(kfs)} keyframes, {'corrected' if poses.exists() else 'recorded'} poses")
    return meta, kfs


def load_image(walk, kf, div=1):
    im = Image.open(walk / "rgb" / f"{kf['i']:06d}.jpg")
    if div > 1:
        im.draft("RGB", (im.width // div, im.height // div))  # JPEG DCT scaling, much faster than a full decode
    return np.asarray(im.convert("RGB"))


def project(pts, kf):
    """World points -> RGB pixel coords (u, v) and planar depth z, same maths as fuse.project_into_keyframe."""
    fx, fy, cx, cy = kf["intrinsics"]
    T = kf["T"]
    cam = (pts - T[:3, 3]) @ T[:3, :3]
    z = -cam[:, 2]
    zs = np.maximum(z, 1e-6)
    return fx * cam[:, 0] / zs + cx, cy - fy * cam[:, 1] / zs, z


def sample(img, u, v):
    """Bilinear sample at pixel coords (pixel centres at integers, as fuse.py), float32 (n, 3)."""
    h, w = img.shape[:2]
    u, v = np.clip(u, 0, w - 1.001), np.clip(v, 0, h - 1.001)
    x0, y0 = u.astype(int), v.astype(int)
    fx, fy = (u - x0)[:, None].astype(np.float32), (v - y0)[:, None].astype(np.float32)
    top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    bot = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bot * fy


def sharpness(walk, kfs):
    """Variance of the Laplacian of the downscaled grey image, relative to the median keyframe, clipped."""
    s = np.array([ndimage.laplace(load_image(walk, kf, SMALL_DIV).mean(2)).var() for kf in kfs])
    return np.clip(s / np.median(s), 0.25, 1.25)


def score_faces(walk, meta, kfs, verts, tris, normals):
    """Top CANDIDATES (score, keyframe index) per face over all keyframes."""
    nf = len(tris)
    samples = np.concatenate([verts[tris].mean(1)[:, None], verts[tris]], 1)  # (F, 4, 3): centroid + vertices
    cent = samples[:, 0]
    cand_s = np.zeros((nf, CANDIDATES), np.float32)
    cand_id = np.full((nf, CANDIDATES), -1, np.int32)
    sharp = sharpness(walk, kfs)
    W, H = meta["rgb_w"], meta["rgb_h"]
    dw, dh = meta["depth_w"], meta["depth_h"]
    cw = np.array(CONF_WEIGHT, np.float32)
    for n, kf in enumerate(kfs):
        d = cent - kf["T"][:3, 3]
        dist = np.linalg.norm(d, axis=1)
        sel = np.nonzero(dist < fuse.MAX_DEPTH + 0.3)[0]
        cos = np.abs((normals[sel] * d[sel]).sum(1)) / dist[sel]  # both sides: TSDF normals are not reliably outward
        sel = sel[cos > MIN_COS]
        if not len(sel):
            continue
        u, v, z = (a.reshape(len(sel), 4) for a in project(samples[sel].reshape(-1, 3), kf))
        kd, kc = fuse.read_depth(walk, meta, kf["i"])
        ud = np.clip(np.round(u * dw / W).astype(int), 0, dw - 1)
        vd = np.clip(np.round(v * dh / H).astype(int), 0, dh - 1)
        depth, conf = kd[vd, ud], kc[vd, ud]
        tol = np.maximum(DEPTH_TOL[0], DEPTH_TOL[1] * z)
        good = ((z > 0.1) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
                & (depth >= fuse.MIN_DEPTH) & (np.abs(depth - z) < tol))
        n_good = good.sum(1)
        ok = n_good >= MIN_SAMPLES
        conf_w = (cw[np.minimum(conf, 2)] * good).sum(1) / np.maximum(n_good, 1)
        sel, u, v, z, conf_w = sel[ok], u[ok, 0], v[ok, 0], z[ok, 0], conf_w[ok]
        if not len(sel):
            continue
        edge = np.minimum.reduce([u / W, 1 - u / W, v / H, 1 - v / H])
        s = (cos[cos > MIN_COS][ok] / np.maximum(z, 0.4) * sharp[n] * conf_w * np.clip(edge / 0.1, 0.3, 1)).astype(np.float32)
        m = cand_s[sel].argmin(1)
        better = s > cand_s[sel, m]
        cand_s[sel[better], m[better]] = s[better]
        cand_id[sel[better], m[better]] = n
    return cand_s, cand_id


def choose_views(cand_s, cand_id, fa, fb, iters=12):
    """ICM on a Potts model over the face adjacency graph. Returns a keyframe index per face (-1 = unseen)."""
    nf = len(cand_s)
    best = cand_s.max(1)
    unary = np.where(cand_id >= 0, 1 - cand_s / np.maximum(best, 1e-9)[:, None], 1e3)
    rows = np.arange(nf)
    lab = cand_id[rows, cand_s.argmax(1)]
    rng = np.random.default_rng(0)
    da, db = np.r_[fa, fb], np.r_[fb, fa]  # directed edges a -> b
    for _ in range(iters):
        nb = lab[db]
        known = nb >= 0
        cost = unary.copy()
        for c in range(CANDIDATES):
            cost[:, c] += POTTS * np.bincount(da, weights=known & (cand_id[da, c] != nb), minlength=nf)
        new = cand_id[rows, cost.argmin(1)]
        move = (rng.random(nf) < 0.5) & (lab >= 0)  # half the faces at a time, a simultaneous update oscillates
        lab = np.where(move, new, lab)
    return lab


def seam_gains(walk, meta, kfs, verts, lab, adj):
    """Per-keyframe RGB gains so colours agree on both sides of every seam between differently-viewed faces."""
    fa, fb, p, q = adj[:4]
    seam = (lab[fa] >= 0) & (lab[fb] >= 0) & (lab[fa] != lab[fb])
    fa, fb, p, q = fa[seam], fb[seam], p[seam], q[seam]
    ts = np.array([0.25, 0.5, 0.75])
    pts = verts[p][:, None] * (1 - ts[None, :, None]) + verts[q][:, None] * ts[None, :, None]  # (E, 3, 3)
    col = np.zeros((2, len(fa), 3, 3), np.float32)
    for k in np.unique(np.r_[lab[fa], lab[fb]]):
        img = load_image(walk, kfs[k], SMALL_DIV).astype(np.float32)
        s = img.shape[1] / meta["rgb_w"]
        for side, f in enumerate((fa, fb)):
            e = np.nonzero(lab[f] == k)[0]
            if len(e):
                u, v, _ = project(pts[e].reshape(-1, 3), kfs[k])
                col[side, e] = sample(img, u * s, v * s).reshape(len(e), 3, 3)
    d = (np.log(col[1] + 10) - np.log(col[0] + 10)).mean(1)  # (E, 3): wanted y_a - y_b
    use = np.abs(d).max(1) < 0.6  # misaligned or occluded samples are not exposure differences
    gains = np.ones((len(kfs), 3), np.float32)
    ne = use.sum()
    if ne:
        A = sparse.coo_matrix((np.r_[np.ones(ne), -np.ones(ne), np.full(len(kfs), 3.0)],
                               (np.r_[np.arange(ne), np.arange(ne), ne + np.arange(len(kfs))],
                                np.r_[lab[fa][use], lab[fb][use], np.arange(len(kfs))])), shape=(ne + len(kfs), len(kfs))).tocsr()
        for c in range(3):
            y = lsqr(A, np.r_[d[use, c], np.zeros(len(kfs))])[0]
            gains[:, c] = np.clip(np.exp(y), *GAIN_RANGE)
    print(f"seams: {seam.sum()} edges between views, {ne} used; gain range {gains.min():.2f}..{gains.max():.2f}")
    return gains


# ---------------------------------------------------------------- atlas


def shelf_pack(sizes, width):
    """Rectangles (n, 2) into rows of the given width, tallest first. Returns positions (n, 2) and the used height."""
    pos = np.zeros((len(sizes), 2), np.int64)
    x = y = shelf = 0
    for i in np.argsort(-sizes[:, 1], kind="stable"):
        w, h = sizes[i]
        if x + w > width:
            x, y, shelf = 0, y + shelf, 0
        pos[i] = x, y
        x += w
        shelf = max(shelf, h)
    return pos, y + shelf


def parametrize(verts, tris, lab, adj, kfs, atlas_size):
    """View-based charts: each connected set of faces sharing a view is one chart, UV = that view's image coords
    (no unfolding, so it is fast and charts never overlap), scaled to fit and shelf-packed into the atlas.
    Each face no keyframe saw gets a tiny chart of its own (a MINI-leg right triangle, never scaled).
    Returns split vertices V, faces T (same order as tris), uv in atlas pixels (y down) per vertex, and a bool
    per vertex: it belongs to an unseen face."""
    nf = len(tris)
    seen = lab >= 0
    same = (lab[adj[0]] == lab[adj[1]]) & (lab[adj[0]] >= 0)
    graph = sparse.coo_matrix((np.ones(same.sum()), (adj[0][same], adj[1][same])), shape=(nf, nf))
    _, comp = sparse.csgraph.connected_components(graph, directed=False)
    unseen = np.nonzero(~seen)[0]
    nu = len(unseen)
    chart = np.zeros(nf, np.int64)  # charts 0..nu-1: one per unseen face, then the view patches
    chart[unseen] = np.arange(nu)
    chart[seen] = np.unique(comp[seen], return_inverse=True)[1] + nu
    nc = chart.max() + 1
    # corner pixel coords in the face's own view
    Ts = np.array([kf["T"] for kf in kfs])
    K = np.array([kf["intrinsics"] for kf in kfs])
    k = np.where(seen, lab, 0)
    cam = np.einsum("fcj,fji->fci", verts[tris] - Ts[k][:, None, :3, 3], Ts[k][:, :3, :3])
    z = np.maximum(-cam[..., 2], 0.05)
    uv = np.stack([K[k][:, None, 0] * cam[..., 0] / z + K[k][:, None, 2],
                   K[k][:, None, 3] - K[k][:, None, 1] * cam[..., 1] / z], -1)  # (F, 3, 2)
    lo, hi = np.full((nc, 2), np.inf), np.full((nc, 2), -np.inf)
    for c in range(3):
        np.minimum.at(lo, chart, uv[:, c])
        np.maximum.at(hi, chart, uv[:, c])
    extent = hi - lo
    is_mini = np.arange(nc) < nu
    pad = np.where(is_mini, 1, 4)
    lo[is_mini], extent[is_mini] = 0, MINI
    uv[unseen] = [[0, 0], [MINI, 0], [0, MINI]]
    fixed = nu * (MINI + 3) ** 2
    s = min(1.0, np.sqrt(max(0.9 * atlas_size**2 - fixed, 0.05 * atlas_size**2) / (extent[~is_mini] + 8).prod(1).sum()))
    while True:
        scale = np.where(is_mini, 1.0, s)
        sizes = np.ceil(extent * scale[:, None]).astype(np.int64) + 2 * pad[:, None] + 1
        pos, height = shelf_pack(sizes, atlas_size)
        if height <= atlas_size and sizes.max() <= atlas_size:
            break
        s *= 0.93
    print(f"charts: {nc - nu} view patches + {nu} unseen faces, photo pixels scaled by {s:.2f}, "
          f"atlas rows used {height / atlas_size:.0%}")
    origin = pos + pad[:, None] - lo * scale[:, None]  # photo px -> atlas px: p * scale + origin[chart]
    px = uv * scale[chart][:, None, None] + origin[chart][:, None]
    key = (tris * nc + chart[:, None]).ravel()
    key, first, inv = np.unique(key, return_index=True, return_inverse=True)
    V = verts[tris.ravel()[first]]
    return V, inv.reshape(-1, 3), px.reshape(-1, 2)[first], ~seen[first // 3]


def rasterize(tri_uv):
    """Texels (centre inside the triangle, small tolerance) of triangles given in pixel coords (n, 3, 2).
    Yields chunks of (triangle index, x, y, barycentrics (m, 3))."""
    lo = np.floor(tri_uv.min(1)).astype(np.int64)
    size = np.maximum(np.ceil(tri_uv.max(1)).astype(np.int64) - lo, 1)
    cells = size[:, 0] * size[:, 1]
    bounds = np.searchsorted(np.cumsum(cells), np.arange(0, cells.sum() + RASTER_CELLS, RASTER_CELLS))
    bounds = np.unique(np.r_[0, np.minimum(bounds, len(cells)), len(cells)])
    for a, b in zip(bounds[:-1], bounds[1:]):
        c = cells[a:b]
        tri = np.repeat(np.arange(a, b), c)
        off = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
        x = lo[tri, 0] + off % size[tri, 0]
        y = lo[tri, 1] + off // size[tri, 0]
        A, B, C = (tri_uv[tri, i] for i in range(3))
        px, py = x + 0.5, y + 0.5
        den = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
        den = np.where(np.abs(den) < 1e-12, np.inf, den)
        l0 = ((B[:, 1] - C[:, 1]) * (px - C[:, 0]) + (C[:, 0] - B[:, 0]) * (py - C[:, 1])) / den
        l1 = ((C[:, 1] - A[:, 1]) * (px - C[:, 0]) + (A[:, 0] - C[:, 0]) * (py - C[:, 1])) / den
        bary = np.stack([l0, l1, 1 - l0 - l1], 1)
        ok = (bary > -0.02).all(1)
        yield tri[ok], x[ok], y[ok], bary[ok]


def blend_in(atlas, weight, x, y, col, w):
    """Running weighted mean per texel. Order-independent, so views can be baked one at a time."""
    cur, wc = atlas[y, x].astype(np.float32), weight[y, x].astype(np.float32)[:, None]
    w = w[:, None].astype(np.float32)
    atlas[y, x] = np.clip((cur * wc + col * w) / (wc + w) + 0.5, 0, 255)
    weight[y, x] = (wc + w)[:, 0]


def dilate(img, filled, iters):
    h, w = filled.shape
    for _ in range(iters):
        start = filled.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx):
                    continue
                yd, ys = slice(max(-dy, 0), h + min(-dy, 0)), slice(max(dy, 0), h + min(dy, 0))
                xd, xs = slice(max(-dx, 0), w + min(-dx, 0)), slice(max(dx, 0), w + min(dx, 0))
                take = ~start[yd, xd] & start[ys, xs]
                img[yd, xd][take] = img[ys, xs][take]
                filled[yd, xd] |= take


def point_colours(src, V, mask):
    """Colour per vertex in `mask` (0-255): mean of the points.ply points within POINT_RADIUS, else the nearest point."""
    pcd = o3d.io.read_point_cloud(str(src))
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors) * 255
    tree = cKDTree(pts)
    out = np.full((len(V), 3), 128.0)
    d, idx = tree.query(V[mask], k=8, distance_upper_bound=POINT_RADIUS)
    found = np.isfinite(d)
    n = found.sum(1)
    mean = (cols[np.minimum(idx, len(pts) - 1)] * found[..., None]).sum(1) / np.maximum(n, 1)[:, None]
    nearest = cols[tree.query(V[mask])[1]]
    out[mask] = np.where((n > 0)[:, None], mean, nearest)
    return out


def bake(walk, kfs, V, T, uv, lab, gains, adj, vcol, atlas_size):
    """Atlas image (uint8, rows top-down) plus the fraction of texels written. vcol: colour of the vertices of
    unseen faces (n, 3), baked into those faces' tiny charts."""
    atlas = np.zeros((atlas_size, atlas_size, 3), np.uint8)
    weight = np.zeros((atlas_size, atlas_size), np.float16)
    tri_uv = uv[T]
    fa, fb, _, _, oa, ob = adj
    # seam entries: face f gets the view of its neighbour g, weighted by distance to the shared edge
    sf = np.r_[fa, fb]
    sg = np.r_[fb, fa]
    so = np.r_[oa, ob]
    diff = (lab[sf] >= 0) & (lab[sg] >= 0) & (lab[sf] != lab[sg])
    sf, sg, so = sf[diff], sg[diff], so[diff]

    def paint(faces, k, img, opp=None):
        for loc, x, y, b in rasterize(tri_uv[faces]):
            f = faces[loc]
            if opp is None:
                w = np.ones(len(loc))
            else:
                w = np.clip(1 - b[np.arange(len(loc)), opp[loc]] / SEAM_RAMP, 0, 1)
                keep = w > 0
                loc, x, y, b, f, w = loc[keep], x[keep], y[keep], b[keep], f[keep], w[keep]
            pos = np.einsum("nc,ncd->nd", b, V[T[f]])
            u, v, _ = project(pos, kfs[k])
            blend_in(atlas, weight, x, y, sample(img, u, v) * gains[k], w)

    un = np.nonzero(lab < 0)[0]
    for loc, x, y, b in rasterize(tri_uv[un]):
        col = np.einsum("nc,ncd->nd", b, vcol[T[un[loc]]])
        blend_in(atlas, weight, x, y, col.astype(np.float32), np.ones(len(x)))
    by_view = {k: np.nonzero(lab == k)[0] for k in np.unique(lab[lab >= 0])}
    seam_view = lab[sg]
    for n, k in enumerate(sorted(by_view)):
        img = load_image(walk, kfs[k]).astype(np.float32)
        paint(by_view[k], k, img)
        e = np.nonzero(seam_view == k)[0]
        if len(e):
            paint(sf[e], k, img, opp=so[e])
        if n % 50 == 0:
            print(f"  bake {n}/{len(by_view)} views")
    filled = weight > 0
    coverage = filled.mean()
    if filled.any():
        empty_colour = atlas[filled].mean(0).astype(np.uint8)
        dilate(atlas, filled, 5)
        atlas[~filled] = empty_colour
    return atlas, coverage


# ---------------------------------------------------------------- writers


def write_glb(path, V, T, uv, jpeg):
    """glTF 2.0 binary, one textured unlit material. uv: (n, 2) with v up (OBJ convention)."""
    pos = V.astype("<f4")
    tex = np.stack([uv[:, 0], 1 - uv[:, 1]], 1).astype("<f4")  # glTF v runs down
    idx = T.astype("<u4").ravel()
    parts = [(idx.tobytes(), 34963), (pos.tobytes(), 34962), (tex.tobytes(), 34962), (jpeg, None)]
    blob, views, at = b"", [], 0
    for data, target in parts:
        blob += b"\0" * (-len(blob) % 4)
        at = len(blob)
        blob += data
        view = {"buffer": 0, "byteOffset": at, "byteLength": len(data)}
        if target:
            view["target"] = target
        views.append(view)
    doc = {
        "asset": {"version": "2.0", "generator": "texture.py"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 1, "TEXCOORD_0": 2}, "indices": 0, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0, "roughnessFactor": 1},
                       "doubleSided": True, "extensions": {"KHR_materials_unlit": {}}}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": 3, "mimeType": "image/jpeg"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5126, "count": len(pos), "type": "VEC3",
             "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
            {"bufferView": 2, "componentType": 5126, "count": len(tex), "type": "VEC2"}],
        "bufferViews": views, "buffers": [{"byteLength": len(blob)}],
    }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    Path(path).write_bytes(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
                           + struct.pack("<I4s", len(blob), b"BIN\0") + blob)


def write_obj(dst, V, T, uv):
    (dst / "model.mtl").write_text("newmtl mat\nKd 1 1 1\nmap_Kd texture.jpg\n")
    lines = ["mtllib model.mtl", "usemtl mat"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in V]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in uv]
    lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in T + 1]
    (dst / "model.obj").write_text("\n".join(lines) + "\n")


def jpeg_bytes(img, max_side=None, quality=88):
    im = Image.fromarray(img)
    if max_side and max(im.size) > max_side:
        im = im.resize((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------- main


def texture(walk, out, max_faces=300_000, atlas_size=8192):
    walk, out = Path(walk), Path(out)
    dst = out / "textured"
    dst.mkdir(exist_ok=True)
    t0 = time.time()

    def lap(msg):
        print(f"{msg} [{time.time() - t0:.0f} s]")

    verts, tris, normals = clean_mesh(out / "mesh.ply", max_faces)
    meta, kfs = load_views(walk, out)
    cand_s, cand_id = score_faces(walk, meta, kfs, verts, tris, normals)
    lap(f"scored: {(cand_id[:, 0] >= 0).mean():.1%} of faces seen by at least one keyframe")
    adj = adjacency(tris)
    lab = choose_views(cand_s, cand_id, adj[0], adj[1])
    seam = (lab[adj[0]] != lab[adj[1]]) & (lab[adj[0]] >= 0) & (lab[adj[1]] >= 0)
    lap(f"labels: {len(np.unique(lab[lab >= 0]))} views used, {seam.mean():.1%} of edges are seams")
    gains = seam_gains(walk, meta, kfs, verts, lab, adj)
    V, T, uv, unseen_v = parametrize(verts, tris, lab, adj, kfs, atlas_size)
    lap(f"atlas: {len(V)} vertices ({len(V) / len(verts):.2f}x split at chart borders)")
    vcol = point_colours(out / "points.ply", V, unseen_v)
    img, coverage = bake(walk, kfs, V, T, uv, lab, gains, adj, vcol, atlas_size)
    lap(f"baked: {coverage:.0%} of the atlas used, {(lab < 0).mean():.1%} of faces coloured from points")

    uv_out = np.stack([uv[:, 0] / atlas_size, 1 - uv[:, 1] / atlas_size], 1)  # v up
    uv_out = np.clip(uv_out, 0, 1)
    full = jpeg_bytes(img)
    (dst / "texture.jpg").write_bytes(full)
    write_glb(dst / "model.glb", V, T, uv_out, jpeg_bytes(img, GLB_MAX_TEX, 85))
    write_glb(dst / "model_mobile.glb", V, T, uv_out, jpeg_bytes(img, MOBILE_MAX_TEX, 80))
    write_obj(dst, V, T, uv_out)
    export_sketchup.write_dae(dst / "model.dae", "textured", export_sketchup.to_su(V), T, uv_out, "texture.jpg")
    export_sketchup.check_dae(dst / "model.dae")
    sizes = ", ".join(f"{n} {(dst / n).stat().st_size / 1e6:.1f} MB"
                      for n in ("model.glb", "model_mobile.glb", "model.obj", "texture.jpg", "model.dae"))
    lap(f"done: {len(T)} faces, {len(V)} vertices, atlas {atlas_size}^2; {sizes}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("walk_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--max-faces", type=int, default=300_000)
    ap.add_argument("--atlas", type=int, default=8192)
    a = ap.parse_args()
    texture(a.walk_dir, a.out_dir, a.max_faces, a.atlas)
