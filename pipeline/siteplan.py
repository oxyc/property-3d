"""2D site plan (contours, trees, walls) from a fused walk, for landscaping / architect conversations.

usage: siteplan.py <out_dir> [--interval 0.1] [--min-height 1.5]

Reads points.ply and ground.npz from out_dir (fuse.py output) and writes out_dir/siteplan/:
  siteplan.svg   vector, one <g class="layer-NAME"> per layer (ortho, hillshade, tint, grid, contours-minor,
                 contours-major, contour-labels, buildings, trees, annotations). The hillshade starts hidden;
                 toggle layers with CSS, e.g. `.layer-ortho {display: none}`, or in an SVG editor.
  siteplan.pdf   A3 landscape at a real scale (1:50 ... 1:500, the largest that fits), stated on the sheet
  siteplan.png   preview of the same page
  features.json  trees, building outlines, ground range. Coordinates: metres east / north of the walk start.

Contours come from the cleaned ground grid (export_sketchup.clean_grid) smoothed over ~0.4 m, labelled with
height above the lowest ground point. Everything above `--min-height` over that ground surface is split into
walls/buildings (vertical planar points forming long structures) and trees/shrubs (the rest, clustered into canopies).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
from contourpy import contour_generator  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import ConvexHull, cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))  # not on sys.path when run with -I
import export_sketchup  # noqa: E402

CELL = 0.2  # m, ground surface grid
SMOOTH = 2.0  # cells, gaussian sigma for contours and hillshade
FILL_MAX = 6.0  # m, ground under unscanned cells is taken from the nearest scanned cell up to this far
MIN_LINE = 1.0  # m, shorter contour fragments are dropped
RASTER = 0.1  # m, wall raster
WALL_CURV, WALL_NY, WALL_MIN_PTS, WALL_MIN_LEN = 0.02, 0.3, 3, 1.5
TREE_CELL, TREE_MIN_PTS, PEAK_WINDOW, PEAK_SIGMA = 0.2, 150, 1.6, 0.5  # m, count, m, m
TRUNK_BAND = (0.3, 1.3)  # m above ground
ORTHO_FILL = 0.2  # m, empty orthophoto pixels copy the nearest point up to this far
PAGE_W, PAGE_H = 420.0, 297.0  # mm, A3 landscape
MAP_BOX = (20.0, 52.0, 385.0, 233.0)  # mm: left, bottom, width, height of the area the map may use
SCALES = [20, 25, 50, 100, 125, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 5000]
TINT = LinearSegmentedColormap.from_list("tint", ["#7fb3d5", "#e8e4a0", "#c9885a"])
BROWN, GREEN = "#7a4210", "#2e7d32"


# ---------------------------------------------------------------- ground surface


def ground_surface(g):
    """ground.npz -> (cleaned grid, smoothed grid, west, north). NaN where nothing was scanned."""
    a = export_sketchup.clean_grid(export_sketchup.median_resample(g, CELL), CELL)
    a, west, north = export_sketchup.crop(a, float(g["west"]), float(g["north"]), CELL)
    ok = ~np.isnan(a)
    den = ndimage.gaussian_filter(ok.astype(float), SMOOTH)
    num = ndimage.gaussian_filter(np.where(ok, a, 0.0), SMOOTH)
    smooth = np.where(ok & (den > 0.2), num / np.maximum(den, 1e-9), np.nan)
    return a, smooth, west, north


def trace(z, west, north, cell, level):
    """Lines (list of (n, 2) east/north arrays) where the north-up grid z crosses `level`; NaN cells are holes."""
    z = np.ma.masked_invalid(z[::-1])
    xs = west + (np.arange(z.shape[1]) + 0.5) * cell
    ys = north - (z.shape[0] - 1 - np.arange(z.shape[0]) + 0.5) * cell
    return contour_generator(xs, ys, z).lines(level)


def length(line):
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def contour_levels(smooth, interval):
    """Levels above the lowest ground point (0): all multiples of interval up to the highest ground point."""
    top = np.nanmax(smooth) - np.nanmin(smooth)
    return [round(k * interval, 6) for k in range(1, int(top / interval + 1e-9) + 1)]


def contours(smooth, west, north, interval):
    """{level above lowest ground: [lines]} for every multiple of interval."""
    rel = smooth - np.nanmin(smooth)
    out = {}
    for lv in contour_levels(smooth, interval):
        lines = [ln for ln in trace(rel, west, north, CELL, lv) if len(ln) >= 4 and length(ln) >= MIN_LINE]
        if lines:
            out[lv] = lines
    return out


def hillshade(z, cell, exaggeration=3.0):
    gr, gc = np.gradient(z, cell)  # rows run south
    n = np.stack([-gc * exaggeration, gr * exaggeration, np.ones_like(z)])
    n /= np.linalg.norm(n, axis=0)
    az, alt = np.radians(315), np.radians(45)
    light = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])
    return np.clip(np.tensordot(light, n, 1), 0, 1)


# ---------------------------------------------------------------- height above ground


def height_above_ground(pts, a, west, north):
    """Height of every world point (X east, Y up, Z south) over the cleaned ground grid; NaN outside it."""
    dist, (ir, ic) = ndimage.distance_transform_edt(np.isnan(a), return_indices=True)
    filled = np.where(np.isnan(a) & (dist * CELL <= FILL_MAX), a[ir, ic], a)
    r, c = ((north + pts[:, 2]) / CELL).astype(int), ((pts[:, 0] - west) / CELL).astype(int)
    inside = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
    ground = np.full(len(pts), np.nan)
    ground[inside] = filled[r[inside], c[inside]]
    return pts[:, 1] - ground, ground


def simplify(ring, tol):
    """Douglas-Peucker on a closed ring (first point repeated at the end)."""
    def rdp(p):
        if len(p) < 3:
            return p
        d = p[-1] - p[0]
        n = np.hypot(*d)
        dist = np.hypot(*(p - p[0]).T) if n < 1e-9 else np.abs(d[0] * (p[:, 1] - p[0, 1]) - d[1] * (p[:, 0] - p[0, 0])) / n
        k = int(dist.argmax())
        if dist[k] <= tol:
            return np.array([p[0], p[-1]])
        return np.vstack([rdp(p[:k + 1])[:-1], rdp(p[k:])])
    ring = ring[:-1]
    k = int(np.hypot(*(ring - ring[0]).T).argmax())
    return np.vstack([rdp(ring[:k + 1])[:-1], rdp(np.vstack([ring[k:], ring[:1]]))])


# ---------------------------------------------------------------- walls / buildings


def detect_walls(pts, h, min_h):
    """Long, tall, vertical planar structures. Returns [{outline (n, 2) east/north, height, length}] and a
    point mask of their (dilated) footprint, to keep them out of the tree search."""
    hi = pts[h > min_h]
    walls, footprint = [], np.zeros(len(pts), bool)
    if len(hi) < 100:
        return walls, footprint
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hi))
    pcd = pcd.voxel_down_sample(0.05)
    pcd.estimate_covariances(o3d.geometry.KDTreeSearchParamHybrid(0.2, 30))
    w, v = np.linalg.eigh(np.asarray(pcd.covariances))
    curv = w[:, 0] / np.maximum(w.sum(1), 1e-12)
    vert = (curv < WALL_CURV) & (np.abs(v[:, 1, 0]) < WALL_NY)  # smallest-eigenvalue direction = surface normal
    p = np.asarray(pcd.points)[vert]
    if len(p) < WALL_MIN_PTS:
        return walls, footprint

    west, top = p[:, 0].min() - 0.5, (-p[:, 2]).max() + 0.5
    nc, nr = int((p[:, 0].max() + 0.5 - west) / RASTER) + 1, int((top + p[:, 2].max() + 0.5) / RASTER) + 1
    r, c = ((top + p[:, 2]) / RASTER).astype(int), ((p[:, 0] - west) / RASTER).astype(int)
    count = np.zeros((nr, nc), int)
    np.add.at(count, (r, c), 1)
    mask = ndimage.binary_closing(count >= WALL_MIN_PTS, np.ones((3, 3)), iterations=2)
    lab, n = ndimage.label(mask, np.ones((3, 3)))
    keep = np.zeros((nr, nc), bool)
    # wall height per raster cell: the tallest of all high points there
    r2, c2 = ((top + hi[:, 2]) / RASTER).astype(int), ((hi[:, 0] - west) / RASTER).astype(int)
    ok2 = (r2 >= 0) & (r2 < nr) & (c2 >= 0) & (c2 < nc)
    tall = np.zeros((nr, nc))
    np.maximum.at(tall, (r2[ok2], c2[ok2]), h[h > min_h][ok2])
    for k in range(1, n + 1):
        rr, cc = np.nonzero(lab == k)
        if len(rr) < 3:
            continue
        xy = np.stack([cc, rr], 1) * RASTER
        span = float(np.ptp(xy @ np.linalg.eigh(np.cov(xy.T))[1][:, 1]))  # extent along the main axis
        height = float(np.percentile(tall[rr, cc], 95))
        if span < WALL_MIN_LEN or height < min_h + 0.2:
            continue
        sub = np.pad(lab[rr.min():rr.max() + 1, cc.min():cc.max() + 1] == k, 2)
        sub = ndimage.binary_dilation(sub).astype(float)  # thin walls get a body
        ring = max(trace(sub, west + (cc.min() - 2) * RASTER, top - (rr.min() - 2) * RASTER, RASTER, 0.5), key=len)
        walls.append(dict(outline=simplify(ring, 0.08), height=height, length=span))
        keep |= lab == k
    if walls:  # keep the walls themselves out of the tree search
        keep = ndimage.binary_dilation(keep, iterations=4)
        r3, c3 = ((top + pts[:, 2]) / RASTER).astype(int), ((pts[:, 0] - west) / RASTER).astype(int)
        ok3 = (r3 >= 0) & (r3 < nr) & (c3 >= 0) & (c3 < nc)
        footprint[ok3] = keep[r3[ok3], c3[ok3]]
    return walls, footprint


# ---------------------------------------------------------------- trees / shrubs


def hull_of(xy):
    if len(xy) < 3 or np.ptp(xy, axis=0).min() < 1e-3:
        c = xy.mean(0)
        t = np.linspace(0, 2 * np.pi, 13)
        return c + 0.1 * np.stack([np.cos(t), np.sin(t)], 1)
    return xy[ConvexHull(xy).vertices]


def detect_trees(pts, cols, h, skip, min_h):
    """Canopy clusters of the points above min_h that are not walls. One tree per height peak; the canopy is the
    outline of its dense cells, the trunk the densest spot of the 0.3-1.3 m band below it (else the canopy top).
    `green` is the mean excess-green (2G - R - B, 0-1 colours) of the cluster: low means probably not vegetation."""
    sel = (h > min_h) & (h < 15) & ~skip
    p, hp, cp = pts[sel], h[sel], cols[sel]
    exg = 2 * cp[:, 1] - cp[:, 0] - cp[:, 2]
    if len(p) < TREE_MIN_PTS:
        return []
    east, north = p[:, 0], -p[:, 2]
    west, top = east.min() - 1, north.max() + 1
    nc, nr = int((east.max() + 1 - west) / TREE_CELL) + 1, int((top - north.min() + 1) / TREE_CELL) + 1
    r, c = ((top - north) / TREE_CELL).astype(int), ((east - west) / TREE_CELL).astype(int)
    cell = r * nc + c
    count = np.bincount(cell, minlength=nr * nc).reshape(nr, nc)
    chm = np.zeros(nr * nc)
    np.maximum.at(chm, cell, hp)
    chm = chm.reshape(nr, nc)
    mask = ndimage.binary_closing(count >= 3, np.ones((3, 3)), iterations=1)
    lab, n = ndimage.label(mask, np.ones((3, 3)))

    smooth = ndimage.gaussian_filter(np.where(mask, chm, 0), PEAK_SIGMA / TREE_CELL)
    win = 2 * int(round(PEAK_WINDOW / TREE_CELL / 2)) + 1
    peaks = mask & (smooth == ndimage.maximum_filter(smooth, win)) & (smooth > 0)
    pk_lab, npk = ndimage.label(peaks, np.ones((3, 3)))
    # cells go to the nearest peak of their own component
    owner = np.zeros((nr, nc), int)
    for k in range(1, n + 1):
        comp = lab == k
        ids = np.unique(pk_lab[comp & (pk_lab > 0)])
        if len(ids) == 0:
            rr, cc = np.nonzero(comp)
            ids = [npk + 1 + k]
            pk_lab[rr[np.argmax(chm[rr, cc])], cc[np.argmax(chm[rr, cc])]] = ids[0]
        if len(ids) == 1:
            owner[comp] = ids[0]
            continue
        seeds = np.isin(pk_lab, ids)
        _, (ir, ic) = ndimage.distance_transform_edt(~seeds, return_indices=True)
        owner[comp] = pk_lab[ir, ic][comp]
    pt_owner = owner[r, c]

    low = (h > TRUNK_BAND[0]) & (h < TRUNK_BAND[1])
    low_xy = np.stack([pts[low, 0], -pts[low, 2]], 1)
    low_tree = cKDTree(low_xy) if len(low_xy) else None
    trees = []
    for k in np.unique(pt_owner[pt_owner > 0]):
        m = pt_owner == k
        if m.sum() < TREE_MIN_PTS:
            continue
        xy = np.stack([east[m], north[m]], 1)
        own = ndimage.binary_closing(owner == k, np.ones((3, 3)), iterations=2)
        own = ndimage.binary_fill_holes(own)
        rr, cc = np.nonzero(own)
        sub = np.pad(own[rr.min():rr.max() + 1, cc.min():cc.max() + 1], 1).astype(float)
        rings = trace(sub, west + (cc.min() - 1) * TREE_CELL, top - (rr.min() - 1) * TREE_CELL, TREE_CELL, 0.5)
        hull = simplify(max(rings, key=len), 0.1) if rings and len(max(rings, key=len)) >= 4 else hull_of(xy)
        area = float(own.sum() * TREE_CELL**2)
        top_cell = np.argmax(hp[m])
        top_xy = xy[top_cell]
        centre = xy.mean(0)
        radius = max(float(np.sqrt(area / np.pi)), 0.1)
        trunk, found = top_xy, False
        if low_tree is not None:
            idx = low_tree.query_ball_point(centre, radius + 0.3)
            if len(idx) >= 30:
                q = low_xy[idx]
                hist, ex, ey = np.histogram2d(q[:, 0], q[:, 1], bins=[np.arange(q[:, 0].min(), q[:, 0].max() + 0.2, 0.1),
                                                                      np.arange(q[:, 1].min(), q[:, 1].max() + 0.2, 0.1)])
                hist = ndimage.uniform_filter(hist, 3)
                i, j = np.unravel_index(hist.argmax(), hist.shape)
                near = q[np.hypot(q[:, 0] - (ex[i] + 0.05), q[:, 1] - (ey[j] + 0.05)) < 0.3]
                if len(near) >= 20 and len(near) / len(q) > 0.25:
                    trunk, found = near.mean(0), True
        trees.append(dict(trunk=trunk, trunk_found=found, height=float(np.percentile(hp[m], 98)), radius=radius,
                          area=area, hull=hull, points=int(m.sum()), green=float(exg[m].mean())))
    trees.sort(key=lambda t: (-t["trunk"][1], t["trunk"][0]))
    return trees


# ---------------------------------------------------------------- drawing


def nice_length(m):
    for step in (1, 2, 5):
        for e in (-1, 0, 1, 2, 3):
            if step * 10.0**e >= m:
                return step * 10.0**e
    return m


def pick_scale(w, h):
    for s in SCALES:
        if w * 1000 / s <= MAP_BOX[2] and h * 1000 / s <= MAP_BOX[3]:
            return s
    return SCALES[-1]


def place_label(ax, line, text, **kw):
    """Upright text at the middle of a line, rotated along it."""
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    k = int(np.searchsorted(np.cumsum(seg), seg.sum() / 2))
    k = min(k, len(line) - 2)
    d = line[min(k + 2, len(line) - 1)] - line[max(k - 1, 0)]
    ang = np.degrees(np.arctan2(d[1], d[0]))
    ang = ang + 180 if ang > 90 or ang < -90 else ang
    ax.text(*line[k], text, rotation=ang, rotation_mode="anchor", ha="center", va="center", fontsize=5,
            color=BROWN, bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.75), gid="layer:contour-labels", **kw)


def draw(title, pts, cols, smooth, west, north, cont, trees, walls, base, interval):
    nr, nc = smooth.shape
    ext = (west - 1, west + nc * CELL + 1, north - nr * CELL - 1, north + 1)  # metres, padded
    for t in trees:  # features may lie beyond the ground grid
        ext = (min(ext[0], t["hull"][:, 0].min() - 0.5), max(ext[1], t["hull"][:, 0].max() + 0.5),
               min(ext[2], t["hull"][:, 1].min() - 0.5), max(ext[3], t["hull"][:, 1].max() + 0.5))
    for wl in walls:
        ext = (min(ext[0], wl["outline"][:, 0].min() - 0.5), max(ext[1], wl["outline"][:, 0].max() + 0.5),
               min(ext[2], wl["outline"][:, 1].min() - 0.5), max(ext[3], wl["outline"][:, 1].max() + 0.5))
    x0, x1, y0, y1 = np.floor(ext[0]), np.ceil(ext[1]), np.floor(ext[2]), np.ceil(ext[3])
    w_m, h_m = x1 - x0, y1 - y0
    scale = pick_scale(w_m, h_m)
    mm = 1 / 25.4
    map_w, map_h = w_m * 1000 / scale, h_m * 1000 / scale
    left = MAP_BOX[0] + (MAP_BOX[2] - map_w) / 2
    bottom = MAP_BOX[1] + (MAP_BOX[3] - map_h) / 2

    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans", "hatch.linewidth": 0.4})
    fig = plt.figure(figsize=(PAGE_W * mm, PAGE_H * mm))
    ax = fig.add_axes([left / PAGE_W, bottom / PAGE_H, map_w / PAGE_W, map_h / PAGE_H])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6, length=2, width=0.4)
    for s in ax.spines.values():
        s.set_linewidth(0.5)

    imext = (west, west + nc * CELL, north - nr * CELL, north)
    # --- base layers
    px = max(export_sketchup.ORTHO_PX, max(w_m, h_m) / export_sketchup.ORTHO_MAX)
    img = export_sketchup.orthophoto(pts, cols, x0, y1, w_m, h_m, fill=ORTHO_FILL)  # a plan shouldn't invent ground
    empty = (img == np.array((110, 110, 100), np.uint8)).all(2)
    ax.imshow(img, extent=(x0, x0 + img.shape[1] * px, y1 - img.shape[0] * px, y1), alpha=np.where(empty, 0, 0.5),
              interpolation="bilinear", zorder=1, gid="layer:ortho")
    shade = hillshade(smooth, CELL)
    ax.imshow(np.ma.masked_invalid(np.where(np.isnan(smooth), np.nan, shade)), cmap="gray", vmin=0.2, vmax=1.0, alpha=0.6,
              extent=imext, interpolation="bilinear", zorder=1.5, gid="layer:hillshade")
    rel = smooth - base
    tint_top = max(float(np.nanpercentile(rel, 98)), 1e-6)
    tint = TINT(np.clip(rel / tint_top, 0, 1))
    tint[..., 3] = np.where(np.isnan(smooth), 0, 0.45)
    ax.imshow(tint, extent=imext, interpolation="bilinear", zorder=2, gid="layer:tint")

    # --- grid
    step = next(s for s in (1, 2, 5, 10, 20, 50, 100, 200) if s * 1000 / scale >= 40)
    gx, gy = np.arange(np.ceil(x0 / step) * step, x1 + 1e-9, step), np.arange(np.ceil(y0 / step) * step, y1 + 1e-9, step)
    ax.add_collection(LineCollection([[(x, y0), (x, y1)] for x in gx] + [[(x0, y), (x1, y)] for y in gy],
                                     colors="#555", linewidths=0.2, alpha=0.5, zorder=3, gid="layer:grid"))
    ax.set_xticks(gx)
    ax.set_yticks(gy)
    ax.set_xlabel(f"east of walk start (m), grid {step:g} m", fontsize=6)
    ax.set_ylabel("north of walk start (m)", fontsize=6)

    # --- contours
    minor = [ln for lv, lines in cont.items() if not np.isclose(lv % 1.0, 0) and not np.isclose(lv % 1.0, 1) for ln in lines]
    major = {lv: lines for lv, lines in cont.items() if np.isclose(lv % 1.0, 0) or np.isclose(lv % 1.0, 1)}
    ax.add_collection(LineCollection(minor, colors=BROWN, linewidths=0.4, zorder=4, gid="layer:contours-minor"))
    ax.add_collection(LineCollection([ln for ls in major.values() for ln in ls], colors=BROWN, linewidths=1.0,
                                     zorder=4.1, gid="layer:contours-major"))
    for lv, lines in major.items():
        for ln in lines:
            if length(ln) >= 2:
                place_label(ax, ln, f"{lv:g} m")

    # --- features
    for i, wl in enumerate(walls, 1):
        ax.add_patch(Polygon(wl["outline"], closed=True, fc="#444", ec="#111", lw=0.7, alpha=0.7, zorder=5, gid="layer:buildings"))
        c = wl["outline"].mean(0)
        ax.text(c[0], c[1], f"B{i}  {wl['height']:.1f} m", fontsize=5.5, ha="center", va="bottom", color="#111",
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8), zorder=7, gid="layer:buildings")
    for i, t in enumerate(trees, 1):
        ax.add_patch(Polygon(t["hull"], closed=True, fc=GREEN, ec=GREEN, lw=0.6, alpha=0.2, zorder=5, gid="layer:trees"))
        ax.add_patch(Polygon(t["hull"], closed=True, fill=False, ec=GREEN, lw=0.6, ls="-" if t["trunk_found"] else "--",
                             zorder=5.1, gid="layer:trees"))
        ax.plot(*t["trunk"], marker="+" if t["trunk_found"] else "x", ms=3.5, mew=0.7, color="#111", zorder=6, gid="layer:trees")
        ax.text(t["trunk"][0] + 0.08, t["trunk"][1] + 0.08, f"T{i}  {t['height']:.1f} m", fontsize=5.5, color="#14501a",
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.75), zorder=7, gid="layer:trees")
    ax.plot(0, 0, marker="o", ms=4, mfc="white", mec="#d00", mew=1, zorder=8, gid="layer:annotations")
    ax.text(0.15, 0.15, "walk start", fontsize=5.5, color="#d00", zorder=8, gid="layer:annotations")

    # --- sheet annotations, in mm on the page
    ov = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ov.set_xlim(0, PAGE_W)
    ov.set_ylim(0, PAGE_H)
    ov.axis("off")
    kw = dict(gid="layer:annotations")
    ov.text(20, 40, "Site plan", fontsize=18, weight="bold", va="center", **kw)
    ov.text(20, 31, title, fontsize=8, va="center", color="#444", **kw)
    ov.text(20, 24, f"Scale 1:{scale} on A3 (landscape). Contours every {interval:g} m, labelled every 1 m, as height above the "
                    f"lowest\nscanned ground point (0 m; {base:+.2f} m in the walk frame). "
                    "Trees and walls are estimated from a\nhandheld phone scan: canopy hulls, trunk positions and heights are approximate.",
            fontsize=6, va="top", color="#444", linespacing=1.5, **kw)

    lx, ly = 150, 44
    ov.text(lx, ly, "Legend", fontsize=7, weight="bold", va="center", **kw)
    rows = [("minor contour", dict(color=BROWN, lw=0.4)), ("major contour (1 m)", dict(color=BROWN, lw=1.0))]
    for k, (txt, st) in enumerate(rows):
        y = ly - 6 - 5.5 * k
        ov.plot([lx, lx + 10], [y, y], solid_capstyle="butt", **st, **kw)
        ov.text(lx + 13, y, txt, fontsize=6, va="center", **kw)
    y = ly - 6 - 5.5 * 2
    ov.add_patch(Polygon([(lx, y - 1.6), (lx + 4, y + 1.8), (lx + 10, y + 0.5), (lx + 8, y - 1.8)], fc=GREEN, ec=GREEN, alpha=0.3, lw=0.6, **kw))
    ov.plot(lx + 5, y, "+", ms=3.5, mew=0.7, color="#111", **kw)
    ov.text(lx + 13, y, "tree / shrub canopy, trunk, height (dashed: trunk not seen)", fontsize=6, va="center", **kw)
    y = ly - 6 - 5.5 * 3
    ov.add_patch(Rectangle((lx, y - 1.6), 10, 3.2, fc="#444", ec="#111", alpha=0.7, lw=0.6, **kw))
    ov.text(lx + 13, y, "wall / building outline, height", fontsize=6, va="center", **kw)
    y = ly - 6 - 5.5 * 4
    ov.imshow(np.linspace(0, 1, 64)[None, :], cmap=TINT, alpha=0.5, extent=(lx, lx + 10, y - 1.6, y + 1.6), aspect="auto", **kw)
    ov.text(lx + 13, y, f"ground height tint, 0 to {tint_top:.1f} m and above", fontsize=6, va="center", **kw)
    ov.set_xlim(0, PAGE_W)
    ov.set_ylim(0, PAGE_H)

    bar = nice_length(min(map_w * scale / 1000 / 3, 20))
    bar = min(bar, nice_length(60 * scale / 1000)) if bar * 1000 / scale > 80 else bar
    bx, by, bw = 300, 30, bar * 1000 / scale
    for k in range(4):
        ov.add_patch(Rectangle((bx + k * bw / 4, by), bw / 4, 2.5, fc="black" if k % 2 == 0 else "white", ec="black", lw=0.5, **kw))
    for f in (0, 0.5, 1):
        ov.text(bx + f * bw, by - 2.5, f"{f * bar:g}", fontsize=6, ha="center", va="top", **kw)
    ov.text(bx + bw + 4, by + 1.2, "m", fontsize=6, va="center", **kw)
    ov.text(bx, by + 7, f"1:{scale}", fontsize=8, weight="bold", **kw)

    nx, ny = 395, 32
    ov.add_patch(Polygon([(nx, ny + 12), (nx - 3.5, ny), (nx, ny + 3), (nx + 3.5, ny)], fc="black", ec="black", lw=0.5, **kw))
    ov.text(nx, ny + 14, "N", fontsize=9, weight="bold", ha="center", **kw)

    return fig, scale


def siteplan(out, interval=0.1, min_h=1.5):
    out = Path(out)
    dst = out / "siteplan"
    dst.mkdir(exist_ok=True)
    pcd = o3d.io.read_point_cloud(str(out / "points.ply"))
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors)
    g = np.load(out / "ground.npz")
    a, smooth, west, north = ground_surface(g)
    base = float(np.nanmin(smooth))
    cont = contours(smooth, west, north, interval)
    if len(cont) < 3:
        print(f"siteplan: only {len(cont)} contour levels, the ground is flat; try a smaller --interval")

    h, _ = height_above_ground(pts, a, west, north)
    h = np.nan_to_num(h, nan=-1.0)
    walls, footprint = detect_walls(pts, h, min_h)
    trees = detect_trees(pts, cols, h, footprint, min_h)
    print(f"siteplan: ground {np.nanmin(a):.2f} to {np.nanmax(a):.2f} m, {sum(map(len, cont.values()))} contour lines, "
          f"{len(trees)} trees/shrubs, {len(walls)} walls/buildings")

    su = np.stack([pts[:, 0], -pts[:, 2], pts[:, 1]], 1)
    fig, scale = draw(out.name, su, cols, smooth, west, north, cont, trees, walls, base, interval)
    for art in fig.findobj(lambda o: o.get_gid() == "layer:hillshade"):
        art.set_visible(False)
    fig.savefig(dst / "siteplan.pdf")
    fig.savefig(dst / "siteplan.png", dpi=150)
    for art in fig.findobj(lambda o: o.get_gid() == "layer:hillshade"):
        art.set_visible(True)
    fig.savefig(dst / "siteplan.svg")
    plt.close(fig)
    svg = (dst / "siteplan.svg").read_text()
    svg = re.sub(r'id="layer:([a-z-]+)"', r'class="layer-\1"', svg)
    svg = svg.replace("</defs>", "</defs><style>.layer-hillshade { display: none }</style>", 1)
    (dst / "siteplan.svg").write_text(svg)

    feats = dict(
        frame="metres east (x) / north (y) of the walk start",
        scale=f"1:{scale} on A3 landscape",
        ground=dict(min_y=float(np.nanmin(a)), max_y=float(np.nanmax(a)), range_m=float(np.nanmax(a) - np.nanmin(a)),
                    contour_interval_m=interval, contour_base_y=base),
        min_feature_height_m=min_h,
        trees=[dict(id=i, x=float(t["trunk"][0]), y=float(t["trunk"][1]), trunk_observed=bool(t["trunk_found"]),
                    height_m=round(t["height"], 2), canopy_radius_m=round(t["radius"], 2), canopy_area_m2=round(t["area"], 2),
                    points=t["points"], greenness=round(t["green"], 3), canopy=np.round(t["hull"], 2).tolist()) for i, t in enumerate(trees, 1)],
        buildings=[dict(id=i, height_m=round(w["height"], 2), length_m=round(w["length"], 2),
                        outline=np.round(w["outline"], 2).tolist()) for i, w in enumerate(walls, 1)],
        notes="Heuristic: heights are above the lowest-point ground grid, trunks are the densest spot 0.3-1.3 m up "
              "(biased towards the scanned side), walls are long vertical planar structures only.",
    )
    (dst / "features.json").write_text(json.dumps(feats, indent=1))
    print(f"siteplan: 1:{scale}, wrote {dst}")
    return feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--interval", type=float, default=0.1, help="minor contour interval, m")
    ap.add_argument("--min-height", type=float, default=1.5, help="m above ground for trees / walls")
    a = ap.parse_args()
    siteplan(a.out_dir, a.interval, a.min_height)
