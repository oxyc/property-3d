"""Independent accuracy audit of a reconstructed walk against national aerial elevation data.

usage: audit_ide.py <out_dir> --dsm <DSM.tif> --dtm <DTM.tif> [--ortho <RGB.tif>] [--report <dir>]

out_dir is a georef.py output folder (georef.json, georef_points.ply, optionally ground.tif and
poses.npz / optimize.json). The reference is a high-resolution DSM (surface incl. trees and roofs), a
coarser DTM (bare ground) and optionally an orthophoto, all GeoTIFFs in the same projected CRS as
georef.json (read from the files, only checked for a match with georef.json).

The GPS georef is translation-only and several metres off, so the audit first FITS a rigid placement
(E/N translation + yaw about the model centroid + vertical offset) of the model onto the reference:
  - model ground cells are compared with the DTM, model hard structures (cells well above the ground
    that are not green) with the DSM; model-only things (a newer house) and vegetation that grew or
    was cut are handled by a robust (Geman-McClure / Cauchy) loss, not by hand-picked masks;
  - coarse grid search over translation and yaw on a blurred reference, then robust least squares from
    the best few candidates; confidence from the cost surface (is there a single clear minimum?) and a
    spatial block bootstrap.
After the fit it measures residuals on open ground (DTM) and smooth hard surfaces (DSM), and looks for
DRIFT: a tilt plane and a quadratic bowl in the ground residual, residual along the walk (path
distance, if poses.npz exists), and a translation fit per 10 m tile and per walk segment (a spread of
offsets well above their noise means horizontal drift or a torn model).

Writes to the report dir (default out_dir): audit.json, georef_ide.json (improved placement; georef.json
is never touched), audit_residual.png, audit_tiles.png, audit_distance.png.

Heuristics worth knowing: cell statistics use 0.25 m cells; tile/segment uncertainties come from the
Gauss-Newton covariance inflated x8 because neighbouring cells are strongly correlated; a model seen from
the ground under-reports roofs, so the hard-surface residual is biased low; the DTM is itself only
~0.1-0.3 m good, especially under trees.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from scipy import ndimage
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree

CELL = 0.25  # model grid cell, m
MIN_PTS = 3  # points per cell
MAX_PTS = 3_000_000
GM_C = 0.35  # robust loss scale, m
NAN_RES = 2.0  # residual used where the shifted model leaves the reference coverage
STRUCT_H = 1.5  # model cell this far above the local ground counts as a structure/vegetation top
YAW_ARC = 10.0  # the optimiser works in "yaw arc" units: yaw_rad * 10 m, so all four parameters are ~metres
CORR_INFLATE = 50.0  # variance inflation: cells are correlated and the DTM is 1 m; calibrated on the synthetic test
CHANGED_M = 0.75  # model ground this far above the DTM, over a compact patch, counts as changed since the flight
CANOPY_ND = 3.0  # DSM this far above the DTM = canopy, where the DTM is an interpolation
MIN_OPEN_CELLS = 1500  # ~95 m2 of judgeable open ground needed before tilt/drift are reported
MAX_AMBIG_M, MAX_AMBIG_DEG = 3.0, 3.0  # competing minima further apart than this make the placement ambiguous
ROUGH_MAX = 0.08  # reference texture (m) above which a DSM cell is treated as vegetation


# ---------------------------------------------------------------- model

class Cells:
    """Fit cells: x, y, z (ground height for g cells, surface top for the others), g, w, with a yaw pivot."""

    def __init__(self, x, y, z, g, w, pivot, extra=None):
        self.x, self.y, self.z, self.g, self.w, self.pivot = x, y, z, g, w, pivot
        self.extra = extra or {}

    def sub(self, idx):
        return Cells(self.x[idx], self.y[idx], self.z[idx], self.g[idx], self.w[idx], self.pivot,
                     {k: v[idx] for k, v in self.extra.items()})

    def __len__(self):
        return len(self.x)


def load_points(out):
    import open3d as o3d
    info = json.loads((out / "georef.json").read_text())
    pcd = o3d.io.read_point_cloud(str(out / "georef_points.ply"))
    p = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors) if pcd.has_colors() else np.full((len(p), 3), 0.5)
    k = max(1, len(p) // MAX_PTS)
    p, rgb = p[::k], rgb[::k]
    return info, p + np.array(info["offset"]), rgb


def _nearest_fill(a):
    bad = np.isnan(a)
    if not bad.any():
        return a
    if bad.all():
        raise SystemExit("no ground cells")
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def _fallback_ground(z05, spread, size=48):
    """No ground.tif: ground = flat cells that sit on the morphological opening of the lowest heights."""
    big = np.where(np.isnan(z05), 1e9, z05)
    lo = ndimage.minimum_filter(big, size=size)
    lo[lo > 1e8] = -1e9
    op = ndimage.maximum_filter(lo, size=size)
    ok = ~np.isnan(z05) & (op > -1e8) & (z05 - op < 0.3) & (spread < 0.2)
    return np.where(ok, z05, np.nan)


def build_cells(p, rgb, ground_path=None):
    """Grid the points: cells with ground height (from ground.tif, else a low-envelope fallback) and structure tops."""
    E, N, Z = p.T
    e0, n0 = E.min(), N.min()
    ix, iy = ((E - e0) // CELL).astype(np.int64), ((N - n0) // CELL).astype(np.int64)
    nx, ny = ix.max() + 1, iy.max() + 1
    cid = iy * nx + ix
    order = np.lexsort((Z, cid))
    cs, zs = cid[order], Z[order]
    uid, start, cnt = np.unique(cs, return_index=True, return_counts=True)
    exg_pt = (2 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2])[order]
    exg_c = np.bincount(np.repeat(np.arange(len(uid)), cnt), exg_pt) / cnt
    z05 = zs[start + ((cnt - 1) * 0.05).astype(int)]
    z95 = zs[start + ((cnt - 1) * 0.95).astype(int)]
    ok = cnt >= MIN_PTS
    gi, gj = (uid // nx)[ok], (uid % nx)[ok]
    G05, G95, GEXG = (np.full((ny, nx), np.nan) for _ in range(3))
    G05[gi, gj], G95[gi, gj], GEXG[gi, gj] = z05[ok], z95[ok], exg_c[ok]
    cx = e0 + (np.arange(nx) + 0.5) * CELL
    cy = n0 + (np.arange(ny) + 0.5) * CELL
    if ground_path is not None and Path(ground_path).exists():
        with rasterio.open(ground_path) as ds:
            t, a = ds.transform, ds.read(1).astype(np.float64)
            a[(a == ds.nodata) | ~np.isfinite(a)] = np.nan
        col = np.floor((cx - t.c) / t.a).astype(int)
        row = np.floor((t.f - cy) / -t.e).astype(int)
        inside = ((row >= 0) & (row < a.shape[0]))[:, None] & ((col >= 0) & (col < a.shape[1]))[None, :]
        gg = np.where(inside, a[np.clip(row, 0, a.shape[0] - 1)[:, None], np.clip(col, 0, a.shape[1] - 1)[None, :]], np.nan)
        GG = np.where(np.isnan(G05), np.nan, gg)
        source = "ground.tif"
    else:
        GG = _fallback_ground(G05, G95 - G05)
        source = "low-envelope fallback (no ground.tif)"
    near = _nearest_fill(GG)
    height = G95 - near
    struct = ~np.isnan(G95) & (height > STRUCT_H) & (GEXG < 0.1)
    ground = ~np.isnan(GG)
    X, Y = np.meshgrid(cx, cy)
    pivot = (float(np.mean(X[ground | struct])), float(np.mean(Y[ground | struct])))
    gx, gy = X[ground], Y[ground]
    sx, sy = X[struct], Y[struct]
    c = Cells(np.r_[gx, sx], np.r_[gy, sy], np.r_[GG[ground], G95[struct]], np.r_[np.ones(len(gx), bool), np.zeros(len(sx), bool)],
              np.r_[np.ones(len(gx)), np.full(len(sx), 2.0)], pivot,
              {"height": np.r_[height[ground], height[struct]], "exg": np.r_[GEXG[ground], GEXG[struct]]})
    return c, source


# ---------------------------------------------------------------- reference

def _read_window(path, bounds):
    with rasterio.open(path) as ds:
        win = from_bounds(*bounds, ds.transform).round_offsets().round_lengths()
        try:
            win = win.intersection(Window(0, 0, ds.width, ds.height))
        except Exception as e:
            raise SystemExit(f"{path} does not cover the model area: {e}")
        a = ds.read(1, window=win).astype(np.float32)
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
        return a, ds.window_transform(win), str(ds.crs), ds.res[0]


def _sample(a, t, x, y):
    return ndimage.map_coordinates(a, [(t.f - y) / -t.e - 0.5, (x - t.c) / t.a - 0.5], order=1, mode="constant", cval=np.nan)


def _nan_blur(a, sigma):
    v = np.isfinite(a)
    num = ndimage.gaussian_filter(np.where(v, a, 0.0), sigma)
    den = ndimage.gaussian_filter(v.astype(float), sigma)
    return np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)


class Ref:
    """DSM block-averaged to ~0.3 m with the DTM, nDSM, roughness and a blurred DSM on the same grid."""

    def __init__(self, dsm, dtm, bounds):
        a, t, self.crs, res = _read_window(dsm, bounds)
        f = max(1, round(0.3 / res))
        h, w = a.shape[0] // f * f, a.shape[1] // f * f
        b = a[:h, :w].reshape(h // f, f, w // f, f)
        v = np.isfinite(b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = np.where(v.sum((1, 3)) > f * f / 2, np.where(v, b, 0).sum((1, 3)) / np.maximum(v.sum((1, 3)), 1), np.nan)
        self.s = res * f
        self.t = rasterio.Affine(self.s, 0, t.c, 0, -self.s, t.f)
        ny, nx = c.shape
        X = self.t.c + (np.arange(nx) + 0.5) * self.s
        Y = self.t.f - (np.arange(ny) + 0.5) * self.s
        d, dt, crs2, _ = _read_window(dtm, bounds)
        XX, YY = np.meshgrid(X, Y)
        self.crs_dtm = crs2
        dtm_c = _sample(d, dt, XX, YY)
        filled = np.where(np.isfinite(c), c, np.nanmean(c))
        # texture = distance from the 3x3 median, locally averaged: ~0 on flat ground and roofs AND on a clean edge
        # (a median keeps steps), large on a tree crown. Plain local std would flag every roof edge.
        tex = ndimage.uniform_filter(np.abs(filled - ndimage.median_filter(filled, size=3)), 3)
        self.arr = {"dsm": c, "dtm": dtm_c, "ndsm": c - dtm_c,
                    "rough": np.where(np.isfinite(c), tex, np.nan),
                    "dsm_b": _nan_blur(c, 0.6 / self.s)}

    def at(self, name, x, y):
        return _sample(self.arr[name], self.t, x, y)


# ---------------------------------------------------------------- fit

def xf(p, S, x=None, y=None):
    """Apply (dE, dN, yaw_arc, dz) to cell coordinates: rotate about the pivot, then translate."""
    psi = p[2] / YAW_ARC
    c, s = np.cos(psi), np.sin(psi)
    x = S.x if x is None else x
    y = S.y if y is None else y
    dx, dy = x - S.pivot[0], y - S.pivot[1]
    return S.pivot[0] + c * dx - s * dy + p[0], S.pivot[1] + s * dx + c * dy + p[1]


def residuals(p, S, ref, blur=False):
    x, y = xf(p, S)
    z = S.z + p[3]
    if len(p) > 4:  # optional tilt nuisance (p[4], p[5] = height change per 10 m toward E and N): on gentle terrain a tilt
        # is partly confounded with a horizontal shift, so it has to be fitted jointly to keep the translation unbiased
        z = z + (p[4] * (x - S.pivot[0] - p[0]) + p[5] * (y - S.pivot[1] - p[1])) / 10
    g = S.g
    r = np.empty(len(S))
    r[g] = z[g] - ref.at("dtm", x[g], y[g])
    r[~g] = z[~g] - ref.at("dsm_b" if blur else "dsm", x[~g], y[~g])
    # a model structure may only match a smooth reference surface (roof, wall top): matching it to a rough tree
    # crown would let model-only objects (a newer house) fit against vegetation
    t = ~g
    r[t] = np.where(ref.at("rough", x[t], y[t]) > ROUGH_MAX, NAN_RES, r[t])
    return np.where(np.isnan(r), NAN_RES, r)


def gm_cost(r, w):
    return float(np.sum(w * r ** 2 / (r ** 2 + GM_C ** 2)) / np.sum(w))


def coarse_search(S, ref, R, A_deg, step_m=1.0, step_deg=2.5, max_cells=6000, seed=0):
    rng = np.random.default_rng(seed)
    Ss = S.sub(np.sort(rng.choice(len(S), max_cells, replace=False))) if len(S) > max_cells else S
    psis = np.deg2rad(np.arange(-A_deg, A_deg + 1e-9, step_deg))
    sh = np.arange(-R, R + 1e-9, step_m)
    C = np.empty((len(psis), len(sh), len(sh)))
    DZ = np.empty_like(C)
    g = Ss.g
    for a, psi in enumerate(psis):
        for b, dE in enumerate(sh):
            x, y = xf([dE, 0, psi * YAW_ARC, 0], Ss)
            X, Y = np.broadcast_to(x, (len(sh), len(x))), y[None, :] + sh[:, None]
            r = np.empty(X.shape)
            r[:, g] = Ss.z[g] - ref.at("dtm", X[:, g], Y[:, g])
            r[:, ~g] = Ss.z[~g] - ref.at("dsm_b", X[:, ~g], Y[:, ~g])
            r[:, ~g] = np.where(ref.at("rough", X[:, ~g], Y[:, ~g]) > ROUGH_MAX, NAN_RES, r[:, ~g])
            r = np.where(np.isnan(r), NAN_RES, r)
            dz = np.median(r, axis=1)
            r = r - dz[:, None]
            C[a, b] = np.sum(Ss.w * r ** 2 / (r ** 2 + GM_C ** 2), axis=1) / np.sum(Ss.w)
            DZ[a, b] = -dz
    return psis, sh, C, DZ


def window(lim, n):
    """Parameter bounds for (dE, dN, yaw_arc, dz[, tiltE, tiltN]): lim = (search_m, search_yaw_arc)."""
    R, A = lim
    return np.array([-R, -R, -A, -60, -3, -3][:n]), np.array([R, R, A, 60, 3, 3][:n])


def refine(S, ref, p0, lim, f_scale=0.3, max_nfev=40):
    """Robust fit from p0, every stage confined to the search window `lim`."""
    sw = np.sqrt(S.w)
    lo, hi = window(lim, len(p0))
    p = np.clip(np.asarray(p0, float), lo + 1e-6, hi - 1e-6)
    for blur in (True, False):
        res = least_squares(lambda q: sw * residuals(q, S, ref, blur), p, loss="cauchy", f_scale=f_scale, diff_step=0.02,
                            bounds=(lo, hi), max_nfev=max_nfev)
        p = res.x
    # least_squares tends to stop early on this ragged cost (1 m DTM kinks, saturated cells); finish derivative-free
    return polish(lambda q: gm_cost(residuals(q, S, ref), S.w), p, [0.4, 0.4, 0.15, 0.1, 0.1, 0.1][:len(p)], lo, hi)


def polish(cost, x0, steps, lo=None, hi=None, maxfev=600):
    x0 = np.asarray(x0, float)
    clip = (lambda q: q) if lo is None else (lambda q: np.clip(q, lo, hi))
    f = lambda q: cost(clip(q)) + 1e-2 * float(np.abs(q - clip(q)).sum())  # leaving the window is penalised, never used
    sim = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * steps[i] for i in range(len(x0))])
    r = minimize(f, x0, method="Nelder-Mead", options={"initial_simplex": sim, "maxfev": maxfev, "xatol": 0.003, "fatol": 1e-7})
    c0 = cost(x0)
    xb = clip(r.x)
    cb = cost(xb)
    return (xb, float(cb)) if cb < c0 else (x0, float(c0))


def fit_shift(S, ref, R=3.0, step=0.5):
    """Translation (dE, dN, dz) of a cell subset relative to its current placement, with a rough 1-sigma."""
    sh = np.arange(-R, R + 1e-9, step)
    sw = np.sqrt(S.w)
    starts = [[0.0, 0.0, -np.median(residuals([0, 0, 0, 0], S, ref))]]  # the current placement is always a candidate
    for blur in (True, False):  # blurred ref finds the basin, the sharp one the exact edge fit
        best = (np.inf, None)
        for dE in sh:
            for dN in sh:
                r = residuals([dE, dN, 0, 0], S, ref, blur)
                dz = -np.median(r)
                c = gm_cost(r + dz, S.w)
                if c < best[0]:
                    best = (c, [dE, dN, dz])
        starts.append(best[1])
    f = lambda q: sw * residuals([q[0], q[1], 0, q[2]], S, ref)
    fits = [least_squares(f, q0, loss="cauchy", f_scale=0.3, diff_step=0.02, max_nfev=30) for q0 in starts]
    res = min(fits, key=lambda r: gm_cost(residuals([r.x[0], r.x[1], 0, r.x[2]], S, ref), S.w))
    cost3 = lambda q: gm_cost(residuals([q[0], q[1], 0, q[2]], S, ref), S.w)
    x3, _ = polish(cost3, res.x, [0.3, 0.3, 0.05])
    res.x = x3
    raw = f(x3) / sw
    sig = max(1.4826 * np.median(np.abs(raw - np.median(raw))), 0.05)
    H = res.jac.T @ res.jac
    try:
        cov = sig ** 2 * np.linalg.inv(H) * CORR_INFLATE
        # practical floor: a roof edge pins one direction tightly, but on the synthetic world a 1 m DTM and 0.3 m
        # cells still left ~0.3-0.5 m of scatter in the nominal 'constrained' directions
        se = np.maximum(np.sqrt(np.diag(cov)), 0.3)
    except np.linalg.LinAlgError:
        se = np.full(3, np.inf)
    return {"dE": float(res.x[0]), "dN": float(res.x[1]), "dz": float(res.x[2]), "sigma_E": float(se[0]), "sigma_N": float(se[1]),
            "n": int(len(S))}


def bootstrap(S, ref, p, lim, reps=16, block=10.0, seed=1):
    rng = np.random.default_rng(seed)
    x, y = xf(p, S)
    key = np.floor((x - x.min()) / block).astype(int) * 1000 + np.floor((y - y.min()) / block).astype(int)
    keys = np.unique(key)
    groups = [np.flatnonzero(key == k) for k in keys]
    out = []
    for _ in range(reps):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        sw = np.sqrt(S.w[idx])
        Sb = S.sub(idx)
        lo, hi = window(lim, len(p))
        r = least_squares(lambda q: sw * residuals(q, Sb, ref), np.clip(p, lo + 1e-6, hi - 1e-6), loss="cauchy", f_scale=0.3,
                          diff_step=0.02, bounds=(lo, hi), max_nfev=15)
        out.append(r.x)
    o = np.array(out)
    return np.array([o[:, 0].std(), o[:, 1].std(), np.rad2deg(o[:, 2].std() / YAW_ARC), o[:, 3].std()])


def stats(r):
    r = np.asarray(r)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {"n": 0}
    med = float(np.median(r))
    return {"n": int(len(r)), "median": med, "mad_sigma": float(1.4826 * np.median(np.abs(r - med))),
            "p90_abs": float(np.percentile(np.abs(r), 90))}


# ---------------------------------------------------------------- drift

def robust_fit(A, y, iters=20):
    w = np.ones(len(y))
    for _ in range(iters):
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        r = y - A @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        u = r / (4.685 * s)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
    return beta


def _plane_design(x, y):
    return np.c_[np.ones(len(x)), x, y]


def changed_mask(x, y, r, g):
    """Ground cells in a compact patch (>= ~4 m2) where the model is far above the DTM: a new building or earthworks."""
    bad = g & np.isfinite(r) & (r > CHANGED_M)
    ix, iy = ((x - x.min()) // CELL).astype(int), ((y - y.min()) // CELL).astype(int)
    G = np.zeros((iy.max() + 1, ix.max() + 1), bool)
    G[iy[bad], ix[bad]] = True
    lab, n = ndimage.label(ndimage.binary_closing(G, iterations=2))
    big = np.bincount(lab.ravel()) >= 64
    big[0] = False
    return ndimage.binary_dilation(big[lab], iterations=4)[iy, ix]


def ground_resid(p, cells, ref):
    """Residuals at placement p plus the masks that decide which ground cells can be judged against the DTM."""
    x2, y2 = xf(p, cells)
    z2 = cells.z + p[3]
    r = np.where(cells.g, z2 - ref.at("dtm", x2, y2), z2 - ref.at("dsm", x2, y2))
    nd, rough = ref.at("ndsm", x2, y2), ref.at("rough", x2, y2)
    changed = changed_mask(x2, y2, r, cells.g)
    open_g = cells.g & (nd < 0.3) & (rough < 0.05) & np.isfinite(r) & ~changed
    return {"x": x2, "y": y2, "z": z2, "r": r, "nd": nd, "rough": rough, "changed": changed, "open": open_g,
            "canopy": cells.g & (nd > CANOPY_ND)}


def plane_tilt(x, y, r):
    u, v = x - x.mean(), y - y.mean()
    return _tilt_from(robust_fit(_plane_design(u, v), r, iters=10))


def _tilt_from(beta):
    gE, gN = beta[1], beta[2]
    return 1000 * float(np.hypot(gE, gN)), float(np.rad2deg(np.arctan2(gE, gN)) % 360)


def drift_fit(x, y, r, boot=40, block=5.0, seed=2):
    """Plane and quadratic of the residual over (E, N). Direction = azimuth (deg clockwise from north) in which
    the model is higher than the reference. Bootstrap by spatial block, since neighbouring cells are correlated."""
    cx, cy = x.mean(), y.mean()
    u, v = x - cx, y - cy
    plane = robust_fit(_plane_design(u, v), r)
    mm, az = _tilt_from(plane)
    q = robust_fit(np.c_[np.ones(len(u)), u / 10, v / 10, (u / 10) ** 2, u * v / 100, (v / 10) ** 2], r)
    surf = np.c_[np.ones(len(u)), u / 10, v / 10, (u / 10) ** 2, u * v / 100, (v / 10) ** 2] @ q
    bend = surf - _plane_design(u, v) @ robust_fit(_plane_design(u, v), surf)  # quadratic part beyond any plane
    rng = np.random.default_rng(seed)
    key = np.floor(u / block).astype(int) * 1000 + np.floor(v / block).astype(int)
    groups = [np.flatnonzero(key == k) for k in np.unique(key)]
    tilts = []
    for _ in range(boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        b = robust_fit(_plane_design(u[idx], v[idx]), r[idx], iters=8)
        tilts.append(_tilt_from(b)[0])
    sig = lambda a: float(1.4826 * np.median(np.abs(a - np.median(a))))
    return {"n": int(len(r)), "plane_tilt_mm_per_m": mm, "plane_uphill_azimuth_deg": az,
            "plane_gradient_E_mm_per_m": 1000 * float(plane[1]), "plane_gradient_N_mm_per_m": 1000 * float(plane[2]),
            "plane_tilt_bootstrap_std_mm_per_m": float(np.std(tilts)),
            "plane_tilt_significant": bool(mm > 2 * np.std(tilts) and mm > 1.0),
            "plane_range_p5_p95_cm": 100 * float(np.percentile(_plane_design(u, v) @ plane, 95) - np.percentile(_plane_design(u, v) @ plane, 5)),
            "bowl_coeffs_mm_per_10m_sq": {"uu": 1000 * float(q[3]), "uv": 1000 * float(q[4]), "vv": 1000 * float(q[5])},
            "bowl_range_p5_p95_cm": 100 * float(np.percentile(bend, 95) - np.percentile(bend, 5)),
            "residual_mad_sigma_m": {"raw": sig(r), "after_plane": sig(r - _plane_design(u, v) @ plane), "after_quadratic": sig(r - surf)}}


def offset_field_affine(tiles):
    """Scale and rotation of the per-tile offset field: dE = a0 + a1 dx + a2 dy, dN = b0 + b1 dx + b2 dy."""
    t = [k for k in tiles if k["constrained"]]
    if len(t) < 4:
        return None
    cx, cy = np.mean([k["cx"] for k in t]), np.mean([k["cy"] for k in t])
    A = np.array([[1, k["cx"] - cx, k["cy"] - cy] for k in t])
    w = np.array([1 / max(k["sigma_E"] ** 2 + k["sigma_N"] ** 2, 1e-4) for k in t]) ** 0.5
    a = np.linalg.lstsq(A * w[:, None], np.array([k["dE"] for k in t]) * w, rcond=None)[0]
    b = np.linalg.lstsq(A * w[:, None], np.array([k["dN"] for k in t]) * w, rcond=None)[0]
    return {"n_tiles": len(t), "scale_mm_per_m": 1000 * float((a[1] + b[2]) / 2),
            "rotation_deg_ccw": float(np.rad2deg((b[1] - a[2]) / 2)), "shear_mm_per_m": 1000 * float((a[2] + b[1]) / 2)}


# ---------------------------------------------------------------- path / poses

def load_path(out, info):
    """Pose positions in the georef frame (before the fit) and cumulative path distance, or None."""
    f = Path(out) / "poses.npz"
    if not f.exists():
        return None
    z = np.load(f)
    o = np.argsort(z["i"])
    T = z["T"][o][:, :3, 3]
    E, N = info["E0"] + T[:, 0], info["N0"] - T[:, 2]
    s = np.r_[0, np.cumsum(np.hypot(np.diff(E), np.diff(N)))]
    return {"i": z["i"][o], "E": E, "N": N, "s": s}


# ---------------------------------------------------------------- plots

def _background(ref, ortho, bounds):
    """(image, extent) in absolute coordinates: the ortho if given, else a DSM hillshade."""
    if ortho:
        try:
            with rasterio.open(ortho) as ds:
                win = from_bounds(*bounds, ds.transform).round_offsets().round_lengths().intersection(Window(0, 0, ds.width, ds.height))
                k = max(1, int(win.width // 1400) + 1)
                img = ds.read([1, 2, 3], window=win, out_shape=(3, int(win.height // k), int(win.width // k)))
                t = ds.window_transform(win)
            return np.moveaxis(img, 0, 2), (t.c, t.c + t.a * win.width, t.f + t.e * win.height, t.f)
        except Exception as e:
            print(f"ortho unusable ({e}), using hillshade")
    d = np.nan_to_num(ref.arr["dsm"], nan=float(np.nanmean(ref.arr["dsm"])))
    gy, gx = np.gradient(d, ref.s)
    hs = np.clip(0.5 + 0.5 * (gx - gy) / np.sqrt(2) * 1.5 / (1 + np.hypot(gx, gy)), 0, 1)
    ny, nx = d.shape
    return np.dstack([hs] * 3), (ref.t.c, ref.t.c + nx * ref.s, ref.t.f - ny * ref.s, ref.t.f)


def make_plots(rep, ref, ortho, cells, fitted, tiles, segs, path_curve, snaps, drift, origin, tile_m, full=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ox, oy = origin
    x, y = fitted["x"] - ox, fitted["y"] - oy
    pad = 6
    bounds = (fitted["x"].min() - pad, fitted["y"].min() - pad, fitted["x"].max() + pad, fitted["y"].max() + pad)
    img, ext = _background(ref, ortho, bounds)
    ext = (ext[0] - ox, ext[1] - ox, ext[2] - oy, ext[3] - oy)
    lim = (x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(img, extent=ext, origin="upper")
    g = fitted["g"] & np.isfinite(fitted["r"])
    sc = ax.scatter(x[g], y[g], c=np.clip(fitted["r"][g], -0.6, 0.6), s=3, marker="s", cmap="RdBu_r", vmin=-0.6, vmax=0.6, alpha=0.75, linewidths=0)
    if fitted.get("path") is not None:
        ax.plot(fitted["path"][0] - ox, fitted["path"][1] - oy, "k-", lw=0.8)
    plt.colorbar(sc, ax=ax, shrink=0.7, label="model ground - DTM after fit (m)")
    ax.set_xlim(lim[:2]); ax.set_ylim(lim[2:]); ax.set_aspect("equal")
    ax.set_title("ground residual vs IDE DTM (red = model too high), walk path in black" if full else
                 "AMBIGUOUS PLACEMENT: GPS georef shown, residuals not meaningful")
    ax.set_xlabel("m east of model corner"); ax.set_ylabel("m north of model corner")
    fig.savefig(rep / "audit_residual.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    if not full:
        return

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(img, extent=ext, origin="upper")
    for k in tiles:
        c = "yellow" if k["constrained"] else "grey"
        ax.add_patch(plt.Rectangle((k["cx"] - ox - tile_m / 2, k["cy"] - oy - tile_m / 2), tile_m, tile_m, fill=False, ec=c, lw=0.6, alpha=0.6))
        ax.annotate("", xy=(k["cx"] - ox + 4 * k["dE"], k["cy"] - oy + 4 * k["dN"]), xytext=(k["cx"] - ox, k["cy"] - oy),
                    arrowprops=dict(arrowstyle="->", color="red" if k["constrained"] else "grey", lw=2 if k["constrained"] else 1))
    ax.set_xlim(lim[:2]); ax.set_ylim(lim[2:]); ax.set_aspect("equal")
    ax.set_title(f"per-{tile_m:g} m-tile extra offset after global fit (arrows x4; grey = weakly constrained)")
    ax.set_xlabel("m east of model corner"); ax.set_ylabel("m north of model corner")
    fig.savefig(rep / "audit_tiles.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    xs, med, lo, hi, label = path_curve
    axs[0].fill_between(xs, lo, hi, color="C0", alpha=0.25)
    axs[0].plot(xs, med, "C0-o", ms=3)
    axs[0].axhline(0, color="k", lw=0.5)
    axs[0].set_ylabel("open-ground residual (m)\nmedian, +/-1 sigma-MAD band")
    for sn in snaps:
        for a in axs:
            a.axvline(sn, color="r", ls="--", lw=0.8)
    if segs:
        sx = [k["s_mid"] for k in segs]
        for key, c in (("dE", "C1"), ("dN", "C2"), ("dz", "C3")):
            axs[1].errorbar(sx, [k[key] for k in segs], yerr=[k["sigma_E"] if key == "dE" else k["sigma_N"] if key == "dN" else 0 for k in segs],
                            fmt="-o", color=c, ms=3, label=key)
        axs[1].legend(loc="upper right", ncol=3)
    axs[1].axhline(0, color="k", lw=0.5)
    axs[1].set_ylabel("extra offset per segment (m)")
    axs[1].set_xlabel(label)
    axs[0].set_title("drift along the walk (red dashed = pipeline-reported tracking snaps)" if snaps else "drift along the walk")
    fig.savefig(rep / "audit_distance.png", dpi=110, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- main

def audit(out_dir, dsm, dtm, ortho=None, report=None, search_m=15.0, search_deg=15.0, tile_m=10.0, boot=16, plots=True, verbose=True):
    out = Path(out_dir)
    rep = Path(report) if report else out
    rep.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)
    info, pts, rgb = load_points(out)
    cells, ground_source = build_cells(pts, rgb, out / "ground.tif")
    log(f"{len(pts)} points -> {int(cells.g.sum())} ground cells, {int((~cells.g).sum())} structure cells (ground from {ground_source})")
    bounds = (pts[:, 0].min() - search_m - 15, pts[:, 1].min() - search_m - 15, pts[:, 0].max() + search_m + 15, pts[:, 1].max() + search_m + 15)
    ref = Ref(dsm, dtm, bounds)
    if ref.crs != info["crs"] or ref.crs_dtm != info["crs"]:
        log(f"WARNING: CRS mismatch: georef {info['crs']}, DSM {ref.crs}, DTM {ref.crs_dtm}")

    # 1. rigid fit
    lim = (search_m, np.deg2rad(search_deg) * YAW_ARC)
    psis, sh, C, DZ = coarse_search(cells, ref, search_m, search_deg)
    mins = (ndimage.minimum_filter(C, size=3, mode="nearest") == C)
    cand = sorted(zip(C[mins], *np.nonzero(mins)))[:5]
    best, alts = None, []
    for c0, a, b, c in cand:
        p0 = [sh[b], sh[c], psis[a] * YAW_ARC, DZ[a, b, c]]
        p, cost = refine(cells, ref, p0, lim)
        alts.append({"dE": float(p[0]), "dN": float(p[1]), "dyaw_deg": float(np.rad2deg(p[2] / YAW_ARC)), "dz": float(p[3]), "cost": float(cost)})
        if best is None or cost < best[1]:
            best = (p, cost)
    p, cost = best
    near = [a for a in alts if a["cost"] < cost * 1.10]  # candidates that fit nearly as well as the best
    near_spread = {k: float(np.ptp([a[k] for a in near])) for k in ("dE", "dN", "dyaw_deg", "dz")}
    C2 = C.min(axis=0)
    bi = np.unravel_index(np.argmin(C2), C2.shape)
    dist = np.hypot(sh[:, None] - sh[bi[0]], sh[None, :] - sh[bi[1]])
    second = float(C2[dist > 3.0].min()) if (dist > 3.0).any() else float("nan")
    gap = (second - C2.min()) / max(float(np.median(C2)) - C2.min(), 1e-9)
    at_edge = lambda q: bool(np.any(np.abs(q[:2]) > search_m - 0.5) or abs(np.rad2deg(q[2] / YAW_ARC)) > search_deg - 0.5)
    reasons = []
    if gap <= 0.05:  # coarse, blurred, min-over-yaw measure: only a weak signal, competing refined minima are the real test
        reasons.append(f"no single clear minimum in the cost surface (uniqueness {gap:.2f})")
    contrast = (float(np.median(C2)) - float(C2.min())) / max(float(np.median(C2)), 1e-9)
    if contrast < 0.1:
        reasons.append(f"cost surface is flat: the best shift is only {contrast:.0%} better than a typical one")
    if at_edge(p):
        reasons.append("best fit sits on the search-window boundary")
    if max(near_spread["dE"], near_spread["dN"]) > MAX_AMBIG_M or near_spread["dyaw_deg"] > MAX_AMBIG_DEG:
        reasons.append(f"competing minima within 10% cost differ by {max(near_spread['dE'], near_spread['dN']):.1f} m / {near_spread['dyaw_deg']:.1f} deg")
    # tilt and a horizontal shift are confounded on gently sloping ground: refit jointly with a tilt plane and keep
    # that translation if the tilt is real (> 2 mm/m and the robust cost clearly improves)
    cost4, joint, p_full = cost, None, p
    if not reasons:
        p6, cost6 = refine(cells, ref, np.r_[p, 0, 0], lim)
        tilt6 = 100 * float(np.hypot(p6[4], p6[5]))
        use6 = bool(tilt6 > 2.0 and cost6 < 0.97 * cost)
        joint = {"tilt_mm_per_m": tilt6, "model_higher_toward_azimuth_deg": float(np.rad2deg(np.arctan2(-p6[4], -p6[5])) % 360),
                 "robust_cost": cost6, "used_for_translation": use6}
        if use6:
            p, cost, p_full = p6[:4], cost6, p6
            if at_edge(p):
                reasons.append("joint tilt fit ends on the search-window boundary")
    ambiguous = bool(reasons)
    rejected = None
    if ambiguous:  # keep the GPS horizontal placement; only the vertical offset is well determined
        rejected = {"dE": float(p[0]), "dN": float(p[1]), "dyaw_deg": float(np.rad2deg(p[2] / YAW_ARC)), "dz": float(p[3])}
        r0 = residuals(np.zeros(4), cells, ref)
        ok0 = cells.g & (r0 != NAN_RES)
        p = np.array([0.0, 0.0, 0.0, -float(np.median(r0[ok0])) if ok0.any() else p[3]])
        p_full, cost = p, gm_cost(residuals(p, cells, ref), cells.w)
        log("\n*** PLACEMENT AMBIGUOUS, using the GPS georef (plus a vertical offset): " + "; ".join(reasons) + " ***")
    start_cost = gm_cost(residuals(np.zeros(4), cells, ref), cells.w)
    sd = bootstrap(cells, ref, p_full, lim, reps=boot) if (boot and not ambiguous) else np.full(4, np.nan)
    r_fit = residuals(p_full, cells, ref)
    inl = np.abs(r_fit) < 0.3
    yaw_deg = float(np.rad2deg(p[2] / YAW_ARC))
    edge = at_edge(p)
    fit = {"placement_status": "ambiguous_gps_fallback" if ambiguous else "determined", "ambiguity_reasons": reasons,
           "rejected_search_result": rejected, "search_window": {"m": search_m, "deg": search_deg},
           "dE": float(p[0]), "dN": float(p[1]), "dyaw_deg_ccw": yaw_deg, "dz": float(p[3]),
           "pivot_E": cells.pivot[0], "pivot_N": cells.pivot[1],
           "bootstrap_std": {"dE": float(sd[0]), "dN": float(sd[1]), "dyaw_deg": float(sd[2]), "dz": float(sd[3])},
           "robust_cost_start": start_cost, "robust_cost_fit": cost, "robust_cost_rigid_only": cost4, "joint_tilt_fit": joint,
           "cells_used": {"ground": int(cells.g.sum()), "structure": int((~cells.g).sum())},
           "inlier_fraction_within_0.3m": {"ground": float(inl[cells.g].mean()), "structure": float(inl[~cells.g].mean()) if (~cells.g).any() else None},
           "refined_candidates": alts,
           "candidates_within_10pct_cost_spread": near_spread,
           "cost_surface": {"best": float(C2.min()), "second_minimum_outside_3m": second, "median": float(np.median(C2)),
                            "uniqueness_0_to_1": float(gap), "clear_single_minimum": bool(gap > 0.05), "at_search_boundary": edge,
                            "yaw_profile_deg_cost": [[float(np.rad2deg(a)), float(C[i].min())] for i, a in enumerate(psis)]}}

    # post-fit cell values
    gr = ground_resid(p, cells, ref)
    x2, y2, z2, r, nd, rough, open_g = gr["x"], gr["y"], gr["z"], gr["r"], gr["nd"], gr["rough"], gr["open"]
    n_open = int(open_g.sum())
    open_info = {"ground_cells": int(cells.g.sum()), "masked_changed_since_flight": int((gr["changed"] & cells.g).sum()),
                 "masked_under_tall_canopy": int(gr["canopy"].sum()), "open_ground_cells": n_open,
                 "open_ground_m2": n_open * CELL ** 2, "open_ground_fraction": n_open / max(int(cells.g.sum()), 1),
                 "enough_for_drift_analysis": bool(n_open >= MIN_OPEN_CELLS)}
    cand =~cells.g & (nd > STRUCT_H) & (rough < ROUGH_MAX) & np.isfinite(r)
    hard = cand & (np.abs(r) < 1.0)  # a model structure counts as 'matched' when the DSM shows a smooth surface within 1 m
    # (a walk under tall trees sees low bushes and walls where the DSM shows the canopy, which must not be scored)
    resid = {"open_ground_budget": open_info, "open_ground_vs_DTM": stats(r[open_g]), "hard_surfaces_vs_DSM": stats(r[hard]),
             "all_ground_vs_DTM": stats(r[cells.g]),
             "hard_surface_match": {"model_structure_cells": int((~cells.g).sum()), "with_smooth_dsm_surface": int(cand.sum()),
                                    "matched_within_1m": int(hard.sum())},
             "note": "hard-surface stats cover only matched cells (selection by |r|<1 m flatters them, see match counts); the model "
                     "is built from the ground so it under-reports roofs; the DTM itself is ~0.1-0.3 m, worse under trees"}
    # same stats with the GPS-only placement, to show what the fit bought
    p0z = np.zeros(4)
    r0 = residuals(p0z, cells, ref)
    resid["open_ground_vs_DTM_gps_only"] = stats(r0[cells.g & np.isfinite(r0) & (r0 != NAN_RES)])

    # 3. drift
    drift = {}
    # drift is judged on open ground only: not under tall canopy (the DTM there is an interpolation) and not where
    # the site changed since the flight. Too little of it, or an ambiguous placement, means "not determinable".
    dset = open_g
    drift["ground_cells_used"] = open_info
    if ambiguous or not open_info["enough_for_drift_analysis"]:
        why = ("placement ambiguous: " + "; ".join(reasons)) if ambiguous else \
            f"only {n_open} judgeable open-ground cells ({open_info['open_ground_m2']:.0f} m2), need {MIN_OPEN_CELLS}"
        drift["status"] = "not_determinable"
        drift["reason"] = why
        drift["ground_plane_and_bowl"] = None
        if ambiguous:
            # indicative only: tilt at the GPS placement (right only to the extent GPS is, +-3-5 m) and how much it
            # changes across the competing minima
            sens = []
            for q in [dict(dE=0.0, dN=0.0, dyaw_deg=0.0, dz=float(p[3]), cost=float(cost))] + [a for a in alts if a["cost"] < 1.25 * best[1]]:
                pq = np.array([q["dE"], q["dN"], np.deg2rad(q["dyaw_deg"]) * YAW_ARC, q["dz"]])
                g2 = ground_resid(pq, cells, ref)
                e = {k: q[k] for k in ("dE", "dN", "dyaw_deg", "cost")}
                e["open_cells"] = int(g2["open"].sum())
                if e["open_cells"] > 200:
                    e["tilt_mm_per_m"], e["model_higher_toward_azimuth_deg"] = plane_tilt(g2["x"][g2["open"]], g2["y"][g2["open"]], g2["r"][g2["open"]])
                sens.append(e)
            tl = [e["tilt_mm_per_m"] for e in sens if "tilt_mm_per_m" in e]
            drift["tilt_sensitivity"] = {"note": "first entry = GPS placement; tilt is NOT reported as significant", "placements": sens,
                                         "tilt_range_mm_per_m": [min(tl), max(tl)] if tl else None}
    else:
        drift["status"] = "determined"
        drift["ground_plane_and_bowl"] = drift_fit(x2[dset], y2[dset], r[dset])
    S2 = Cells(x2, y2, z2, cells.g, cells.w, cells.pivot)
    tiles = []
    tk = (np.floor((x2 - x2.min()) / tile_m)).astype(int) * 1000 + np.floor((y2 - y2.min()) / tile_m).astype(int)
    for k in ([] if ambiguous else np.unique(tk)):
        idx = np.flatnonzero(tk == k)
        if len(idx) < 400:
            continue
        f = fit_shift(S2.sub(idx), ref)
        f.update(cx=float(x2[idx].mean()), cy=float(y2[idx].mean()), n_structure=int((~cells.g[idx]).sum()))
        f["constrained"] = bool(f["sigma_E"] < 0.5 and f["sigma_N"] < 0.5)
        tiles.append(f)
    ct = [t for t in tiles if t["constrained"]]
    if len(ct) >= 3:
        mad = lambda a: float(1.4826 * np.median(np.abs(np.asarray(a) - np.median(a))))
        spread = {"n_constrained": len(ct), "n_tiles": len(tiles),
                  "dE_median": float(np.median([t["dE"] for t in ct])), "dN_median": float(np.median([t["dN"] for t in ct])),
                  "dE_mad_sigma": mad([t["dE"] for t in ct]), "dN_mad_sigma": mad([t["dN"] for t in ct]),
                  "max_abs_offset_m": float(max(np.hypot(t["dE"], t["dN"]) for t in ct)),
                  "median_noise_sigma_m": float(np.median([max(t["sigma_E"], t["sigma_N"]) for t in ct])),
                  "dz_mad_sigma": mad([t["dz"] for t in ct]), "affine": offset_field_affine(tiles)}
        zE = (np.array([t["dE"] for t in ct]) - spread["dE_median"]) / np.array([t["sigma_E"] for t in ct])
        zN = (np.array([t["dN"] for t in ct]) - spread["dN_median"]) / np.array([t["sigma_N"] for t in ct])
        spread["rms_z_about_median"] = float(np.sqrt(np.mean(np.r_[zE, zN] ** 2)))  # ~1 or a bit more when offsets are only noise
        spread["tiles_beyond_3_sigma"] = int(np.sum(np.hypot(zE, zN) > 3 * np.sqrt(2)))
        spread["horizontal_drift_suspected"] = bool(spread["rms_z_about_median"] > 2.5 and spread["max_abs_offset_m"] > 1.0)
    else:
        spread = {"n_constrained": len(ct), "n_tiles": len(tiles), "horizontal_drift_suspected": None,
                  "status": "not_determinable",
                  "note": "placement ambiguous, tile offsets not computed" if ambiguous else "too few constrained tiles"}
    drift["tiles"] = {"size_m": tile_m, "spread": spread, "offsets": tiles}

    # path analysis (needs poses.npz). Cells are mapped to the NEAREST POSE in plan view, i.e. "which part of
    # the walk is closest", not to the frame that actually observed the point (the ply has no frame index).
    path = load_path(out, info)
    segs, snaps_s, path_info = [], [], None
    sv = np.nan
    if ambiguous:
        curve = (np.array([]), np.array([]), np.array([]), np.array([]), "")
        path_info = {"status": "not_determinable", "reason": "placement ambiguous, residuals along the walk would be meaningless"}
        fpath = None
    elif path is not None:
        px, py = xf(p, S2, path["E"], path["N"])
        _, nearest = cKDTree(np.c_[path["E"], path["N"]]).query(np.c_[x2, y2])
        s_cell = path["s"][nearest]
        L = path["s"][-1]
        nseg = int(np.clip(L // 15, 3, 8))
        edges = np.linspace(0, L, nseg + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            idx = np.flatnonzero((s_cell >= a) & (s_cell <= b))
            if len(idx) >= 300:
                f = fit_shift(S2.sub(idx), ref)
                f.update(s_from=float(a), s_to=float(b), s_mid=float((a + b) / 2))
                segs.append(f)
        bins = np.linspace(0, L, 21)
        xs, med, lo, hi = [], [], [], []
        for a, b in zip(bins[:-1], bins[1:]):
            m = dset & (s_cell >= a) & (s_cell < b)
            if m.sum() >= 30:
                v = r[m]
                md, sg = np.median(v), 1.4826 * np.median(np.abs(v - np.median(v)))
                xs.append((a + b) / 2); med.append(md); lo.append(md - sg); hi.append(md + sg)
        curve = (np.array(xs), np.array(med), np.array(lo), np.array(hi), "path distance from walk start (m)")
        if len(xs) > 3:
            sv = float(np.polyfit(xs, med, 1)[0])
        seg_ok = [k for k in segs if max(k["sigma_E"], k["sigma_N"]) < 0.5]
        steps = np.hypot(np.diff(path["E"]), np.diff(path["N"]))
        jumps = [{"pose_index": int(path["i"][k + 1]), "path_distance_m": float(path["s"][k]), "step_m": float(steps[k])}
                 for k in np.argsort(steps)[::-1][:3] if steps[k] > 0.5]
        snaps = []
        of = out / "optimize.json"
        if of.exists():
            for sn in json.loads(of.read_text()).get("snaps", []):
                k = int(np.argmin(np.abs(path["i"] - sn["n"])))
                snaps.append({"t": sn.get("t"), "path_distance_m": float(path["s"][k]), "jump_m": sn.get("jump_m"),
                              "corrected_jump_m": sn.get("corrected_jump_m"), "rot_deg": sn.get("rot_deg")})
        snaps_s = [s["path_distance_m"] for s in snaps if (s["jump_m"] or 0) > 1.0]
        path_info = {"mapping": "nearest pose in plan view (approximate; points carry no frame index), not time",
                     "length_m": float(L), "ground_residual_trend_mm_per_10m": None if np.isnan(sv) else 10000 * sv,
                     "segments": segs, "biggest_pose_steps": jumps, "pipeline_snaps": snaps,
                     "segment_offset_spread_m": {
                         "dE": float(np.ptp([k["dE"] for k in seg_ok])) if seg_ok else None,
                         "dN": float(np.ptp([k["dN"] for k in seg_ok])) if seg_ok else None,
                         "dz": float(np.ptp([k["dz"] for k in seg_ok])) if seg_ok else None}}
        fpath = (px, py)
    else:
        # spatial only: project onto the model's principal axis
        u = np.c_[x2 - x2.mean(), y2 - y2.mean()]
        ax = np.linalg.svd(u, full_matrices=False)[2][0]
        s_cell = u @ ax
        s_cell = s_cell - s_cell.min()
        bins = np.linspace(0, s_cell.max(), 21)
        xs, med, lo, hi = [], [], [], []
        for a, b in zip(bins[:-1], bins[1:]):
            m = dset & (s_cell >= a) & (s_cell < b)
            if m.sum() >= 30:
                v = r[m]
                md, sg = np.median(v), 1.4826 * np.median(np.abs(v - np.median(v)))
                xs.append((a + b) / 2); med.append(md); lo.append(md - sg); hi.append(md + sg)
        curve = (np.array(xs), np.array(med), np.array(lo), np.array(hi), "distance along model's long axis (m); no poses.npz, spatial only")
        path_info = {"mapping": "no poses.npz: spatial only (principal axis)"}
        fpath = None
    drift["path"] = path_info

    # georef_ide.json
    E0n = cells.pivot[0] + np.cos(p[2] / YAW_ARC) * (info["E0"] - cells.pivot[0]) - np.sin(p[2] / YAW_ARC) * (info["N0"] - cells.pivot[1]) + p[0]
    N0n = cells.pivot[1] + np.sin(p[2] / YAW_ARC) * (info["E0"] - cells.pivot[0]) + np.cos(p[2] / YAW_ARC) * (info["N0"] - cells.pivot[1]) + p[1]
    new = dict(info)
    new.update({"E0": float(E0n), "N0": float(N0n), "dz": float(info["dz"] + p[3]), "yaw_deg": yaw_deg,
                "offset": [float(np.floor(E0n / 100) * 100), float(np.floor(N0n / 100) * 100), 0.0],
                "model": "E = E0 + X cos(yaw) + Z sin(yaw), N = N0 + X sin(yaw) - Z cos(yaw), Z_up = Y + dz; scale 1; yaw ccw seen from above",
                "ply_convention": "a points ply written with this georef: coordinates + offset = (E, N, Z_up)",
                "note": ("PLACEMENT AMBIGUOUS in the aerial data: this is the GPS georef with only a vertical offset fitted by audit_ide.py"
                         if ambiguous else
                         "GPS georef refined by audit_ide.py against the aerial DSM/DTM; rigid fit only, drift not corrected"),
                "placement_status": fit["placement_status"],
                "gps_georef": {"E0": info["E0"], "N0": info["N0"], "dz": info["dz"]},
                "correction": {"dE": fit["dE"], "dN": fit["dN"], "dyaw_deg": yaw_deg, "dz": fit["dz"], "bootstrap_std": fit["bootstrap_std"]}})
    (rep / "georef_ide.json").write_text(json.dumps(new, indent=2))

    result = {"model": {"points_used": int(len(pts)), "ground_source": ground_source},
              "fit_vs_gps_georef": fit, "residuals_after_fit": resid, "drift": drift,
              "reference": {"crs": ref.crs, "cell_m": CELL, "dsm_working_res_m": ref.s}}
    (rep / "audit.json").write_text(json.dumps(result, indent=2, default=float))
    if plots:
        fitted = {"x": x2, "y": y2, "r": np.where(cells.g, r, np.nan), "g": cells.g, "path": fpath}
        make_plots(rep, ref, ortho, cells, fitted, tiles, segs, curve, snaps_s, drift, (x2.min(), y2.min()), tile_m, full=not ambiguous)
    summary(result, log)
    return result


def summary(a, log=print):
    f, r, d = a["fit_vs_gps_georef"], a["residuals_after_fit"], a["drift"]
    sd = f["bootstrap_std"]
    log("\n=== audit vs aerial elevation data ===")
    amb = f["placement_status"] != "determined"
    if amb:
        rj = f["rejected_search_result"]
        log("!!! PLACEMENT AMBIGUOUS: the aerial data does not pin down a horizontal placement. Using the GPS georef plus a vertical offset.")
        for why in f["ambiguity_reasons"]:
            log(f"    - {why}")
        log(f"    (rejected search result: dE {rj['dE']:+.1f} m, dN {rj['dN']:+.1f} m, yaw {rj['dyaw_deg']:+.1f} deg)")
        log(f"placement used: GPS georef, dz={f['dz']:+.2f} m")
    else:
        log(f"correction to GPS georef: dE={f['dE']:+.2f} m (+/-{sd['dE']:.2f}), dN={f['dN']:+.2f} m (+/-{sd['dN']:.2f}), "
        f"yaw={f['dyaw_deg_ccw']:+.2f} deg (+/-{sd['dyaw_deg']:.2f}), dz={f['dz']:+.2f} m (+/-{sd['dz']:.2f})")
    cs = f["cost_surface"]
    lg = (lambda *a, **k: None) if amb else log  # the fit-confidence lines describe a fit that was rejected
    lg(f"confidence: {f['cells_used']['ground']} ground + {f['cells_used']['structure']} structure cells, "
        f"inliers(<0.3 m) ground {f['inlier_fraction_within_0.3m']['ground']:.0%}; cost {f['robust_cost_start']:.3f} -> {f['robust_cost_fit']:.3f}; "
        f"single clear minimum: {cs['clear_single_minimum']} (uniqueness {cs['uniqueness_0_to_1']:.2f}), at search boundary: {cs['at_search_boundary']}")
    sp5 = f["candidates_within_10pct_cost_spread"]
    lg(f"candidate minima within 10% of the best cost differ by dE {sp5['dE']:.2f} m, dN {sp5['dN']:.2f} m, yaw {sp5['dyaw_deg']:.1f} deg "
        f"(the real placement uncertainty is at least this, not just the bootstrap std)")
    for k, name in (("open_ground_vs_DTM", "open ground vs DTM"), ("all_ground_vs_DTM", "all ground vs DTM"),
                    ("hard_surfaces_vs_DSM", "hard surfaces vs DSM (matched cells only)"),
                    ("open_ground_vs_DTM_gps_only", "ground, GPS georef as-is (incl. its altitude error)")):
        s = r[k]
        if s["n"]:
            log(f"{name}{' (at GPS placement, indicative)' if amb else ''}: n={s['n']} median {s['median']:+.3f} m, MAD-sigma {s['mad_sigma']:.3f} m, p90|r| {s['p90_abs']:.3f} m")
    ob = r["open_ground_budget"]
    log(f"open ground judgeable: {ob['open_ground_cells']} of {ob['ground_cells']} ground cells ({ob['open_ground_m2']:.0f} m2; "
        f"{ob['masked_under_tall_canopy']} under tall canopy, {ob['masked_changed_since_flight']} changed since the flight)")
    g = d["ground_plane_and_bowl"]
    if d["status"] != "determined":
        log(f"tilt / bowl / drift: NOT DETERMINABLE ({d['reason']})")
        ts = d.get("tilt_sensitivity")
        if ts and ts["tilt_range_mm_per_m"]:
            lo_, hi_ = ts["tilt_range_mm_per_m"]
            log(f"  indicative only: tilt at the GPS placement {ts['placements'][0].get('tilt_mm_per_m', float('nan')):.1f} mm/m; "
                f"across the competing minima {lo_:.1f}-{hi_:.1f} mm/m, so not a finding (GPS placement is only good to +-3-5 m)")
    if g:
        log(f"tilt: {g['plane_tilt_mm_per_m']:.1f} mm/m (bootstrap +/-{g['plane_tilt_bootstrap_std_mm_per_m']:.1f}) model higher toward azimuth "
            f"{g['plane_uphill_azimuth_deg']:.0f} deg; plane range {g['plane_range_p5_p95_cm']:.1f} cm; bowl range {g['bowl_range_p5_p95_cm']:.1f} cm; "
            f"significant: {g['plane_tilt_significant']}")
        log(f"  residual MAD-sigma: raw {g['residual_mad_sigma_m']['raw']:.3f} -> plane {g['residual_mad_sigma_m']['after_plane']:.3f} -> quadratic {g['residual_mad_sigma_m']['after_quadratic']:.3f} m")
    sp = d["tiles"]["spread"]
    if sp.get("status") == "not_determinable":
        log(f"tiles: not determinable ({sp['note']})")
    if "dE_mad_sigma" in sp:
        log(f"tiles: {sp['n_constrained']}/{sp['n_tiles']} constrained, offset spread (MAD-sigma) dE {sp['dE_mad_sigma']:.2f} dN {sp['dN_mad_sigma']:.2f} m, "
            f"max {sp['max_abs_offset_m']:.2f} m, typical noise {sp['median_noise_sigma_m']:.2f} m -> horizontal drift suspected: {sp['horizontal_drift_suspected']}")
        if sp["affine"]:
            log(f"  offset field: scale {sp['affine']['scale_mm_per_m']:+.1f} mm/m, rotation {sp['affine']['rotation_deg_ccw']:+.2f} deg")
    pi = d["path"]
    if pi and "segments" in pi:
        sg = pi["segment_offset_spread_m"]
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
        log(f"walk segments ({len(pi['segments'])}): offset range dE {fmt(sg['dE'])}, dN {fmt(sg['dN'])}, dz {fmt(sg['dz'])} m "
            f"(segments with sigma<0.5 m only); mapping is nearest-pose, not time")
    elif pi:
        log(f"path: {pi.get('mapping') or 'not determinable (' + pi['reason'] + ')'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--dsm", required=True)
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--ortho")
    ap.add_argument("--report", help="output folder (default: out_dir)")
    ap.add_argument("--search-m", type=float, default=15.0)
    ap.add_argument("--search-deg", type=float, default=15.0)
    a = ap.parse_args()
    sys.exit(0 if audit(a.out_dir, a.dsm, a.dtm, a.ortho, a.report, a.search_m, a.search_deg) else 1)
