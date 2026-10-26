"""audit_ide check on a synthetic world (made-up location): a fake DSM/DTM, a model sampled from it with a
known placement error (translation, yaw, z), noise, differing trees, an extra house, and optionally a tilt.

Run: uv run --project . python test_audit_ide.py     (or pytest)
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).parent))  # not on sys.path when run with -I
import audit_ide

W = np.array([412300.0, 5987600.0])  # made-up local origin (E, N), EPSG:5382
CRS = "EPSG:5382"
ERR_T, ERR_YAW, ERR_Z = (4.0, -3.0), np.deg2rad(3.0), 2.0


FLAT = False  # featureless scene for the ambiguity test: dead-flat ground, no houses, no trees


def ground(x, y):
    if FLAT:
        return 30 + 0 * x
    return 30 + 0.03 * x + 0.01 * y + 1.5 * np.sin(x / 17) * np.cos(y / 13)


def dome(x, y, cx, cy, h=6.0, r=3.0):
    return np.where(np.hypot(x - cx, y - cy) < r, h * np.sqrt(np.maximum(1 - ((x - cx) ** 2 + (y - cy) ** 2) / r ** 2, 0)), 0.0)


def box(x, y, x0, x1, y0, y1, top):
    return np.where((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1), top, -np.inf)


def houses(x, y):
    if FLAT:
        return np.full(np.shape(x), -np.inf)
    return np.maximum(box(x, y, -10, -2, 4, 14, 36.0), box(x, y, 10, 16, -12, -6, 33.5))


IDE_TREES = [(-18, -10), (5, 12), (18, 8), (-5, -18)]
MODEL_TREES = [(-15, 15), (2, -8), (20, -15)]


def ide_dsm(x, y):
    g = ground(x, y)
    rng = np.random.default_rng(5)
    t = sum((dome(x, y, cx, cy) for cx, cy in ([] if FLAT else IDE_TREES)), np.zeros(np.shape(x)))
    return np.maximum(np.maximum(g + t * (1 + 0.15 * rng.standard_normal(x.shape)), g), houses(x, y))


def model_top(x, y):
    g = ground(x, y)
    if FLAT:
        return g
    new_house = box(x, y, -4, 2, -24, -18, 34.0)  # newer than the aerial data
    return np.maximum(np.maximum(g + sum(dome(x, y, cx, cy) for cx, cy in MODEL_TREES), houses(x, y)), new_house)


def in_tree(x, y):
    return sum((dome(x, y, cx, cy) for cx, cy in ([] if FLAT else MODEL_TREES)), np.zeros(np.shape(x))) > 0


def write_ide(d):
    x = np.arange(-60, 60, 0.1) + 0.05
    X, Y = np.meshgrid(x, -x)  # row 0 = north
    for name, arr, res, ext in (("dsm.tif", ide_dsm(X, Y), 0.1, 60), ("dtm.tif", None, 1.0, 62)):
        if arr is None:
            xs = np.arange(-ext, ext, res) + res / 2
            XX, YY = np.meshgrid(xs, -xs)
            arr = ground(XX, YY) + 0.03 * np.random.default_rng(6).standard_normal(XX.shape)
        with rasterio.open(d / name, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32", crs=CRS,
                           nodata=-32767, transform=from_origin(W[0] - ext, W[1] + ext, res, res)) as dst:
            dst.write(arr.astype(np.float32), 1)


def placement_error(x, y):
    """local (x, y) true position -> georef-frame offset position (E, N), rotation about W."""
    c, s = np.cos(ERR_YAW), np.sin(ERR_YAW)
    return W[0] + c * x - s * y + ERR_T[0], W[1] + s * x + c * y + ERR_T[1]


def write_model(out, tilt_mm_per_m=0.0, tilt_az_deg=60.0, with_ground=True, n=300_000, tilt_region=None):
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-25, 25, (2, n))
    z = model_top(x, y) + 0.03 * rng.standard_normal(n)
    E, N = placement_error(x, y)
    E, N = E + 0.02 * rng.standard_normal(n), N + 0.02 * rng.standard_normal(n)
    az = np.deg2rad(tilt_az_deg)
    tilt = lambda e, nn: 1e-3 * tilt_mm_per_m * ((e - W[0]) * np.sin(az) + (nn - W[1]) * np.cos(az))
    zt = z + ERR_Z + tilt(E, N)
    rgb = np.where(in_tree(x, y)[:, None], [0.2, 0.7, 0.2], [0.5, 0.5, 0.5])
    E0, N0 = W
    offset = np.array([np.floor(E0 / 100) * 100, np.floor(N0 / 100) * 100, 0.0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.c_[E, N, zt] - offset)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    out.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out / "georef_points.ply"), pcd)
    (out / "georef.json").write_text(json.dumps({"crs": CRS, "E0": E0, "N0": N0, "dz": 0.0, "offset": offset.tolist()}))
    if with_ground:  # ground.tif as georef.py writes it: 0.1 m grid, -32767 where no ground was seen
        gx = np.arange(-27, 27, 0.1) + 0.05
        Q = np.meshgrid(W[0] + gx, W[1] - gx)  # georef frame (E, N), row 0 = north
        dE, dN = Q[0] - W[0] - ERR_T[0], Q[1] - W[1] - ERR_T[1]
        c, s = np.cos(-ERR_YAW), np.sin(-ERR_YAW)
        tx, ty = c * dE - s * dN, s * dE + c * dN  # back to true local position
        g = ground(tx, ty) + ERR_Z + tilt(Q[0], Q[1])
        seen = (np.abs(tx) < 25) & (np.abs(ty) < 25) & (model_top(tx, ty) - ground(tx, ty) < 0.5)
        with rasterio.open(out / "ground.tif", "w", driver="GTiff", height=g.shape[0], width=g.shape[1], count=1, dtype="float32", crs=CRS,
                           nodata=-32767, transform=from_origin(W[0] - 27, W[1] + 27, 0.1, 0.1)) as dst:
            dst.write(np.where(seen, g, -32767).astype(np.float32), 1)
    # a walk: along three sides of the area, in the local ARKit frame (X = E - E0, Z = -(N - N0))
    s = np.linspace(0, 1, 400)
    px = np.interp(s, [0, .4, .6, 1], [-20, 20, 20, -20])
    py = np.interp(s, [0, .4, .6, 1], [-10, -10, 15, 15])
    pe, pn = placement_error(px, py)
    T = np.tile(np.eye(4), (400, 1, 1))
    T[:, 0, 3], T[:, 2, 3] = pe - E0, -(pn - N0)
    np.savez(out / "poses.npz", i=np.arange(400), T=T)


def apply_fit(fit, E, N):
    psi, c, s = np.deg2rad(fit["dyaw_deg_ccw"]), np.cos(np.deg2rad(fit["dyaw_deg_ccw"])), np.sin(np.deg2rad(fit["dyaw_deg_ccw"]))
    dx, dy = E - fit["pivot_E"], N - fit["pivot_N"]
    return fit["pivot_E"] + c * dx - s * dy + fit["dE"], fit["pivot_N"] + s * dx + c * dy + fit["dN"]


def run(tmp, **kw):
    tmp = Path(tmp)
    write_ide(tmp)
    out = tmp / "out"
    write_model(out, **kw)
    return audit_ide.audit(out, tmp / "dsm.tif", tmp / "dtm.tif", report=tmp / "rep", boot=8, plots=kw.get("with_ground", True), verbose=False)


def check_recovery(a, tol=0.10):
    fit = a["fit_vs_gps_georef"]
    px, py = np.meshgrid(np.linspace(-20, 20, 5), np.linspace(-20, 20, 5))
    E, N = placement_error(px.ravel(), py.ravel())
    E2, N2 = apply_fit(fit, E, N)
    err = np.hypot(E2 - (W[0] + px.ravel()), N2 - (W[1] + py.ravel()))
    assert err.max() < tol, f"placement recovered to {err.max():.3f} m"
    assert abs(fit["dyaw_deg_ccw"] + 3.0) < 0.3, fit["dyaw_deg_ccw"]
    assert abs(fit["dz"] + ERR_Z) < 0.10, fit["dz"]
    assert fit["cost_surface"]["clear_single_minimum"]


def test_recovers_offset_and_detects_tilt():
    with tempfile.TemporaryDirectory() as tmp:
        a = run(tmp, tilt_mm_per_m=5.0, tilt_az_deg=60.0)
        check_recovery(a, tol=0.2)  # a tilt is partly confounded with a shift on sloping ground, so a little looser
        t = a["drift"]["ground_plane_and_bowl"]
        assert abs(t["plane_tilt_mm_per_m"] - 5.0) < 1.5, t["plane_tilt_mm_per_m"]
        d = (t["plane_uphill_azimuth_deg"] - 60 + 180) % 360 - 180
        assert abs(d) < 10, t["plane_uphill_azimuth_deg"]
        assert t["plane_tilt_significant"]
        assert (Path(tmp) / "rep" / "audit.json").exists() and (Path(tmp) / "rep" / "audit_tiles.png").exists()
        ide = json.loads((Path(tmp) / "rep" / "georef_ide.json").read_text())
        assert abs(ide["yaw_deg"] + 3.0) < 0.3 and "note" in ide
        assert a["residuals_after_fit"]["open_ground_vs_DTM"]["median"] < 0.1  # the 'MAD' of the plain model is tiny too, tilt aside


def test_clean_model_reports_no_drift():
    with tempfile.TemporaryDirectory() as tmp:
        a = run(tmp)
        check_recovery(a)
        t = a["drift"]["ground_plane_and_bowl"]
        assert t["plane_tilt_mm_per_m"] < 1.5, t["plane_tilt_mm_per_m"]
        sp = a["drift"]["tiles"]["spread"]
        assert sp["n_constrained"] >= 3, sp
        off = [np.hypot(k["dE"], k["dN"]) for k in a["drift"]["tiles"]["offsets"] if k["constrained"]]
        assert np.median(off) < 0.8 and max(off) < 2.0, off  # per-tile scatter on smooth synthetic terrain is ~0.5 m
        assert not sp["horizontal_drift_suspected"]
        assert a["drift"]["path"]["segments"]


def test_ground_from_points_only():
    with tempfile.TemporaryDirectory() as tmp:
        a = run(tmp, with_ground=False)
        fit = a["fit_vs_gps_georef"]  # checked loosely: roofs and trees pollute the fallback ground
        px, py = np.meshgrid(np.linspace(-20, 20, 5), np.linspace(-20, 20, 5))
        E, N = placement_error(px.ravel(), py.ravel())
        E2, N2 = apply_fit(fit, E, N)
        assert np.hypot(E2 - (W[0] + px.ravel()), N2 - (W[1] + py.ravel())).max() < 0.3


def test_ambiguous_scene_is_not_determinable():
    global FLAT
    FLAT = True
    try:
        with tempfile.TemporaryDirectory() as tmp:
            a = run(tmp, tilt_mm_per_m=8.0)  # a real tilt is present but must not be reported as a finding
            f = a["fit_vs_gps_georef"]
            assert f["placement_status"] == "ambiguous_gps_fallback" and f["ambiguity_reasons"], f["placement_status"]
            assert (f["dE"], f["dN"], f["dyaw_deg_ccw"]) == (0.0, 0.0, 0.0)  # GPS placement kept, only dz fitted
            assert abs(f["dz"] + ERR_Z) < 0.2
            d = a["drift"]
            assert d["status"] == "not_determinable" and d["ground_plane_and_bowl"] is None
            assert d["tiles"]["spread"]["horizontal_drift_suspected"] is None
            ide = json.loads((Path(tmp) / "rep" / "georef_ide.json").read_text())
            assert ide["placement_status"] == "ambiguous_gps_fallback" and ide["yaw_deg"] == 0.0
    finally:
        FLAT = False


if __name__ == "__main__":
    test_ambiguous_scene_is_not_determinable()
    test_clean_model_reports_no_drift()
    test_recovers_offset_and_detects_tilt()
    test_ground_from_points_only()
    print("OK")
