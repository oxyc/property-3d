"""Generate a synthetic walk in the FORMAT.md layout from a known analytic scene.

Scene (ARKit world: X east, Y up, Z south, origin = walk start): sloped, undulating
ground, four yawed boxes ("rocks") and four vertical cylinders ("trunks"), described by
signed distance functions. Depth/RGB are rendered by sphere tracing, so the exact surface
is known and `scene_sdf` can be used to measure reconstruction error.

The camera walks one loop (radius 4.5 m) looking inward, with sway and pitch variation.
Depth is noisy, with low-confidence pixels (random, depth edges, far range) and a few
confidence-1 flying-pixel outliers. GPS is simulated around ORIGIN_EN with iid noise and
two gross outliers that carry a large h_acc. Ground truth goes to truth.json (not part of
the format).

usage: synth.py <out_dir> [--frames 240] [--seed 0] [--drift]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Transformer

# Walk start in EPSG:5382 (centre of data/crop).
ORIGIN_EN = (600000.0, 6150000.0)  # arbitrary round point in EPSG:5382
ORIGIN_ALT = 12.0
DEPTH_W, DEPTH_H = 256, 192
RGB_W, RGB_H = 1920, 1440
FX = FY = 1500.0
CX, CY = RGB_W / 2, RGB_H / 2
HZ = 10
LOOP_R = 4.5
LOOP_CENTRE = np.array([0.0, -LOOP_R])  # (x, z); loop starts at the world origin
SPEED = 1.3
CAM_HEIGHT = 1.3
RGB_EVERY = 10
MAX_RANGE = 5.0  # sensor returns nothing beyond this

# ---------------------------------------------------------------- scene


def ground_height(x, z):
    return -1.3 + 0.08 * x - 0.10 * z + 0.15 * np.sin(0.5 * x) * np.sin(0.4 * z)


def _ground_sdf(p):
    x, z = p[:, 0], p[:, 2]
    hx = 0.08 + 0.075 * np.cos(0.5 * x) * np.sin(0.4 * z)
    hz = -0.10 + 0.06 * np.sin(0.5 * x) * np.cos(0.4 * z)
    return (p[:, 1] - ground_height(x, z)) / np.sqrt(1 + hx**2 + hz**2)


# (cx, cz, half extents x/y/z, yaw)
BOXES = [
    (1.2, -3.2, (0.5, 0.35, 0.4), 0.5),
    (-1.6, -5.4, (0.7, 0.5, 0.5), -0.3),
    (0.4, -6.2, (0.3, 0.25, 0.3), 1.0),
    (2.0, -4.8, (0.4, 0.3, 0.6), 0.2),
]
# (cx, cz, radius, height)
CYLINDERS = [
    (-0.8, -3.6, 0.15, 2.5),
    (0.0, -4.5, 0.25, 3.0),
    (1.6, -6.5, 0.20, 2.5),
    (-2.2, -3.2, 0.30, 2.8),
]
ALBEDO = [  # index 0 = ground (checker, see albedo()), then boxes, then cylinders
    None,
    (150, 140, 130), (170, 110, 90), (120, 120, 130), (190, 170, 120),
    (110, 70, 40), (90, 60, 35), (130, 90, 55), (100, 65, 45),
]


def _box_sdf(p, cx, cz, half, yaw):
    cy = ground_height(cx, cz) + half[1] - 0.15  # sunk slightly into the slope
    d = p - np.array([cx, cy, cz])
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.stack([c * d[:, 0] - s * d[:, 2], d[:, 1], s * d[:, 0] + c * d[:, 2]], 1)
    q = np.abs(local) - np.array(half)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)


def _cyl_sdf(p, cx, cz, r, h):
    base = ground_height(cx, cz) - 0.2
    w = np.stack([np.hypot(p[:, 0] - cx, p[:, 2] - cz) - r, np.abs(p[:, 1] - (base + h / 2)) - h / 2], 1)
    return np.linalg.norm(np.maximum(w, 0), axis=1) + np.minimum(w.max(1), 0)


def scene_sdf_all(p):
    """(K, N) signed distances; row 0 ground, then boxes, then cylinders."""
    return np.stack([_ground_sdf(p)] + [_box_sdf(p, *b) for b in BOXES] + [_cyl_sdf(p, *c) for c in CYLINDERS])


def scene_sdf(p):
    return scene_sdf_all(p).min(0)


def albedo(p, ids):
    """RGB (uint8) of the surface at points p belonging to object ids."""
    out = np.array([a or (0, 0, 0) for a in ALBEDO], np.uint8)[ids]
    check = (np.floor(p[:, 0] / 0.5) + np.floor(p[:, 2] / 0.5)) % 2 == 0
    ground = np.where(check[:, None], (70, 120, 60), (100, 145, 80)).astype(np.uint8)
    return np.where((ids == 0)[:, None], ground, out)


# ---------------------------------------------------------------- rendering


def cam_rotation(forward):
    """Camera-to-world rotation, ARKit axes: columns are camera +X (right), +Y (up), +Z (= -forward)."""
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, -forward], 1)


def trace(origin, dirs, t_max, iters=64):
    """Sphere-trace unit rays. Returns (t, hit)."""
    t = np.zeros(len(dirs))
    hit = np.zeros(len(dirs), bool)
    active = np.arange(len(dirs))
    for _ in range(iters):
        d = scene_sdf(origin + t[active, None] * dirs[active])
        done = d < 1e-3
        hit[active[done]] = True
        t[active[~done]] += d[~done]
        active = active[~done & (t[active] < t_max)]
        if not len(active):
            break
    return t, hit


def render(R, pos, u, v, t_max):
    """Trace rays through RGB-pixel coords (u, v). Returns z-depth, hit mask, world hit points, object ids."""
    dc = np.stack([(u - CX) / FX, -(v - CY) / FY, -np.ones_like(u)], 1)  # camera-space ray, z = -1
    norm = np.linalg.norm(dc, axis=1)
    dirs = (dc / norm[:, None]) @ R.T
    t, hit = trace(pos, dirs, t_max)
    pts = pos + t[:, None] * dirs
    ids = scene_sdf_all(pts).argmin(0)
    return t / norm, hit, pts, ids  # t along unit ray -> planar depth = t * |dir_z| = t / norm


# ---------------------------------------------------------------- walk


def camera_pose(t, phase=0.0, shift=(0.0, 0.0)):
    """Scene-frame pose at time t. phase: where on the loop the walk starts (rad); shift: loop centre offset (x, z)."""
    a = SPEED * t / LOOP_R + phase
    centre = LOOP_CENTRE + np.array(shift)
    x, z = centre[0] + LOOP_R * np.sin(a), centre[1] + LOOP_R * np.cos(a)
    pos = np.array([x, ground_height(x, z) + CAM_HEIGHT, z])
    d = centre - pos[[0, 2]]
    d /= np.linalg.norm(d)
    sway = 0.35 * np.sin(0.8 * t)
    d = np.array([np.cos(sway) * d[0] + np.sin(sway) * d[1], -np.sin(sway) * d[0] + np.cos(sway) * d[1]])
    pitch = -0.5 + 0.08 * np.sin(1.3 * t)
    fwd = np.array([d[0] * np.cos(pitch), np.sin(pitch), d[1] * np.cos(pitch)])
    return cam_rotation(fwd), pos, fwd


def simulate_depth(rng, R, pos):
    u, v = np.meshgrid(np.arange(DEPTH_W), np.arange(DEPTH_H))
    # FORMAT.md: depth pixel -> RGB pixel by plain scaling
    z, hit, _, _ = render(R, pos, (u * RGB_W / DEPTH_W).ravel(), (v * RGB_H / DEPTH_H).ravel(), MAX_RANGE + 1)
    z = z.reshape(DEPTH_H, DEPTH_W)
    hit = hit.reshape(DEPTH_H, DEPTH_W) & (z <= MAX_RANGE)
    depth = np.where(hit, z + rng.normal(size=z.shape) * (0.003 + 0.004 * z), 0.0)
    conf = np.where(hit, 2, 0)
    conf[hit & (depth > 3) & (rng.random(z.shape) < 0.4)] = 1
    gy, gx = np.gradient(np.where(hit, depth, 10.0))
    conf[np.maximum(abs(gy), abs(gx)) > 0.1 * np.maximum(depth, 1)] = 0  # depth edges
    bad = rng.random(z.shape) < 0.04
    conf[bad] = 0
    depth[bad] = rng.uniform(0.2, 4.5, bad.sum())
    fly = rng.random(z.shape) < 0.001  # confidence-1 flying pixels, left for outlier removal
    conf[fly] = 1
    depth[fly] = rng.uniform(0.3, 4.0, fly.sum())
    return depth.astype(np.float32), conf.astype(np.uint8)


def simulate_rgb(R, pos, step=4):
    w, h = RGB_W // step, RGB_H // step
    u, v = np.meshgrid(np.arange(w) * step + (step - 1) / 2, np.arange(h) * step + (step - 1) / 2)
    _, hit, pts, ids = render(R, pos, u.ravel(), v.ravel(), 12)
    rgb = np.where(hit[:, None], albedo(pts, ids), (150, 190, 230)).astype(np.uint8).reshape(h, w, 3)
    return Image.fromarray(rgb).resize((RGB_W, RGB_H), Image.BICUBIC)


def simulate_gps(rng, duration, to_ll, **path):
    rows = []
    for k, t in enumerate(np.arange(0, duration, 1.0)):
        R, pos, fwd = camera_pose(t, **path)
        acc = rng.uniform(3, 5)
        err = rng.normal(size=2) * acc
        if k in (5, 17):  # gross outliers, honestly reported as poor
            acc, err = 35.0, err + 30
        e, n = ORIGIN_EN[0] + pos[0] + err[0], ORIGIN_EN[1] - pos[2] + err[1]
        lon, lat = to_ll.transform(e, n)
        rows.append({"t": float(t), "lat": lat, "lon": lon, "alt": ORIGIN_ALT + pos[1] + rng.normal() * 3,
                     "h_acc": float(acc), "v_acc": 3.0,
                     "heading": float(np.degrees(np.arctan2(fwd[0], -fwd[2])) % 360 + rng.normal() * 3),
                     "heading_acc": 10})
    return rows


def drift_offsets(rng, true_pos):
    """Accumulating ARKit-style drift per frame: (yaw about +Y at the world origin, translation).
    Yaw grows linearly to 1-2 degrees over the walk. Translation integrates 1.5% of every step
    walked in a slowly wandering horizontal direction, plus 0.5% vertical, so it grows over time."""
    n = len(true_pos)
    yaw = np.radians(rng.uniform(1, 2) * rng.choice([-1, 1])) * np.arange(n) / max(n - 1, 1)
    step = np.r_[0, np.linalg.norm(np.diff(true_pos, axis=0), axis=1)]
    heading = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(size=n) * 0.05)
    rate = np.stack([0.015 * np.cos(heading), np.full(n, 0.005), 0.015 * np.sin(heading)], 1)
    return yaw, np.cumsum(rate * step[:, None], axis=0)


def generate(out_dir, n_frames=240, seed=0, drift=False, phase=0.0, yaw_deg=0.0, shift=(0.0, 0.0)):
    """drift=True records drifted transforms (the true ones go to truth.json) but renders depth from the true poses.
    phase / shift start the walk elsewhere on the loop / move the loop (a different part of the ground); the recorded
    ARKit frame then has its origin at the walk's start and is rotated by yaw_deg about +Y (compass error) relative to
    the scene, while GPS still follows the true position. truth.json gets `to_scene` (recorded frame -> scene frame)."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    for sub in ("depth", "conf", "rgb"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    true_M = []
    for i in range(n_frames):
        R, pos, _ = camera_pose(i / HZ, phase, shift)
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R, pos
        true_M.append(M)
    rec_M = true_M
    to_scene = None
    if phase or yaw_deg or any(shift):
        a = np.radians(yaw_deg)
        D = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        F = np.eye(4)  # scene -> recorded frame
        F[:3, :3], F[:3, 3] = D, -D @ true_M[0][:3, 3]
        rec_M = [F @ M for M in true_M]
        to_scene = np.linalg.inv(F)
    true_rec = rec_M  # the true poses in the recorded frame
    if drift:
        yaw, offset = drift_offsets(np.random.default_rng(seed + 1), np.array([M[:3, 3] for M in true_rec]))
        rec_M = []
        for M, a, d in zip(true_rec, yaw, offset):
            c, s = np.cos(a), np.sin(a)
            D = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            rec_M.append(np.block([[D @ M[:3, :3], (D @ M[:3, 3] + d)[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))

    frames = []
    for i in range(n_frames):
        t = i / HZ
        R, pos = true_M[i][:3, :3], true_M[i][:3, 3]
        M = rec_M[i]
        depth, conf = simulate_depth(rng, R, pos)
        depth.astype("<f2").tofile(out / "depth" / f"{i:06d}.bin")  # row-major float16 (format v2)
        conf.tofile(out / "conf" / f"{i:06d}.bin")
        has_rgb = i % RGB_EVERY == 0
        if has_rgb:
            simulate_rgb(R, pos).save(out / "rgb" / f"{i:06d}.jpg", quality=80)
        frames.append({"i": i, "t": t, "transform": M.flatten(order="F").tolist(),  # column-major
                       "intrinsics": [FX, FY, CX, CY], "tracking": "normal", "rgb": has_rgb})
    (out / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))

    to_ll = Transformer.from_crs("EPSG:5382", "EPSG:4326", always_xy=True)
    gps = simulate_gps(rng, n_frames / HZ, to_ll, phase=phase, shift=shift)
    (out / "gps.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gps))
    (out / "meta.json").write_text(json.dumps({
        "format_version": 2, "depth_dtype": "f16", "app_version": "synth", "device": "synthetic",
        "started_at": "2026-10-19T15:00:00Z", "ended_at": "2026-10-19T15:00:24Z",
        "world_alignment": "gravityAndHeading", "depth_w": DEPTH_W, "depth_h": DEPTH_H,
        "rgb_w": RGB_W, "rgb_h": RGB_H, "frame_count": n_frames}, indent=2))
    truth = {"origin_en": ORIGIN_EN, "origin_alt": ORIGIN_ALT}
    if drift:
        truth["true_transforms"] = [M.flatten(order="F").tolist() for M in true_rec]  # column-major, like frames.jsonl
    if to_scene is not None:
        truth["to_scene"] = to_scene.flatten(order="F").tolist()
    (out / "truth.json").write_text(json.dumps(truth))
    print(f"wrote {n_frames} frames, {sum(f['rgb'] for f in frames)} rgb, {len(gps)} gps fixes to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift", action="store_true", help="record drifted poses (use --frames 440 for two loops)")
    ap.add_argument("--phase", type=float, default=0.0, help="start angle on the loop, radians")
    ap.add_argument("--yaw", type=float, default=0.0, help="recorded frame yaw error, degrees")
    ap.add_argument("--shift", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Z"), help="move the loop, metres")
    a = ap.parse_args()
    generate(a.out_dir, a.frames, a.seed, a.drift, a.phase, a.yaw, tuple(a.shift))
