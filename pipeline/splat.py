"""Gaussian splats from a walk: export a COLMAP-format training set, train with Brush, convert for the web.

usage: splat.py export <walk_dir> <out_dir> [--width 1600] [--max-points 200000] [--drop-blurry 0.35]
       splat.py export-merged <merge_out_dir> [--walks ../walks] [same options as export]
       splat.py train <out_dir> [--steps 30000] [--max-splats 1200000]
       splat.py convert <out_dir> [--min-opacity 0.05] [--max-scale 0.5] [--max-gap 0.5] [--max-splats 1250000]

out_dir is the folder fuse.py / optimize.py wrote (points.ply, poses.npz). export writes
out_dir/splat/data/ (images/ and sparse/0/{cameras,images,points3D}.txt, COLMAP text format); train
runs Brush on it and writes out_dir/splat/scene.ply (standard 3DGS PLY) and scene.splat (compact
antimatter15 format, 32 bytes per gaussian, what splat.html loads). The splat stays in the ARKit
world frame (X east, Y up, Z south, metres), the same frame as points.ply.

The trainer is Brush (https://github.com/ArthurBrussee/brush): $BRUSH, else brush_app on PATH, else
the binary under ../tools/.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script dir from sys.path
import fuse  # noqa: E402

HERE = Path(__file__).resolve().parent
SH_C0 = 0.28209479177387814
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def colmap_pose(T):
    """ARKit camera-to-world T (camera +X right, +Y up, -Z forward) -> COLMAP (qw, qx, qy, qz, tx, ty, tz).
    COLMAP's camera is +X right, +Y down, +Z forward and the pose is world-to-camera:
    flip the camera's Y and Z, then invert."""
    w2c = np.linalg.inv(T @ FLIP)
    x, y, z, w = Rotation.from_matrix(w2c[:3, :3]).as_quat()
    return np.array([w, x, y, z, *w2c[:3, 3]])


def thin(pcd, max_points):
    if len(pcd.points) > 8 * max_points:  # a merged cloud is tens of millions of points: don't voxel-copy all of it
        pcd = pcd.random_down_sample(8 * max_points / len(pcd.points))
    voxel = 0.01
    while len(pcd.points) > max_points:
        voxel += 0.01
        pcd = pcd.voxel_down_sample(voxel)
    return pcd


def sharpness(img):
    """Variance of the Laplacian of a 640 px wide grey copy: low means motion blur or defocus."""
    g = np.asarray(img.convert("L").resize((640, round(640 * img.height / img.width)), Image.BOX), np.float64)
    return float(ndimage.laplace(g).var())


def select_sharp(frames, scores, drop=0.35, cover_dist=0.5, cover_deg=25):
    """Indices into frames to keep. Scores are compared with the median of the neighbouring keyframes
    (so a texture-poor stretch isn't dropped wholesale) and the blurriest `drop` fraction goes, except
    a frame that no kept frame already covers (within cover_dist metres and cover_deg of viewing
    direction) stays, sharpest first."""
    s = np.array(scores)
    rel = np.array([s[k] / np.median(s[max(0, k - 5):k + 6]) for k in range(len(s))])
    pos = np.array([f["T"][:3, 3] for f in frames])
    fwd = np.array([-f["T"][:3, 2] for f in frames])
    keep = list(np.nonzero(rel >= np.quantile(rel, drop))[0])
    for k in np.argsort(-rel):
        if rel[k] >= np.quantile(rel, drop):
            continue
        near = np.linalg.norm(pos[keep] - pos[k], axis=1) < cover_dist
        if not (near & (fwd[keep] @ fwd[k] > np.cos(np.radians(cover_deg)))).any():
            keep.append(k)
    return sorted(int(k) for k in keep)


def export(walk, out, width=1600, max_points=200_000, drop_blurry=0.35):
    out = Path(out)
    poses = out / "poses.npz"
    if not poses.exists():
        print("no poses.npz in out_dir: using the recorded (uncorrected) poses")
    write_dataset([(Path(walk), poses if poses.exists() else None)], out, width, max_points, drop_blurry)


def export_merged(out, walks_root, width=1600, max_points=200_000, drop_blurry=0.35):
    """Training set from a merge.py output: every placed piece's frames, at its merged poses."""
    out, walks_root = Path(out), Path(walks_root)
    pieces = json.loads((out / "merge.json").read_text())["walks"]
    units = [(walks_root / p["walk"], out / f"walk{k}" / "poses_merged.npz") for k, p in enumerate(pieces)]
    write_dataset([u for u in units if u[1].exists()], out, width, max_points, drop_blurry)


def write_dataset(units, out, width, max_points, drop_blurry):
    """units: (walk_dir, poses.npz or None) pairs. A poses file also limits its walk to the frames it lists
    (a split walk's piece lists only its own). Image names get a unit prefix when there is more than one."""
    data = out / "splat" / "data"
    shutil.rmtree(data, ignore_errors=True)
    (data / "images").mkdir(parents=True)
    (data / "sparse" / "0").mkdir(parents=True)

    cameras, images = {}, []
    for u, (walk, poses) in enumerate(units):
        meta, frames = fuse.read_walk(walk)
        if poses is not None:
            z = np.load(poses)
            corrected = dict(zip(z["i"].tolist(), z["T"]))
            frames = [f for f in frames if f["i"] in corrected]
            for f in frames:
                f["T"] = corrected[f["i"]]
        cand, scores = [], []
        for f in frames:
            if not (f["rgb"] and f["tracking"] == "normal"):
                continue
            try:  # a walk cut short by a full disk can end on an empty or truncated JPEG
                scores.append(sharpness(Image.open(walk / "rgb" / f"{f['i']:06d}.jpg")))
            except OSError:
                print(f"{walk.name}: skipped unreadable rgb/{f['i']:06d}.jpg")
                continue
            cand.append(f)
        chosen = select_sharp(cand, scores, drop_blurry)
        print(f"{walk.name}: {len(chosen)} of {len(cand)} keyframes kept (sharpness median {np.median(scores):.0f}, "
              f"kept median {np.median([scores[k] for k in chosen]):.0f})")
        for f in (cand[k] for k in chosen):
            name = f"{u}_{f['i']:06d}.jpg" if len(units) > 1 else f"{f['i']:06d}.jpg"
            img = Image.open(walk / "rgb" / f"{f['i']:06d}.jpg").convert("RGB")
            w, h = width, round(width * img.height / img.width)
            img.resize((w, h), Image.LANCZOS).save(data / "images" / name, quality=92)
            s = w / img.width
            fx, fy, cx, cy = (v * s for v in f["intrinsics"])  # intrinsics are in full-size RGB pixels
            cam = cameras.setdefault((w, h, round(fx, 3), round(fy, 3), round(cx, 3), round(cy, 3)), len(cameras) + 1)
            images.append((name, cam, colmap_pose(f["T"])))
    (data / "sparse" / "0" / "cameras.txt").write_text("".join(
        f"{cid} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n" for (w, h, fx, fy, cx, cy), cid in cameras.items()))
    # second line of each image is its 2D points: none
    (data / "sparse" / "0" / "images.txt").write_text("".join(
        f"{k + 1} {' '.join(repr(float(v)) for v in pose)} {cam} {name}\n\n" for k, (name, cam, pose) in enumerate(images)))

    pcd = thin(o3d.io.read_point_cloud(str(out / "points.ply")), max_points)
    pts, cols = np.asarray(pcd.points), (np.asarray(pcd.colors) * 255).round().astype(int)
    (data / "sparse" / "0" / "points3D.txt").write_text("".join(
        f"{k + 1} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]} 0\n" for k, (p, c) in enumerate(zip(pts, cols))))
    print(f"{len(images)} images at {width} px, {len(cameras)} camera(s), {len(pts)} init points -> {data}")


def find_brush():
    for cand in (os.environ.get("BRUSH"), shutil.which("brush_app"), *sorted((HERE.parent / "tools").glob("*/*/brush_app"))):
        if cand and Path(cand).exists():
            return str(cand)
    sys.exit("Brush not found: set $BRUSH or put brush_app on PATH (https://github.com/ArthurBrussee/brush/releases)")


def read_ply(path):
    """Binary little-endian PLY with one vertex element -> structured array."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        fields, n = [], 0
        for line in iter(f.readline, b"end_header\n"):
            tok = line.split()
            if tok[0] == b"format" and tok[1] != b"binary_little_endian":
                sys.exit(f"{path}: only binary_little_endian PLY is supported")
            if tok[0] == b"element":
                assert tok[1] == b"vertex", "unexpected PLY element " + tok[1].decode()
                n = int(tok[2])
            elif tok[0] == b"property":
                fields.append((tok[2].decode(), "<" + {b"float": "f4", b"double": "f8", b"uchar": "u1", b"int": "i4"}[tok[1]]))
        return np.fromfile(f, np.dtype(fields), n)


def near_scan(pos, points, margin, max_gap):
    """Mask of positions inside the bounding box of `points` (+ margin) and within max_gap of one of them:
    the sky and everything else the depth sensor never saw is floaters."""
    pts = np.asarray(points.voxel_down_sample(0.05).points)
    inside = ((pos >= pts.min(0) - margin) & (pos <= pts.max(0) + margin)).all(1)
    gap = np.full(len(pos), np.inf)
    gap[inside] = cKDTree(pts).query(pos[inside])[0]
    return gap <= max_gap


def ply_to_splat(ply, dest, min_opacity=0.05, max_scale=0.5, max_splats=1_250_000, points=None, margin=0.5, max_gap=0.5):
    """Standard 3DGS PLY -> antimatter15 .splat: per gaussian float32 xyz, float32 scale (linear, metres),
    uint8 RGBA (SH degree 0 colour, opacity), uint8 quaternion (w, x, y, z; q * 128 + 128). Gaussians below
    min_opacity or with a semi-axis over max_scale metres are dropped (floaters), as are, if `points` (an
    Open3D cloud of the scan) is given, those outside its bounding box + margin or farther than max_gap from
    any point. Then the most visible (opacity x volume) max_splats are kept, largest first."""
    v = read_ply(ply)
    pos = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
    alpha = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
    keep = np.isfinite(pos).all(1) & np.isfinite(scale).all(1) & (alpha >= min_opacity) & (scale.max(1) <= max_scale)
    if points is not None:
        keep &= near_scan(np.nan_to_num(pos), points, margin, max_gap)
    order = np.argsort(-(alpha * scale.prod(1)) * keep)[:min(max_splats, keep.sum())]
    rgb = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    rec = np.empty(len(order), [("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    rec["pos"], rec["scale"] = pos[order], scale[order]
    rec["rgba"][:, :3] = np.clip(rgb[order] * 255, 0, 255).round()
    rec["rgba"][:, 3] = np.clip(alpha[order] * 255, 0, 255).round()
    rec["rot"] = np.clip(q[order] * 128 + 128, 0, 255).round()
    rec.tofile(dest)
    print(f"{len(v)} gaussians -> {len(rec)} kept, {Path(dest).stat().st_size / 1e6:.1f} MB")
    return len(rec)


def train(out, steps=30000, max_splats=1_200_000, **prune):
    out = Path(out)
    sp = out / "splat"
    cmd = [find_brush(), str(sp / "data"), "--total-steps", str(steps), "--max-splats", str(max_splats),
           "--growth-stop-iter", str(steps // 2),  # Brush's default stops growth at half of its 30000 steps
           "--export-path", str(sp), "--export-name", "scene.ply", "--export-every", str(max(1000, steps // 4))]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"trained {steps} steps in {(time.time() - t0) / 60:.1f} min")
    convert(out, **prune)


def convert(out, **kw):
    sp = Path(out) / "splat"
    return ply_to_splat(sp / "scene.ply", sp / "scene.splat", points=o3d.io.read_point_cloud(str(Path(out) / "points.ply")), **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("walk_dir")
    e.add_argument("out_dir")
    e.add_argument("--width", type=int, default=1600, help="training image width in px")
    e.add_argument("--max-points", type=int, default=200_000, help="initial points")
    e.add_argument("--drop-blurry", type=float, default=0.35, help="fraction of blurriest keyframes to drop (coverage kept)")
    m = sub.add_parser("export-merged", help="training set from a merge.py output (all pieces, merged poses)")
    m.add_argument("out_dir")
    m.add_argument("--walks", default=str(HERE.parent / "walks"), help="folder holding the walk_* folders")
    for flag, kw in (("--width", dict(type=int, default=1600)), ("--max-points", dict(type=int, default=200_000)),
                     ("--drop-blurry", dict(type=float, default=0.35))):
        m.add_argument(flag, **kw)
    for name in ("train", "convert"):
        p = sub.add_parser(name)
        p.add_argument("out_dir")
        p.add_argument("--min-opacity", type=float, default=0.05)
        p.add_argument("--max-scale", type=float, default=0.5, help="drop gaussians with a semi-axis over this (m)")
        p.add_argument("--max-gap", type=float, default=0.5, help="drop gaussians farther than this from any scan point (m)")
        if name == "train":
            p.add_argument("--steps", type=int, default=30000)
            p.add_argument("--max-splats", type=int, default=1_200_000, help="Brush's cap while training")
        else:
            p.add_argument("--max-splats", type=int, default=1_250_000, help="web file cap (32 bytes each)")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.walk_dir, a.out_dir, a.width, a.max_points, a.drop_blurry)
    elif a.cmd == "export-merged":
        export_merged(a.out_dir, a.walks, a.width, a.max_points, a.drop_blurry)
    else:
        prune = dict(min_opacity=a.min_opacity, max_scale=a.max_scale, max_gap=a.max_gap)
        if a.cmd == "train":
            train(a.out_dir, a.steps, a.max_splats, **prune)
        else:
            convert(a.out_dir, max_splats=a.max_splats, **prune)
