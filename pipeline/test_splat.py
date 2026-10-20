"""Splat export check: synth walk -> splat.export -> project known scene points through the exported
COLMAP cameras and compare with where (and what colour) they are in the exported images.

Run: uv run --project . python -I test_splat.py     (or pytest)
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))  # not on sys.path when run with -I
import fuse
import splat
import synth


def read_colmap(sp):
    """Parse the exported text model independently of the writer: {name: (W, H, fx, fy, cx, cy, R, t)} with
    x_cam = R x_world + t, plus the points3D rows."""
    cams = {}
    for line in (sp / "cameras.txt").read_text().splitlines():
        c = line.split()
        assert c[1] == "PINHOLE"
        cams[int(c[0])] = [int(c[2]), int(c[3])] + [float(v) for v in c[4:8]]
    out = {}
    lines = (sp / "images.txt").read_text().split("\n")
    for k in range(0, len(lines) - 1, 2):
        c = lines[k].split()
        qw, qx, qy, qz, tx, ty, tz = map(float, c[1:8])
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        out[c[9]] = (*cams[int(c[8])], R, np.array([tx, ty, tz]))
    return out, [l.split() for l in (sp / "points3D.txt").read_text().splitlines()]


def project(model, X):
    w, h, fx, fy, cx, cy, R, t = model
    c = X @ R.T + t
    return np.stack([fx * c[:, 0] / c[:, 2] + cx, fy * c[:, 1] / c[:, 2] + cy], 1), c[:, 2]


def test_export_conventions():
    with tempfile.TemporaryDirectory() as tmp:
        walk, out = Path(tmp) / "walk", Path(tmp) / "out"
        synth.generate(walk, n_frames=60, drift=True)  # recorded poses drift; poses.npz below holds the true ones
        meta, frames = fuse.read_walk(walk)
        true_T = [np.array(t).reshape(4, 4, order="F") for t in json.loads((walk / "truth.json").read_text())["true_transforms"]]
        out.mkdir()
        np.savez(out / "poses.npz", i=np.array([f["i"] for f in frames]), T=np.array(true_T))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.default_rng(0).uniform(-3, 3, (500, 3)))
        pcd.colors = o3d.utility.Vector3dVector(np.random.default_rng(1).uniform(0, 1, (500, 3)))
        o3d.io.write_point_cloud(str(out / "points.ply"), pcd)

        width = 640
        splat.export(walk, out, width=width, max_points=300)
        data = out / "splat" / "data"
        model, pts = read_colmap(data / "sparse" / "0")
        assert len(model) == 6 and len(list((data / "images").glob("*.jpg"))) == 6
        assert 0 < len(pts) <= 300 and len(pts[0]) == 8

        # Rays through known pixels of the full-size render hit known world points; those points must
        # project onto the same pixel (scaled) in the exported camera, in front of it, at the planar depth.
        s = width / synth.RGB_W
        errs, derrs, cerrs = [], [], []
        for f, true in zip(frames, true_T):
            if not f["rgb"]:
                continue
            name = f"{f['i']:06d}.jpg"
            u, v = np.meshgrid(np.arange(100, synth.RGB_W - 100, 150.0), np.arange(100, synth.RGB_H - 100, 150.0))
            u, v = u.ravel(), v.ravel()
            z, hit, X, ids = synth.render(true[:3, :3], true[:3, 3], u, v, 12)
            uv, depth = project(model[name], X[hit])
            errs.append(np.linalg.norm(uv - np.stack([u[hit], v[hit]], 1) * s, axis=1))
            derrs.append(depth - z[hit])
            img = np.asarray(Image.open(data / "images" / name))
            assert img.shape[:2] == model[name][1::-1]  # (H, W) as the camera says
            px = img[np.clip(uv[:, 1].astype(int), 0, img.shape[0] - 1), np.clip(uv[:, 0].astype(int), 0, img.shape[1] - 1)]
            cerrs.append(np.abs(px.astype(float) - synth.albedo(X[hit], ids[hit])).mean(1))
        errs, derrs, cerrs = map(np.concatenate, (errs, derrs, cerrs))
        print(f"{len(errs)} points: pixel error median {np.median(errs):.2f} max {errs.max():.2f}, "
              f"depth error max {np.abs(derrs).max() * 100:.2f} cm, colour error median {np.median(cerrs):.1f}")
        assert len(errs) > 100
        assert errs.max() < 1.0 and np.abs(derrs).max() < 1e-3
        assert np.median(cerrs) < 12  # same surface colour in the exported JPEG (edges and JPEG blur aside)

        # a camera looking along its own forward axis must see a point ahead at positive z
        T = true_T[0]
        ahead = T[:3, 3] + 2 * (-T[:3, 2])
        assert project(model["000000.jpg"], ahead[None])[1][0] > 0


def test_ply_to_splat():
    with tempfile.TemporaryDirectory() as tmp:
        n = 3
        names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        v = np.zeros(n, [(k, "<f4") for k in names])
        v["x"], v["y"], v["z"] = [1, 2, 3], [4, 5, 6], [7, 8, 9]
        v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = 1.0, 0.0, -1.0
        v["opacity"] = [0.0, 5.0, -9.0]  # alpha 0.5, 0.993, 0.0001 (dropped)
        v["scale_0"] = v["scale_1"] = v["scale_2"] = [np.log(0.1), np.log(0.2), np.log(0.1)]
        v["rot_0"], v["rot_1"] = 2.0, 0.0  # unnormalised identity
        with open(Path(tmp) / "a.ply", "wb") as f:
            f.write(("ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
                     + "".join(f"property float {k}\n" for k in names) + "end_header\n").encode())
            f.write(v.tobytes())
        assert splat.ply_to_splat(Path(tmp) / "a.ply", Path(tmp) / "a.splat") == 2
        raw = np.fromfile(Path(tmp) / "a.splat", np.uint8)
        assert raw.size == 2 * 32
        rec = np.frombuffer(raw.tobytes(), [("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
        assert rec["pos"][0].tolist() == [2, 5, 8]  # the larger, more opaque gaussian is first
        assert np.allclose(rec["scale"][0], 0.2)
        assert rec["rgba"][0].tolist() == [199, 128, 56, 253]
        assert rec["rot"][0].tolist() == [255, 128, 128, 128]


def test_select_and_crop():
    sharp = Image.fromarray(np.random.default_rng(0).integers(0, 255, (360, 480), np.uint8))
    blurred = sharp.resize((24, 18)).resize((480, 360), Image.BILINEAR)
    assert splat.sharpness(sharp) > 20 * splat.sharpness(blurred)

    def frame(x, yaw):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("y", yaw, degrees=True).as_matrix()
        T[0, 3] = x
        return {"T": T}
    # 0..3 form a run 0.2 m apart; 4 is alone 10 m away. Blurry ones go unless nothing else covers them.
    frames = [frame(0.0, 0), frame(0.2, 0), frame(0.4, 0), frame(0.6, 0), frame(10, 0), frame(0.3, 90)]
    keep = splat.select_sharp(frames, [100, 90, 5, 100, 1, 3], drop=0.5)
    assert 2 not in keep  # blurry and covered by its neighbours
    assert 4 in keep and 5 in keep  # blurry but the only view of their place / direction

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(np.random.default_rng(0).uniform(0, 4, (5000, 3)))
    pos = np.array([[2, 2, 2], [2, 2, 4.3], [2, 2, 9], [-3, 2, 2]], float)
    assert splat.near_scan(pos, pts, margin=0.5, max_gap=0.5).tolist() == [True, True, False, False]


if __name__ == "__main__":
    test_export_conventions()
    test_select_and_crop()
    test_ply_to_splat()
    print("OK")
