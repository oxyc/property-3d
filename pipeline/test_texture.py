"""Texturing check: synth -> fuse -> texture, baked colours compared with the synthetic scene's known albedo.

Run: uv run --project . python test_texture.py     (or pytest)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))  # not on sys.path when run with -I
import export_sketchup
import fuse
import synth
import texture

ATLAS = 2048


def test_texture():
    with tempfile.TemporaryDirectory() as tmp:
        walk, out = Path(tmp) / "walk", Path(tmp) / "out"
        synth.generate(walk)
        fuse.fuse(walk, out)
        max_faces = 20_000
        texture.texture(walk, out, max_faces, ATLAS)
        dst = out / "textured"
        for name in ("model.glb", "model_mobile.glb", "model.obj", "model.mtl", "texture.jpg", "model.dae"):
            assert (dst / name).stat().st_size > 0, name
        small = trimesh.load(dst / "model_mobile.glb", process=False)
        small = next(iter(small.geometry.values()))
        assert len(small.faces) > 1000 and small.visual.material.baseColorTexture.size == (texture.MOBILE_MAX_TEX,) * 2

        # --- GLB loads, within the face budget, UVs in range, texture embedded
        scene = trimesh.load(dst / "model.glb", process=False)
        assert len(scene.geometry) == 1
        mesh = next(iter(scene.geometry.values()))
        V, F, uv = np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.visual.uv)
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))
        print(f"glb: {len(F)} faces, {len(V)} vertices, texture {tex.shape[1]}x{tex.shape[0]}")
        assert 1000 < len(F) <= max_faces and tex.shape[:2] == (ATLAS, ATLAS)
        assert uv.min() >= 0 and uv.max() <= 1 and len(uv) == len(V)
        assert V[:, 2].mean() < -2 and -3 < V[:, 1].mean() < 1  # world frame: the loop lies in -Z (north), ground near Y = -1
        assert not np.isnan(V).any() and F.max() < len(V)

        # --- OBJ and DAE describe the same mesh
        obj = trimesh.load(dst / "model.obj", process=False, force="mesh")
        assert len(obj.faces) == len(F) and np.allclose(obj.vertices, V, atol=1e-4)
        assert np.allclose(obj.visual.uv, uv, atol=1e-5)
        assert "texture.jpg" in (dst / "model.mtl").read_text()
        d = export_sketchup.check_dae(dst / "model.dae")
        assert d["texture"].name == "texture.jpg" and d["unit"] == "1" and d["up_axis"] == "Z_UP"
        assert np.allclose(d["verts"][:, 2], V[:, 1], atol=1e-4) and np.allclose(d["verts"][:, 1], -V[:, 2], atol=1e-4)
        assert (d["tris"] == F).all()
        assert Image.open(dst / "texture.jpg").size == (ATLAS, ATLAS)

        # --- baked colours match the known albedo on surfaces away from colour boundaries
        rng = np.random.default_rng(1)
        f = rng.integers(len(F), size=60_000)
        b = rng.dirichlet([1, 1, 1], size=len(f))
        p = np.einsum("nc,ncd->nd", b, V[F[f]])
        q = np.einsum("nc,ncd->nd", b, uv[F[f]])
        sdf = np.sort(synth.scene_sdf_all(p), axis=0)
        ids = synth.scene_sdf_all(p).argmin(0)
        on_surface = np.abs(sdf[0]) < 0.05
        isolated = sdf[1] - sdf[0] > 0.15  # not near another object (ground meeting a box, ...)
        off_checker = (np.abs(((p[:, [0, 2]] / 0.5 + 0.5) % 1) - 0.5) < 0.35).all(1) | (ids != 0)  # 0.15 m from lines
        use = on_surface & isolated & off_checker
        got = tex[np.clip(((1 - q[use, 1]) * ATLAS).astype(int), 0, ATLAS - 1), np.clip((q[use, 0] * ATLAS).astype(int), 0, ATLAS - 1)].astype(float)
        want = synth.albedo(p[use], ids[use]).astype(float)
        err = np.abs(got - want).max(1)
        # ~75% of the synthetic faces are seen by a keyframe and match the albedo; the others are coloured from the
        # points, which are grey where fuse.py found no keyframe, so only the photo-textured share is checked
        print(f"baked colours: {use.sum()} sample points, error median {np.median(err):.1f}, "
              f"p60 {np.percentile(err, 60):.1f}, p90 {np.percentile(err, 90):.1f} (of 255)")
        assert np.median(err) < 8 and np.percentile(err, 60) < 10


if __name__ == "__main__":
    test_texture()
    print("OK")
