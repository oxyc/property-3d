"""SketchUp export check: synth -> fuse -> export_sketchup against the known synthetic scene.

Run: uv run --project . python test_export.py     (or pytest)
"""
import sys
import tempfile
from pathlib import Path

import laspy
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))  # not on sys.path when run with -I
import export_sketchup
import fuse
import synth


def to_world(p):
    """SketchUp (east, north, up) -> world (X east, Y up, Z south)."""
    return np.stack([p[:, 0], p[:, 2], -p[:, 1]], 1)


def test_export():
    with tempfile.TemporaryDirectory() as tmp:
        walk, out = Path(tmp) / "walk", Path(tmp) / "out"
        synth.generate(walk)
        fuse.fuse(walk, out)
        max_faces = 20_000
        export_sketchup.export(out, 0.25, max_faces)
        sk = out / "sketchup"
        for name in ("terrain.dae", "terrain.png", "scene.dae", "terrain.stl", "scene.stl", "points.las"):
            assert (sk / name).stat().st_size > 0, name

        # --- terrain: heights, orientation, texture
        t = export_sketchup.check_dae(sk / "terrain.dae")
        v, tris, uvs = t["verts"], t["tris"], t["uvs"]
        assert t["unit"] == "1" and t["up_axis"] == "Z_UP"
        assert 1000 < len(tris) <= export_sketchup.TERRAIN_MAX_FACES
        err = v[:, 2] - synth.ground_height(v[:, 0], -v[:, 1])  # world Z = -north
        print(f"terrain: {len(tris)} faces, height error mean {err.mean()*100:.1f} cm, mean abs {np.abs(err).mean()*100:.1f} cm")
        assert np.abs(err).mean() < 0.05
        assert v[:, 1].mean() > 2  # the loop lies north of the start: north = +Y
        n = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
        assert (n[:, 2] > 0).mean() > 0.99  # facing up
        assert t["texture"].name == "terrain.png"
        w, h = Image.open(t["texture"]).size
        assert w > 100 and h > 100
        # planar UVs, not mirrored: u grows east, v grows north, same metres per pixel on both axes
        assert np.corrcoef(uvs[:, 0], v[:, 0])[0, 1] > 0.999 and np.corrcoef(uvs[:, 1], v[:, 1])[0, 1] > 0.999
        assert abs((np.ptp(v[:, 0]) / np.ptp(uvs[:, 0]) / w) / (np.ptp(v[:, 1]) / np.ptp(uvs[:, 1]) / h) - 1) < 0.02

        # --- scene: within the face budget, lies on the true surfaces in world coordinates
        s = export_sketchup.check_dae(sk / "scene.dae")
        d = np.abs(synth.scene_sdf(to_world(s["verts"])))
        print(f"scene:   {len(s['tris'])} faces, |dist to scene| median {np.median(d)*100:.2f} cm, p90 {np.percentile(d, 90)*100:.2f} cm")
        assert 1000 < len(s["tris"]) <= max_faces and s["uvs"] is None and s["texture"] is None
        assert np.median(d) < 0.03 and np.percentile(d, 90) < 0.08

        # --- points
        las = laspy.read(str(sk / "points.las"))
        assert las.header.version.minor == 4 and len(las.points) == len(fuse.o3d.io.read_point_cloud(str(out / "points.ply")).points)
        lp = np.stack([las.x, las.y, las.z], 1)
        dp = np.abs(synth.scene_sdf(to_world(lp)))
        assert np.median(dp) < 0.015 and las.red.max() > 0
        print(f"points:  {len(lp)} in LAS, |dist to scene| median {np.median(dp)*100:.2f} cm")


if __name__ == "__main__":
    test_export()
    print("OK")
