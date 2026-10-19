"""Site plan check: synth -> fuse -> siteplan against the known synthetic scene, plus a hand-made wall.

Run: uv run --project . python test_siteplan.py     (or pytest)
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # not on sys.path when run with -I
import fuse
import siteplan
import synth


def test_siteplan():
    with tempfile.TemporaryDirectory() as tmp:
        walk, out = Path(tmp) / "walk", Path(tmp) / "out"
        synth.generate(walk)
        fuse.fuse(walk, out)
        feats = siteplan.siteplan(out, interval=0.25, min_h=1.2)
        sp = out / "siteplan"
        for name in ("siteplan.svg", "siteplan.pdf", "siteplan.png", "features.json"):
            assert (sp / name).stat().st_size > 0, name
        assert json.loads((sp / "features.json").read_text())["scale"].startswith("1:")
        svg = (sp / "siteplan.svg").read_text()
        assert 'class="layer-contours-minor"' in svg and 'class="layer-trees"' in svg
        assert (sp / "siteplan.pdf").read_bytes().startswith(b"%PDF")

        # --- contours: every vertex lies at its level above the lowest ground point, within a few cm
        a, smooth, west, north = siteplan.ground_surface(np.load(out / "ground.npz"))
        base = float(np.nanmin(smooth))
        cont = siteplan.contours(smooth, west, north, 0.25)
        assert sorted(cont) == [0.25, 0.5, 0.75]  # the scanned ground spans a little over 1 m
        errs = []
        for lv, lines in cont.items():
            for ln in lines:
                if siteplan.length(ln) < 4:  # small loops are boxes: objects taller than the cell show up as terrain bumps
                    continue
                errs.append(synth.ground_height(ln[:, 0], -ln[:, 1]) - (base + lv))
        errs = np.concatenate(errs)
        print(f"contours: {len(cont)} levels, {len(errs)} vertices, height error mean {errs.mean()*100:.1f} cm, "
              f"p95 abs {np.percentile(np.abs(errs), 95)*100:.1f} cm")
        assert abs(errs.mean()) < 0.05 and np.percentile(np.abs(errs), 95) < 0.1

        # --- trunks: each synthetic cylinder has a tree within 0.4 m, with a sensible height
        trees = feats["trees"]
        print(f"trees: {len(trees)} found")
        for cx, cz, r, hgt in synth.CYLINDERS:
            d = [np.hypot(t["x"] - cx, t["y"] + cz) for t in trees]
            k = int(np.argmin(d))
            print(f"  trunk at ({cx}, {-cz}): nearest tree {d[k]*100:.0f} cm away, {trees[k]['height_m']} m (true height {hgt} m, 0.2 m of it below ground)")
            assert d[k] < 0.4
            assert 0.8 * hgt - 1.5 < trees[k]["height_m"] < hgt + 0.5
        assert len(trees) <= 2 * len(synth.CYLINDERS)

        # --- boxes are under the 1.2 m threshold: neither features nor terrain bumps (clean_grid drops them)
        loops = [ln.mean(0) for lines in cont.values() for ln in lines if siteplan.length(ln) < 4]
        near = sum(any(np.hypot(p[0] - bx, p[1] + bz) < 1.0 for p in loops) for bx, bz, _, _ in synth.BOXES)
        print(f"boxes: {near} of {len(synth.BOXES)} show up as contour loops")
        assert near <= 1
        assert not feats["buildings"]  # nothing is both tall and wall-like


def test_wall():
    """A 6 m x 2.5 m planar wall next to a noisy bush, on flat ground."""
    rng = np.random.default_rng(1)
    gx, gz = np.meshgrid(np.arange(-3, 9, 0.02), np.arange(-4, 4, 0.02))
    ground = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], 1)
    wx, wy = np.meshgrid(np.arange(0, 6, 0.02), np.arange(0.0, 2.5, 0.02))
    wall = np.stack([wx.ravel(), wy.ravel(), np.full(wx.size, 3.0) + rng.normal(size=wx.size) * 0.005], 1)
    bush = rng.normal(size=(20000, 3)) * [0.4, 0.4, 0.4] + [-1.5, 1.8, -2.0]
    pts = np.vstack([ground, wall, bush])
    h = pts[:, 1].copy()
    walls, footprint = siteplan.detect_walls(pts, h, 1.5)
    print(f"wall test: {len(walls)} walls", [(round(w['length'], 1), round(w['height'], 1)) for w in walls])
    assert len(walls) == 1
    w = walls[0]
    assert abs(w["length"] - 6) < 0.5 and 2.0 < w["height"] < 2.7
    c = (w["outline"].min(0) + w["outline"].max(0)) / 2
    assert abs(c[0] - 3) < 0.3 and abs(c[1] + 3) < 0.3  # north = -Z
    assert footprint[len(ground):len(ground) + len(wall)].mean() > 0.95 and not footprint[-20000:].any()


if __name__ == "__main__":
    test_wall()
    test_siteplan()
    print("OK")
