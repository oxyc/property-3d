"""Simplified-model check: synth -> fuse -> simplify against the known synthetic scene.

Run: uv run --project . python test_simplify.py     (or pytest)
"""
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, str(Path(__file__).parent))  # not on sys.path when run with -I
import fuse
import simplify
import synth

NS = "{http://www.collada.org/2005/11/COLLADASchema}"


def box_faces():
    """Vertical faces of the synth boxes in the SketchUp frame: (box index, unit normal, offset n.p, centre xy)."""
    faces = []
    for k, (cx, cz, half, yaw) in enumerate(synth.BOXES):
        c, s = np.cos(yaw), np.sin(yaw)
        axes = [np.array([c, s]), np.array([s, -c])]  # box x / z axes as (east, north)
        for ax, h_along, h_across in ((axes[0], half[0], half[2]), (axes[1], half[2], half[0])):
            for sign in (1, -1):
                centre = np.array([cx, -cz]) + sign * h_along * ax
                faces.append((k, sign * ax, float(sign * ax @ centre), centre))
    return faces


def test_simplify():
    with tempfile.TemporaryDirectory() as tmp:
        walk, out = Path(tmp) / "walk", Path(tmp) / "out"
        synth.generate(walk)
        fuse.fuse(walk, out)
        r = simplify.simplify(out, min_height=0.15, min_area=0.15)
        dst = out / "simple"
        el = json.loads((dst / "elements.json").read_text())
        parts = r["parts"]

        # --- terrain: close to the true surface, facing up, few faces
        group, _, _, v, t, uv = parts[0]
        assert group == "terrain" and uv.min() >= 0 and uv.max() <= 1
        err = v[:, 2] - synth.ground_height(v[:, 0], -v[:, 1])
        n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
        sdf = synth.scene_sdf_all(np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1))[1:].min(0)
        free = sdf > 0.8  # the ground grid keeps the lowest point seen, so it follows the tops of objects
        print(f"terrain: {len(t)} faces, height error away from objects: median abs {np.median(np.abs(err[free])) * 100:.1f} cm, "
              f"p90 {np.percentile(np.abs(err[free]), 90) * 100:.1f} cm; model error vs grid p99 {np.percentile(r['terrain_error'], 99) * 100:.1f} cm")
        assert len(t) < 5000 and (n[:, 2] > 0).mean() > 0.99 and free.sum() > 0.3 * len(v)
        assert np.median(np.abs(err[free])) < 0.04 and np.percentile(np.abs(err[free]), 90) < 0.08
        assert np.percentile(r["terrain_error"], 99) <= simplify.TERRAIN_TOL + 1e-9

        # --- planes: box tops become horizontal rectangles, detected vertical planes lie on box faces
        walls = [p for p in el["planes"] if p["kind"] == "vertical"]
        tops = [p for p in el["planes"] if p["kind"] == "horizontal"]
        assert all(abs(p["normal"][2]) < 1e-6 for p in walls) and all(p["normal"] == [0, 0, 1] for p in tops)  # snapped
        matched = 0
        for cx, cz, half, yaw in synth.BOXES:
            top = synth.ground_height(cx, cz) - 0.15 + 2 * half[1]
            for p in tops:
                poly = np.array(p["polygon"])
                if np.hypot(*(poly[:, :2].mean(0) - [cx, -cz])) > 0.3 or abs(p["offset"] - top) > 0.06:
                    continue
                edges = poly[1:3, :2] - poly[:2, :2]
                ang = np.degrees(np.arctan2(edges[0, 1], edges[0, 0]) - yaw) % 90
                ang = min(ang, 90 - ang)
                dims = np.sort(np.linalg.norm(edges, axis=1))
                true = np.sort([2 * half[0], 2 * half[2]])  # a top is only partly seen, so the rectangle may be smaller
                assert p["shape"] == "rect" and ang < 8 and (dims < true + 0.1).all() and (dims > 0.5 * true).all(), (p, ang, dims)
                matched += 1
                break
        faces = box_faces()
        for p in walls:
            n2, centre = np.array(p["normal"][:2]), np.array(p["polygon"])[:, :2].mean(0)
            assert any(np.degrees(np.arccos(np.clip(abs(n2 @ fn), -1, 1))) < 5 and abs(centre @ fn - fd) < 0.05
                       and np.linalg.norm(centre - fc) < 1.0 for _, fn, fd, fc in faces), f"{p['id']} is not on a box face"
        print(f"planes: {matched} of {len(synth.BOXES)} box tops matched (z < 6 cm, yaw < 8 deg), "
              f"{len(walls)} vertical planes, all on box faces within 5 deg / 5 cm")
        assert matched >= 3 and len(walls) >= 1

        # --- trunks: position and radius
        trunks = [t_ for tree in el["trees"] for t_ in tree.get("trunks", [])]
        hits = 0
        for cx, cz, rad, h in synth.CYLINDERS:
            near = [t_ for t_ in trunks if np.hypot(t_["x"] - cx, t_["y"] + cz) < 0.15]
            if near:
                assert abs(near[0]["radius"] - rad) < 0.05, (near[0], rad)
                assert 0.5 < near[0]["height"] < h, (near[0], h)  # only the lower part is within the sensor's reach
                hits += 1
        print(f"trunks: {len(trunks)} detected, {hits} of {len(synth.CYLINDERS)} true trunks matched (position < 15 cm, radius < 5 cm)")
        assert hits >= 3 and len(trunks) <= 6

        # --- files load
        assert r["faces"] < 50_000
        root = ET.parse(dst / "simple.dae").getroot()
        assert root.find(f"{NS}asset/{NS}unit").get("meter") == "1" and root.find(f"{NS}asset/{NS}up_axis").text == "Z_UP"
        assert len(root.findall(f"{NS}library_geometries/{NS}geometry")) == len(parts)
        groups = [g.get("name") for g in root.findall(f"{NS}library_visual_scenes/{NS}visual_scene/{NS}node")]
        assert groups[0] == "terrain" and "walls" in groups and "trees" in groups, groups
        assert (dst / root.find(f"{NS}library_images/{NS}image/{NS}init_from").text).is_file()
        stl = o3d.io.read_triangle_mesh(str(dst / "simple.stl"))
        assert len(stl.triangles) == r["faces"]
        glb = trimesh.load(str(dst / "simple.glb"))
        assert sum(len(g.faces) for g in glb.geometry.values()) == r["faces"]
        assert glb.bounds[1][1] > glb.bounds[0][1] + 1  # world frame: Y is up, the model is more than a metre tall
        print(f"files: {r['faces']} faces; dae {(dst / 'simple.dae').stat().st_size // 1024} kB, "
              f"stl {(dst / 'simple.stl').stat().st_size // 1024} kB, glb {(dst / 'simple.glb').stat().st_size // 1024} kB")


if __name__ == "__main__":
    test_simplify()
    print("OK")
