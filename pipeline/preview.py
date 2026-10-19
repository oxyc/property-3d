"""Render quick PNG previews of a fused point cloud: top-down and a side view.

Usage: preview.py <out_dir>   (reads points.ply, writes preview_top.png, preview_side.png)
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


def splat(u, v, colors, depth, size=1000):
    """Orthographic splat: nearest point (smallest depth) wins each pixel."""
    span = max(np.ptp(u), np.ptp(v), 1e-6)
    px = ((u - u.min()) / span * (size - 1)).astype(int)
    py = ((v.max() - v) / span * (size - 1)).astype(int)
    order = np.argsort(-depth)  # far first, near overwrites
    img = np.full((size, size, 3), 255, np.uint8)
    img[py[order], px[order]] = (colors[order] * 255).astype(np.uint8)
    return Image.fromarray(img)


out = Path(sys.argv[1])
pcd = o3d.io.read_point_cloud(str(out / "points.ply"))
p = np.asarray(pcd.points)
c = np.asarray(pcd.colors)

# World: X east, Y up, Z south. Top-down = north up, east right; nearest = highest.
splat(p[:, 0], -p[:, 2], c, -p[:, 1]).save(out / "preview_top.png")
# Side view looking north: X right, Y up; nearest = most southern.
splat(p[:, 0], p[:, 1], c, -p[:, 2]).save(out / "preview_side.png")
print("wrote", out / "preview_top.png", out / "preview_side.png")
