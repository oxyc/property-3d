# pipeline

## Normal use

```
# 1. Pull a walk off the phone (cable or Wi-Fi; no AirDrop needed). Find <device-id> with
#    `xcrun devicectl list devices`, then list walks:
xcrun devicectl device info files --device <device-id> --domain-type appDataContainer --domain-identifier fi.genero.walkscan --subdirectory Documents
xcrun devicectl device copy from --device <device-id> --domain-type appDataContainer --domain-identifier fi.genero.walkscan --source Documents/<walk_name> --destination ../walks/<walk_name>

# 2. Walk folder (or an AirDropped .zip) -> point cloud, mesh, ground grid, georef, previews, viewer
uv run python process.py ../walks/<walk_name>

# 3. Look at it in 3D
python3 -m http.server -d ../out/<walk_name> 8000    # then open http://localhost:8000/viewer.html
```

## Individual steps

Run from this directory. Python 3.11/3.12 via uv (open3d is pinned to 0.19; 0.20 needs a Homebrew libusb).

```
uv run python synth.py walks/synth            # synthetic walk in FORMAT.md layout (240 frames, ~1 min)
uv run python fuse.py walks/synth out/synth   # points.ply, mesh.ply, ground.npz, ground.png
uv run python georef.py out/synth walks/synth # georef.json, georef_points.ply, ground.tif (EPSG:5382, ~3-5 m accuracy)
uv run python test_pipeline.py                # all three against the known scene, plus the drift case (440 frames, a few minutes), in a temp dir
uv run python optimize.py walks/<walk> out/<walk>   # drift correction: poses.npz + optimize.json (process.py runs this first)
uv run python fuse.py walks/<walk> out/<walk> --poses out/<walk>/poses.npz
uv run python closure.py walks/<walk> out/<walk>/poses.npz   # start-vs-end error, for walks that start and end on a marked spot
```

`optimize.py` splits the walk into ~100-frame fragments, finds loop closures by yaw-only ICP between fragments whose paths revisit each other, and prunes bad ones in an Open3D pose graph. `synth.py --drift --frames 440` makes a two-loop walk with accumulating drift for testing.

`georef_points.ply` stores coordinates minus `offset` from `georef.json`; add it back to get EPSG:5382.
