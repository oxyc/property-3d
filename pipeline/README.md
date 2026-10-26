# pipeline

## Normal use

```
# 1a. Pull a walk off the phone over Wi-Fi (fast, resumable, no zip, no extra space on the phone):
#     in the app open Walks > (...) > Send to Mac, then run the command it shows, e.g.
python3 pull.py http://192.168.1.23:51234            # -> ../walks/<walk_name>, verified; add --tar for one streamed tar
#     The app then offers "Delete from phone". Re-run the same command to resume after a dropped connection.

# 1b. Or by cable with devicectl (slow over Wi-Fi with 50k small files). Find <device-id> with
#    `xcrun devicectl list devices`, then list walks:
xcrun devicectl device info files --device <device-id> --domain-type appDataContainer --domain-identifier fi.genero.walkscan --subdirectory Documents
xcrun devicectl device copy from --device <device-id> --domain-type appDataContainer --domain-identifier fi.genero.walkscan --source Documents/<walk_name> --destination ../walks/<walk_name>

# 2. Walk folder (or an AirDropped .zip) -> point cloud, mesh, ground grid, georef, previews, viewer
#    (process.py, merge.py, optimize.py and fuse.py take --jobs N, default 3: caps Open3D's threads and runs the
#    mesh, colouring and per-walk work in parallel; the results are identical for any N. See cores.py.)
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

Experimental `fuse.py --consensus` drops points that most of the frames able to judge them saw straight through (people walking by, flying pixels); something present in 1 of 2 passes is a tie and stays, in 1 of 3 it goes. Points seen too rarely to judge are kept. Off by default: on a real garden walk it also removed real structure along walls and edges, because ARKit's smoothed depth makes silhouettes look see-through.

`optimize.py` first removes ARKit relocalization snaps (single-frame jumps > 3 m/s): post-snap poses are trusted and the offset is ramped back over the frames since the previous snap. It then splits the walk into ~100-frame fragments, finds loop closures by yaw-only ICP between fragments whose paths revisit each other, and prunes bad ones in an Open3D pose graph. `synth.py --drift --frames 440` makes a two-loop walk with accumulating drift for testing.

## Merging walks

```
uv run python merge.py ../out/merged ../walks/<walk_a> ../walks/<walk_b> [...]   # same outputs as process.py, plus merge.json
```

Each walk is optimised on its own, then registered into the first walk's frame (GPS guess, yaw/translation search, ICP; yaw + translation only) and tied to it by inter-walk loop closures in a joint pose graph. A walk that cannot be registered confidently keeps its GPS placement and is flagged in `merge.json` and on stderr. Tip: start every walk on the same taped mark and the initial alignment is nearly exact.

Bad-tracking stretches (ARKit sliding, then relocalising) are cut out first and the clean segments are merged like walks; a segment ICP cannot place is placed by photo features (`visual.py`: SIFT matches made metric with the lidar depth; at least 3 constraints must agree on one placement, searched from the GPS guess out to 18 m) or, failing that, in its neighbour segment's frame (ARKit's pose is continuous across the cut), flagged as unverified. Photo constraints are also loop edges in the pose graphs (needs 2 agreeing constraints per fragment pair), and ARKit's snaps are checked against the photos (a snap the photos do not confirm is real motion, not a glitch). `--no-visual` turns all of that off. Per piece, `merge.json` says how it was placed (`registered` / `visual` / `continuity_placement` / `gps_fallback` / `excluded`) and how well it fits the pieces before it.

## Publishing to the site (GitHub Pages)

```
uv run python publish.py ../out/<walk> <name> "<title>"   # thins to <=3M points into ../site/models/<name>/, updates the index
```

Only `points.ply` (local frame, metres from the walk start) is published, never the georeferenced outputs. Then run `./push_models.sh`: `site/models/` goes to the `site-models` branch as its only commit (replacing the previous one, so model versions don't pile up in history) and Pages redeploys. Viewer changes (`site/index.html`) are committed to `main` as usual. The published model is public.

## SketchUp

```
uv run python export_sketchup.py ../out/<walk>   # sketchup/: terrain.dae+png, scene.dae, terrain.stl, scene.stl, points.las (Z-up, metres)
uv run python test_export.py                     # synth -> fuse -> export, checks axes, heights, face counts
```

Paid plans: File > Import > COLLADA for the `.dae` files. SketchUp Free officially imports only `.skp`/`.jpg`/`.png`; try the `.stl` files via Insert as Component (choose metres) and lay `terrain.png` over the terrain. `points.las` is for Scan Essentials (Studio, Windows).

## Site plan

```
uv run python siteplan.py ../out/<walk> [--interval 0.1]   # siteplan/: siteplan.svg (layers), .pdf (A3, real scale), .png, features.json
uv run python test_siteplan.py                              # contour heights, trunk positions, a hand-made wall
```

Contours, tree canopies and wall outlines are estimated from the scan: use the plan for discussion, not as a survey.

## Simplified model (experimental)

```
uv run python simplify.py ../out/<walk>   # simple/: simple.dae+terrain.png, simple.stl, simple.glb, elements.json
uv run python test_simplify.py
```

The decimated terrain is good (e.g. ~40k grid faces -> ~1.5k within a few cm) and light enough for SketchUp. Walls are only recovered as small slabs on real scans, and anything tall and non-planar (including tanks or parts of houses) becomes a crown ellipsoid — check `elements.json` before relying on it.

`georef_points.ply` stores coordinates minus `offset` from `georef.json`; add it back to get EPSG:5382.
