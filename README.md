# property-3d

Map a few thousand m² of land in 3D by walking it once with an iPhone Pro (LiDAR).

- **`ios/`** — WalkScan, a SwiftUI + ARKit app. Records LiDAR depth, camera poses, photos and
  GPS to disk during one continuous walk, with a live coverage minimap (red = not yet seen) and
  haptic warnings (too fast, tracking lost, pointing at the sky, heat, battery, storage).
- **`pipeline/`** — Python (uv + Open3D) on the Mac: drift correction (fragment pose graph with
  loop closures), coloured point cloud, TSDF mesh, ground height grid, GPS georeferencing, and a
  three.js browser viewer. See [`pipeline/README.md`](pipeline/README.md).
- **`FORMAT.md`** — the on-disk walk format shared by the app and the pipeline.
- **`scripts/`** — fetch/crop IDE Uruguay national orthophoto and elevation tiles for context.

## Quick start

1. Open `ios/WalkScan.xcodeproj`, pick your team under Signing, run on an iPhone/iPad with LiDAR.
2. Walk the area. Start and end on the same marked spot if you want an accuracy check.
3. AirDrop the walk (share button in the app) or pull it with `xcrun devicectl` (see pipeline README).
4. `cd pipeline && uv run python process.py <walk.zip or walk folder>`, then open the printed viewer URL.

Recordings (`walks/`), outputs (`out/`) and downloaded map data (`data/`) are git-ignored.
