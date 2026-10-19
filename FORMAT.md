# Walk recording format (v2)

v2 changes from v1: depth is float16 (`meta.depth_dtype = "f16"`; absent means float32),
RGB keyframes ~2 Hz, gps.jsonl written once per location fix.

The iPhone app writes one folder per walk into its Documents directory. The Mac
pipeline reads that folder. This file is the contract between them.

```
walk_YYYYMMDD_HHMMSS/
  meta.json
  frames.jsonl
  gps.jsonl
  depth/000000.bin   Float16 little-endian (v1: Float32), metres, row-major, depth_w × depth_h
  conf/000000.bin    UInt8 ARConfidenceLevel (0 low, 1 medium, 2 high), same layout
  rgb/000000.jpg     only on some frames (~2 Hz), JPEG quality 0.8, full camera res
```

Frame indices are zero-padded to 6 digits and shared between depth/conf/rgb.

Depth is ARKit `sceneDepth`: planar z-depth (not ray length). Invalid pixels may be
0 or NaN; readers should treat anything outside a sane range as invalid.

Both `t` fields (frames and gps) are seconds since the same walk-start instant:
frames use `ARFrame.timestamp − startUptime`, GPS uses `CLLocation.timestamp − startDate`,
with `startUptime`/`startDate` captured together. GPS `t` can be slightly negative
(cached fix); readers may drop those.

## World coordinates

ARKit session with `worldAlignment = .gravityAndHeading`:
+X = east, +Y = up, +Z = south, metres, origin = where the walk started.

## meta.json

```json
{
  "format_version": 2,
  "app_version": "0.2",
  "depth_dtype": "f16",
  "device": "iPhone16,1",
  "started_at": "2026-10-19T15:00:00Z",   // ISO 8601 UTC
  "ended_at": "2026-10-19T15:45:00Z",      // null if the app crashed mid-walk
  "world_alignment": "gravityAndHeading",
  "depth_w": 256, "depth_h": 192,
  "rgb_w": 1920, "rgb_h": 1440,
  "frame_count": 12345
}
```

## frames.jsonl

One JSON object per recorded depth frame (~10 Hz), appended as recorded:

```json
{
  "i": 0,
  "t": 12.345,                       // seconds since walk start
  "transform": [16 floats],          // ARCamera.transform, camera-to-world, COLUMN-major
  "intrinsics": [fx, fy, cx, cy],    // ARCamera.intrinsics, in RGB image pixels
  "tracking": "normal",              // normal | limited_* | not_available
  "rgb": true                        // whether rgb/<i>.jpg exists
}
```

Camera frame is ARKit's: +X right, +Y up, −Z forward (looking direction).
Depth pixel (u, v) maps to RGB pixel (u · rgb_w/depth_w, v · rgb_h/depth_h).
Image v grows downward, so unprojecting depth pixel (u, v) with depth d:
`x = (u_rgb − cx)/fx · d`, `y = −(v_rgb − cy)/fy · d`, `z = −d` in camera space.

## gps.jsonl

One object per CLLocation update, carrying the latest heading:

```json
{"t": 12.3, "lat": -34.906, "lon": -56.191, "alt": 8.1, "h_acc": 4.2, "v_acc": 3.0, "heading": 271.5, "heading_acc": 10}
```

`heading`/`heading_acc` are true-north degrees, null when unavailable.
`alt` is `CLLocation.altitude` (above mean sea level).

Whether ARKit's `gravityAndHeading` −Z is true or magnetic north (≈10° apart in
Uruguay) is unverified; the pipeline should fit yaw once a real walk is long enough.
