# Walk recording format (v3)

v3: no conf/; depth 0 = invalid or low confidence. Also new in v3: RGB keyframes only when the camera moved
or turned (or 3 s passed), and the optional files loops.jsonl, motion.jsonl and markers.jsonl, plus the
`degenerate_view` reason in events.jsonl. A reader of v2 walks still finds conf/ and uses it.

v2 changes from v1: depth is float16 (`meta.depth_dtype = "f16"`; absent means float32),
RGB keyframes ~2 Hz, gps.jsonl written once per location fix.

The iPhone app writes one folder per walk into its Documents directory. The Mac
pipeline reads that folder. This file is the contract between them.

```
walk_YYYYMMDD_HHMMSS/
  meta.json
  frames.jsonl
  gps.jsonl
  events.jsonl       optional: intervals where tracking was wrong (absent in older walks = none recorded)
  baro.jsonl         optional: barometer samples, an independent vertical reference (absent in older walks)
  diag.jsonl         optional: app health every ~5 s (memory, heat, battery, frame backlog), for debugging crashes
  loops.jsonl        optional (v3): the user came back to the start mark (a loop closure)
  motion.jsonl       optional (v3): accelerometer and pedometer, an independent check of distance walked
  markers.jsonl      optional (v3): sightings of printed markers with their pose
  depth/000000.bin   Float16 little-endian (v1: Float32), metres, row-major, depth_w × depth_h
  conf/000000.bin    v1 and v2 only: UInt8 ARConfidenceLevel (0 low, 1 medium, 2 high), same layout
  rgb/000000.jpg     only on some frames (at most 2 Hz; v3: only if the camera moved >= 0.25 m or turned >= 8°
                     since the last one, or >= 3 s passed), JPEG quality 0.8, full camera res
```

Frame indices are zero-padded to 6 digits and shared between depth/rgb (and conf in v1/v2).

Depth is ARKit `sceneDepth`: planar z-depth (not ray length). Invalid pixels may be
0 or NaN; readers should treat anything outside a sane range as invalid.

**v3: no conf/; depth 0 = invalid or low confidence.** The app writes 0 for every pixel whose ARKit
confidence was < 1 (low) and for invalid pixels (NaN, <= 0). All other pixels are medium or high confidence.
A v3 walk has no `conf/` directory; for v1/v2 walks keep applying `conf` as before.

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
  "format_version": 3,                     // 2: has conf/ ; 3: no conf/, depth 0 = invalid or low confidence
  "app_version": "0.6",
  "depth_dtype": "f16",
  "device": "iPhone16,1",
  "started_at": "2026-10-19T15:00:00Z",   // ISO 8601 UTC
  "ended_at": "2026-10-19T15:45:00Z",      // null while recording, or if the app died and was never reopened
  "end_reason": "user",                    // optional: "user" | "back_at_start" | "low_space" | "write_failed" | "thermal" | "interrupted"
  "last_checkpoint": "2026-10-19T15:44:40Z", // optional: last 30 s metadata rewrite while recording
  "world_alignment": "gravityAndHeading",
  "depth_w": 256, "depth_h": 192,
  "rgb_w": 1920, "rgb_h": 1440,
  "frame_count": 12345,
  "site_anchor": {                         // optional: walk was started on a loaded coverage.json
    "coverage_id": "garden-v3",            // the file's "id", or null
    "east": 12.5, "north": 40.0,           // where the walk origin (ARKit 0,0) sits in the coverage frame, metres
    "heading_deg": 271.5,                  // phone compass heading at start, clockwise from north, or null
    "heading_is_true": true,               // true north (else magnetic)
    "method": "tap"
  }
}
```

`end_reason` says how the walk ended:

- `user`: Stop was pressed.
- `back_at_start`: after 20 minutes the user walked back to within 1.5 m of the start pose and stayed there,
  looking around, for 5 s; the app then ended the walk. The last seconds re-observe the start view, for the
  pipeline to close the loop on (the 1.5 m is only the app's "you are back" test, not the closure accuracy).
- `low_space` (free space fell below 500 MB), `write_failed` (a file write failed), `thermal` (the phone
  reached the critical heat state; iOS would have paused the camera): the app closed the walk itself,
  cleanly, with `ended_at` and an exact `frame_count`; every frame listed in `frames.jsonl` has its files.
- `interrupted`: the app was killed or crashed (typically out of memory) and repaired the walk the next
  time it was opened. `ended_at` is the walk start plus the `t` of the last complete line of
  `frames.jsonl`, and `frame_count` the number of complete lines. A killed app may have written the files
  of one more frame than listed (extra files, never missing ones).

While recording, `meta.json` is rewritten atomically every 30 s with the current `frame_count` and
`last_checkpoint`, so even a walk the app could not repair (`ended_at: null`: not reopened yet) has a
frame count at most 30 s stale. Walks from older app versions have no `end_reason`; `ended_at: null`
there means the app died and the last few frames may list files that were never written.

`site_anchor` is the initial guess for merging with the earlier model: the coverage frame is east/north
metres, ARKit's world is already north-aligned (`gravityAndHeading`, +X east, −Z north), so a walk point
`(x, z)` is at `(east + x, north − z)`. Only a translation is stored, not a rotation: the compass heading
is recorded for the pipeline to refine yaw, but is not applied because ARKit and the compass share the
same magnetometer error.

## frames.jsonl

One JSON object per recorded depth frame (~10 Hz), appended as recorded:

```json
{
  "i": 0,
  "t": 12.345,                       // seconds since walk start
  "transform": [16 floats],          // ARCamera.transform, camera-to-world, COLUMN-major
  "intrinsics": [fx, fy, cx, cy],    // ARCamera.intrinsics, in RGB image pixels
  "tracking": "normal",              // normal | limited_* | not_available
  "rgb": true                        // whether rgb/<i>.jpg exists (v3: not on every ~0.5 s slot, see above)
}
```

Frames are recorded only while ARKit reports `tracking: "normal"`; limited stretches leave a gap in `t`.

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

## events.jsonl (optional)

One object per interval where ARKit tracking was clearly wrong, as opposed to the user walking fast.
Times are seconds since walk start, the same clock as `frames.jsonl`:

```json
{"t_start": 226.8, "t_end": 239.7, "reason": "implausible_speed"}
```

The pipeline should drop (or re-register) frames with `t_start <= t <= t_end`. `reason`:

- `implausible_speed`: poses implied more than 2.5 m/s over about 1 s while the accelerometer showed no
  running-level motion (ARKit sliding, then usually snapping back).
- `jump`: two consecutive poses (0.1 s apart) more than 1.5 m apart.
- `tracking_limited` / `relocalizing`: ARKit tracking state was limited or relocalizing for over 1 s
  (shorter dips are only a warning). Frames are not recorded while it lasts, so there is a gap.

- `degenerate_view` (v3): the depth view was dominated by one flat surface (more than 85% of the sampled
  points within a few cm of one plane) or had only one strong direction, for over 1.5 s. ARKit's pose can
  slide along such a surface (e.g. a long uniform fence or wall). **Treat these intervals as low-trust
  odometry, not as drops**: the frames are valid and recorded; down-weight their poses in optimisation
  and prefer other constraints (loop closures, markers, GPS, barometer) over them. A ground-only view
  counts too. Unlike the first two reasons, `t_start` is the first degenerate frame (no 10 s pre-roll) and
  `t_end` is 1 s after the last one. A long interval may be written in several consecutive pieces
  (the end of one is the start of the next): the app flushes the open interval every 30 s, on memory
  warnings and when the walk stops, so a killed app loses at most 30 s of it. Adjacent pieces can be merged.

`t_start` is 10 s before detection for the first two reasons (the pose drifts for a while before it
is detectable; on the recorded failures it lands about at the real onset) and the moment tracking
degraded for the last two. `t_end` is 1 s after the last anomaly, and the interval stays open until
3 s of clean frames. The app also keeps these frames out of its live coverage map. An interval of these
four reasons still open when the app dies is not written (`degenerate_view` is flushed as above).

## baro.jsonl (optional)

Barometer samples from `CMAltimeter`, for checking vertical drift of the ARKit poses against something
independent. `t` is on the same clock as `frames.jsonl` (seconds since walk start: CoreMotion's
`timestamp` and `ARFrame.timestamp` are both seconds since boot, minus the walk's `startUptime`).
Two line types, told apart by `type`; either may be missing on a device without the sensor:

```json
{"type": "relative", "t": 12.4, "relative_altitude_m": 0.31, "pressure_kpa": 100.912}
{"type": "absolute", "t": 12.9, "altitude_m": 41.2, "accuracy_m": 8.0, "precision_m": 0.6}
```

- `relative`: altitude change since the first barometer sample of the walk (that sample is ~0, not exactly
  at `t = 0`), about 1 Hz, resolution a few cm, short-term very good. Weather moves the pressure, so over
  tens of minutes it can wander by decimetres: use it for differences between nearby times or places, and
  fit a slow drift out. Walking up a staircase or stopping at a taped mark gives the cleanest checks.
- `absolute`: GPS-assisted altitude above sea level with its `accuracy_m` (metres, 68%) and
  `precision_m` (the barometric part); irregular, only when the system has an estimate.

## loops.jsonl (optional, v3)

Moments when the user physically came back to the walk's start mark, i.e. a loop closure with the start. This
is good news, not bad tracking, so it is not in events.jsonl. `t` is on the frame clock:

```json
{"kind": "start_revisit", "t": 812.4}
```

- `start_revisit`: the camera was within 1.5 m (horizontal, in ARKit's own world coordinates) of the pose where
  recording started, having been more than 5 m away since the start or the previous revisit. The user is
  physically at the start at that `t`, while the ARKit pose there has drifted by the accumulated error: the
  pipeline should use the time, not the position, and tie that frame's pose to the first frames of the walk.
- The app prompts for this every ~8 min of walking or ~250 m of path since the last revisit, so a long walk
  has them roughly that often if the user follows the prompt. The file may be empty or absent: no revisit.

## motion.jsonl (optional, v3)

An independent measure of distance walked, to detect ARKit sliding along a uniform surface. One JSON object per
line, short keys, numbers at float32 precision. Two line types by `k`:

```json
{"k": "a", "t": 12.345, "a": [0.012, -0.034, 0.101]}
{"k": "p", "t": 13.2, "s": 18, "d": 12.7}
```

- `a`: `CMDeviceMotion.userAcceleration` (gravity removed) in g, in the phone's own axes, about 20 Hz. `t` is
  on the frame clock (CoreMotion and ARKit timestamps share the boot clock). The phone's orientation is not
  logged here; use frames.jsonl poses to rotate it to the world frame. The accelerometer is noisy:
  use its magnitude/variance as a "walking or standing" cue, not for double integration.
- `p`: `CMPedometer` update, cumulative since the walk started: `s` step count, `d` distance in metres
  (omitted when the device gives none). Updates come in batches every few seconds. `t` is the
  update's `endDate` minus the walk start date (the GPS clock: wall time, which can differ from the frame clock
  by milliseconds; negligible here). Pedometer distance is rough (stride model, about 5 to 10%) but
  independent of ARKit: ARKit path length far above it over the same window means the pose slid.
- Lines are buffered, so a killed app may be missing the last few seconds.

## markers.jsonl (optional, v3)

Sightings of printed reference markers taped to fixed spots (house wall, gate posts, start mark), detected by
ARKit image detection. Each sighting gives the marker's exact pose in the walk's world frame, an absolute
anchor that the same physical marker shares across passes and walks. `ios/markers/make_markers.py` generates the
markers (ids `m1` to `m12`, A4 pages in `ios/markers/walkscan_markers.pdf`, printed with the square exactly
18.0 cm wide).

```json
{"t": 12.3, "id": "m1", "T": [16 floats], "tracked": true, "size_m": 0.18, "scale": 1.0}
```

- `t`: frame clock, seconds since walk start. At most 2 lines per second per marker.
- `id`: marker id, `m1` ... `m12`; the same id is always the same physical sheet.
- `T`: `ARImageAnchor.transform`, marker-to-world, COLUMN-major like frames.jsonl, in this walk's world
  frame. The marker's local +X is along the printed square's right edge, +Y is its surface normal
  (out of the sheet) and +Z points to the bottom edge of the image (ARKit image anchor convention).
- `tracked`: ARKit was actively tracking the image in this frame. When false the pose is the last known one
  (the marker is out of view); prefer lines with `tracked: true`.
- `size_m`: physical width the app assumed (0.18).
- `scale`: `estimatedScaleFactor`, ARKit's estimate of how much larger the printed marker is than
  `size_m` (1.0 = as assumed; a badly scaled print shows up here, and the pose then has the wrong range).
- Pipeline use: the same `id` must have one pose in the merged model across every pass and every walk, so a
  sighting is a loop constraint between the walk's drifting pose chain and a fixed point. Fit one world pose
  per id and constrain each sighting to it. Poses near the edge of the field of view or at a grazing angle
  are less accurate; the orientation is better than the position along the viewing ray.

## diag.jsonl (optional)

One line about every 5 s while recording (plus one when the app starts shedding load), to find out why a
long walk died. `t` is on the frame clock. For debugging only; the pipeline can ignore it.

```json
{"t": 305.0, "avail_mb": 2810, "footprint_mb": 640, "thermal": "fair", "battery": 0.71,
 "queue_lag_s": 0.04, "frames_dropped": 0, "mesh_anchors": 0, "frames": 3050,
 "coarse_cells": 41000, "fine_tiles": 120, "hole_cells": 312, "shedding": false}
```

- `avail_mb`: `os_proc_available_memory()`, what the app may still allocate before iOS kills it (0 if the OS
  cannot say). `footprint_mb`: `phys_footprint`, the figure iOS compares against that limit.
- `thermal`: `nominal | fair | serious | critical`. `battery`: 0 to 1, or null.
- `queue_lag_s`: the worst age of an ARKit frame when the app got to it since the previous line (the
  write/processing backlog: it should be well under 0.1). `frames_dropped`: frames skipped so far because
  they were more than 0.5 s stale (they leave a gap in `frames.jsonl`).
- `mesh_anchors`: ARKit mesh anchors alive (0 unless the mesh overlay is on, which is off by default).
- `coarse_cells`, `fine_tiles` (64 x 64-cell tiles of the 10 cm hole grid), `hole_cells`: sizes of the
  app's coverage structures.
- `shedding`: gap search and mesh were switched off because available memory fell below 500 MB or iOS sent a
  memory warning. The line written at that moment has a `note` (`low_memory` or `memory_warning`).

## coverage.json (Mac → phone, "continue a site")

Written by the pipeline for a merged model and loaded in the app (Files picker, AirDrop / "Open in
WalkScan", or fetched from a URL). The app shows it as already-covered ground, so its red frontier and
gap arrows point only at what is still missing, and lets the user tap where they stand to place the new
walk (`site_anchor` above). Everything is in a local east/north frame in metres (any origin; no CRS
needed):

```json
{
  "format_version": 1,
  "id": "garden-v3",
  "cell_size": 0.25,
  "origin": {"east": 120.0, "north": 340.0},
  "width": 240, "height": 200,
  "rows": ["....####....", "...#####x...", "..."],
  "outline": [[[120, 340], [180, 340], [180, 290], [120, 290]]],
  "ref": {"lat": -34.906, "lon": -56.191}
}
```

- `origin` is the **north-west corner** of the grid; column 0 is the westernmost, row 0 the **northernmost**.
  Cell `(col, row)` covers east `[origin.east + col·cell_size, +cell_size)` and north
  `(origin.north − (row+1)·cell_size, origin.north − row·cell_size]`.
- `rows` has `height` strings of exactly `width` ASCII characters: `#` covered (enough points to model
  it), `.` unseen, `x` blocked (looked at from several sides but cannot be seen: water, glass, dense
  thicket; the app treats it as done). Writing the grid as text keeps it readable and small.
- `id` (optional string) is copied into `site_anchor.coverage_id`.
- `outline` (optional) is a list of rings of `[east, north]` points, drawn on the map (e.g. the plot boundary).
- `ref` (optional) is the lat/lon of east 0 / north 0; with it the map shows the phone's GPS position
  to help the user tap the right spot.
- Limits: `cell_size` 0.05 to 5 m, at most 4 000 000 cells. 0.25 m matches the app's own grid.
