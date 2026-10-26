import Foundation
import simd

// Pure logic for degenerate-view detection, loop prompts and the RGB keyframe gate. No ARKit, so it can be
// compiled and tested on its own.

enum DegenerateConfig {
    /// Every 8th depth pixel is used (32 x 24 = 768 points for a 256 x 192 map).
    static let pixelStride = 8
    /// Fewer valid points than this: no verdict (the "no valid depth" warning covers it).
    static let minPoints = 150
    /// This share of the points on one plane means the view is degenerate...
    static let planeFraction = 0.85
    /// ...and so does a point spread with a single strong direction: the 2nd principal std-dev below this (m).
    static let minSpread = 0.12
    /// Degenerate for this long before the warning shows and an interval is logged.
    static let confirmSeconds = 1.5
    /// Clean for this long before the interval ends.
    static let clearSeconds = 1.0
    static let reason = "degenerate_view"
    static let message = "Too little to track: include posts, trees or ground in view, walk slower"
}

enum ViewShape {
    /// Whether the depth map (row-major metres, 0 or NaN invalid) is dominated by one plane or one line.
    /// Intrinsics are in RGB pixels; `scale` is RGB pixels per depth pixel; `up` is world up in camera coordinates.
    /// A level plane (ground, gravel) does not count: it has texture to track, and walls are what let the pose slide.
    /// Returns nil with too few valid points.
    static func isDegenerate(depth: UnsafeBufferPointer<Float>, width: Int, height: Int,
                             fx: Float, fy: Float, cx: Float, cy: Float, scale: Float, up: SIMD3<Double>) -> Bool? {
        var pts: [SIMD3<Double>] = []
        pts.reserveCapacity((width / DegenerateConfig.pixelStride + 1) * (height / DegenerateConfig.pixelStride + 1))
        for v in stride(from: 0, to: height, by: DegenerateConfig.pixelStride) {
            for u in stride(from: 0, to: width, by: DegenerateConfig.pixelStride) {
                let z = depth[v * width + u]
                guard z > 0.2, z <= 5 else { continue }  // also rejects NaN
                pts.append(SIMD3(Double((Float(u) * scale - cx) / fx * z), Double(-(Float(v) * scale - cy) / fy * z), Double(-z)))
            }
        }
        guard pts.count >= DegenerateConfig.minPoints else { return nil }

        let n = Double(pts.count)
        let mean = pts.reduce(SIMD3<Double>.zero, +) / n
        var c = [Double](repeating: 0, count: 6)  // xx yy zz xy xz yz
        for p in pts {
            let d = p - mean
            c[0] += d.x * d.x; c[1] += d.y * d.y; c[2] += d.z * d.z
            c[3] += d.x * d.y; c[4] += d.x * d.z; c[5] += d.y * d.z
        }
        c = c.map { $0 / n }

        // Eigenvalues of the symmetric 3x3 covariance, largest first (closed form).
        let q = (c[0] + c[1] + c[2]) / 3
        let p2 = (c[0] - q) * (c[0] - q) + (c[1] - q) * (c[1] - q) + (c[2] - q) * (c[2] - q)
            + 2 * (c[3] * c[3] + c[4] * c[4] + c[5] * c[5])
        let p = (p2 / 6).squareRoot()
        guard p > 1e-9 else { return true }  // all points in one spot
        let b = [(c[0] - q) / p, (c[1] - q) / p, (c[2] - q) / p, c[3] / p, c[4] / p, c[5] / p]
        let det = b[0] * (b[1] * b[2] - b[5] * b[5]) - b[3] * (b[3] * b[2] - b[5] * b[4]) + b[4] * (b[3] * b[5] - b[1] * b[4])
        let phi = acos(max(-1, min(1, det / 2))) / 3
        let l1 = q + 2 * p * cos(phi)
        let l3 = q + 2 * p * cos(phi + 2 * .pi / 3)
        let l2 = 3 * q - l1 - l3
        if max(l2, 0).squareRoot() < DegenerateConfig.minSpread { return true }

        // Plane normal = eigenvector of the smallest eigenvalue: the longest cross product of two rows of (C - l3 I).
        let r0 = SIMD3(c[0] - l3, c[3], c[4]), r1 = SIMD3(c[3], c[1] - l3, c[5]), r2 = SIMD3(c[4], c[5], c[2] - l3)
        let normal = [simd_cross(r0, r1), simd_cross(r0, r2), simd_cross(r1, r2)].max { simd_length($0) < simd_length($1) }!
        let len = simd_length(normal)
        guard len > 1e-12 else { return true }
        let nrm = normal / len
        if abs(simd_dot(nrm, simd_normalize(up))) > cos(30 * Double.pi / 180) { return false }  // level ground

        var inliers = 0
        for pt in pts where abs(simd_dot(nrm, pt - mean)) <= 0.03 + 0.01 * -pt.z { inliers += 1 }
        return Double(inliers) / n > DegenerateConfig.planeFraction
    }
}

/// Turns the per-frame verdict into `degenerate_view` intervals: a view must stay degenerate for 1.5 s to count,
/// and `t_start` is the first degenerate frame (no pre-roll). Feed it every depth frame, in time order.
struct DegenerateMonitor {
    private var runStart: Double?
    private var lastDegenerate = 0.0
    private var cleanSince: Double?
    private var lastSeen = -Double.infinity
    private var confirmed = false

    /// The warning is on from confirmation until the interval ends.
    var warning: Bool { confirmed }

    /// Returns an interval when one has just ended.
    mutating func update(t: Double, degenerate: Bool) -> TrackingEvent? {
        var closed: TrackingEvent?
        // Frames are not recorded while tracking is limited: a long gap ends the run.
        if runStart != nil, t - lastSeen > 1 { closed = close(upTo: lastDegenerate + TrackingConfig.endMargin) }
        lastSeen = t
        if degenerate {
            if runStart == nil { runStart = t }
            lastDegenerate = t
            cleanSince = nil
            if t - runStart! >= DegenerateConfig.confirmSeconds { confirmed = true }
        } else if runStart != nil {
            if cleanSince == nil { cleanSince = t }
            if t - cleanSince! >= DegenerateConfig.clearSeconds { closed = closed ?? close(upTo: lastDegenerate + TrackingConfig.endMargin) }
        }
        return closed
    }

    /// Writes out a confirmed, still-open interval so far, and carries on from there: for checkpoints and
    /// memory warnings, so a killed app loses at most the time since the last flush.
    mutating func flush() -> TrackingEvent? {
        guard confirmed, let start = runStart, lastDegenerate > start else { return nil }
        runStart = lastDegenerate
        return TrackingEvent(tStart: start, tEnd: lastDegenerate, reason: DegenerateConfig.reason)
    }

    /// Closes an interval that is still open when the walk ends.
    mutating func finish(t: Double) -> TrackingEvent? { close(upTo: min(t, lastDegenerate + TrackingConfig.endMargin)) }

    private mutating func close(upTo end: Double) -> TrackingEvent? {
        defer { runStart = nil; cleanSince = nil; confirmed = false }
        guard confirmed, let start = runStart else { return nil }
        return TrackingEvent(tStart: start, tEnd: max(end, start), reason: DegenerateConfig.reason)
    }
}

enum LoopConfig {
    /// Prompt to return to the start after this much walking time or path length since the last revisit.
    static let promptSeconds = 480.0
    static let promptMetres: Float = 250
    /// Back within this distance of the start counts as a revisit...
    static let revisitRadius: Float = 1.5
    /// ...once the user has been at least this far away since the last one.
    static let armDistance: Float = 5
    /// Slower than this is standing, not walking.
    static let walkingSpeed: Float = 0.2
}

/// Tracks the start pose, how far and how long the user has walked since they last were there, and revisits.
/// Positions are ARKit world metres (horizontal distances use x and z).
struct LoopMonitor {
    private(set) var start: SIMD3<Float>?
    private var last: SIMD3<Float>?
    private var lastT = 0.0
    private var armed = false
    private var walkedSeconds = 0.0
    private var walkedMetres: Float = 0

    var prompting: Bool { walkedSeconds >= LoopConfig.promptSeconds || walkedMetres >= LoopConfig.promptMetres }

    /// Horizontal distance to the start, 0 before there is one.
    func distanceToStart(from pos: SIMD3<Float>) -> Float {
        guard let s = start else { return 0 }
        return hypot(pos.x - s.x, pos.z - s.z)
    }

    /// Feed every recorded frame. Returns true at the moment the user comes back to the start.
    mutating func update(t: Double, pos: SIMD3<Float>) -> Bool {
        guard let s = start, let prev = last else {
            start = pos
            last = pos
            lastT = t
            return false
        }
        let dt = t - lastT
        if dt > 0, dt <= 1 {
            let step = hypot(pos.x - prev.x, pos.z - prev.z)
            let v = step / Float(dt)
            // Steps faster than 5 m/s are tracking glitches (as in Engine's speed estimate), not walking.
            if v >= LoopConfig.walkingSpeed, v < 5 {
                walkedMetres += step
                walkedSeconds += dt
            }
        }
        last = pos
        lastT = t

        let d = hypot(pos.x - s.x, pos.z - s.z)
        if d > LoopConfig.armDistance { armed = true }
        if armed, d <= LoopConfig.revisitRadius {
            armed = false
            walkedSeconds = 0
            walkedMetres = 0
            return true
        }
        return false
    }
}

/// Decides which camera frames also get an RGB keyframe: at most every 0.48 s (2 Hz, as before), and only if the
/// camera moved 0.25 m, turned 8 degrees, or 3 s passed since the last one.
struct KeyframeGate {
    static let minGap = 0.48
    static let maxGap = 3.0
    static let minMove: Float = 0.25
    static let minTurnCos: Float = cos(8 * .pi / 180)

    private var lastT = -Double.infinity
    private var lastPos: SIMD3<Float>?
    private var lastFwd: SIMD3<Float>?

    func due(t: Double, pos: SIMD3<Float>, fwd: SIMD3<Float>) -> Bool {
        let gap = t - lastT
        guard gap >= Self.minGap else { return false }
        guard let p = lastPos, let f = lastFwd else { return true }
        return gap >= Self.maxGap || simd_distance(pos, p) >= Self.minMove || simd_dot(fwd, f) <= Self.minTurnCos
    }

    mutating func saved(t: Double, pos: SIMD3<Float>, fwd: SIMD3<Float>) {
        lastT = t
        lastPos = pos
        lastFwd = fwd
    }
}
