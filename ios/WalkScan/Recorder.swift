import ARKit
import CoreLocation
import SwiftUI
import UIKit

enum Storage {
    static var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
}

/// Snapshot of recording state sent from the recording queue to the UI.
struct Update: @unchecked Sendable {
    var snapshot: CoverageSnapshot?
    var x: Float
    var z: Float
    var yaw: Float
    var duration: TimeInterval
    var frames: Int
    var issues: [String]
    var holes: [HoleMark]
    var gaps: Int
    var target: SIMD3<Float>?
    var targetDistance: Float
    /// Target bearing relative to where the phone points, radians clockwise (0 = straight ahead).
    var targetAngle: Float
    var targetOnScreen: Bool
    /// Open holes that were filled since the previous update.
    var filled: Int
}

/// A blind spot for the minimap. Position is world X/Z in metres.
struct HoleMark: Sendable {
    var x: Float
    var z: Float
    var radius: Float
    var blocked: Bool
}

/// UI-facing recorder. All disk and ARKit frame work lives in `Engine` on its own queue.
@MainActor
final class Recorder: ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var snapshot: CoverageSnapshot?
    @Published private(set) var position = SIMD2<Float>(0, 0)
    @Published private(set) var yaw: Float = 0
    @Published private(set) var duration: TimeInterval = 0
    @Published private(set) var frames = 0
    @Published private(set) var warnings: [String] = []
    @Published private(set) var holes: [HoleMark] = []
    @Published private(set) var gaps = 0
    @Published private(set) var target: SIMD3<Float>?
    @Published private(set) var targetDistance: Float = 0
    @Published private(set) var targetAngle: Float = 0
    @Published private(set) var targetOnScreen = false
    @Published var errorMessage: String?

    let session = ARSession()
    private let engine = Engine()
    private let location = CLLocationManager()
    private let haptics = UINotificationFeedbackGenerator()
    private let fillHaptic = UIImpactFeedbackGenerator(style: .light)
    private var shownWarnings = Set<String>()
    private var lastHaptic = Date.distantPast
    private var lastFillHaptic = Date.distantPast

    func setViewport(_ size: CGSize) { engine.setViewport(size) }

    var areaM2: Double { snapshot?.areaM2 ?? 0 }

    init() {
        session.delegate = engine
        session.delegateQueue = engine.q
        location.delegate = engine
        location.desiredAccuracy = kCLLocationAccuracyBest
        engine.onUpdate = { [weak self] update in
            Task { @MainActor in self?.apply(update) }
        }
    }

    private static func makeConfig() -> ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravityAndHeading
        config.frameSemantics = .sceneDepth
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }
        return config
    }

    /// Runs the session so the camera and mesh preview show before a walk starts.
    func startPreview() {
        location.requestWhenInUseAuthorization()
        session.run(Self.makeConfig())
    }

    func start() {
        guard !isRecording else { return }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let started = Date()
        let dir = Storage.documents.appendingPathComponent("walk_" + formatter.string(from: started))
        let startUptime = ProcessInfo.processInfo.systemUptime
        do {
            try engine.begin(dir: dir, startUptime: startUptime, startDate: started)
        } catch {
            errorMessage = "Could not create walk folder: \(error.localizedDescription)"
            return
        }
        // Reset so the world origin (and gravityAndHeading axes) are where the walk starts.
        session.run(Self.makeConfig(), options: [.resetTracking, .removeExistingAnchors])
        location.startUpdatingLocation()
        location.startUpdatingHeading()
        UIDevice.current.isBatteryMonitoringEnabled = true
        UIApplication.shared.isIdleTimerDisabled = true
        snapshot = nil
        position = .zero
        duration = 0
        frames = 0
        warnings = []
        shownWarnings = []
        holes = []
        gaps = 0
        target = nil
        isRecording = true
    }

    func stop() {
        guard isRecording else { return }
        isRecording = false
        location.stopUpdatingLocation()
        location.stopUpdatingHeading()
        UIApplication.shared.isIdleTimerDisabled = false
        engine.end()
        warnings = []
    }

    private func apply(_ u: Update) {
        guard isRecording else { return }
        if let s = u.snapshot { snapshot = s }
        position = SIMD2(u.x, u.z)
        yaw = u.yaw
        duration = u.duration
        frames = u.frames
        holes = u.holes
        gaps = u.gaps
        target = u.target
        targetDistance = u.targetDistance
        targetAngle = u.targetAngle
        targetOnScreen = u.targetOnScreen

        var list = u.issues
        let thermal = ProcessInfo.processInfo.thermalState
        if thermal == .serious || thermal == .critical { list.append("Phone is overheating (\(thermal == .critical ? "critical" : "serious"))") }
        let battery = UIDevice.current.batteryLevel
        if battery >= 0, battery < 0.2 { list.append("Battery low (\(Int(battery * 100))%)") }
        warnings = list

        // Haptic on any new warning, then repeat every few seconds while one persists.
        let current = Set(list)
        let now = Date()
        if !current.isEmpty, !current.isSubset(of: shownWarnings) || now.timeIntervalSince(lastHaptic) > 4 {
            haptics.notificationOccurred(.warning)
            lastHaptic = now
        }
        shownWarnings = current

        // Light tap when a gap gets filled; never within 1.5 s of another tap.
        if u.filled > 0, now.timeIntervalSince(lastFillHaptic) > 1.5, now.timeIntervalSince(lastHaptic) > 1.5 {
            fillHaptic.impactOccurred()
            lastFillHaptic = now
        }
    }
}

// MARK: - Recording engine

private struct FrameRecord: Encodable {
    let i: Int
    let t: Double
    let transform: [Float]
    let intrinsics: [Float]
    let tracking: String
    let rgb: Bool
}

/// A hole cell that can be looked at; `y` is the estimated ground height.
private struct LookCell {
    var key: Int64
    var x: Float
    var y: Float
    var z: Float
    var blocked: Bool
}

private struct HoleComp {
    var keys: [Int64]
    var groundY: Float
    var center: SIMD2<Float>
    var radius: Float
    var blocked = false
}

/// Handles ARKit frames and location updates on `q`: writes to disk, updates coverage, reports to the UI.
/// Everything except `onUpdate` and `q` is confined to `q`.
final class Engine: NSObject, ARSessionDelegate, CLLocationManagerDelegate, @unchecked Sendable {
    let q = DispatchQueue(label: "walkscan.record", qos: .userInitiated)
    var onUpdate: (@Sendable (Update) -> Void)?

    private var walk: Walk?
    private var viewport = CGSize(width: 390, height: 844)
    private let ciContext = CIContext()
    private let holeQueue = DispatchQueue(label: "walkscan.holes", qos: .utility)

    private static let iso: ISO8601DateFormatter = ISO8601DateFormatter()

    private final class Walk {
        let dir: URL
        let startUptime: TimeInterval
        let startDate: Date
        let framesFile: FileHandle
        let gpsFile: FileHandle
        let encoder = JSONEncoder()
        var meta: [String: Any]

        var count = 0
        var lastDepthT = -Double.infinity
        var lastRGBT = -Double.infinity
        var lastPublish = 0.0
        var lastSnapshot = 0.0
        var lastDiskCheck = -Double.infinity
        var dirty = false
        var grid = CoverageGrid()

        // Blind-spot guidance
        var fine = FineGrid()
        var fineDirty = false
        var lastHoleJob = -Double.infinity
        var holeJobRunning = false
        var firstSeen: [Int64: Double] = [:]
        var looks: [Int64: LookTracker] = [:]
        var blockedKeys = Set<Int64>()
        var comps: [HoleComp] = []
        var lookCells: [LookCell] = []
        var targetKey: Int64?
        var target: SIMD3<Float>?
        var targetOnScreen = false
        var pendingFilled = 0

        var pos = SIMD3<Float>(0, 0, 0)
        var yaw: Float = 0
        var lastPos: SIMD3<Float>?
        var lastPosT = 0.0
        var speed: Float = 0
        var lastValidDepth: TimeInterval

        var trackingIssue: String?
        var sessionIssue: String?
        var diskFree = Int64.max
        var writeFailed = false
        var lastLocation: CLLocation?

        init(dir: URL, startUptime: TimeInterval, startDate: Date) throws {
            self.dir = dir
            self.startUptime = startUptime
            self.startDate = startDate
            lastValidDepth = startUptime
            let fm = FileManager.default
            for sub in ["depth", "conf", "rgb"] {
                try fm.createDirectory(at: dir.appendingPathComponent(sub), withIntermediateDirectories: true)
            }
            for name in ["frames.jsonl", "gps.jsonl"] {
                guard fm.createFile(atPath: dir.appendingPathComponent(name).path, contents: nil) else {
                    throw CocoaError(.fileWriteUnknown)
                }
            }
            framesFile = try FileHandle(forWritingTo: dir.appendingPathComponent("frames.jsonl"))
            gpsFile = try FileHandle(forWritingTo: dir.appendingPathComponent("gps.jsonl"))
            meta = [
                "format_version": 2,
                "app_version": "0.2",
                "depth_dtype": "f16",
                "device": Engine.deviceModel,
                "started_at": Engine.iso.string(from: startDate),
                "ended_at": NSNull(),
                "world_alignment": "gravityAndHeading",
                "depth_w": 256, "depth_h": 192,
                "rgb_w": 1920, "rgb_h": 1440,
                "frame_count": 0,
            ]
            writeMeta()
        }

        func writeMeta() {
            guard let data = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]) else { return }
            do { try data.write(to: dir.appendingPathComponent("meta.json"), options: .atomic) } catch { writeFailed = true }
        }

        func write(_ data: Data, to file: FileHandle) {
            do { try file.write(contentsOf: data) } catch { writeFailed = true }
        }

        func write(_ data: Data, path: String) {
            do { try data.write(to: dir.appendingPathComponent(path)) } catch { writeFailed = true }
        }
    }

    private static var deviceModel: String {
        var info = utsname()
        uname(&info)
        let machine = info.machine
        return withUnsafePointer(to: machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: MemoryLayout.size(ofValue: machine)) { String(cString: $0) }
        }
    }

    // MARK: Start / stop

    func begin(dir: URL, startUptime: TimeInterval, startDate: Date) throws {
        try q.sync { walk = try Walk(dir: dir, startUptime: startUptime, startDate: startDate) }
    }

    func end() {
        q.async {
            guard let w = self.walk else { return }
            self.walk = nil
            w.meta["ended_at"] = Self.iso.string(from: Date())
            w.meta["frame_count"] = w.count
            w.writeMeta()
            try? w.framesFile.close()
            try? w.gpsFile.close()
        }
    }

    func setViewport(_ size: CGSize) {
        q.async { self.viewport = size }
    }

    // MARK: ARSessionDelegate (called on q)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let w = walk else { return }
        let now = frame.timestamp
        // Frames shortly after the reset may still carry the old world origin.
        guard now > w.startUptime + 0.5 else { return }

        let cam = frame.camera
        let tr = cam.transform
        w.pos = SIMD3(tr.columns.3.x, tr.columns.3.y, tr.columns.3.z)
        let fwd = -tr.columns.2
        // Heading clockwise from north (north = -Z). Looking steeply up or down, the forward vector says
        // nothing about heading, so use the phone's top edge (camera -X in portrait) instead.
        var heading = SIMD2(fwd.x, -fwd.z)
        if abs(fwd.y) > 0.85 {
            let top = -tr.columns.0
            heading = SIMD2(top.x, -top.z) * (fwd.y < 0 ? 1 : -1)
        }
        w.yaw = atan2(heading.x, heading.y)
        defer { publish(w) }

        let (trackingName, issue) = Self.describe(cam.trackingState)
        w.trackingIssue = issue
        guard trackingName == "normal" else {
            w.lastPos = nil
            return
        }
        guard now - w.lastDepthT >= 0.098, let depth = frame.sceneDepth,
              let conf = depth.confidenceMap else { return }
        w.lastDepthT = now

        if let last = w.lastPos, now > w.lastPosT {
            let v = hypot(w.pos.x - last.x, w.pos.z - last.z) / Float(now - w.lastPosT)
            w.speed = 0.8 * w.speed + 0.2 * v
        }
        w.lastPos = w.pos
        w.lastPosT = now

        if let t = w.target {
            let toTarget = SIMD3(t.x - w.pos.x, t.y - w.pos.y, t.z - w.pos.z)
            var onScreen = false
            if simd_dot(toTarget, SIMD3(fwd.x, fwd.y, fwd.z)) > 0 {
                let p = cam.projectPoint(t, orientation: .portrait, viewportSize: viewport)
                onScreen = p.x > viewport.width * 0.08 && p.x < viewport.width * 0.92
                    && p.y > viewport.height * 0.15 && p.y < viewport.height * 0.85
            }
            w.targetOnScreen = onScreen
        } else {
            w.targetOnScreen = false
        }

        let depthW = CVPixelBufferGetWidth(depth.depthMap)
        let depthH = CVPixelBufferGetHeight(depth.depthMap)
        // Copy out of the ARKit buffers right away; nothing from the frame is retained.
        let depthData = Self.packed(depth.depthMap, bytesPerPixel: 4)
        let confData = Self.packed(conf, bytesPerPixel: 1)

        let stem = String(format: "%06d", w.count)
        w.write(Self.float16(depthData), path: "depth/\(stem).bin")
        w.write(confData, path: "conf/\(stem).bin")

        var hasRGB = false
        if now - w.lastRGBT >= 0.48 {
            let image = CIImage(cvPixelBuffer: frame.capturedImage)
            let options = [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.8]
            if let jpeg = ciContext.jpegRepresentation(of: image, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!, options: options) {
                w.write(jpeg, path: "rgb/\(stem).jpg")
                hasRGB = true
                w.lastRGBT = now
            }
        }

        if w.count == 0 {
            w.meta["depth_w"] = depthW
            w.meta["depth_h"] = depthH
            w.meta["rgb_w"] = CVPixelBufferGetWidth(frame.capturedImage)
            w.meta["rgb_h"] = CVPixelBufferGetHeight(frame.capturedImage)
            w.writeMeta()
        }

        let record = FrameRecord(
            i: w.count,
            t: now - w.startUptime,
            transform: (0..<4).flatMap { c in (0..<4).map { r in tr[c][r] } },
            intrinsics: [cam.intrinsics[0][0], cam.intrinsics[1][1], cam.intrinsics[2][0], cam.intrinsics[2][1]],
            tracking: trackingName,
            rgb: hasRGB)
        if var line = try? w.encoder.encode(record) {
            line.append(10)
            w.write(line, to: w.framesFile)
        }
        w.count += 1

        markCoverage(w, frame: frame, depth: depthData, conf: confData, width: depthW, height: depthH)
        registerLooks(w, frame: frame, depth: depthData, conf: confData, width: depthW, height: depthH)
    }

    /// Unprojects every 4th depth pixel (confidence >= 1, depth <= 4 m) into the 25 cm coverage cells,
    /// and every 2nd pixel (depth <= 3.5 m) into the 10 cm grid used for hole detection.
    private func markCoverage(_ w: Walk, frame: ARFrame, depth: Data, conf: Data, width: Int, height: Int) {
        let cam = frame.camera
        let tr = cam.transform
        let k = cam.intrinsics
        let fx = k[0][0], fy = k[1][1], cx = k[2][0], cy = k[2][1]
        let scale = Float(cam.imageResolution.width) / Float(width)
        var total = 0, valid = 0
        depth.withUnsafeBytes { dRaw in
            conf.withUnsafeBytes { cRaw in
                let d = dRaw.bindMemory(to: Float.self)
                let c = cRaw.bindMemory(to: UInt8.self)
                for v in stride(from: 0, to: height, by: 2) {
                    for u in stride(from: 0, to: width, by: 2) {
                        let coarse = u % 4 == 0 && v % 4 == 0
                        if coarse { total += 1 }
                        let i = v * width + u
                        let z = d[i]
                        guard c[i] >= 1, z > 0, z <= 4 else { continue }
                        let p = SIMD4<Float>((Float(u) * scale - cx) / fx * z, -(Float(v) * scale - cy) / fy * z, -z, 1)
                        let wp = tr * p
                        if coarse {
                            valid += 1
                            w.grid.mark(x: wp.x, y: wp.y, z: wp.z)
                        }
                        if z <= HoleConfig.maxFineDistance { w.fine.mark(x: wp.x, y: wp.y, z: wp.z) }
                    }
                }
            }
        }
        if valid * 20 >= total { w.lastValidDepth = frame.timestamp }
        w.dirty = true
        w.fineDirty = true
    }

    /// For every settled hole cell near the camera: if it projects into the image and the depth there
    /// is invalid or shorter than the cell (something in front), count a look from this viewpoint.
    private func registerLooks(_ w: Walk, frame: ARFrame, depth: Data, conf: Data, width: Int, height: Int) {
        guard !w.lookCells.isEmpty else { return }
        let cam = frame.camera
        let inv = cam.transform.inverse
        let k = cam.intrinsics
        let fx = k[0][0], fy = k[1][1], cx = k[2][0], cy = k[2][1]
        let resW = Float(cam.imageResolution.width), resH = Float(cam.imageResolution.height)
        let scale = resW / Float(width)
        let view = SIMD2(w.pos.x, w.pos.z)
        depth.withUnsafeBytes { dRaw in
            conf.withUnsafeBytes { cRaw in
                let d = dRaw.bindMemory(to: Float.self)
                let c = cRaw.bindMemory(to: UInt8.self)
                for i in w.lookCells.indices {
                    let cell = w.lookCells[i]
                    if cell.blocked { continue }
                    let dx = cell.x - view.x, dz = cell.z - view.y
                    if dx * dx + dz * dz > 20 { continue }
                    let p = inv * SIMD4<Float>(cell.x, cell.y, cell.z, 1)
                    let cellDepth = -p.z
                    guard cellDepth > 0.3, cellDepth <= 4 else { continue }
                    let u = fx * p.x / cellDepth + cx
                    let v = cy - fy * p.y / cellDepth
                    // Central 90% of the image only: the camera is really pointed there.
                    guard u > 0.05 * resW, u < 0.95 * resW, v > 0.05 * resH, v < 0.95 * resH else { continue }
                    let idx = Int(v / scale) * width + Int(u / scale)
                    guard HoleLook.isLook(depth: d[idx], confidence: c[idx], cellDepth: cellDepth) else { continue }
                    var tracker = w.looks[cell.key] ?? LookTracker(cell: SIMD2(cell.x, cell.z))
                    tracker.add(viewpoint: view)
                    w.looks[cell.key] = tracker
                    if tracker.isBlocked() {
                        w.lookCells[i].blocked = true
                        w.blockedKeys.insert(cell.key)
                    }
                }
            }
        }
    }

    /// Takes the result of a hole search: ages holes, drops stale look state, detects filled holes.
    private func applyHoles(_ w: Walk, _ found: [FoundHole]) {
        guard walk === w else { return }
        w.holeJobRunning = false
        let now = ProcessInfo.processInfo.systemUptime

        var raw = Set<Int64>()
        var seen: [Int64: Double] = [:]
        var comps: [HoleComp] = []
        for hole in found {
            var settled = false
            var sum = SIMD2<Float>(0, 0)
            for key in hole.keys {
                raw.insert(key)
                let first = w.firstSeen[key] ?? now
                seen[key] = first
                if now - first >= HoleConfig.settleSeconds { settled = true }
                sum += FineGrid.center(key)
            }
            if settled {
                let n = Float(hole.keys.count)
                let radius = max(HoleConfig.cellSize, (n / .pi).squareRoot() * HoleConfig.cellSize)
                comps.append(HoleComp(keys: hole.keys, groundY: hole.groundY, center: sum / n, radius: radius))
            }
        }
        w.firstSeen = seen

        // A hole that was open and has no hole cells left was filled (blocked ones don't count).
        for c in w.comps where !c.blocked && !c.keys.contains(where: raw.contains) { w.pendingFilled += 1 }

        w.looks = w.looks.filter { raw.contains($0.key) }
        w.blockedKeys.formIntersection(raw)
        w.comps = comps
        w.lookCells = comps.flatMap { comp in
            comp.keys.map { key -> LookCell in
                let c = FineGrid.center(key)
                return LookCell(key: key, x: c.x, y: comp.groundY, z: c.y, blocked: w.blockedKeys.contains(key))
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        guard let w = walk else { return }
        w.sessionIssue = "AR session failed: \(error.localizedDescription)"
        publish(w, force: true)
    }

    func sessionWasInterrupted(_ session: ARSession) {
        guard let w = walk else { return }
        w.sessionIssue = "AR session interrupted"
        publish(w, force: true)
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        walk?.sessionIssue = nil
    }

    // MARK: Publishing to the UI

    private func publish(_ w: Walk, force: Bool = false) {
        let now = ProcessInfo.processInfo.systemUptime
        guard force || now - w.lastPublish >= 0.2 else { return }
        w.lastPublish = now

        var snapshot: CoverageSnapshot?
        if w.dirty, now - w.lastSnapshot >= 1 {
            snapshot = w.grid.render()
            w.lastSnapshot = now
            w.dirty = false
        }
        // Hole search runs on its own queue on a copy of the fine grid, at most once a second.
        if w.fineDirty, !w.holeJobRunning, now - w.lastHoleJob >= 1 {
            w.fineDirty = false
            w.holeJobRunning = true
            w.lastHoleJob = now
            let fine = w.fine
            holeQueue.async {
                let found = fine.holes()
                self.q.async { self.applyHoles(w, found) }
            }
        }
        let guidance = guide(w)
        if now - w.lastDiskCheck >= 5 {
            w.lastDiskCheck = now
            let values = try? w.dir.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            w.diskFree = values?.volumeAvailableCapacityForImportantUsage ?? .max
        }

        var issues: [String] = []
        if let t = w.trackingIssue { issues.append(t) }
        if let s = w.sessionIssue { issues.append(s) }
        if w.speed > 1 { issues.append(String(format: "Walking too fast (%.1f m/s), slow down", w.speed)) }
        if w.trackingIssue == nil, now - w.lastValidDepth > 2 { issues.append("No valid depth (sky or too far), aim at the ground") }
        if w.diskFree < 2_000_000_000 { issues.append(String(format: "Low disk space (%.1f GB free)", Double(w.diskFree) / 1e9)) }
        if w.writeFailed { issues.append("Writing to disk failed") }

        onUpdate?(Update(snapshot: snapshot, x: w.pos.x, z: w.pos.z, yaw: w.yaw,
                         duration: now - w.startUptime, frames: w.count, issues: issues,
                         holes: guidance.marks, gaps: guidance.open, target: w.target,
                         targetDistance: guidance.distance, targetAngle: guidance.angle,
                         targetOnScreen: w.targetOnScreen, filled: guidance.filled))
    }

    /// Marks blocked holes, counts open ones and picks the nearest open hole cell as the target.
    private func guide(_ w: Walk) -> (marks: [HoleMark], open: Int, distance: Float, angle: Float, filled: Int) {
        var marks: [HoleMark] = []
        var open = 0
        for i in w.comps.indices {
            w.comps[i].blocked = w.comps[i].keys.allSatisfy(w.blockedKeys.contains)
            let c = w.comps[i]
            if !c.blocked { open += 1 }
            marks.append(HoleMark(x: c.center.x, z: c.center.y, radius: c.radius, blocked: c.blocked))
        }

        // Nearest open cell, sticking with the current target unless another is clearly (1 m) closer.
        func dist(_ c: LookCell) -> Float { hypot(c.x - w.pos.x, c.z - w.pos.z) }
        var best: LookCell?
        var current: LookCell?
        for c in w.lookCells where !c.blocked {
            if best == nil || dist(c) < dist(best!) { best = c }
            if c.key == w.targetKey { current = c }
        }
        var chosen = best
        if let current, let best, dist(current) <= dist(best) + 1 { chosen = current }

        let filled = w.pendingFilled
        w.pendingFilled = 0
        guard let chosen else {
            w.target = nil
            w.targetKey = nil
            return (marks, open, 0, 0, filled)
        }
        w.target = SIMD3(chosen.x, chosen.y, chosen.z)
        w.targetKey = chosen.key
        let bearing = atan2(chosen.x - w.pos.x, -(chosen.z - w.pos.z))
        var rel = bearing - w.yaw
        while rel > .pi { rel -= 2 * .pi }
        while rel < -.pi { rel += 2 * .pi }
        return (marks, open, dist(chosen), rel, filled)
    }

    // MARK: CLLocationManagerDelegate (called on main; hop to q)

    private var lastHeading: CLHeading?

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        q.async {
            guard let w = self.walk else { return }
            for l in locations {
                w.lastLocation = l
                self.logGPS(w, at: l.timestamp)
            }
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        q.async {
            // Heading is only recorded alongside the next location fix, not on every compass tick.
            guard self.walk != nil else { return }
            self.lastHeading = newHeading
        }
    }

    private func logGPS(_ w: Walk, at time: Date) {
        guard let l = w.lastLocation else { return }
        let h = lastHeading
        let validHeading = (h?.trueHeading ?? -1) >= 0
        let entry: [String: Any] = [
            "t": time.timeIntervalSince(w.startDate),
            "lat": l.coordinate.latitude,
            "lon": l.coordinate.longitude,
            "alt": l.altitude,
            "h_acc": l.horizontalAccuracy,
            "v_acc": l.verticalAccuracy,
            "heading": validHeading ? h!.trueHeading : NSNull(),
            "heading_acc": validHeading ? h!.headingAccuracy : NSNull(),
        ]
        guard var line = try? JSONSerialization.data(withJSONObject: entry) else { return }
        line.append(10)
        w.write(line, to: w.gpsFile)
    }

    // MARK: Helpers

    private static func describe(_ state: ARCamera.TrackingState) -> (String, String?) {
        switch state {
        case .normal:
            return ("normal", nil)
        case .notAvailable:
            return ("not_available", "Tracking unavailable")
        case .limited(let reason):
            switch reason {
            case .excessiveMotion: return ("limited_excessive_motion", "Tracking limited: moving too fast")
            case .insufficientFeatures: return ("limited_insufficient_features", "Tracking limited: not enough features")
            case .relocalizing: return ("limited_relocalizing", "Relocalizing: return to a place you already scanned")
            case .initializing: return ("limited_initializing", "Tracking is initializing")
            @unknown default: return ("limited_unknown", "Tracking limited")
            }
        }
    }

    /// Float32 depth bytes -> Float16 bytes (halves storage; ~2 mm precision at 4 m).
    private static func float16(_ data: Data) -> Data {
        data.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: Float.self)
            return src.map { Float16($0) }.withUnsafeBufferPointer { Data(buffer: $0) }
        }
    }

    /// Copies a pixel buffer into tightly packed bytes (row padding removed).
    private static func packed(_ buffer: CVPixelBuffer, bytesPerPixel: Int) -> Data {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let rowBytes = width * bytesPerPixel
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return Data() }
        if stride == rowBytes { return Data(bytes: base, count: rowBytes * height) }
        var out = Data(capacity: rowBytes * height)
        for y in 0..<height { out.append(Data(bytes: base + y * stride, count: rowBytes)) }
        return out
    }
}
