import ARKit
import CoreLocation
import CoreMotion
import SwiftUI
import UIKit

enum Storage {
    static var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// Recording stops cleanly below this much free space; it warns below `lowSpaceWarn`.
    static let lowSpaceStop: Int64 = 500_000_000
    static let lowSpaceWarn: Int64 = 2_000_000_000

    /// Truly free bytes (not counting purgeable space, which a full disk cannot rely on).
    static func freeBytes() -> Int64? {
        (try? documents.resourceValues(forKeys: [.volumeAvailableCapacityKey]))?.volumeAvailableCapacity.map(Int64.init)
    }

    private static let rateKey = "bytesPerSecond"

    /// Bytes written per second of recording: measured on the last walk, else about 8 GB/hour.
    static var estimatedRate: Double {
        let measured = UserDefaults.standard.double(forKey: rateKey)
        return measured > 0 ? measured : 8e9 / 3600
    }

    static func saveRate(_ bytesPerSecond: Double) { UserDefaults.standard.set(bytesPerSecond, forKey: rateKey) }
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
    var trackingLost: Bool
    var freeBytes: Int64
    var bytesPerSecond: Double
    /// The engine has dropped optional work (gap search) because memory is short.
    var shedding: Bool
    /// Time to return to the start mark (or a place already walked): distance and bearing like the hole target.
    var loopPrompt: Bool
    var loopDistance: Float
    var loopAngle: Float
    /// Set once on the update after a printed marker was first seen.
    var markerToast: String?
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
    @Published private(set) var trackingLost = false
    @Published private(set) var loopPrompt = false
    @Published private(set) var loopDistance: Float = 0
    @Published private(set) var loopAngle: Float = 0
    @Published private(set) var markerToast: String?
    @Published private(set) var freeBytes: Int64?
    @Published private(set) var bytesPerSecond: Double = 0
    /// Live compass heading in degrees (true north if available, else magnetic) and GPS position.
    @Published private(set) var heading: Double?
    @Published private(set) var headingIsTrue = false
    @Published private(set) var coordinate: CLLocationCoordinate2D?
    /// ARKit scene mesh + its RealityKit rendering. Off by default: the mesh grows with the walked area and was
    /// the largest memory consumer; coverage comes from depth, not from the mesh. When switched on it turns
    /// itself off again after `meshOverlaySeconds`, and it is dropped on any memory pressure.
    @Published private(set) var meshOverlay = false
    @Published var errorMessage: String?

    private static let meshOverlaySeconds = 180.0
    private var meshOffTask: Task<Void, Never>?
    private var toastTask: Task<Void, Never>?
    private var memoryObserver: NSObjectProtocol?

    let session = ARSession()
    private let engine = Engine()
    private let location = CLLocationManager()
    private let haptics = UINotificationFeedbackGenerator()
    private let fillHaptic = UIImpactFeedbackGenerator(style: .light)
    private let lostHaptic = UIImpactFeedbackGenerator(style: .heavy)
    private var shownWarnings = Set<String>()
    private var lastHaptic = Date.distantPast
    private var lastFillHaptic = Date.distantPast
    private var lastLostHaptic = Date.distantPast

    func setViewport(_ size: CGSize) { engine.setViewport(size) }

    var areaM2: Double { snapshot?.areaM2 ?? 0 }

    /// Minutes of recording left before the safe stop, from the measured (or estimated) write rate.
    var minutesLeft: Int? {
        guard let free = freeBytes else { return nil }
        let rate = bytesPerSecond > 0 ? bytesPerSecond : Storage.estimatedRate
        return max(0, Int(Double(free - Storage.lowSpaceStop) / rate / 60))
    }

    init() {
        session.delegate = engine
        session.delegateQueue = engine.q
        location.delegate = engine
        location.desiredAccuracy = kCLLocationAccuracyBest
        engine.onUpdate = { [weak self] update in
            Task { @MainActor in self?.apply(update) }
        }
        engine.onStopped = { [weak self] reason in
            Task { @MainActor in self?.handleStopped(reason) }
        }
        engine.onHeading = { [weak self] degrees, isTrue in
            Task { @MainActor in
                self?.heading = degrees
                self?.headingIsTrue = isTrue
            }
        }
        engine.onLocation = { [weak self] coordinate in
            Task { @MainActor in self?.coordinate = coordinate }
        }
        memoryObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.memoryWarning() }
        }
    }

    /// m1, m2, ... from the bundled PNGs (ios/markers/make_markers.py, 12 by default), printed 18 cm wide.
    private static let markerImages: Set<ARReferenceImage> = {
        var images = Set<ARReferenceImage>()
        for id in (1...32).map({ "m\($0)" }) {
            guard let url = Bundle.main.url(forResource: id, withExtension: "png"),
                  let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else { continue }
            let image = ARReferenceImage(cg, orientation: .up, physicalWidth: 0.18)
            image.name = id
            images.insert(image)
        }
        return images
    }()

    private static func makeConfig(mesh: Bool) -> ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravityAndHeading
        config.frameSemantics = .sceneDepth
        // 30 fps at the same 1920 x 1440 as the default: the app only uses ~10 depth frames and 2 keyframes a
        // second, so the extra 30 callbacks/s of a 60 fps format are wasted power. Falls back to ARKit's default.
        if let format = ARWorldTrackingConfiguration.supportedVideoFormats.first(where: {
            $0.framesPerSecond == 30 && $0.imageResolution == CGSize(width: 1920, height: 1440)
        }) {
            config.videoFormat = format
        }
        // Printed markers taped to fixed spots: absolute anchors across passes and walks.
        if !markerImages.isEmpty {
            config.detectionImages = markerImages
            config.maximumNumberOfTrackedImages = 1
        }
        if mesh, ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }
        return config
    }

    /// Mesh overlay on or off without resetting tracking (re-running a config keeps the world). Turning it off
    /// makes ARKit drop all mesh anchors, which frees their memory.
    func setMeshOverlay(_ on: Bool) {
        guard on != meshOverlay else { return }
        meshOverlay = on
        session.run(Self.makeConfig(mesh: on))
        meshOffTask?.cancel()
        if on {
            meshOffTask = Task { [weak self] in
                try? await Task.sleep(for: .seconds(Self.meshOverlaySeconds))
                if !Task.isCancelled { self?.setMeshOverlay(false) }
            }
        }
    }

    /// iOS says memory is short: drop the mesh and pause the gap search, which frees the most.
    private func memoryWarning() {
        setMeshOverlay(false)
        engine.shed(reason: "memory_warning")
    }

    /// Runs the session so the camera and mesh preview show before a walk starts, and keeps
    /// location and compass running (the compass and GPS are used to place the start on a loaded site).
    func startPreview() {
        location.requestWhenInUseAuthorization()
        location.startUpdatingLocation()
        location.startUpdatingHeading()
        session.run(Self.makeConfig(mesh: meshOverlay))
        refreshFreeSpace()
    }

    func refreshFreeSpace() {
        if let free = Storage.freeBytes() { freeBytes = free }
    }

    /// Starts a walk. With a loaded `site` and the tapped `anchor`, the old coverage counts as already covered.
    func start(site: Site?, anchor: SiteAnchor?) {
        guard !isRecording else { return }
        refreshFreeSpace()
        if let free = freeBytes, free < Storage.lowSpaceStop + 200_000_000 {
            errorMessage = "Only \(ByteCountFormatter.string(fromByteCount: free, countStyle: .file)) free: not enough to record. Send walks to your Mac and delete them from the phone, then try again."
            return
        }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let started = Date()
        let dir = Storage.documents.appendingPathComponent("walk_" + formatter.string(from: started))
        let startUptime = ProcessInfo.processInfo.systemUptime
        var anchor = anchor
        anchor?.headingDeg = heading
        anchor?.headingIsTrue = headingIsTrue
        anchor?.coverageId = site?.id
        do {
            try engine.begin(dir: dir, startUptime: startUptime, startDate: started, site: site, anchor: anchor)
        } catch {
            errorMessage = "Could not create walk folder: \(error.localizedDescription)"
            return
        }
        // Reset so the world origin (and gravityAndHeading axes) are where the walk starts. Every walk starts
        // without the mesh; it can be switched on from the screen for a few minutes.
        meshOffTask?.cancel()
        meshOverlay = false
        session.run(Self.makeConfig(mesh: false), options: [.resetTracking, .removeExistingAnchors])
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
        trackingLost = false
        loopPrompt = false
        markerToast = nil
        bytesPerSecond = 0
        isRecording = true
    }

    func stop() {
        guard isRecording else { return }
        isRecording = false
        UIApplication.shared.isIdleTimerDisabled = false
        engine.end()
        warnings = []
        trackingLost = false
        refreshFreeSpace()
    }

    /// The engine ended the walk itself (disk nearly full, or writes failing). The walk is saved and closed.
    private func handleStopped(_ reason: String) {
        guard isRecording else { return }
        isRecording = false
        UIApplication.shared.isIdleTimerDisabled = false
        warnings = []
        trackingLost = false
        refreshFreeSpace()
        haptics.notificationOccurred(.error)
        let free = freeBytes.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) } ?? "little"
        switch reason {
        case "thermal":
            errorMessage = "Recording stopped: the phone is critically hot and iOS was about to shut the camera down. The walk so far was saved and closed cleanly. Let the phone cool in the shade, then start a new walk (use Continue a site to carry on)."
        case "low_space":
            errorMessage = "Recording stopped: the phone is almost out of space (\(free) free). The walk so far was saved and closed cleanly. Send it to your Mac and delete it from the phone before starting another."
        default:
            errorMessage = "Recording stopped: writing to disk failed, most likely because the phone is full (\(free) free). The walk so far was saved and closed cleanly. Send it to your Mac and delete it from the phone before starting another."
        }
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
        trackingLost = u.trackingLost
        loopPrompt = u.loopPrompt
        loopDistance = u.loopDistance
        loopAngle = u.loopAngle
        if let toast = u.markerToast {
            markerToast = toast
            toastTask?.cancel()
            toastTask = Task { [weak self] in
                try? await Task.sleep(for: .seconds(3))
                if !Task.isCancelled { self?.markerToast = nil }
            }
        }
        bytesPerSecond = u.bytesPerSecond
        if u.freeBytes != .max { freeBytes = u.freeBytes }
        if u.shedding, meshOverlay { setMeshOverlay(false) }

        var list = u.issues
        let thermal = ProcessInfo.processInfo.thermalState
        if thermal == .serious || thermal == .critical { list.append("Phone is hot: shade it, or pause and let it cool") }
        let battery = UIDevice.current.batteryLevel
        engine.setBattery(battery)
        if battery >= 0, battery < 0.2 { list.append("Battery low (\(Int(battery * 100))%)") }
        warnings = list

        // Tracking lost gets its own, heavier double tap every 4 s until it clears.
        let now = Date()
        if u.trackingLost, now.timeIntervalSince(lastLostHaptic) > 4 {
            lastLostHaptic = now
            lostHaptic.impactOccurred()
            Task {
                try? await Task.sleep(for: .milliseconds(180))
                lostHaptic.impactOccurred()
            }
        }

        // Haptic on any other new warning, then repeat every few seconds while one persists.
        let current = Set(list).subtracting([Engine.lostMessage])
        if !current.isEmpty, !current.isSubset(of: shownWarnings) || now.timeIntervalSince(lastHaptic) > 4 {
            haptics.notificationOccurred(.warning)
            lastHaptic = now
        }
        shownWarnings = current

        // Light tap when a gap gets filled; never within 1.5 s of another tap.
        if u.filled > 0, now.timeIntervalSince(lastFillHaptic) > 1.5, now.timeIntervalSince(lastHaptic) > 1.5,
           now.timeIntervalSince(lastLostHaptic) > 1.5 {
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

/// World points of one recorded frame, held back before they count towards coverage so that a tracking
/// failure detected a moment later can still discard them.
private struct PointBatch {
    var t: Double
    var coarse: [SIMD3<Float>]
    var fine: [SIMD3<Float>]
}

/// Handles ARKit frames and location updates on `q`: writes to disk, updates coverage, reports to the UI.
/// Everything except the callbacks and `q` is confined to `q`.
final class Engine: NSObject, ARSessionDelegate, CLLocationManagerDelegate, @unchecked Sendable {
    static let lostMessage = "Tracking lost: stop, then point the phone at something you already scanned"
    /// Coverage is applied this long after capture (see `PointBatch`).
    private static let coverageDelay = 2.5

    let q = DispatchQueue(label: "walkscan.record", qos: .userInitiated)
    var onUpdate: (@Sendable (Update) -> Void)?
    /// The engine ended the walk itself; the reason is "low_space" or "write_failed".
    var onStopped: (@Sendable (String) -> Void)?
    var onHeading: (@Sendable (Double, Bool) -> Void)?
    var onLocation: (@Sendable (CLLocationCoordinate2D) -> Void)?

    private var walk: Walk?
    private var viewport = CGSize(width: 390, height: 844)
    /// No intermediate cache: it would keep render buffers of 1920 x 1440 frames alive between keyframes.
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let pedometer = CMPedometer()
    private let holeQueue = DispatchQueue(label: "walkscan.holes", qos: .utility)
    private let motionManager = CMMotionManager()
    private let motionQueue = OperationQueue()
    private let altimeter = CMAltimeter()
    private let altimeterQueue = OperationQueue()
    private var meshAnchors = 0
    private var battery: Float = -1

    private static let iso: ISO8601DateFormatter = ISO8601DateFormatter()

    private final class Walk {
        let dir: URL
        let startUptime: TimeInterval
        let startDate: Date
        let framesFile: FileHandle
        let gpsFile: FileHandle
        let eventsFile: FileHandle
        let baroFile: FileHandle
        let diagFile: FileHandle
        let loopsFile: FileHandle
        let motionFile: FileHandle
        let markersFile: FileHandle
        let encoder = JSONEncoder()
        var meta: [String: Any]

        // Health: memory/thermal log, metadata checkpoints, frame backlog
        var lastDiag = -Double.infinity
        var lastCheckpoint = 0.0
        var queueLag = 0.0  // worst age of a frame when we got to it, since the last diag line
        var framesDropped = 0
        var shedding = false
        var lastLowMemory = false

        var count = 0
        var bytesWritten: Int64 = 0
        var lastDepthT = -Double.infinity
        var keyframes = KeyframeGate()
        var lastCoverageT = -Double.infinity
        var lastPublish = 0.0
        var lastSnapshot = 0.0
        var lastDiskCheck = -Double.infinity
        var dirty = false
        var grid = CoverageGrid()
        var pending: [PointBatch] = []

        // Previous coverage of a loaded site, resampled to this walk's world cells.
        var prior25: [Int64: UInt8] = [:]
        var priorFine: FineGrid?

        // Tracking failure detection
        var monitor = TrackingMonitor()
        var motion = MotionWindow()
        var lost = false
        var degenerate = DegenerateMonitor()

        // Return-to-start prompts, motion.jsonl buffer, marker sightings
        var loop = LoopMonitor()
        var motionBuf = Data()
        var markerLastLogged: [String: Double] = [:]
        var markerLastSeen: [String: Double] = [:]
        var markerToast: String?

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

        init(dir: URL, startUptime: TimeInterval, startDate: Date, site: Site?, anchor: SiteAnchor?) throws {
            self.dir = dir
            self.startUptime = startUptime
            self.startDate = startDate
            lastValidDepth = startUptime
            let fm = FileManager.default
            for sub in ["depth", "rgb"] {
                try fm.createDirectory(at: dir.appendingPathComponent(sub), withIntermediateDirectories: true)
            }
            for name in ["frames.jsonl", "gps.jsonl", "events.jsonl", "baro.jsonl", "diag.jsonl", "loops.jsonl",
                         "motion.jsonl", "markers.jsonl"] {
                guard fm.createFile(atPath: dir.appendingPathComponent(name).path, contents: nil) else {
                    throw CocoaError(.fileWriteUnknown)
                }
            }
            framesFile = try FileHandle(forWritingTo: dir.appendingPathComponent("frames.jsonl"))
            gpsFile = try FileHandle(forWritingTo: dir.appendingPathComponent("gps.jsonl"))
            eventsFile = try FileHandle(forWritingTo: dir.appendingPathComponent("events.jsonl"))
            baroFile = try FileHandle(forWritingTo: dir.appendingPathComponent("baro.jsonl"))
            diagFile = try FileHandle(forWritingTo: dir.appendingPathComponent("diag.jsonl"))
            loopsFile = try FileHandle(forWritingTo: dir.appendingPathComponent("loops.jsonl"))
            motionFile = try FileHandle(forWritingTo: dir.appendingPathComponent("motion.jsonl"))
            markersFile = try FileHandle(forWritingTo: dir.appendingPathComponent("markers.jsonl"))
            meta = [
                "format_version": 3,
                "app_version": "0.6",
                "depth_dtype": "f16",
                "device": Engine.deviceModel,
                "started_at": Engine.iso.string(from: startDate),
                "ended_at": NSNull(),
                "world_alignment": "gravityAndHeading",
                "depth_w": 256, "depth_h": 192,
                "rgb_w": 1920, "rgb_h": 1440,
                "frame_count": 0,
            ]
            if let site, let anchor {
                prior25 = site.worldCells(cellSize: CoverageGrid.cellSize, anchor: anchor)
                var fine = FineGrid()
                for key in site.worldCells(cellSize: HoleConfig.cellSize, anchor: anchor).keys { fine.insert(key: key) }
                priorFine = fine
                meta["site_anchor"] = anchor.metaJSON
                dirty = true  // draw the old coverage right away
            }
            writeMeta()
            if writeFailed { throw CocoaError(.fileWriteOutOfSpace) }
        }

        func writeMeta() {
            guard let data = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]) else { return }
            do { try data.write(to: dir.appendingPathComponent("meta.json"), options: .atomic) } catch { writeFailed = true }
        }

        @discardableResult
        func write(_ data: Data, to file: FileHandle) -> Bool {
            do {
                try file.write(contentsOf: data)
                bytesWritten += Int64(data.count)
                return true
            } catch {
                writeFailed = true
                return false
            }
        }

        @discardableResult
        func write(_ data: Data, path: String) -> Bool {
            do {
                try data.write(to: dir.appendingPathComponent(path))
                bytesWritten += Int64(data.count)
                return true
            } catch {
                writeFailed = true
                return false
            }
        }

        func logDiag(_ entry: [String: Any]) {
            guard var line = try? JSONSerialization.data(withJSONObject: entry, options: [.sortedKeys]) else { return }
            line.append(10)
            write(line, to: diagFile)
        }

        /// Rewrites meta.json so a killed walk still has a sane frame count and a recent checkpoint time.
        func checkpoint() {
            flushIntervals()
            meta["frame_count"] = count
            meta["last_checkpoint"] = Engine.iso.string(from: Date())
            writeMeta()
        }

        func logBaro(_ entry: [String: Any]) {
            guard var line = try? JSONSerialization.data(withJSONObject: entry, options: [.sortedKeys]) else { return }
            line.append(10)
            write(line, to: baroFile)
        }

        /// Writes what a killed app would otherwise lose: the open degenerate_view interval so far, and buffered motion lines.
        func flushIntervals() {
            if let event = degenerate.flush() { log(event) }
            flushMotion()
        }

        /// motion.jsonl lines are buffered (20 Hz would otherwise be 20 writes a second) and written every ~4 s.
        func logMotion(_ line: String) {
            motionBuf.append(Data(line.utf8))
            if motionBuf.count >= 4096 { flushMotion() }
        }

        func flushMotion() {
            guard !motionBuf.isEmpty else { return }
            write(motionBuf, to: motionFile)
            motionBuf.removeAll(keepingCapacity: true)
        }

        func logLine(_ entry: [String: Any], to file: FileHandle) {
            guard var line = try? JSONSerialization.data(withJSONObject: entry, options: [.sortedKeys]) else { return }
            line.append(10)
            write(line, to: file)
        }

        func log(_ event: TrackingEvent) {
            let entry: [String: Any] = ["t_start": (event.tStart * 1000).rounded() / 1000,
                                        "t_end": (event.tEnd * 1000).rounded() / 1000, "reason": event.reason]
            guard var line = try? JSONSerialization.data(withJSONObject: entry, options: [.sortedKeys]) else { return }
            line.append(10)
            write(line, to: eventsFile)
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

    func begin(dir: URL, startUptime: TimeInterval, startDate: Date, site: Site?, anchor: SiteAnchor?) throws {
        try q.sync { walk = try Walk(dir: dir, startUptime: startUptime, startDate: startDate, site: site, anchor: anchor) }
        // The accelerometer tells genuine fast motion from ARKit sliding while the phone is still.
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1.0 / 20
            motionManager.startDeviceMotionUpdates(to: motionQueue) { [weak self] motion, _ in
                guard let self, let motion else { return }
                let a = motion.userAcceleration
                let magnitude = Float((a.x * a.x + a.y * a.y + a.z * a.z).squareRoot())
                // motion.jsonl: user acceleration in g, device axes, shortest exact Float text.
                let line = String(format: "{\"k\":\"a\",\"t\":%.3f,\"a\":[", WalkClock.t(motion.timestamp, startUptime: startUptime))
                    + "\(Float(a.x)),\(Float(a.y)),\(Float(a.z))]}\n"
                self.q.async {
                    self.walk?.motion.add(t: motion.timestamp, magnitudeG: magnitude)
                    self.walk?.logMotion(line)
                }
            }
        }
        startBarometer(startUptime: startUptime)
        startPedometer(startDate: startDate)
    }

    /// Step count and walked distance from the pedometer (cumulative since the walk started), on updates.
    private func startPedometer(startDate: Date) {
        guard CMPedometer.isStepCountingAvailable() else { return }
        pedometer.startUpdates(from: startDate) { [weak self] data, _ in
            guard let self, let data else { return }
            var line = String(format: "{\"k\":\"p\",\"t\":%.3f,\"s\":%d", data.endDate.timeIntervalSince(startDate), data.numberOfSteps.intValue)
            if let d = data.distance { line += ",\"d\":\(Float(d.doubleValue))" }
            line += "}\n"
            self.q.async { self.walk?.logMotion(line) }
        }
    }

    /// Barometer as an independent vertical reference, logged to baro.jsonl on the frame clock.
    /// Relative altitude is measured from the first sample after start; absolute altitude (GPS + pressure) is optional.
    private func startBarometer(startUptime: TimeInterval) {
        if CMAltimeter.isRelativeAltitudeAvailable() {
            altimeter.startRelativeAltitudeUpdates(to: altimeterQueue) { [weak self] data, _ in
                guard let self, let data else { return }
                let entry: [String: Any] = [
                    "type": "relative",
                    "t": WalkClock.t(data.timestamp, startUptime: startUptime),
                    "relative_altitude_m": data.relativeAltitude.doubleValue,
                    "pressure_kpa": data.pressure.doubleValue,
                ]
                self.q.async { self.walk?.logBaro(entry) }
            }
        }
        if CMAltimeter.isAbsoluteAltitudeAvailable() {
            altimeter.startAbsoluteAltitudeUpdates(to: altimeterQueue) { [weak self] data, _ in
                guard let self, let data else { return }
                let entry: [String: Any] = [
                    "type": "absolute",
                    "t": WalkClock.t(data.timestamp, startUptime: startUptime),
                    "altitude_m": data.altitude,
                    "accuracy_m": data.accuracy,
                    "precision_m": data.precision,
                ]
                self.q.async { self.walk?.logBaro(entry) }
            }
        }
    }

    func end() {
        q.async {
            if let w = self.walk { self.finish(w, reason: "user") }
        }
    }

    /// Closes the walk cleanly: events, meta.json with ended_at and the reason, files closed. Runs on `q`.
    private func finish(_ w: Walk, reason: String) {
        guard walk === w else { return }
        walk = nil
        motionManager.stopDeviceMotionUpdates()
        pedometer.stopUpdates()
        altimeter.stopRelativeAltitudeUpdates()
        altimeter.stopAbsoluteAltitudeUpdates()
        let duration = ProcessInfo.processInfo.systemUptime - w.startUptime
        if let event = w.monitor.finish(t: duration) { w.log(event) }
        if let event = w.degenerate.finish(t: duration) { w.log(event) }
        w.flushMotion()
        w.meta["ended_at"] = Self.iso.string(from: Date())
        w.meta["end_reason"] = reason
        w.meta["frame_count"] = w.count
        w.writeMeta()
        try? w.framesFile.close()
        try? w.gpsFile.close()
        try? w.eventsFile.close()
        try? w.baroFile.close()
        try? w.diagFile.close()
        try? w.loopsFile.close()
        try? w.motionFile.close()
        try? w.markersFile.close()
        if duration > 30 { Storage.saveRate(Double(w.bytesWritten) / duration) }
        if reason != "user" { onStopped?(reason) }
    }

    func setViewport(_ size: CGSize) {
        q.async { self.viewport = size }
    }

    func setBattery(_ level: Float) {
        q.async { self.battery = level }
    }

    /// Memory is short: stop the optional gap search and free its structures. Stays on for the rest of the walk.
    func shed(reason: String) {
        q.async {
            self.walk?.flushIntervals()  // a memory warning may precede a kill
            guard let w = self.walk, !w.shedding else { return }
            self.beginShedding(w, reason: reason)
        }
    }

    private func beginShedding(_ w: Walk, reason: String) {
        w.flushIntervals()
        w.shedding = true
        w.comps = []
        w.lookCells = []
        w.looks = [:]
        w.blockedKeys = []
        w.firstSeen = [:]
        w.target = nil
        w.targetKey = nil
        w.priorFine = nil
        logDiag(w, note: reason)
    }

    // MARK: ARSessionDelegate (called on q)

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        meshAnchors += anchors.reduce(0) { $0 + ($1 is ARMeshAnchor ? 1 : 0) }
        for case let image as ARImageAnchor in anchors { markerSeen(image) }
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        for case let image as ARImageAnchor in anchors { markerSeen(image) }
    }

    /// A printed marker's pose in the walk's world frame, at most 2 Hz per marker, to markers.jsonl.
    private func markerSeen(_ anchor: ARImageAnchor) {
        guard let w = walk, let id = anchor.referenceImage.name else { return }
        let t = ProcessInfo.processInfo.systemUptime - w.startUptime
        // Toast the first sighting, and again when it was out of sight for a minute (a new pass).
        if t - (w.markerLastSeen[id] ?? -.infinity) > 60 { w.markerToast = "Marker \(id) seen" }
        w.markerLastSeen[id] = t
        guard t - (w.markerLastLogged[id] ?? -.infinity) >= 0.5 else { return }
        w.markerLastLogged[id] = t
        let tr = anchor.transform
        let entry: [String: Any] = [
            "t": (t * 1000).rounded() / 1000, "id": id,
            "T": (0..<4).flatMap { c in (0..<4).map { r in tr[c][r] } },
            "tracked": anchor.isTracked, "size_m": Double(anchor.referenceImage.physicalSize.width),
            "scale": Double(anchor.estimatedScaleFactor),
        ]
        w.logLine(entry, to: w.markersFile)
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        meshAnchors = max(0, meshAnchors - anchors.reduce(0) { $0 + ($1 is ARMeshAnchor ? 1 : 0) })
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let w = walk else { return }
        let now = frame.timestamp
        // If we are behind (slow disk, heat), ARKit's frames queue up here, each holding several MB. Skip
        // stale frames instead of piling them up; the gap shows in frames.jsonl and in diag.jsonl.
        let lag = ProcessInfo.processInfo.systemUptime - now
        w.queueLag = max(w.queueLag, lag)
        if lag > 0.5 {
            w.framesDropped += 1
            return
        }
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

        // Is tracking clearly wrong (as opposed to the user walking fast)? Everything inside such an
        // interval is logged to events.jsonl for the pipeline and kept out of the coverage.
        let verdict = w.monitor.update(t: now - w.startUptime, x: w.pos.x, z: w.pos.z, normal: trackingName == "normal",
                                       state: trackingName, motionRMS: w.motion.rms(at: now))
        if verdict.opened { w.pending.removeAll() }
        if let event = verdict.closed { w.log(event) }
        w.lost = w.monitor.lost

        guard trackingName == "normal" else {
            w.lastPos = nil
            return
        }
        guard now - w.lastDepthT >= 0.098, let depth = frame.sceneDepth,
              let conf = depth.confidenceMap else { return }
        w.lastDepthT = now

        if let last = w.lastPos, now > w.lastPosT {
            let v = hypot(w.pos.x - last.x, w.pos.z - last.z) / Float(now - w.lastPosT)
            // Steps faster than 5 m/s are tracking glitches (the monitor handles them), not walking.
            if v < 5 { w.speed = 0.8 * w.speed + 0.2 * v }
        }
        w.lastPos = w.pos
        w.lastPosT = now

        // Back at the start mark: tell the pipeline (loops.jsonl, not events.jsonl: this is good, not bad, tracking).
        if w.loop.update(t: now - w.startUptime, pos: w.pos) {
            w.logLine(["t": ((now - w.startUptime) * 1000).rounded() / 1000, "kind": "start_revisit"], to: w.loopsFile)
        }

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

        // A frame is only logged if every file it refers to was written. Any failed write ends the walk
        // cleanly instead of logging frames whose files are missing.
        let stem = String(format: "%06d", w.count)
        w.write(Self.float16(depthData, confidence: confData), path: "depth/\(stem).bin")

        // A view that is mostly one flat surface lets ARKit's pose slide: log it as low-trust odometry.
        let fwd3 = SIMD3(fwd.x, fwd.y, fwd.z)
        let planar: Bool? = depthData.withUnsafeBytes { raw in
            let k = cam.intrinsics
            return ViewShape.isDegenerate(depth: raw.bindMemory(to: Float.self), width: depthW, height: depthH,
                                          fx: k[0][0], fy: k[1][1], cx: k[2][0], cy: k[2][1],
                                          scale: Float(cam.imageResolution.width) / Float(depthW),
                                          up: SIMD3(Double(cam.transform.columns.0.y), Double(cam.transform.columns.1.y),
                                                    Double(cam.transform.columns.2.y)))  // world +Y in camera axes
        }
        if let planar, let event = w.degenerate.update(t: now - w.startUptime, degenerate: planar) { w.log(event) }

        var hasRGB = false
        if w.keyframes.due(t: now, pos: w.pos, fwd: fwd3) {
            let image = CIImage(cvPixelBuffer: frame.capturedImage)
            let options = [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.8]
            if let jpeg = ciContext.jpegRepresentation(of: image, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!, options: options) {
                hasRGB = w.write(jpeg, path: "rgb/\(stem).jpg")
                w.keyframes.saved(t: now, pos: w.pos, fwd: fwd3)
            }
        }
        guard !w.writeFailed else { return finish(w, reason: "write_failed") }

        if w.count == 0 {
            w.meta["depth_w"] = depthW
            w.meta["depth_h"] = depthH
            w.meta["rgb_w"] = CVPixelBufferGetWidth(frame.capturedImage)
            w.meta["rgb_h"] = CVPixelBufferGetHeight(frame.capturedImage)
            w.writeMeta()
        }

        let record = FrameRecord(
            i: w.count,
            t: WalkClock.t(now, startUptime: w.startUptime),
            transform: (0..<4).flatMap { c in (0..<4).map { r in tr[c][r] } },
            intrinsics: [cam.intrinsics[0][0], cam.intrinsics[1][1], cam.intrinsics[2][0], cam.intrinsics[2][1]],
            tracking: trackingName,
            rgb: hasRGB)
        if var line = try? w.encoder.encode(record) {
            line.append(10)
            w.write(line, to: w.framesFile)
        }
        guard !w.writeFailed else { return finish(w, reason: "write_failed") }
        w.count += 1

        // Coverage and hole bookkeeping is the heaviest per-frame work after encoding, and 3 Hz is plenty for it.
        guard now - w.lastCoverageT >= 0.33 else { return }
        w.lastCoverageT = now
        collectCoverage(w, frame: frame, depth: depthData, conf: confData, width: depthW, height: depthH)
        if !w.lost { registerLooks(w, frame: frame, depth: depthData, conf: confData, width: depthW, height: depthH) }
    }

    /// Unprojects every 4th depth pixel (confidence >= 1, depth <= 4 m) for the 25 cm coverage cells,
    /// and every 2nd pixel (depth <= 3.5 m) for the 10 cm grid used for hole detection. The points wait
    /// `coverageDelay` seconds in `pending` before they are counted, and are dropped if tracking fails.
    private func collectCoverage(_ w: Walk, frame: ARFrame, depth: Data, conf: Data, width: Int, height: Int) {
        let cam = frame.camera
        let tr = cam.transform
        let k = cam.intrinsics
        let fx = k[0][0], fy = k[1][1], cx = k[2][0], cy = k[2][1]
        let scale = Float(cam.imageResolution.width) / Float(width)
        var total = 0, valid = 0
        var batch = PointBatch(t: frame.timestamp, coarse: [], fine: [])
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
                            batch.coarse.append(SIMD3(wp.x, wp.y, wp.z))
                        }
                        if z <= HoleConfig.maxFineDistance { batch.fine.append(SIMD3(wp.x, wp.y, wp.z)) }
                    }
                }
            }
        }
        if valid * 20 >= total { w.lastValidDepth = frame.timestamp }
        if !w.lost { w.pending.append(batch) }

        // Count everything that has waited long enough.
        while let first = w.pending.first, first.t <= frame.timestamp - Self.coverageDelay {
            for p in first.coarse { w.grid.mark(x: p.x, y: p.y, z: p.z) }
            for p in first.fine { w.fine.mark(x: p.x, y: p.y, z: p.z) }
            w.pending.removeFirst()
            w.dirty = true
            w.fineDirty = true
        }
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
        guard !w.shedding else { return }
        let now = ProcessInfo.processInfo.systemUptime

        // Every hole cell, to tell "filled" from "merely not tracked right now".
        var raw = Set<Int64>()
        // Only the holes nearest to the user are tracked (bounded count and cells), recomputed every second.
        var candidates: [(hole: FoundHole, center: SIMD2<Float>, distance: Float)] = []
        for hole in found {
            var sum = SIMD2<Float>(0, 0)
            for key in hole.keys {
                raw.insert(key)
                sum += FineGrid.center(key)
            }
            let center = sum / Float(hole.keys.count)
            candidates.append((hole, center, hypot(center.x - w.pos.x, center.y - w.pos.z)))
        }
        candidates.sort { $0.distance < $1.distance }

        var seen: [Int64: Double] = [:]
        var comps: [HoleComp] = []
        var cellBudget = HoleConfig.maxTrackedCells
        for (hole, center, _) in candidates {
            guard comps.count < HoleConfig.maxTrackedHoles, hole.keys.count <= cellBudget else { break }
            cellBudget -= hole.keys.count
            var settled = false
            for key in hole.keys {
                let first = w.firstSeen[key] ?? now
                seen[key] = first
                if now - first >= HoleConfig.settleSeconds { settled = true }
            }
            if settled {
                let n = Float(hole.keys.count)
                let radius = max(HoleConfig.cellSize, (n / .pi).squareRoot() * HoleConfig.cellSize)
                // Holes bordered only by old coverage have no ground height of their own: assume the phone is held ~1.4 m up.
                let ground = hole.groundY.isNaN ? w.pos.y - 1.4 : hole.groundY
                comps.append(HoleComp(keys: hole.keys, groundY: ground, center: center, radius: radius))
            }
        }
        w.firstSeen = seen

        // A hole that was open and has no hole cells left was filled (blocked ones don't count).
        for c in w.comps where !c.blocked && !c.keys.contains(where: raw.contains) { w.pendingFilled += 1 }

        // Look state is kept only for cells that are tracked (at most maxTrackedCells).
        let tracked = Set(comps.lazy.flatMap(\.keys))
        w.looks = w.looks.filter { tracked.contains($0.key) }
        w.blockedKeys.formIntersection(tracked)
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
        guard walk === w, force || now - w.lastPublish >= 0.2 else { return }
        w.lastPublish = now

        // Free space: warn early, and stop cleanly well before the disk is really full.
        if now - w.lastDiskCheck >= 2 {
            w.lastDiskCheck = now
            w.diskFree = Storage.freeBytes() ?? .max
            if w.diskFree < Storage.lowSpaceStop { return finish(w, reason: "low_space") }
        }

        // Memory first: iOS kills an app that goes over its limit without warning. Below the headroom
        // threshold, drop the optional work (gap search; the screen also drops the mesh) while there is still time.
        let available = Memory.availableBytes()
        if !w.shedding, available > 0, available < Memory.lowAvailable { beginShedding(w, reason: "low_memory") }

        // Heat is secondary: iOS throttles first and only pauses the camera at critical. Close the walk cleanly before that.
        if ProcessInfo.processInfo.thermalState == .critical { return finish(w, reason: "thermal") }

        // Crash safety: frame_count and a checkpoint time in meta.json every 30 s, and a health line every 5 s.
        if now - w.lastCheckpoint >= 30 {
            w.lastCheckpoint = now
            w.checkpoint()
        }
        if now - w.lastDiag >= 5 {
            w.lastDiag = now
            logDiag(w)
        }

        var snapshot: CoverageSnapshot?
        if w.dirty, now - w.lastSnapshot >= 1 {
            snapshot = w.grid.render(prior: w.prior25)
            w.lastSnapshot = now
            w.dirty = false
        }
        // Hole search runs on its own queue on a copy of the fine grid, at most once a second.
        if w.fineDirty, !w.holeJobRunning, !w.shedding, now - w.lastHoleJob >= 1 {
            w.fineDirty = false
            w.holeJobRunning = true
            w.lastHoleJob = now
            let fine = w.fine  // shallow copy: tiles are copy-on-write, so this costs a few hundred pointers
            let prior = w.priorFine
            holeQueue.async {
                let found = fine.holes(prior: prior)
                self.q.async { self.applyHoles(w, found) }
            }
        }
        let guidance = guide(w)

        var issues: [String] = []
        if w.lost {
            issues.append(Self.lostMessage)
        } else {
            if let t = w.trackingIssue { issues.append(t) }
            if w.speed > 1 { issues.append(String(format: "Walking too fast (%.1f m/s), slow down", w.speed)) }
            if w.trackingIssue == nil, now - w.lastValidDepth > 2 { issues.append("No valid depth (sky or too far), aim at the ground") }
        }
        if w.degenerate.warning { issues.append(DegenerateConfig.message) }
        if let s = w.sessionIssue { issues.append(s) }
        if w.shedding { issues.append("Low memory: gap search and mesh paused to keep recording") }
        if w.diskFree < Storage.lowSpaceWarn {
            issues.append(String(format: "Low disk space (%.1f GB free), recording stops at %.1f GB",
                                 Double(w.diskFree) / 1e9, Double(Storage.lowSpaceStop) / 1e9))
        }
        if w.writeFailed { issues.append("Writing to disk failed") }

        var loopAngle: Float = 0
        if let s = w.loop.start {
            let bearing = atan2(s.x - w.pos.x, -(s.z - w.pos.z))
            loopAngle = bearing - w.yaw
            while loopAngle > .pi { loopAngle -= 2 * .pi }
            while loopAngle < -.pi { loopAngle += 2 * .pi }
        }
        let toast = w.markerToast
        w.markerToast = nil

        let duration = now - w.startUptime
        onUpdate?(Update(snapshot: snapshot, x: w.pos.x, z: w.pos.z, yaw: w.yaw,
                         duration: duration, frames: w.count, issues: issues,
                         holes: guidance.marks, gaps: guidance.open, target: w.target,
                         targetDistance: guidance.distance, targetAngle: guidance.angle,
                         targetOnScreen: w.targetOnScreen, filled: guidance.filled,
                         trackingLost: w.lost, freeBytes: w.diskFree,
                         bytesPerSecond: duration > 15 ? Double(w.bytesWritten) / duration : 0,
                         shedding: w.shedding, loopPrompt: w.loop.prompting && !w.lost,
                         loopDistance: w.loop.distanceToStart(from: w.pos), loopAngle: loopAngle, markerToast: toast))
    }

    /// One health line for diag.jsonl: memory headroom and footprint, heat, battery, frame backlog, structure sizes.
    private func logDiag(_ w: Walk, note: String? = nil) {
        let now = ProcessInfo.processInfo.systemUptime
        var entry: [String: Any] = [
            "t": WalkClock.t(now, startUptime: w.startUptime),
            "avail_mb": Memory.availableBytes() / 1_000_000,
            "footprint_mb": Int(Memory.footprintBytes() / 1_000_000),
            "thermal": Self.thermalName(ProcessInfo.processInfo.thermalState),
            "battery": battery >= 0 ? Double(battery) as Any : NSNull(),
            "queue_lag_s": (w.queueLag * 1000).rounded() / 1000,
            "frames_dropped": w.framesDropped,
            "mesh_anchors": meshAnchors,
            "frames": w.count,
            "coarse_cells": w.grid.cellCount,
            "fine_tiles": w.fine.tileCount,
            "hole_cells": w.lookCells.count,
            "shedding": w.shedding,
        ]
        if let note { entry["note"] = note }
        w.queueLag = 0
        w.logDiag(entry)
    }

    private static func thermalName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
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
        if let last = locations.last { onLocation?(last.coordinate) }
        q.async {
            guard let w = self.walk else { return }
            for l in locations {
                w.lastLocation = l
                self.logGPS(w, at: l.timestamp)
            }
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        let isTrue = newHeading.trueHeading >= 0
        onHeading?(isTrue ? newHeading.trueHeading : newHeading.magneticHeading, isTrue)
        // Heading is only recorded alongside the next location fix, not on every compass tick.
        q.async { self.lastHeading = newHeading }
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

    /// Float32 depth bytes -> Float16 bytes (halves storage; ~2 mm precision at 4 m). Format v3: pixels that are
    /// invalid or have low confidence (< 1) are written as 0, so no separate confidence file is needed.
    private static func float16(_ data: Data, confidence: Data) -> Data {
        data.withUnsafeBytes { raw in
            confidence.withUnsafeBytes { conf in
                let src = raw.bindMemory(to: Float.self)
                let c = conf.bindMemory(to: UInt8.self)
                return src.indices.map { i -> Float16 in c[i] >= 1 && src[i].isFinite && src[i] > 0 ? Float16(src[i]) : 0 }
                    .withUnsafeBufferPointer { Data(buffer: $0) }
            }
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
