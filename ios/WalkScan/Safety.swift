import Foundation

enum WalkClock {
    /// Walk-relative seconds, the `t` of frames.jsonl and baro.jsonl. ARFrame.timestamp and CMLogItem.timestamp
    /// (CoreMotion, including CMAltimeter) are both seconds since boot, the same base as `ProcessInfo.systemUptime`,
    /// which is what `startUptime` is captured from.
    static func t(_ timestamp: TimeInterval, startUptime: TimeInterval) -> Double { timestamp - startUptime }
}

/// A stretch of the walk where ARKit tracking was clearly wrong. Times are seconds since walk start
/// (same clock as frames.jsonl). `tStart` includes a margin before detection because the pose slides
/// for a while before it is detectable; the pipeline should drop frames inside [tStart, tEnd].
struct TrackingEvent: Equatable {
    var tStart: Double
    var tEnd: Double
    var reason: String
}

enum TrackingConfig {
    /// Implied speed (over ~1 s of poses) above which tracking is assumed wrong, in m/s.
    static let maxSpeed: Float = 2.5
    /// ...unless the accelerometer shows vigorous motion (RMS of user acceleration in g, ~running).
    static let runningMotion: Float = 0.35
    /// Distance between consecutive samples that no person can cover in 0.1 s, in metres.
    static let jumpMetres: Float = 1.5
    /// Tracking limited / unavailable for this long counts as lost.
    static let limitedSeconds = 1.0
    /// Seconds before detection that are included in the event (the slide starts before it is visible).
    static let backoff = 10.0
    /// Clean frames needed before the event is closed, and margin added after the last anomaly.
    static let settle = 3.0
    static let endMargin = 1.0
    /// No limited-tracking events right after start (ARKit is still initialising).
    static let startGrace = 5.0
}

/// Decides from the stream of camera poses whether tracking has gone wrong, as opposed to the user
/// simply walking fast. Pure logic: feed it every frame, in time order.
struct TrackingMonitor {
    struct Result {
        var opened = false
        var closed: TrackingEvent?
    }

    private var samples: [(t: Double, x: Float, z: Float)] = []
    private var lastSampleT = -Double.infinity
    private var nonNormalSince: Double?
    private var current: (start: Double, reason: String)?
    private var lastAnomaly = 0.0
    private var lastEventEnd = 0.0

    private(set) var lost = false

    /// - motionRMS: RMS of the accelerometer's user acceleration over the last ~1.5 s in g, nil if unknown.
    mutating func update(t: Double, x: Float, z: Float, normal: Bool, state: String, motionRMS: Float?) -> Result {
        var anomaly: String?
        if normal {
            nonNormalSince = nil
            if t - lastSampleT >= 0.09 {
                if let last = samples.last, t - last.t <= 0.5, hypot(x - last.x, z - last.z) >= TrackingConfig.jumpMetres {
                    anomaly = "jump"
                }
                samples.append((t, x, z))
                lastSampleT = t
                while let first = samples.first, first.t < t - 1.2 { samples.removeFirst() }
                if let old = samples.first, t - old.t >= 0.8 {
                    let speed = Float(hypot(Double(x - old.x), Double(z - old.z)) / (t - old.t))
                    if speed > TrackingConfig.maxSpeed, motionRMS.map({ $0 < TrackingConfig.runningMotion }) ?? true {
                        anomaly = anomaly ?? "implausible_speed"
                    }
                }
            }
        } else {
            if nonNormalSince == nil { nonNormalSince = t }
            if t > TrackingConfig.startGrace, t - nonNormalSince! >= TrackingConfig.limitedSeconds {
                anomaly = state.contains("relocalizing") ? "relocalizing" : "tracking_limited"
            }
            // Positions on either side of a limited stretch are not comparable.
            samples.removeAll()
            lastSampleT = -Double.infinity
        }

        var result = Result()
        if let anomaly {
            lastAnomaly = t
            if current == nil {
                let limited = !normal
                let start = max(lastEventEnd, limited ? (nonNormalSince ?? t) : t - TrackingConfig.backoff, 0)
                current = (start, anomaly)
                lost = true
                result.opened = true
            }
        } else if let c = current, normal, t - lastAnomaly >= TrackingConfig.settle {
            let event = TrackingEvent(tStart: c.start, tEnd: lastAnomaly + TrackingConfig.endMargin, reason: c.reason)
            current = nil
            lost = false
            lastEventEnd = event.tEnd
            result.closed = event
        }
        return result
    }

    /// Closes an event that is still open when the walk ends.
    mutating func finish(t: Double) -> TrackingEvent? {
        guard let c = current else { return nil }
        current = nil
        lost = false
        return TrackingEvent(tStart: c.start, tEnd: t, reason: c.reason)
    }
}

/// Rolling RMS of accelerometer user-acceleration magnitude (g).
struct MotionWindow {
    private var samples: [(t: Double, sq: Float)] = []

    mutating func add(t: Double, magnitudeG: Float) {
        samples.append((t, magnitudeG * magnitudeG))
        while let first = samples.first, first.t < t - 3 { samples.removeFirst() }
    }

    func rms(at now: Double, window: Double = 1.5) -> Float? {
        let recent = samples.filter { $0.t >= now - window && $0.t <= now + 0.1 }
        guard recent.count >= 10 else { return nil }
        return (recent.reduce(0) { $0 + $1.sq } / Float(recent.count)).squareRoot()
    }
}
