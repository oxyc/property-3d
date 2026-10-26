import Foundation

/// Closes walks the app never got to finish (killed or crashed): meta.json still has `ended_at: null`.
enum Recovery {
    /// Marks every unfinished walk in `root` as interrupted. Returns the folder names that were repaired.
    @discardableResult
    static func recoverAll(in root: URL) -> [String] {
        let dirs = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        return dirs.filter { $0.lastPathComponent.hasPrefix("walk_") && recover(walkDir: $0) }.map(\.lastPathComponent)
    }

    /// If the walk has no `ended_at`: sets `end_reason = "interrupted"`, `ended_at` = walk start + `t` of the last
    /// complete line of frames.jsonl, and an exact `frame_count`. Returns true if meta.json was rewritten.
    static func recover(walkDir: URL) -> Bool {
        let metaURL = walkDir.appendingPathComponent("meta.json")
        guard let data = try? Data(contentsOf: metaURL),
              var meta = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              meta["ended_at"] as? String == nil else { return false }

        let (count, lastT) = scanFrames(walkDir.appendingPathComponent("frames.jsonl"))
        let iso = ISO8601DateFormatter()
        var end = (try? walkDir.appendingPathComponent("frames.jsonl").resourceValues(forKeys: [.contentModificationDateKey]))?
            .contentModificationDate ?? Date()
        if let started = (meta["started_at"] as? String).flatMap(iso.date), let lastT { end = started.addingTimeInterval(lastT) }
        meta["ended_at"] = iso.string(from: end)
        meta["end_reason"] = "interrupted"
        meta["frame_count"] = count
        guard let out = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]) else { return false }
        return (try? out.write(to: metaURL, options: .atomic)) != nil
    }

    /// Number of complete lines and the `t` of the last one. A line cut off by the kill (no newline) is ignored.
    static func scanFrames(_ url: URL) -> (count: Int, lastT: Double?) {
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else { return (0, nil) }
        var count = 0
        var lastLineStart = 0, lastLineEnd = -1
        var lineStart = 0
        for (i, byte) in data.enumerated() where byte == 10 {
            count += 1
            lastLineStart = lineStart
            lastLineEnd = i
            lineStart = i + 1
        }
        guard lastLineEnd > lastLineStart,
              let obj = try? JSONSerialization.jsonObject(with: data[lastLineStart..<lastLineEnd]) as? [String: Any] else {
            return (count, nil)
        }
        return (count, obj["t"] as? Double)
    }
}
