import Foundation

struct TarEntry {
    /// Path inside the archive (and relative to the walk folder for `WalkIndex`).
    let path: String
    let size: Int64
    let mtime: Int64
}

/// Streams an uncompressed POSIX ustar archive without ever building it in memory or on disk:
/// call `next()` until it returns nil. `read(path, offset, count)` supplies file bytes.
struct TarStream {
    private enum Phase { case header, body, padding, trailer, done }

    let entries: [TarEntry]
    private let read: (String, Int64, Int) -> Data
    private var phase = Phase.header
    private var index = 0
    private var offset: Int64 = 0

    init(entries: [TarEntry], read: @escaping (String, Int64, Int) -> Data) {
        self.entries = entries
        self.read = read
    }

    /// Exact size of the finished archive, so a Content-Length can be sent up front.
    static func totalSize(of entries: [TarEntry]) -> Int64 {
        entries.reduce(Int64(1024)) { $0 + 512 + (($1.size + 511) / 512) * 512 }
    }

    mutating func next(maxBytes: Int = 256 * 1024) -> Data? {
        while true {
            switch phase {
            case .header:
                guard index < entries.count else { phase = .trailer; continue }
                let e = entries[index]
                offset = 0
                phase = e.size > 0 ? .body : .padding
                return Self.header(e)
            case .body:
                let e = entries[index]
                let n = Int(min(Int64(maxBytes), e.size - offset))
                var data = read(e.path, offset, n)
                if data.count > n { data = data.prefix(n) }
                if data.count < n { data.append(Data(count: n - data.count)) }  // file shrank: keep the archive valid
                offset += Int64(n)
                if offset >= e.size { phase = .padding }
                return data
            case .padding:
                let e = entries[index]
                let pad = Int((512 - e.size % 512) % 512)
                index += 1
                phase = .header
                if pad > 0 { return Data(count: pad) }
            case .trailer:
                phase = .done
                return Data(count: 1024)
            case .done:
                return nil
            }
        }
    }

    // MARK: Header

    private static func header(_ e: TarEntry) -> Data {
        var h = [UInt8](repeating: 0, count: 512)

        func put(_ s: String, at start: Int, max: Int) {
            for (i, b) in s.utf8.prefix(max).enumerated() { h[start + i] = b }
        }
        func octal(_ v: Int64, at start: Int, len: Int) {
            let digits = String(max(v, 0), radix: 8)
            if digits.count < len {
                put(String(repeating: "0", count: len - 1 - digits.count) + digits, at: start, max: len - 1)
            } else {
                // Too big for octal (>= 8 GiB for size): GNU/POSIX base-256.
                var value = UInt64(max(v, 0))
                for i in stride(from: len - 1, to: 0, by: -1) { h[start + i] = UInt8(value & 0xff); value >>= 8 }
                h[start] = 0x80
            }
        }

        // Names over 100 bytes are split into prefix/name at a '/'.
        let bytes = Array(e.path.utf8)
        var name = e.path, prefix = ""
        if bytes.count > 100,
           let i = bytes.indices.last(where: { bytes[$0] == UInt8(ascii: "/") && $0 <= 155 && bytes.count - $0 - 1 <= 100 }) {
            prefix = String(decoding: bytes[..<i], as: UTF8.self)
            name = String(decoding: bytes[(i + 1)...], as: UTF8.self)
        }
        put(name, at: 0, max: 100)
        octal(0o644, at: 100, len: 8)
        octal(0, at: 108, len: 8)
        octal(0, at: 116, len: 8)
        octal(e.size, at: 124, len: 12)
        octal(e.mtime, at: 136, len: 12)
        for i in 148..<156 { h[i] = 0x20 }  // checksum placeholder
        h[156] = UInt8(ascii: "0")
        put("ustar", at: 257, max: 5)  // magic "ustar\0"
        put("00", at: 263, max: 2)
        put(prefix, at: 345, max: 155)

        let sum = h.reduce(0) { $0 + Int($1) }
        put(String(repeating: "0", count: 6 - String(sum, radix: 8).count) + String(sum, radix: 8), at: 148, max: 6)
        h[154] = 0
        h[155] = 0x20
        return Data(h)
    }
}
