import Foundation
import Network

/// Lists the regular files of a walk folder with sizes and mtimes. Metadata files come first so a
/// partial transfer is still useful.
enum WalkIndex {
    static func scan(_ dir: URL) -> [TarEntry] {
        let root = dir.resolvingSymlinksInPath()
        let keys: [URLResourceKey] = [.fileSizeKey, .isRegularFileKey, .contentModificationDateKey]
        guard let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: keys) else { return [] }
        var entries: [TarEntry] = []
        for case let url as URL in walker {
            guard let v = try? url.resourceValues(forKeys: Set(keys)), v.isRegularFile == true else { continue }
            let rel = url.resolvingSymlinksInPath().pathComponents.dropFirst(root.pathComponents.count).joined(separator: "/")
            entries.append(TarEntry(path: rel, size: Int64(v.fileSize ?? 0),
                                    mtime: Int64(v.contentModificationDate?.timeIntervalSince1970 ?? 0)))
        }
        let first = ["meta.json", "frames.jsonl", "gps.jsonl", "events.jsonl"]
        func rank(_ e: TarEntry) -> Int { first.firstIndex(of: e.path) ?? first.count }
        return entries.sorted { (rank($0), $0.path) < (rank($1), $1.path) }
    }
}

/// Wi-Fi IPv4 addresses of this device (en0 first).
func localIPv4Addresses() -> [String] {
    var result: [(name: String, ip: String)] = []
    var head: UnsafeMutablePointer<ifaddrs>?
    guard getifaddrs(&head) == 0, let first = head else { return [] }
    defer { freeifaddrs(head) }
    for ptr in sequence(first: first, next: { $0.pointee.ifa_next }) {
        let ifa = ptr.pointee
        guard let addr = ifa.ifa_addr, addr.pointee.sa_family == UInt8(AF_INET),
              ifa.ifa_flags & UInt32(IFF_UP) != 0, ifa.ifa_flags & UInt32(IFF_LOOPBACK) == 0 else { continue }
        let name = String(cString: ifa.ifa_name)
        guard name.hasPrefix("en") || name.hasPrefix("bridge") else { continue }
        var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
        if getnameinfo(addr, socklen_t(addr.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 {
            result.append((name, String(cString: host)))
        }
    }
    return result.sorted { ($0.name == "en0" ? 0 : 1, $0.name) < ($1.name == "en0" ? 0 : 1, $1.name) }.map(\.ip)
}

/// Serves one walk folder over plain HTTP/1.1 (Network.framework, no dependencies):
///   GET /index.json        {"name", "ended", "total_bytes", "files": [{"path", "size"}]}
///   GET /files/<path>      one file, with Range support so partial downloads can resume
///   GET /walk.tar          the whole walk as an uncompressed tar, generated on the fly (no temp file)
///   POST /verified         the Mac checked every file; the app then offers to delete the walk
final class WalkServer: @unchecked Sendable {
    enum State { case ready(UInt16), failed(String) }

    let root: URL
    var onState: (@Sendable (State) -> Void)?

    private let lock = NSLock()
    private var sent: Int64 = 0
    private var didVerify = false
    private var listener: NWListener?
    private var connections: [ObjectIdentifier: HTTPConnection] = [:]
    fileprivate var entries: [TarEntry] = []
    fileprivate var sizes: [String: Int64] = [:]
    fileprivate var indexJSON = Data()

    var name: String { root.lastPathComponent }
    var bytesSent: Int64 { lock.withLock { sent } }
    var verified: Bool { lock.withLock { didVerify } }

    init(root: URL) { self.root = root }

    func start(advertise: Bool) {
        DispatchQueue.global(qos: .userInitiated).async {
            self.entries = WalkIndex.scan(self.root)
            self.sizes = Dictionary(self.entries.map { ($0.path, $0.size) }, uniquingKeysWith: { a, _ in a })
            let meta = (try? Data(contentsOf: self.root.appendingPathComponent("meta.json")))
                .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]
            let index: [String: Any] = [
                "name": self.name,
                "ended": meta["ended_at"] is String,
                "total_bytes": self.entries.reduce(Int64(0)) { $0 + $1.size },
                "files": self.entries.map { ["path": $0.path, "size": $0.size] as [String: Any] },
            ]
            self.indexJSON = (try? JSONSerialization.data(withJSONObject: index)) ?? Data()
            self.listen(advertise: advertise)
        }
    }

    func stop() {
        listener?.cancel()
        listener = nil
        let open = lock.withLock { Array(connections.values) }
        open.forEach { $0.close() }
    }

    private func listen(advertise: Bool) {
        do {
            let l = try NWListener(using: .tcp)
            if advertise { l.service = NWListener.Service(name: "WalkScan", type: "_walkscan._tcp") }
            l.stateUpdateHandler = { [weak self, weak l] state in
                switch state {
                case .ready: self?.onState?(.ready(l?.port?.rawValue ?? 0))
                case .failed(let error): self?.onState?(.failed(error.localizedDescription))
                default: break
                }
            }
            l.newConnectionHandler = { [weak self] nw in
                guard let self else { return }
                let c = HTTPConnection(nw, server: self)
                self.lock.withLock { self.connections[ObjectIdentifier(c)] = c }
                c.start()
            }
            listener = l
            l.start(queue: DispatchQueue(label: "walkscan.http.accept"))
        } catch {
            onState?(.failed(error.localizedDescription))
        }
    }

    fileprivate func addBytes(_ n: Int) { lock.withLock { sent += Int64(n) } }
    fileprivate func markVerified() { lock.withLock { didVerify = true } }
    fileprivate func remove(_ c: HTTPConnection) { lock.withLock { _ = connections.removeValue(forKey: ObjectIdentifier(c)) } }
}

// MARK: - Connection

private protocol BodySource: AnyObject {
    func next() -> Data?
}

private final class FileSource: BodySource {
    private let handle: FileHandle
    private var remaining: Int64

    init?(url: URL, offset: Int64, length: Int64) {
        guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
        if offset > 0 { try? h.seek(toOffset: UInt64(offset)) }
        handle = h
        remaining = length
    }

    func next() -> Data? {
        guard remaining > 0 else { try? handle.close(); return nil }
        let data = (try? handle.read(upToCount: Int(min(256 * 1024, remaining)))) ?? Data()
        if data.isEmpty { remaining = 0; try? handle.close(); return nil }
        remaining -= Int64(data.count)
        return data
    }
}

private final class TarSource: BodySource {
    private var stream: TarStream!
    private var handle: FileHandle?
    private var handlePath: String?

    /// `entries` are paths relative to `root`; they are stored in the archive under `<root name>/`.
    init(root: URL, entries: [TarEntry]) {
        let prefix = root.lastPathComponent + "/"
        stream = TarStream(entries: entries.map { TarEntry(path: prefix + $0.path, size: $0.size, mtime: $0.mtime) }) { [unowned self] path, offset, count in
            if handlePath != path {
                try? handle?.close()
                handle = try? FileHandle(forReadingFrom: root.appendingPathComponent(String(path.dropFirst(prefix.count))))
                handlePath = path
            }
            guard let h = handle else { return Data() }
            try? h.seek(toOffset: UInt64(offset))
            return (try? h.read(upToCount: count)) ?? Data()
        }
    }

    deinit { try? handle?.close() }

    func next() -> Data? { stream.next() }
}

private final class HTTPConnection {
    private let nw: NWConnection
    private unowned let server: WalkServer
    private let queue = DispatchQueue(label: "walkscan.http.connection")
    private var buffer = Data()
    private var source: BodySource?
    private var expected: Int64 = 0
    private var written: Int64 = 0
    private var keepAlive = true
    private var closed = false

    init(_ nw: NWConnection, server: WalkServer) {
        self.nw = nw
        self.server = server
    }

    func start() {
        nw.stateUpdateHandler = { [self] state in
            switch state {
            case .failed, .cancelled: close()
            default: break
            }
        }
        nw.start(queue: queue)
        receive()
    }

    func close() {
        queue.async { [self] in
            guard !closed else { return }
            closed = true
            source = nil
            nw.cancel()
            server.remove(self)
        }
    }

    // MARK: Requests

    private func receive() {
        if let end = buffer.range(of: Data("\r\n\r\n".utf8)) {
            let head = buffer[..<end.lowerBound]
            buffer.removeSubrange(..<end.upperBound)
            handle(String(decoding: head, as: UTF8.self))
            return
        }
        guard buffer.count < 32 * 1024 else { return close() }
        nw.receive(minimumIncompleteLength: 1, maximumLength: 16 * 1024) { [self] data, _, complete, error in
            if let data { buffer.append(data) }
            if error != nil || (complete && data == nil) { return close() }
            receive()
        }
    }

    private func handle(_ head: String) {
        let lines = head.components(separatedBy: "\r\n")
        let parts = lines[0].split(separator: " ")
        guard parts.count >= 2 else { return respond(400, "bad request") }
        let method = String(parts[0])
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            if let colon = line.firstIndex(of: ":") {
                headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            }
        }
        keepAlive = headers["connection"]?.lowercased() != "close"
        let path = parts[1].split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init) ?? "/"

        if method == "POST", path == "/verified" {
            server.markVerified()
            keepAlive = false
            return respond(200, "ok")
        }
        guard method == "GET" || method == "HEAD" else { return respond(405, "method not allowed") }
        let headOnly = method == "HEAD"

        switch path {
        case "/":
            respond(200, "WalkScan walk server: GET /index.json, /files/<path>, /walk.tar\n", headOnly: headOnly)
        case "/index.json":
            respond(200, data: server.indexJSON, type: "application/json", headOnly: headOnly)
        case "/walk.tar":
            let entries = server.entries
            // Names are longer inside the archive (walk folder prefix) but sizes, hence the total, are the same.
            send(status: "200 OK", type: "application/x-tar", length: TarStream.totalSize(of: entries),
                 extra: [], source: headOnly ? nil : TarSource(root: server.root, entries: entries))
        default:
            guard path.hasPrefix("/files/"), let rel = String(path.dropFirst(7)).removingPercentEncoding,
                  let size = server.sizes[rel] else { return respond(404, "not found") }  // only indexed files: no path traversal
            var start: Int64 = 0, end = size - 1
            var partial = false
            if let range = headers["range"], range.hasPrefix("bytes=") {
                let bounds = range.dropFirst(6).split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
                guard let s = bounds.first.flatMap({ Int64($0) }), s < size else {
                    return send(status: "416 Range Not Satisfiable", type: "text/plain", length: 0,
                                extra: [("Content-Range", "bytes */\(size)")], source: nil)
                }
                start = s
                if bounds.count > 1, let e = Int64(bounds[1]) { end = min(e, size - 1) }
                partial = true
            }
            let length = max(end - start + 1, 0)
            let body = headOnly || length == 0 ? nil : FileSource(url: server.root.appendingPathComponent(rel), offset: start, length: length)
            if !headOnly, length > 0, body == nil { return respond(500, "cannot open file") }
            send(status: partial ? "206 Partial Content" : "200 OK", type: "application/octet-stream", length: length,
                 extra: partial ? [("Content-Range", "bytes \(start)-\(end)/\(size)")] : [], source: body,
                 declared: length)
        }
    }

    // MARK: Responses

    private func respond(_ status: Int, _ text: String, headOnly: Bool = false) {
        respond(status, data: Data(text.utf8), type: "text/plain", headOnly: headOnly)
    }

    private func respond(_ status: Int, data: Data, type: String, headOnly: Bool = false) {
        let reason = [200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed", 500: "Internal Server Error"][status] ?? "Error"
        let body = BufferSource(headOnly ? Data() : data)
        send(status: "\(status) \(reason)", type: type, length: Int64(data.count), extra: [], source: body)
    }

    private func send(status: String, type: String, length: Int64, extra: [(String, String)],
                      source: BodySource?, declared: Int64? = nil) {
        var head = "HTTP/1.1 \(status)\r\nContent-Type: \(type)\r\nContent-Length: \(length)\r\nAccept-Ranges: bytes\r\n"
        head += "Connection: \(keepAlive ? "keep-alive" : "close")\r\n"
        for (k, v) in extra { head += "\(k): \(v)\r\n" }
        head += "\r\n"
        self.source = source
        expected = source == nil ? 0 : (declared ?? length)
        written = 0
        nw.send(content: Data(head.utf8), completion: .contentProcessed { [self] error in
            if error != nil { return close() }
            pump()
        })
    }

    private func pump() {
        guard let source, let chunk = source.next() else { return finish() }
        written += Int64(chunk.count)
        nw.send(content: chunk, completion: .contentProcessed { [self] error in
            if error != nil { return close() }
            server.addBytes(chunk.count)
            pump()
        })
    }

    private func finish() {
        source = nil
        if written != expected || !keepAlive { return close() }
        receive()
    }
}

private final class BufferSource: BodySource {
    private var data: Data?
    init(_ data: Data) { self.data = data.isEmpty ? nil : data }
    func next() -> Data? {
        defer { data = nil }
        return data
    }
}
