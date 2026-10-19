import SwiftUI

struct WalkInfo: Identifiable {
    let url: URL
    let duration: TimeInterval?
    let bytes: Int64
    let frames: Int
    var id: String { url.lastPathComponent }
}

struct ShareItem: Identifiable {
    let url: URL
    var id: URL { url }
}

/// Reads every walk folder in Documents (meta.json plus total size). Slow for big walks, so call off the main thread.
private func scanWalks() -> [WalkInfo] {
    let fm = FileManager.default
    let dirs = (try? fm.contentsOfDirectory(at: Storage.documents, includingPropertiesForKeys: nil)) ?? []
    let iso = ISO8601DateFormatter()
    return dirs.filter { $0.lastPathComponent.hasPrefix("walk_") }.sorted { $0.lastPathComponent > $1.lastPathComponent }.map { dir in
        var bytes: Int64 = 0
        if let files = fm.enumerator(at: dir, includingPropertiesForKeys: [.fileSizeKey]) {
            for case let file as URL in files {
                bytes += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            }
        }
        let meta = (try? Data(contentsOf: dir.appendingPathComponent("meta.json")))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]
        let started = (meta["started_at"] as? String).flatMap(iso.date)
        let ended = (meta["ended_at"] as? String).flatMap(iso.date)
        let duration = started.flatMap { s in ended.map { $0.timeIntervalSince(s) } }
        var frames = meta["frame_count"] as? Int ?? 0
        if ended == nil, let data = try? Data(contentsOf: dir.appendingPathComponent("frames.jsonl")) {
            frames = data.reduce(0) { $1 == 10 ? $0 + 1 : $0 }  // unfinished walk: count lines
        }
        return WalkInfo(url: dir, duration: duration, bytes: bytes, frames: frames)
    }
}

/// Zips a walk folder with NSFileCoordinator and moves the zip to a temp path that outlives the callback.
private func zipWalk(_ dir: URL) -> URL? {
    var result: URL?
    var error: NSError?
    NSFileCoordinator().coordinate(readingItemAt: dir, options: .forUploading, error: &error) { zip in
        let dest = FileManager.default.temporaryDirectory.appendingPathComponent(dir.lastPathComponent + ".zip")
        try? FileManager.default.removeItem(at: dest)
        if (try? FileManager.default.moveItem(at: zip, to: dest)) != nil { result = dest }
    }
    return result
}

struct WalkListView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var walks: [WalkInfo] = []
    @State private var loading = true
    @State private var pendingDelete: WalkInfo?
    @State private var zipping = false
    @State private var shareItem: ShareItem?

    var body: some View {
        NavigationStack {
            List(walks) { walk in
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(walk.id).font(.headline)
                        Text(details(walk)).font(.subheadline).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button { share(walk) } label: { Image(systemName: "square.and.arrow.up") }
                        .buttonStyle(.borderless)
                }
                .swipeActions {
                    Button("Delete", role: .destructive) { pendingDelete = walk }
                }
            }
            .overlay {
                if loading {
                    ProgressView()
                } else if walks.isEmpty {
                    Text("No walks yet").foregroundStyle(.secondary)
                }
            }
            .overlay {
                if zipping { ProgressView("Zipping\u{2026}").padding().background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12)) }
            }
            .navigationTitle("Walks")
            .toolbar { Button("Done") { dismiss() } }
            .task {
                walks = await Task.detached { scanWalks() }.value
                loading = false
            }
            .confirmationDialog("Delete this walk?", isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }),
                                titleVisibility: .visible, presenting: pendingDelete) { walk in
                Button("Delete \(walk.id)", role: .destructive) {
                    walks.removeAll { $0.id == walk.id }
                    Task.detached { try? FileManager.default.removeItem(at: walk.url) }
                }
            } message: { _ in
                Text("This cannot be undone.")
            }
            .sheet(item: $shareItem) { item in ActivityView(url: item.url) }
        }
    }

    private func details(_ w: WalkInfo) -> String {
        let size = ByteCountFormatter.string(fromByteCount: w.bytes, countStyle: .file)
        let time = w.duration.map(clock) ?? "unfinished"
        return "\(time)  \u{00B7}  \(size)  \u{00B7}  \(w.frames) frames"
    }

    private func share(_ walk: WalkInfo) {
        zipping = true
        Task {
            let url = await Task.detached { zipWalk(walk.url) }.value
            zipping = false
            if let url { shareItem = ShareItem(url: url) }
        }
    }
}

struct ActivityView: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> UIActivityViewController {
        let vc = UIActivityViewController(activityItems: [url], applicationActivities: nil)
        vc.completionWithItemsHandler = { _, _, _, _ in try? FileManager.default.removeItem(at: url) }
        return vc
    }

    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
