import SwiftUI
import UniformTypeIdentifiers

/// The loaded previous coverage ("continue a site"): kept in Application Support so it survives relaunches.
@MainActor
final class SiteStore: ObservableObject {
    @Published private(set) var site: Site?
    @Published var message: String?

    private static var fileURL: URL {
        URL.applicationSupportDirectory.appendingPathComponent("site_coverage.json")
    }

    init() {
        if let data = try? Data(contentsOf: Self.fileURL) { site = try? Site.parse(data) }
    }

    var summary: String? {
        site.map { "\($0.id ?? "coverage"), \(Int(Double($0.coveredCount) * $0.cellSize * $0.cellSize)) m\u{00B2} known" }
    }

    /// From the Files picker, AirDrop ("Open in WalkScan") or any file URL.
    func load(_ url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do { try adopt(Data(contentsOf: url)) } catch { message = error.localizedDescription }
    }

    func fetch(_ text: String) async {
        guard let url = URL(string: text.trimmingCharacters(in: .whitespacesAndNewlines)), url.scheme?.hasPrefix("http") == true else {
            message = "That is not an http(s) address."
            return
        }
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            if let http = response as? HTTPURLResponse, http.statusCode != 200 { throw URLError(.badServerResponse) }
            try adopt(data)
        } catch {
            message = "Could not fetch coverage: \(error.localizedDescription)"
        }
    }

    func clear() {
        site = nil
        try? FileManager.default.removeItem(at: Self.fileURL)
    }

    private func adopt(_ data: Data) throws {
        let parsed = try Site.parse(data)
        try FileManager.default.createDirectory(at: .applicationSupportDirectory, withIntermediateDirectories: true)
        try data.write(to: Self.fileURL, options: .atomic)
        site = parsed
        message = "Loaded \(summary ?? "coverage")."
    }
}

/// Maps the coverage frame (east/north metres, north up) onto a view and back.
private struct MapFit {
    let minEast: Double
    let maxNorth: Double
    let scale: Double
    let offset: CGPoint

    init(site: Site, extra: [(east: Double, north: Double)], size: CGSize) {
        var minE = site.originEast, maxE = site.originEast + Double(site.width) * site.cellSize
        var minN = site.originNorth - Double(site.height) * site.cellSize, maxN = site.originNorth
        for p in Array(site.outline.joined()) + extra {
            minE = min(minE, p.east); maxE = max(maxE, p.east)
            minN = min(minN, p.north); maxN = max(maxN, p.north)
        }
        let pad = max(maxE - minE, maxN - minN) * 0.04
        minE -= pad; maxE += pad; minN -= pad; maxN += pad
        scale = min(size.width / (maxE - minE), size.height / (maxN - minN))
        offset = CGPoint(x: (size.width - (maxE - minE) * scale) / 2, y: (size.height - (maxN - minN) * scale) / 2)
        minEast = minE
        maxNorth = maxN
    }

    func point(east: Double, north: Double) -> CGPoint {
        CGPoint(x: (east - minEast) * scale + offset.x, y: (maxNorth - north) * scale + offset.y)
    }

    func coverage(at p: CGPoint) -> (east: Double, north: Double) {
        (minEast + (p.x - offset.x) / scale, maxNorth - (p.y - offset.y) / scale)
    }
}

/// Shows the loaded coverage and lets the user tap where they are standing. That point becomes the
/// walk's origin in the coverage frame (ARKit's own axes already follow the compass).
struct AnchorSheet: View {
    @ObservedObject var rec: Recorder
    let site: Site
    let onStart: (SiteAnchor) -> Void
    let onSkip: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var tapped: (east: Double, north: Double)?
    @State private var image: CGImage?

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                Text("Tap the map where you are standing, face the way you will start walking, then press Start. Teal is already covered, red edges still need walking.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                GeometryReader { geo in
                    let gps = rec.coordinate.flatMap { site.enu(lat: $0.latitude, lon: $0.longitude) }
                    let fit = MapFit(site: site, extra: (gps.map { [$0] } ?? []) + (tapped.map { [$0] } ?? []), size: geo.size)
                    Canvas { ctx, _ in
                        ctx.fill(Path(CGRect(origin: .zero, size: geo.size)), with: .color(Color(white: 0.08)))
                        if let image {
                            let origin = fit.point(east: site.originEast, north: site.originNorth)
                            let rect = CGRect(x: origin.x, y: origin.y, width: Double(site.width) * site.cellSize * fit.scale,
                                              height: Double(site.height) * site.cellSize * fit.scale)
                            ctx.draw(Image(decorative: image, scale: 1).interpolation(.none), in: rect)
                        }
                        for ring in site.outline where ring.count > 1 {
                            var path = Path()
                            path.move(to: fit.point(east: ring[0].east, north: ring[0].north))
                            for p in ring.dropFirst() { path.addLine(to: fit.point(east: p.east, north: p.north)) }
                            path.closeSubpath()
                            ctx.stroke(path, with: .color(.yellow), lineWidth: 1.5)
                        }
                        if let gps {
                            let p = fit.point(east: gps.east, north: gps.north)
                            ctx.fill(Path(ellipseIn: CGRect(x: p.x - 6, y: p.y - 6, width: 12, height: 12)), with: .color(.blue))
                            ctx.stroke(Path(ellipseIn: CGRect(x: p.x - 6, y: p.y - 6, width: 12, height: 12)), with: .color(.white), lineWidth: 1.5)
                        }
                        if let tapped {
                            var c = ctx
                            let p = fit.point(east: tapped.east, north: tapped.north)
                            c.translateBy(x: p.x, y: p.y)
                            c.rotate(by: .degrees(rec.heading ?? 0))
                            var arrow = Path()
                            arrow.move(to: CGPoint(x: 0, y: -14))
                            arrow.addLine(to: CGPoint(x: 8, y: 9))
                            arrow.addLine(to: CGPoint(x: 0, y: 4))
                            arrow.addLine(to: CGPoint(x: -8, y: 9))
                            arrow.closeSubpath()
                            c.fill(arrow, with: .color(.white))
                            c.stroke(arrow, with: .color(.red), lineWidth: 2)
                        }
                    }
                    .contentShape(Rectangle())
                    .onTapGesture { tapped = fit.coverage(at: $0) }
                }
                .clipShape(RoundedRectangle(cornerRadius: 12))

                HStack {
                    Text(rec.heading.map { String(format: "Compass %.0f\u{00B0}%@", $0, rec.headingIsTrue ? " (true)" : " (magnetic)") } ?? "Compass: waiting")
                    Spacer()
                    if rec.coordinate != nil, site.ref == nil { Text("File has no GPS reference") }
                }
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)

                Button {
                    if let tapped {
                        onStart(SiteAnchor(east: tapped.east, north: tapped.north))
                        dismiss()
                    }
                } label: {
                    Text(tapped == nil ? "Tap your position first" : "Start walk here")
                        .font(.title3.bold())
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 56)
                        .background(tapped == nil ? Color.gray : Color.green, in: RoundedRectangle(cornerRadius: 16))
                }
                .disabled(tapped == nil)

                Button("Walk without this site") {
                    onSkip()
                    dismiss()
                }
                .font(.footnote)
            }
            .padding()
            .navigationTitle(site.id ?? "Site coverage")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { Button("Cancel") { dismiss() } }
            .task { image = site.image() }
        }
    }
}
