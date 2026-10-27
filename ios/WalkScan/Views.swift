import ARKit
import RealityKit
import SwiftUI

func clock(_ t: TimeInterval) -> String {
    let s = Int(t)
    return s >= 3600
        ? String(format: "%d:%02d:%02d", s / 3600, s % 3600 / 60, s % 60)
        : String(format: "%02d:%02d", s / 60, s % 60)
}

struct ContentView: View {
    @StateObject private var rec = Recorder()
    @StateObject private var site = SiteStore()
    @State private var showWalks = false
    @State private var confirmStop = false
    @State private var showAnchor = false
    @State private var showImporter = false
    @State private var showFetch = false
    @State private var fetchText = ""
    /// Walks the app died in are closed on launch; Start waits for that so a new walk is never mistaken for one.
    @State private var recovering = true

    var body: some View {
        ZStack {
            ARContainer(session: rec.session, target: rec.isRecording ? rec.target : nil, meshOverlay: rec.meshOverlay)
                .ignoresSafeArea()

            if rec.isRecording, rec.target != nil, !rec.targetOnScreen {
                GuideArrow(angle: rec.targetAngle, distance: rec.targetDistance)
            }
            let homeTime = rec.duration >= LoopConfig.homeSeconds
            if rec.isRecording, rec.loopPrompt || homeTime {
                GuideArrow(angle: rec.loopAngle, distance: rec.loopDistance, color: homeTime ? .orange : .blue)
            }

            VStack(spacing: 10) {
                if rec.isRecording {
                    if homeTime {
                        Text("\(Int(LoopConfig.homeSeconds / 60)) min: head back to the start mark and finish (\(Int(rec.loopDistance)) m)")
                            .font(.subheadline.bold())
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity)
                            .padding(10)
                            .background(Color.orange.opacity(0.9), in: RoundedRectangle(cornerRadius: 10))
                    } else if rec.loopPrompt {
                        Text("Close the loop: walk back to the start mark (\(Int(rec.loopDistance)) m)")
                            .font(.subheadline.bold())
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity)
                            .padding(10)
                            .background(Color.blue.opacity(0.9), in: RoundedRectangle(cornerRadius: 10))
                    }
                    if let toast = rec.markerToast {
                        Text(toast)
                            .font(.footnote.bold())
                            .padding(.horizontal, 12).padding(.vertical, 6)
                            .background(.regularMaterial, in: Capsule())
                    }
                    ForEach(rec.warnings, id: \.self) { w in
                        let lost = w == Engine.lostMessage
                        Text(w)
                            .font(lost ? .headline : .subheadline.bold())
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity)
                            .padding(lost ? 16 : 10)
                            .background((lost ? Color.orange : Color.red).opacity(0.9), in: RoundedRectangle(cornerRadius: 10))
                    }
                }

                HStack(alignment: .top) {
                    if !rec.isRecording {
                        VStack(alignment: .leading, spacing: 8) {
                            Button { showWalks = true } label: {
                                Label("Walks", systemImage: "folder")
                                    .padding(10)
                                    .background(.regularMaterial, in: Capsule())
                            }
                            Menu {
                                Button("Load coverage from Files\u{2026}", systemImage: "folder.badge.plus") { showImporter = true }
                                Button("Fetch coverage from URL\u{2026}", systemImage: "link") { showFetch = true }
                                if site.site != nil {
                                    Button("Forget this site", systemImage: "trash", role: .destructive) { site.clear() }
                                }
                            } label: {
                                Label(site.summary.map { "Site: \($0)" } ?? "Continue a site", systemImage: "map")
                                    .padding(10)
                                    .background(.regularMaterial, in: Capsule())
                            }
                        }
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 8) {
                        MinimapView(rec: rec)
                            .frame(width: 170, height: 170)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                            .overlay(RoundedRectangle(cornerRadius: 12).stroke(.white.opacity(0.6)))
                        // The mesh grows with the walked area and is the most memory-hungry part: off by default,
                        // switches itself off after 3 minutes.
                        Button { rec.setMeshOverlay(!rec.meshOverlay) } label: {
                            Label(rec.meshOverlay ? "Mesh on (3 min)" : "Show mesh", systemImage: "square.grid.3x3")
                                .font(.footnote.bold())
                                .padding(.horizontal, 10).padding(.vertical, 6)
                                .background(.regularMaterial, in: Capsule())
                        }
                    }
                }

                Spacer()

                if rec.isRecording {
                    Text("\(Int(rec.areaM2)) m\u{00B2}  \u{00B7}  \(clock(rec.duration))  \u{00B7}  \(rec.gaps == 0 ? "no gaps" : "\(rec.gaps) gap\(rec.gaps == 1 ? "" : "s") left")")
                        .font(.title3.monospacedDigit().bold())
                        .padding(.horizontal, 14).padding(.vertical, 6)
                        .background(.regularMaterial, in: Capsule())
                }
                spacePill

                Button {
                    if rec.isRecording {
                        confirmStop = true
                    } else if site.site != nil {
                        showAnchor = true
                    } else {
                        rec.start(site: nil, anchor: nil)
                    }
                } label: {
                    Text(rec.isRecording ? "Stop" : "Start walk")
                        .font(.title.bold())
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 72)
                        .background(rec.isRecording ? Color.red : Color.green, in: RoundedRectangle(cornerRadius: 20))
                }
                .disabled(recovering && !rec.isRecording)
            }
            .padding()
        }
        .onGeometryChange(for: CGSize.self, of: \.size) { rec.setViewport($0) }
        .onAppear { rec.startPreview() }
        .task {
            _ = await Task.detached { Recovery.recoverAll(in: Storage.documents) }.value
            recovering = false
        }
        .task {
            // Free space can change while idle (e.g. after deleting walks).
            while !Task.isCancelled {
                if !rec.isRecording { rec.refreshFreeSpace() }
                try? await Task.sleep(for: .seconds(10))
            }
        }
        .onOpenURL { site.load($0) }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.json]) { result in
            if case .success(let url) = result { site.load(url) }
        }
        .sheet(isPresented: $showWalks, onDismiss: { rec.refreshFreeSpace() }) { WalkListView() }
        .sheet(isPresented: $showAnchor) {
            if let loaded = site.site {
                AnchorSheet(rec: rec, site: loaded,
                            onStart: { rec.start(site: loaded, anchor: $0) },
                            onSkip: { rec.start(site: nil, anchor: nil) })
            }
        }
        .confirmationDialog("Stop recording?", isPresented: $confirmStop, titleVisibility: .visible) {
            Button("Stop and save", role: .destructive) { rec.stop() }
        }
        .alert("Fetch coverage", isPresented: $showFetch) {
            TextField("http://\u{2026}/coverage.json", text: $fetchText)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            Button("Fetch") { Task { await site.fetch(fetchText) } }
            Button("Cancel", role: .cancel) {}
        }
        .alert("WalkScan", isPresented: Binding(get: { rec.errorMessage != nil }, set: { _ in rec.errorMessage = nil })) {
            Button("OK") {}
        } message: {
            Text(rec.errorMessage ?? "")
        }
        .alert("Site coverage", isPresented: Binding(get: { site.message != nil }, set: { _ in site.message = nil })) {
            Button("OK") {}
        } message: {
            Text(site.message ?? "")
        }
    }

    /// Free space and the recording time it buys, before and during a walk.
    @ViewBuilder private var spacePill: some View {
        if let free = rec.freeBytes, let minutes = rec.minutesLeft {
            let low = free < Storage.lowSpaceWarn
            let perHour = rec.isRecording && rec.bytesPerSecond > 0 ? String(format: "  \u{00B7}  %.1f GB/h", rec.bytesPerSecond * 3.6e-6) : ""
            Text("\(ByteCountFormatter.string(fromByteCount: free, countStyle: .file)) free  \u{00B7}  about \(minutes >= 120 ? "\(minutes / 60) h \(minutes % 60) min" : "\(minutes) min") of recording\(perHour)")
                .font(.footnote.monospacedDigit().bold())
                .foregroundStyle(low ? .white : .primary)
                .padding(.horizontal, 12).padding(.vertical, 5)
                .background(low ? AnyShapeStyle(Color.red.opacity(0.85)) : AnyShapeStyle(.regularMaterial), in: Capsule())
        }
    }
}

/// Camera feed with ARKit's scene-understanding mesh drawn translucently over it,
/// plus a magenta pole standing on the ground at the current gap (`target`, world coordinates).
struct ARContainer: UIViewRepresentable {
    let session: ARSession
    let target: SIMD3<Float>?
    let meshOverlay: Bool

    final class Coordinator {
        let anchor = AnchorEntity(world: .zero)
        var added = false
        var last: SIMD3<Float>?
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero)
        view.automaticallyConfigureSession = false
        view.session = session
        view.debugOptions = []
        view.renderOptions = [.disableMotionBlur, .disableDepthOfField, .disablePersonOcclusion, .disableGroundingShadows]

        let material = UnlitMaterial(color: .magenta)
        let pole = ModelEntity(mesh: .generateBox(size: [0.03, 1.2, 0.03]), materials: [material])
        pole.position.y = 0.6
        let ball = ModelEntity(mesh: .generateSphere(radius: 0.1), materials: [material])
        ball.position.y = 1.25
        context.coordinator.anchor.addChild(pole)
        context.coordinator.anchor.addChild(ball)
        context.coordinator.anchor.isEnabled = false
        view.scene.addAnchor(context.coordinator.anchor)
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        uiView.debugOptions = meshOverlay ? [.showSceneUnderstanding] : []
        let c = context.coordinator
        guard c.last != target else { return }
        c.last = target
        if let target {
            c.anchor.setPosition(target, relativeTo: nil)
            c.anchor.isEnabled = true
        } else {
            c.anchor.isEnabled = false
        }
    }
}

/// Screen-edge arrow pointing towards an off-screen gap, with the distance.
struct GuideArrow: View {
    let angle: Float
    let distance: Float
    var color: Color = .pink

    var body: some View {
        GeometryReader { geo in
            let dir = CGVector(dx: Double(sin(angle)), dy: -Double(cos(angle)))
            let halfW = max(geo.size.width / 2 - 50, 40), halfH = max(geo.size.height / 2 - 200, 60)
            let t = min(abs(dir.dx) < 0.001 ? .infinity : halfW / abs(dir.dx),
                        abs(dir.dy) < 0.001 ? .infinity : halfH / abs(dir.dy))
            VStack(spacing: 2) {
                Image(systemName: "arrow.up")
                    .font(.system(size: 44, weight: .bold))
                    .rotationEffect(.radians(Double(angle)))
                Text(String(format: "%.1f m", distance)).font(.headline.monospacedDigit())
            }
            .foregroundStyle(.white)
            .padding(10)
            .background(color.opacity(0.85), in: RoundedRectangle(cornerRadius: 16))
            .position(x: geo.size.width / 2 + dir.dx * t, y: geo.size.height / 2 + dir.dy * t)
        }
        .allowsHitTesting(false)
    }
}

/// Top-down coverage map, north up, auto-fitted to the covered extent plus the user.
struct MinimapView: View {
    @ObservedObject var rec: Recorder

    var body: some View {
        Canvas { ctx, size in
            ctx.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(red: 20 / 255, green: 20 / 255, blue: 24 / 255)))

            // Everything below is in cell units (1 unit = 0.25 m), x east, y south.
            let cell = CoverageGrid.cellSize
            let user = CGPoint(x: Double(rec.position.x / cell), y: Double(rec.position.y / cell))
            var extent = CGRect(x: user.x - 20, y: user.y - 20, width: 40, height: 40)
            if let s = rec.snapshot {
                extent = extent.union(CGRect(x: s.originX, y: s.originZ, width: s.width, height: s.height))
            }
            extent = extent.insetBy(dx: -3, dy: -3)
            let scale = min(size.width / extent.width, size.height / extent.height)
            let offset = CGPoint(x: (size.width - extent.width * scale) / 2 - extent.minX * scale,
                                 y: (size.height - extent.height * scale) / 2 - extent.minY * scale)
            func screen(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x * scale + offset.x, y: p.y * scale + offset.y) }

            if let s = rec.snapshot {
                let origin = screen(CGPoint(x: s.originX, y: s.originZ))
                let rect = CGRect(x: origin.x, y: origin.y, width: Double(s.width) * scale, height: Double(s.height) * scale)
                ctx.draw(Image(decorative: s.image, scale: 1).interpolation(.none), in: rect)
            }

            // Gaps: magenta while open, dark grey once declared unseeable. Ring marks the current target.
            for hole in rec.holes {
                let c = screen(CGPoint(x: Double(hole.x / cell), y: Double(hole.z / cell)))
                let r = max(CGFloat(hole.radius / cell) * scale, 3)
                let circle = Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r))
                ctx.fill(circle, with: .color(hole.blocked ? Color(white: 0.35) : Color(red: 1, green: 0, blue: 1)))
            }
            if let t = rec.target {
                let c = screen(CGPoint(x: Double(t.x / cell), y: Double(t.z / cell)))
                ctx.stroke(Path(ellipseIn: CGRect(x: c.x - 7, y: c.y - 7, width: 14, height: 14)), with: .color(.white), lineWidth: 1.5)
            }

            var arrow = ctx
            let p = screen(user)
            arrow.translateBy(x: p.x, y: p.y)
            arrow.rotate(by: .radians(Double(rec.yaw)))
            var path = Path()
            path.move(to: CGPoint(x: 0, y: -9))
            path.addLine(to: CGPoint(x: 6, y: 7))
            path.addLine(to: CGPoint(x: 0, y: 3))
            path.addLine(to: CGPoint(x: -6, y: 7))
            path.closeSubpath()
            arrow.fill(path, with: .color(.white))
            arrow.stroke(path, with: .color(.black), lineWidth: 1)

            ctx.draw(Text("N").font(.caption.bold()).foregroundStyle(.white), at: CGPoint(x: size.width / 2, y: 9))
        }
    }
}
