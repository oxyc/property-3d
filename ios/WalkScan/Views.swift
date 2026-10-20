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
    @State private var showWalks = false
    @State private var confirmStop = false

    var body: some View {
        ZStack {
            ARContainer(session: rec.session, target: rec.isRecording ? rec.target : nil).ignoresSafeArea()

            if rec.isRecording, rec.target != nil, !rec.targetOnScreen {
                GuideArrow(angle: rec.targetAngle, distance: rec.targetDistance)
            }

            VStack(spacing: 10) {
                if rec.isRecording {
                    ForEach(rec.warnings, id: \.self) { w in
                        Text(w)
                            .font(.subheadline.bold())
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity)
                            .padding(10)
                            .background(.red.opacity(0.85), in: RoundedRectangle(cornerRadius: 10))
                    }
                }

                HStack(alignment: .top) {
                    if !rec.isRecording {
                        Button { showWalks = true } label: {
                            Label("Walks", systemImage: "folder")
                                .padding(10)
                                .background(.regularMaterial, in: Capsule())
                        }
                    }
                    Spacer()
                    MinimapView(rec: rec)
                        .frame(width: 170, height: 170)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .overlay(RoundedRectangle(cornerRadius: 12).stroke(.white.opacity(0.6)))
                }

                Spacer()

                if rec.isRecording {
                    Text("\(Int(rec.areaM2)) m\u{00B2}  \u{00B7}  \(clock(rec.duration))  \u{00B7}  \(rec.gaps == 0 ? "no gaps" : "\(rec.gaps) gap\(rec.gaps == 1 ? "" : "s") left")")
                        .font(.title3.monospacedDigit().bold())
                        .padding(.horizontal, 14).padding(.vertical, 6)
                        .background(.regularMaterial, in: Capsule())
                }

                Button {
                    if rec.isRecording { confirmStop = true } else { rec.start() }
                } label: {
                    Text(rec.isRecording ? "Stop" : "Start walk")
                        .font(.title.bold())
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 72)
                        .background(rec.isRecording ? Color.red : Color.green, in: RoundedRectangle(cornerRadius: 20))
                }
            }
            .padding()
        }
        .onGeometryChange(for: CGSize.self, of: \.size) { rec.setViewport($0) }
        .onAppear { rec.startPreview() }
        .sheet(isPresented: $showWalks) { WalkListView() }
        .confirmationDialog("Stop recording?", isPresented: $confirmStop, titleVisibility: .visible) {
            Button("Stop and save", role: .destructive) { rec.stop() }
        }
        .alert("Error", isPresented: Binding(get: { rec.errorMessage != nil }, set: { _ in rec.errorMessage = nil })) {
            Button("OK") {}
        } message: {
            Text(rec.errorMessage ?? "")
        }
    }
}

/// Camera feed with ARKit's scene-understanding mesh drawn translucently over it,
/// plus a magenta pole standing on the ground at the current gap (`target`, world coordinates).
struct ARContainer: UIViewRepresentable {
    let session: ARSession
    let target: SIMD3<Float>?

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
        view.debugOptions = [.showSceneUnderstanding]
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
            .background(Color.pink.opacity(0.85), in: RoundedRectangle(cornerRadius: 16))
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
