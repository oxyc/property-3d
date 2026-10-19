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
            ARContainer(session: rec.session).ignoresSafeArea()

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
                    Text("\(Int(rec.areaM2)) m\u{00B2}  \u{00B7}  \(clock(rec.duration))")
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

/// Camera feed with ARKit's scene-understanding mesh drawn translucently over it.
struct ARContainer: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero)
        view.automaticallyConfigureSession = false
        view.session = session
        view.debugOptions = [.showSceneUnderstanding]
        view.renderOptions = [.disableMotionBlur, .disableDepthOfField, .disablePersonOcclusion, .disableGroundingShadows]
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
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
