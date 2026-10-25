import SwiftUI
import UIKit

/// Runs the Wi-Fi server for one walk and exposes its state to the UI.
@MainActor
final class SendModel: ObservableObject {
    @Published private(set) var addresses: [String] = []
    @Published private(set) var status = "Starting\u{2026}"
    @Published private(set) var sent: Int64 = 0
    @Published private(set) var verified = false

    private var server: WalkServer?
    private var poll: Task<Void, Never>?

    func start(_ walk: WalkInfo) {
        guard server == nil else { return }
        let server = WalkServer(root: walk.url)
        server.onState = { [weak self] state in
            Task { @MainActor in self?.handle(state) }
        }
        server.start(advertise: true)
        self.server = server
        poll = Task { [weak self] in
            while !Task.isCancelled {
                if let self, let server = self.server {
                    self.sent = server.bytesSent
                    self.verified = server.verified
                }
                try? await Task.sleep(for: .milliseconds(500))
            }
        }
    }

    func stop() {
        poll?.cancel()
        server?.stop()
        server = nil
    }

    private func handle(_ state: WalkServer.State) {
        switch state {
        case .ready(let port):
            addresses = localIPv4Addresses().map { "http://\($0):\(port)" }
            status = addresses.isEmpty ? "No Wi-Fi address found. Connect the phone to Wi-Fi." : "Ready"
        case .failed(let message):
            status = "Could not start: \(message)"
        }
    }
}

/// "Send to Mac": serves the walk over Wi-Fi so `pipeline/pull.py` can download it without zipping.
struct SendView: View {
    let walk: WalkInfo
    let onDeleted: () -> Void
    @Environment(\.dismiss) private var dismiss
    @StateObject private var model = SendModel()
    @State private var confirmDelete = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(walk.id).font(.headline)
                    Text("\(ByteCountFormatter.string(fromByteCount: walk.bytes, countStyle: .file)), \(walk.frames) frames")
                        .foregroundStyle(.secondary)

                    if let url = model.addresses.first {
                        Text("On your Mac, in the property-3d folder:").font(.subheadline)
                        Text("python3 pipeline/pull.py \(url)")
                            .font(.system(.callout, design: .monospaced))
                            .textSelection(.enabled)
                            .padding(10)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
                        Button("Copy command", systemImage: "doc.on.doc") {
                            UIPasteboard.general.string = "python3 pipeline/pull.py \(url)"
                        }
                        if model.addresses.count > 1 {
                            Text("Other addresses: " + model.addresses.dropFirst().joined(separator: ", "))
                                .font(.footnote).foregroundStyle(.secondary)
                        }
                    } else {
                        Text(model.status)
                    }

                    Text("Keep this screen open and the phone on the same Wi-Fi as the Mac. The first time, iOS asks to allow local network access: say yes.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                    if model.sent > 0 {
                        Text("Sent \(ByteCountFormatter.string(fromByteCount: model.sent, countStyle: .file))")
                            .font(.callout.monospacedDigit())
                    }

                    if model.verified {
                        Label("The Mac checked every file against the sizes on the phone.", systemImage: "checkmark.seal.fill")
                            .foregroundStyle(.green)
                        Button("Delete from phone", role: .destructive) { confirmDelete = true }
                            .buttonStyle(.borderedProminent)
                    }
                }
                .padding()
            }
            .navigationTitle("Send to Mac")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { Button("Done") { dismiss() } }
            .confirmationDialog("Delete this walk from the phone?", isPresented: $confirmDelete, titleVisibility: .visible) {
                Button("Delete \(walk.id)", role: .destructive) {
                    model.stop()
                    onDeleted()
                    Task.detached { try? FileManager.default.removeItem(at: walk.url) }
                    dismiss()
                }
            } message: {
                Text("It was verified on the Mac, but this cannot be undone.")
            }
        }
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            model.start(walk)
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
            model.stop()
        }
        .interactiveDismissDisabled()
    }
}
