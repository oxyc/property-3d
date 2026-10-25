import SwiftUI

@main
struct WalkScanApp: App {
    init() {
        // Nothing in tmp outlives a launch; leftovers (system image-encoding scratch, half-written share zips)
        // otherwise pile up to gigabytes over a few long walks.
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory
        for url in (try? fm.contentsOfDirectory(at: tmp, includingPropertiesForKeys: nil)) ?? [] {
            try? fm.removeItem(at: url)
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
