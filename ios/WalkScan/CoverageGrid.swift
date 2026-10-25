import CoreGraphics
import Foundation

/// Rendered view of the coverage grid, ready for the minimap.
/// Image column 0 / row 0 is cell (originX, originZ); rows grow towards +Z (south), so north is up.
struct CoverageSnapshot {
    let image: CGImage
    let originX: Int
    let originZ: Int
    let width: Int
    let height: Int
    let coveredCells: Int

    var areaM2: Double { Double(coveredCells) * Double(CoverageGrid.cellSize * CoverageGrid.cellSize) }
}

/// 2.5D coverage map: 0.25 m cells in world X/Z with hit count and min/max Y.
/// Not thread safe; owned by the recording queue.
struct CoverageGrid {
    static let cellSize: Float = 0.25
    /// A cell counts as covered once it has this many depth points.
    static let minHits: UInt16 = 3

    struct Cell {
        var hits: UInt16 = 0
        var minY = Float.infinity
        var maxY = -Float.infinity
    }

    private var cells: [Int64: Cell] = [:]

    private static func key(_ ix: Int32, _ iz: Int32) -> Int64 {
        Int64(ix) << 32 | Int64(UInt32(bitPattern: iz))
    }

    mutating func mark(x: Float, y: Float, z: Float) {
        guard abs(x) < 1e5, abs(z) < 1e5 else { return }
        let ix = Int32((x / Self.cellSize).rounded(.down))
        let iz = Int32((z / Self.cellSize).rounded(.down))
        let k = Self.key(ix, iz)
        var c = cells[k] ?? Cell()
        if c.hits < .max { c.hits += 1 }
        c.minY = min(c.minY, y)
        c.maxY = max(c.maxY, y)
        cells[k] = c
    }

    /// Paints covered cells by hit count and uncovered cells touching a covered cell ("frontier") red.
    /// `prior` are cells from a previous walk (same keys, 25 cm): teal if covered, grey if unseeable. They count
    /// as covered for the frontier but not for the area.
    func render(prior: [Int64: UInt8] = [:], maxSide: Int = 1024) -> CoverageSnapshot? {
        var minX = Int32.max, maxX = Int32.min, minZ = Int32.max, maxZ = Int32.min
        var covered = 0
        func extend(_ k: Int64) {
            let ix = Int32(truncatingIfNeeded: k >> 32), iz = Int32(truncatingIfNeeded: k)
            minX = min(minX, ix); maxX = max(maxX, ix)
            minZ = min(minZ, iz); maxZ = max(maxZ, iz)
        }
        for (k, c) in cells where c.hits >= Self.minHits {
            extend(k)
            covered += 1
        }
        for k in prior.keys { extend(k) }
        guard covered > 0 || !prior.isEmpty else { return nil }

        // One cell of padding so the frontier outside the covered extent is visible.
        let w = min(Int(maxX) - Int(minX) + 3, maxSide)
        let h = min(Int(maxZ) - Int(minZ) + 3, maxSide)
        let ox = Int(minX) - 1, oz = Int(minZ) - 1

        var hits = [UInt16](repeating: 0, count: w * h)
        var old = [UInt8](repeating: 0, count: w * h)
        for (k, state) in prior {
            let col = Int(Int32(truncatingIfNeeded: k >> 32)) - ox
            let row = Int(Int32(truncatingIfNeeded: k)) - oz
            if col >= 0, col < w, row >= 0, row < h { old[row * w + col] = state }
        }
        for (k, c) in cells where c.hits >= Self.minHits {
            let col = Int(Int32(truncatingIfNeeded: k >> 32)) - ox
            let row = Int(Int32(truncatingIfNeeded: k)) - oz
            if col >= 0, col < w, row >= 0, row < h { hits[row * w + col] = c.hits }
        }
        func seen(_ i: Int) -> Bool { hits[i] > 0 || old[i] > 0 }

        var pixels = [UInt8](repeating: 255, count: w * h * 4)
        for row in 0..<h {
            for col in 0..<w {
                let i = row * w + col
                let rgb: (UInt8, UInt8, UInt8)
                if hits[i] > 0 {
                    let t = min(Float(hits[i]) / 50, 1)
                    rgb = (UInt8(30 + 90 * t), UInt8(90 + 140 * t), UInt8(50 + 70 * t))
                } else if old[i] == Site.covered {
                    rgb = (60, 140, 170)
                } else if old[i] == Site.blocked {
                    rgb = (90, 90, 95)
                } else if (col > 0 && seen(i - 1)) || (col < w - 1 && seen(i + 1))
                            || (row > 0 && seen(i - w)) || (row < h - 1 && seen(i + w)) {
                    rgb = (230, 50, 50)
                } else {
                    rgb = (20, 20, 24)
                }
                pixels[i * 4] = rgb.0
                pixels[i * 4 + 1] = rgb.1
                pixels[i * 4 + 2] = rgb.2
            }
        }

        let image = pixels.withUnsafeMutableBytes { buf -> CGImage? in
            CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                      space: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)?.makeImage()
        }
        guard let image else { return nil }
        return CoverageSnapshot(image: image, originX: ox, originZ: oz, width: w, height: h, coveredCells: covered)
    }
}
