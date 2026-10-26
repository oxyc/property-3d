import Foundation
import simd

/// Tunables for blind-spot ("hole") guidance.
enum HoleConfig {
    /// Fine grid used for hole detection (the 25 cm display grid is too coarse for small gaps).
    static let cellSize: Float = 0.10
    /// Smallest enclosed gap that counts, in fine cells. 1 = any gap, however small.
    static let minHoleCells = 1
    /// Enclosed regions bigger than this (40 m2) are unscanned interiors, not blind spots: ignored, so a loop
    /// around a big unseen patch cannot make the hole structures grow without bound.
    static let maxHoleCells = 4000
    /// At most this many holes (nearest to the user first) and this many hole cells are tracked at once.
    static let maxTrackedHoles = 400
    static let maxTrackedCells = 20000
    /// A hole must have existed this long before it is reported (filters sampling speckle while moving).
    static let settleSeconds = 2.0
    /// Depth beyond this is too sparse to mark fine cells reliably.
    static let maxFineDistance: Float = 3.5
    /// Looks needed before a hole can be declared unseeable.
    static let minLooks = 6
    /// ...from viewpoints at least this far apart (metres) or this different in bearing (degrees).
    static let minSeparation: Float = 1.0
    static let minBearing: Float = 30
    /// A depth reading this much shorter than the cell's distance means something is in front of it.
    static let lookMargin: Float = 0.15
}

struct FoundHole {
    var keys: [Int64]
    var groundY: Float
}

/// 10 cm occupancy grid in world X/Z, stored in 64 x 64 cell tiles (6.4 m, 20 KB each).
///
/// It is a value type whose tiles hold arrays, so `let copy = grid` is a shallow copy of a small dictionary
/// (a few hundred tiles) and a later `mark` only duplicates the one tile it touches (copy-on-write),
/// instead of the whole grid. The hole search runs on such a copy.
struct FineGrid {
    static let tileBits: Int32 = 6
    private static let tileCells = 1 << (2 * Int(tileBits))

    struct Tile {
        var hits = [UInt8](repeating: 0, count: FineGrid.tileCells)
        var minY = [Float](repeating: .infinity, count: FineGrid.tileCells)

        mutating func mark(_ i: Int, y: Float) {
            if hits[i] < .max { hits[i] += 1 }
            minY[i] = min(minY[i], y)
        }
    }

    private(set) var tiles: [Int64: Tile] = [:]

    var tileCount: Int { tiles.count }

    static func key(_ ix: Int32, _ iz: Int32) -> Int64 {
        Int64(ix) << 32 | Int64(UInt32(bitPattern: iz))
    }

    static func decode(_ k: Int64) -> (x: Int32, z: Int32) {
        (Int32(truncatingIfNeeded: k >> 32), Int32(truncatingIfNeeded: k))
    }

    /// World X/Z of a cell centre.
    static func center(_ k: Int64) -> SIMD2<Float> {
        let (ix, iz) = decode(k)
        return SIMD2((Float(ix) + 0.5) * HoleConfig.cellSize, (Float(iz) + 0.5) * HoleConfig.cellSize)
    }

    private static func locate(_ ix: Int32, _ iz: Int32) -> (tile: Int64, index: Int) {
        let mask = (Int32(1) << tileBits) - 1
        return (key(ix >> tileBits, iz >> tileBits), Int(((iz & mask) << tileBits) | (ix & mask)))
    }

    mutating func mark(x: Float, y: Float, z: Float) {
        guard abs(x) < 1e5, abs(z) < 1e5 else { return }
        let (tile, index) = Self.locate(Int32((x / HoleConfig.cellSize).rounded(.down)), Int32((z / HoleConfig.cellSize).rounded(.down)))
        tiles[tile, default: Tile()].mark(index, y: y)
    }

    /// Marks a cell by key as seen, with no height (used for the previous walk's coverage).
    mutating func insert(key k: Int64) {
        let (ix, iz) = Self.decode(k)
        let (tile, index) = Self.locate(ix, iz)
        tiles[tile, default: Tile()].hits[index] = 1
    }

    /// Unseen regions fully enclosed by seen cells, between `minCells` and `maxCells` cells each. Ground height is the
    /// mean lowest-Y of the seen cells around each hole (NaN when only `prior` cells surround it, so the caller
    /// picks a fallback). `prior` is coverage from a previous walk: it counts as seen.
    func holes(minCells: Int = HoleConfig.minHoleCells, maxCells: Int = HoleConfig.maxHoleCells,
               maxSide: Int = 2048, prior: FineGrid? = nil) -> [FoundHole] {
        guard !tiles.isEmpty else { return [] }
        let grids = [self] + (prior.map { [$0] } ?? [])

        // Bounds from the tiles (a few cells wider than the data, which is harmless: the extra is "outside").
        var minX = Int32.max, maxX = Int32.min, minZ = Int32.max, maxZ = Int32.min
        let side = Int32(1) << Self.tileBits
        for grid in grids {
            for k in grid.tiles.keys {
                let (tx, tz) = Self.decode(k)
                minX = min(minX, tx * side); maxX = max(maxX, tx * side + side - 1)
                minZ = min(minZ, tz * side); maxZ = max(maxZ, tz * side + side - 1)
            }
        }
        // One cell of padding on every side guarantees the border is "outside".
        let w = min(Int(maxX) - Int(minX) + 3, maxSide), h = min(Int(maxZ) - Int(minZ) + 3, maxSide)
        let ox = Int(minX) - 1, oz = Int(minZ) - 1
        var covered = [Bool](repeating: false, count: w * h)
        var minY = [Float](repeating: .infinity, count: w * h)
        let tileSide = Int(side)
        for grid in grids {  // prior cells carry no height, so only this walk sets minY
            for (k, tile) in grid.tiles {
                let (tx, tz) = Self.decode(k)
                for i in 0..<tile.hits.count where tile.hits[i] != 0 {
                    let col = Int(tx) * tileSide + (i & (tileSide - 1)) - ox
                    let row = Int(tz) * tileSide + (i >> Int(Self.tileBits)) - oz
                    if col >= 0, col < w, row >= 0, row < h {
                        covered[row * w + col] = true
                        if tile.minY[i].isFinite { minY[row * w + col] = tile.minY[i] }
                    }
                }
            }
        }
        return findEnclosed(covered: covered, width: w, height: h, minCells: minCells, maxCells: maxCells).map { comp in
            var sum: Float = 0, n: Float = 0
            var keys: [Int64] = []
            keys.reserveCapacity(comp.count)
            for i in comp {
                let col = i % w, row = i / w
                keys.append(Self.key(Int32(col + ox), Int32(row + oz)))
                for j in [i - 1, i + 1, i - w, i + w] where covered[j] && minY[j].isFinite {
                    sum += minY[j]; n += 1
                }
            }
            return FoundHole(keys: keys, groundY: n > 0 ? sum / n : .nan)
        }
    }
}

/// Connected components (4-neighbour) of uncovered cells that cannot reach the grid border through
/// uncovered cells. Returns flat indices (row * width + col) per component with `minCells...maxCells` cells.
func findEnclosed(covered: [Bool], width: Int, height: Int, minCells: Int, maxCells: Int = .max) -> [[Int]] {
    guard width > 2, height > 2 else { return [] }
    var visited = covered  // covered cells are never visited
    var stack: [Int] = []

    func seed(_ i: Int) {
        if !visited[i] { visited[i] = true; stack.append(i) }
    }
    func seedNeighbours(_ i: Int) {
        let x = i % width, y = i / width
        if x > 0 { seed(i - 1) }
        if x < width - 1 { seed(i + 1) }
        if y > 0 { seed(i - width) }
        if y < height - 1 { seed(i + width) }
    }

    // Flood the outside from every border cell.
    for x in 0..<width { seed(x); seed((height - 1) * width + x) }
    for y in 0..<height { seed(y * width); seed(y * width + width - 1) }
    while let i = stack.popLast() { seedNeighbours(i) }

    // Whatever uncovered cells remain are enclosed; group them.
    var result: [[Int]] = []
    for start in 0..<(width * height) where !visited[start] {
        var comp: [Int] = []
        visited[start] = true
        stack.append(start)
        while let i = stack.popLast() {
            comp.append(i)
            seedNeighbours(i)
        }
        if comp.count >= minCells, comp.count <= maxCells { result.append(comp) }
    }
    return result
}

enum HoleLook {
    /// Whether one depth reading says "the camera aimed at this cell but could not see it":
    /// no valid or confident depth there, or a surface clearly in front of the cell.
    static func isLook(depth: Float, confidence: UInt8, cellDepth: Float, margin: Float = HoleConfig.lookMargin) -> Bool {
        if !(depth > 0) || confidence < 1 { return true }
        return depth < cellDepth - margin
    }
}

/// Counts looks at one hole cell and remembers distinct viewpoints they were made from.
struct LookTracker {
    let cell: SIMD2<Float>
    private(set) var count = 0
    private(set) var views: [SIMD2<Float>] = []

    init(cell: SIMD2<Float>) { self.cell = cell }

    mutating func add(viewpoint: SIMD2<Float>) {
        count += 1
        if views.count < 8, views.allSatisfy({ differs($0, viewpoint, separation: 0.3, bearing: 10) }) {
            views.append(viewpoint)
        }
    }

    /// Enough looks, and two of the viewpoints differ by the configured distance or bearing.
    func isBlocked(minLooks: Int = HoleConfig.minLooks, separation: Float = HoleConfig.minSeparation,
                   bearing: Float = HoleConfig.minBearing) -> Bool {
        guard count >= minLooks else { return false }
        for i in views.indices {
            for j in views.indices where j > i && differs(views[i], views[j], separation: separation, bearing: bearing) {
                return true
            }
        }
        return false
    }

    private func differs(_ a: SIMD2<Float>, _ b: SIMD2<Float>, separation: Float, bearing: Float) -> Bool {
        if simd_distance(a, b) >= separation { return true }
        let ba = atan2(a.x - cell.x, a.y - cell.y), bb = atan2(b.x - cell.x, b.y - cell.y)
        var d = abs(ba - bb)
        if d > .pi { d = 2 * .pi - d }
        return d * 180 / .pi >= bearing
    }
}
