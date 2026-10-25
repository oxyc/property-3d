import CoreGraphics
import Foundation

/// Where the new walk's ARKit origin sits in a loaded site's coverage frame.
/// ARKit's world is gravityAndHeading (+X east, +Z south), so only a translation is needed:
/// the compass heading is stored for the pipeline, not applied as a rotation.
struct SiteAnchor: Equatable {
    var east: Double
    var north: Double
    var headingDeg: Double?
    var headingIsTrue = false
    var coverageId: String?

    /// ARKit world position (metres from walk start) to the coverage frame.
    func coverage(x: Float, z: Float) -> (east: Double, north: Double) {
        (east + Double(x), north - Double(z))
    }

    /// Coverage frame position to ARKit world (x, z).
    func world(east e: Double, north n: Double) -> (x: Float, z: Float) {
        (Float(e - east), Float(north - n))
    }

    /// The `site_anchor` object written to meta.json.
    var metaJSON: [String: Any] {
        ["coverage_id": coverageId ?? NSNull(), "east": east, "north": north,
         "heading_deg": headingDeg ?? NSNull(), "heading_is_true": headingIsTrue, "method": "tap"]
    }
}

/// A previous coverage map (coverage.json, see FORMAT.md): a grid of covered / blocked cells in east/north metres.
struct Site {
    static let unseen: UInt8 = 0, covered: UInt8 = 1, blocked: UInt8 = 2

    struct ParseError: LocalizedError {
        let errorDescription: String?
    }

    let id: String?
    let cellSize: Double
    /// North-west corner of the grid.
    let originEast: Double
    let originNorth: Double
    let width: Int
    let height: Int
    /// Row-major, row 0 = northernmost, column 0 = westernmost.
    let cells: [UInt8]
    /// Optional boundary rings of (east, north).
    let outline: [[(east: Double, north: Double)]]
    let ref: (lat: Double, lon: Double)?

    private struct Raw: Decodable {
        struct Point: Decodable { var east: Double; var north: Double }
        struct LatLon: Decodable { var lat: Double; var lon: Double }
        var formatVersion: Int
        var id: String?
        var cellSize: Double
        var origin: Point
        var width: Int
        var height: Int
        var rows: [String]
        var outline: [[[Double]]]?
        var ref: LatLon?
    }

    static func parse(_ data: Data) throws -> Site {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let raw: Raw
        do { raw = try decoder.decode(Raw.self, from: data) } catch {
            throw ParseError(errorDescription: "Not a WalkScan coverage file: \(error.localizedDescription)")
        }
        guard raw.formatVersion == 1 else { throw ParseError(errorDescription: "Unsupported coverage format_version \(raw.formatVersion)") }
        guard raw.cellSize >= 0.05, raw.cellSize <= 5 else { throw ParseError(errorDescription: "cell_size out of range") }
        guard raw.width > 0, raw.height > 0, raw.width * raw.height <= 4_000_000 else { throw ParseError(errorDescription: "grid size out of range") }
        guard raw.rows.count == raw.height, raw.rows.allSatisfy({ $0.utf8.count == raw.width }) else {
            throw ParseError(errorDescription: "rows do not match width/height")
        }
        let cells = raw.rows.flatMap { row in
            row.utf8.map { c -> UInt8 in c == UInt8(ascii: "#") ? covered : (c == UInt8(ascii: "x") || c == UInt8(ascii: "X")) ? blocked : unseen }
        }
        let rings = (raw.outline ?? []).map { ring in ring.compactMap { $0.count >= 2 ? (east: $0[0], north: $0[1]) : nil } }
        return Site(id: raw.id, cellSize: raw.cellSize, originEast: raw.origin.east, originNorth: raw.origin.north,
                    width: raw.width, height: raw.height, cells: cells, outline: rings,
                    ref: raw.ref.map { ($0.lat, $0.lon) })
    }

    /// Cell state at a point in the coverage frame (unseen outside the grid).
    func state(east: Double, north: Double) -> UInt8 {
        let col = Int(((east - originEast) / cellSize).rounded(.down))
        let row = Int(((originNorth - north) / cellSize).rounded(.down))
        guard col >= 0, col < width, row >= 0, row < height else { return Site.unseen }
        return cells[row * width + col]
    }

    var coveredCount: Int { cells.reduce(0) { $0 + ($1 == Site.unseen ? 0 : 1) } }

    /// Resamples the site into square world cells of `cellSize` metres around the walk's origin, keyed like
    /// `FineGrid.key` / `CoverageGrid` (x index, z index). Only non-unseen cells are stored.
    func worldCells(cellSize cs: Float, anchor: SiteAnchor) -> [Int64: UInt8] {
        let size = Double(cs)
        let x0 = originEast - anchor.east, x1 = x0 + Double(width) * cellSize
        let z0 = anchor.north - originNorth, z1 = z0 + Double(height) * cellSize
        var out: [Int64: UInt8] = [:]
        for iz in Int32((z0 / size).rounded(.down))...Int32((z1 / size).rounded(.up)) {
            let z = (Double(iz) + 0.5) * size
            for ix in Int32((x0 / size).rounded(.down))...Int32((x1 / size).rounded(.up)) {
                let x = (Double(ix) + 0.5) * size
                let s = state(east: anchor.east + x, north: anchor.north - z)
                if s != Site.unseen { out[FineGrid.key(ix, iz)] = s }
            }
        }
        return out
    }

    /// Local east/north of a GPS fix, if the file carries a reference point (equirectangular, fine at plot scale).
    func enu(lat: Double, lon: Double) -> (east: Double, north: Double)? {
        guard let ref else { return nil }
        return ((lon - ref.lon) * 111_320 * cos(ref.lat * .pi / 180), (lat - ref.lat) * 110_574)
    }

    /// One pixel per cell, north up.
    func image() -> CGImage? {
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for (i, s) in cells.enumerated() {
            let rgb: (UInt8, UInt8, UInt8) = s == Site.covered ? (60, 140, 170) : s == Site.blocked ? (90, 90, 95) : (20, 20, 24)
            pixels[i * 4] = rgb.0; pixels[i * 4 + 1] = rgb.1; pixels[i * 4 + 2] = rgb.2
        }
        return pixels.withUnsafeMutableBytes { buf in
            CGContext(data: buf.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                      space: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)?.makeImage()
        }
    }
}
