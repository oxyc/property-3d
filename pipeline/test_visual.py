"""Tests for visual.py.

usage: test_visual.py [test_synthetic] [test_real_walk] [test_cross_walk] [test_forest] ...
Default is test_synthetic only (seconds). The real-data tests need walks/ and out/ next to the pipeline folder and
print their findings (nothing is asserted beyond sanity): they are measurements, not pass/fail checks.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # not on sys.path when run with -I
import visual

ROOT = Path(__file__).parent.parent
W, H, DW, DH = 960, 720, 256, 192
INTR = [800.0, 800.0, 480.0, 360.0]


# ---------------------------------------------------------------- synthetic scene

class Scene:
    """Rough textured ground y = h(x, z) over [-6, 6]^2, rendered by ray marching: exact depth for any pose."""
    N, EXT = 512, 6.0

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        h = cv2.GaussianBlur(rng.normal(size=(self.N, self.N)).astype(np.float32), (0, 0), 14)
        self.h = (h / np.abs(h).max() * 0.3).astype(np.float32)
        tex = np.zeros((4 * self.N, 4 * self.N, 3), np.float32)
        for sigma, amp in ((1.5, 1.0), (4, 1.0), (12, 1.5)):
            n = rng.normal(size=tex.shape).astype(np.float32)
            tex += amp * cv2.GaussianBlur(n, (0, 0), sigma) * sigma ** 0.8
        for _ in range(400):  # blobs and strokes: corners to detect
            c = tuple(int(v) for v in rng.integers(0, tex.shape[0], 2))
            col = tuple(float(v) for v in rng.normal(0, 3, 3))
            if rng.random() < 0.5:
                cv2.circle(tex, c, int(rng.integers(3, 20)), col, -1)
            else:
                cv2.line(tex, c, (c[0] + int(rng.integers(-60, 60)), c[1] + int(rng.integers(-60, 60))), col, 3)
        tex = (tex - tex.mean()) / tex.std()
        self.tex = tex

    def _grid(self, x, z, size):
        return ((x + self.EXT) / (2 * self.EXT) * (size - 1)).astype(np.float32), ((z + self.EXT) / (2 * self.EXT) * (size - 1)).astype(np.float32)

    def render(self, T, w, h, sx, sy, want_colour):
        """Pixels (i * sx, j * sy) of a camera with ARKit pose T: planar depth (NaN on a miss) and colour."""
        u, v = np.meshgrid(np.arange(w) * sx, np.arange(h) * sy)
        fx, fy, cx, cy = INTR
        d_cam = np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u)], -1).reshape(-1, 3)
        d_world = d_cam @ T[:3, :3].T  # t below is planar depth because d_cam has |z| = 1
        o = T[:3, 3]
        n = len(d_world)
        t_hit = np.full(n, np.nan)
        prev_f = None
        step = 0.03
        ts = np.arange(0.3, 9.0, step)
        for t in ts:
            p = o + t * d_world
            gx, gz = self._grid(p[:, 0], p[:, 2], self.N)
            hh = cv2.remap(self.h, gx.reshape(h, w), gz.reshape(h, w), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=-9).ravel()
            f = p[:, 1] - hh
            if prev_f is not None:
                hit = (f < 0) & (prev_f >= 0) & np.isnan(t_hit) & (hh > -8)
                t_hit[hit] = t - step * (-f[hit]) / (prev_f[hit] - f[hit])
            prev_f = f
        colour = None
        if want_colour:
            p = o + np.nan_to_num(t_hit, nan=1.0)[:, None] * d_world
            gx, gz = self._grid(p[:, 0], p[:, 2], 4 * self.N)
            colour = cv2.remap(self.tex, gx.reshape(h, w), gz.reshape(h, w), cv2.INTER_LINEAR)
            colour[np.isnan(t_hit).reshape(h, w)] = 0
        return t_hit.reshape(h, w), colour


def lookat(pos, target):
    """ARKit camera-to-world: +X right, +Y up, -Z forward."""
    f = np.asarray(target, float) - pos
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, up, -f, pos
    return T


def write_walk(dirpath, scene, poses, seed=0):
    dirpath = Path(dirpath)
    for sub in ("rgb", "depth", "conf"):
        (dirpath / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    (dirpath / "meta.json").write_text(json.dumps(dict(depth_dtype="f16", depth_w=DW, depth_h=DH, rgb_w=W, rgb_h=H)))
    lines = []
    for i, T in enumerate(poses):
        _, col = scene.render(T, W, H, 1.0, 1.0, True)
        img = np.clip(128 + 45 * col, 0, 255).astype(np.uint8)
        cv2.imwrite(str(dirpath / "rgb" / f"{i:06d}.jpg"), cv2.GaussianBlur(img, (0, 0), 0.7), [cv2.IMWRITE_JPEG_QUALITY, 92])
        depth, _ = scene.render(T, DW, DH, W / DW, H / DH, False)
        ok = np.isfinite(depth) & (depth < 4.5)
        depth = np.where(ok, depth + rng.normal(0, 0.004, depth.shape), 0).astype("<f2")  # 4 mm lidar noise
        depth.tofile(dirpath / "depth" / f"{i:06d}.bin")
        np.where(ok, 2, 0).astype(np.uint8).tofile(dirpath / "conf" / f"{i:06d}.bin")
        lines.append(json.dumps(dict(i=i, t=i * 30.0, transform=T.flatten(order="F").tolist(), intrinsics=INTR,
                                     tracking="normal", rgb=True)))
    (dirpath / "frames.jsonl").write_text("\n".join(lines) + "\n")


def pose_error(T_est, T_true):
    d = np.linalg.inv(T_true) @ T_est
    return np.linalg.norm(d[:3, 3]), visual._rot_deg(d[:3, :3])


def test_synthetic():
    """Known camera poses over a rough textured scene: the recovered relative pose must match to 1 cm and 0.2 deg.
    A pure translation, a yaw+pitch+roll-ish change, and a swapped pair (axis conventions), plus an unrelated scene
    that must be rejected."""
    scene, other = Scene(1), Scene(2)
    cams = [lookat(np.array([0.0, 1.6, 0.0]), [0.3, 0.0, -3.0]),
            lookat(np.array([0.7, 1.7, -0.3]), [0.2, 0.0, -3.2]),     # 0.8 m sideways + slight yaw
            lookat(np.array([-1.0, 1.9, 0.6]), [1.0, 0.0, -2.0])]     # bigger baseline, ~35 deg yaw
    with tempfile.TemporaryDirectory() as tmp:
        walk, walk2 = Path(tmp) / "walk", Path(tmp) / "other"
        write_walk(walk, scene, cams)
        write_walk(walk2, other, cams[:1])
        kfs = visual.keyframes(walk)
        assert len(kfs) == 3
        for a, b in ((0, 1), (1, 0), (0, 2), (2, 0)):
            r = visual.match_pair(walk, kfs[a], walk, kfs[b])
            assert r["ok"], (a, b, r["reason"], r["stats"])
            true = np.linalg.inv(cams[a]) @ cams[b]
            dt, dr = pose_error(r["T_rel"], true)
            print(f"pair {a}->{b}: {r['stats']['n_inliers']} inliers, reproj {r['stats']['reproj_med']:.2f} px, "
                  f"error {dt * 100:.2f} cm {dr:.3f} deg, baseline {np.linalg.norm(true[:3, 3]):.2f} m")
            assert dt < 0.01 and dr < 0.2, (a, b, dt, dr)
            assert r["info"].shape == (6, 6) and np.all(np.linalg.eigvalsh(r["info"]) > 0)
        kfs2 = visual.keyframes(walk2)
        r = visual.match_pair(walk, kfs[0], walk2, kfs2[0])
        print("unrelated scene:", r["reason"])
        assert not r["ok"]
        # candidate_pairs: the three frames are 30 s apart, so all pairs qualify within one walk
        pairs = visual.candidate_pairs(kfs, kfs, same_walk=True, radius=4.0, min_gap=1.0)
        assert (0, 1) in pairs and all(i < j for i, j in pairs), pairs


# ---------------------------------------------------------------- real data

def yaw_deg(R):
    return float(np.degrees(np.arctan2(R[0, 2] - R[2, 0], R[0, 0] + R[2, 2])))  # about +Y, as optimize.yaw_of


def poses_of(name):
    p = ROOT / "out" / name / "poses.npz"
    return p if p.exists() else None


def implied_world(constraints, kfs_a, kfs_b):
    """Yaw (deg), translation (m) and tilt (deg) of the B-world to A-world transform each constraint implies."""
    Gs = visual.world_alignments(constraints, kfs_a, kfs_b)
    return (np.array([yaw_deg(G[:3, :3]) for G in Gs]), Gs[:, :3, 3],
            np.degrees(np.arccos(np.clip(Gs[:, 1, 1], -1, 1))))


def describe(x, unit):
    x = np.asarray(x)
    if len(x) == 0:
        return "n/a"
    q = np.percentile(x, [10, 50, 90])
    return f"median {q[1]:.2f} {unit} (10-90%: {q[0]:.2f}..{q[2]:.2f}, max {x.max():.2f})"


def short_span_accuracy(name, step=2, n=120, seed=0):
    """Match keyframes `step` apart (~1 s) and compare with ARKit's own relative pose over that span."""
    walk = ROOT / "walks" / name
    kfs = visual.keyframes(walk)  # raw ARKit transforms: odometry is what we compare against
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(kfs) - step, min(n, len(kfs) - step), replace=False)
    meta = visual._meta(walk)
    feats = {}
    dts, drs, rejected, ok = [], [], {}, 0
    t0 = time.time()
    for k in sorted(idx):
        a, b = kfs[k], kfs[k + step]
        if b["t"] - a["t"] > 1.6:  # a gap (tracking lost): not a short span
            continue
        for kf in (a, b):
            if kf["i"] not in feats:
                feats[kf["i"]] = visual.extract(walk, kf)
        r = visual.solve_pair(a, feats[a["i"]], b, feats[b["i"]], meta, meta)
        if not r["ok"]:
            rejected[r["reason"]] = rejected.get(r["reason"], 0) + 1
            continue
        ok += 1
        dt, dr = pose_error(r["T_rel"], np.linalg.inv(a["T"]) @ b["T"])
        dts.append(dt * 100)
        drs.append(dr)
    print(f"{name}: {ok} accepted of {ok + sum(rejected.values())} pairs ~1 s apart, rejected {rejected}, "
          f"{time.time() - t0:.0f} s")
    print(f"  translation vs ARKit: {describe(dts, 'cm')}")
    print(f"  rotation vs ARKit:    {describe(drs, 'deg')}")
    return np.array(dts), np.array(drs)


def test_real_walk():
    """Same-walk self-consistency on the garden walk that starts and ends on the same mark."""
    name = "walk_20261019_175636"
    for n, step in ((name, 2), ("walk_20261019_175018", 2), ("walk_20261024_172426", 1), ("walk_20261024_172426", 2)):
        short_span_accuracy(n, step=step)
    # NB on this walk the photos show a static camera over the last 6 s while ARKit's rotation swings by up to 155 deg
    # (position stays put), so the rotation columns below say more about ARKit than about visual.py
    walk = ROOT / "walks" / name
    kfs = visual.keyframes(walk)
    meta = visual._meta(walk)
    first, last = kfs[:4], kfs[-4:]
    feats = {f["i"]: visual.extract(walk, f) for f in first + last}
    print(f"start/end of {name} (ARKit start-to-end distance {np.linalg.norm(kfs[0]['T'][:3, 3] - kfs[-1]['T'][:3, 3]) * 100:.1f} cm):")
    for a in first:
        for b in last:
            r = visual.solve_pair(a, feats[a["i"]], b, feats[b["i"]], meta, meta)
            if r["ok"]:
                dt, dr = pose_error(r["T_rel"], np.linalg.inv(a["T"]) @ b["T"])
                print(f"  frames {a['i']} / {b['i']}: {r['stats']['n_inliers']} inliers; visual minus ARKit: {dt * 100:.1f} cm, {dr:.2f} deg")
            else:
                print(f"  frames {a['i']} / {b['i']}: rejected ({r['reason']}, {r['stats'].get('n_matches')} matches)")


def report_cross(loops, summary, kfs_a, kfs_b):
    print({k: v for k, v in summary.items() if k != "alignment"})
    if not loops:
        return
    yaws, trans, tilts = implied_world(loops, kfs_a, kfs_b)
    print(f"  implied yaw {describe(yaws, 'deg')}; translation median {np.round(np.median(trans, 0), 2).tolist()} m, "
          f"spread (std) {np.round(trans.std(0), 2).tolist()}; tilt {describe(tilts, 'deg')}")
    # how well do the constraints agree on ONE alignment? place every B camera with the medoid alignment and compare
    # with where its own constraint puts it
    Gs = visual.world_alignments(loops, kfs_a, kfs_b)
    G = visual._medoid_alignment(Gs)
    tb = {f["i"]: f["T"] for f in kfs_b}
    off = [np.linalg.norm((G @ tb[x[3]])[:3, 3] - (g @ tb[x[3]])[:3, 3]) for x, g in zip(loops, Gs)]
    print(f"  medoid alignment: yaw {yaw_deg(G[:3, :3]):.2f} deg, translation {np.round(G[:3, 3], 2).tolist()} m; "
          f"camera position disagreement of the constraints with it: {describe(off, 'm')}")
    near = [(x[1], x[3], x[6]["n_inliers"]) for x in sorted(loops, key=lambda x: x[1])]
    print(f"  accepted (frame in A, frame in B, inliers), {len(near)} total: {near[:12]}{' ...' if len(near) > 12 else ''}")


def test_cross_walk():
    a, b = "walk_20261019_175018", "walk_20261019_175636"
    loops, summary = visual.visual_loops(ROOT / "walks" / a, ROOT / "walks" / b)  # same mark: identity placement
    print(f"{a} vs {b}:")
    report_cross(loops, summary, visual.keyframes(ROOT / "walks" / a), visual.keyframes(ROOT / "walks" / b))


def report_within(loops, kfs):
    """Accepted same-walk loops: where they are, and how far the current poses are from what the photos say."""
    if not loops:
        return
    by = {f["i"]: f for f in kfs}
    rows = []
    for _, i, _, j, T_rel, _, st in sorted(loops, key=lambda x: x[1]):
        dt, dr = pose_error(T_rel, np.linalg.inv(by[i]["T"]) @ by[j]["T"])
        rows.append((i, j, by[j]["t"] - by[i]["t"], st["n_inliers"], dt, dr))
    print(f"  {len(rows)} accepted; (frame, frame, gap s, inliers, pose error vs current poses in m / deg):")
    for r in rows[:15]:
        print(f"    {r[0]} {r[1]} {r[2]:.0f}s {r[3]} inl  {r[4]:.2f} m {r[5]:.2f} deg")
    print(f"  pose error of the current poses at these loops: {describe([r[4] for r in rows], 'm')}; {describe([r[5] for r in rows], 'deg')}")


def test_long_walk():
    """The 38 min plot walk, raw poses, a random sample of the candidate pairs (not exhaustive)."""
    a = "walk_20261024_163126"
    loops, summary = visual.visual_loops(ROOT / "walks" / a, max_pairs=6000)
    print(f"within {a}: {summary}")
    report_within(loops, visual.keyframes(ROOT / "walks" / a))


def gps_init(wa, wb, pa=None, pb=None):
    """B's world frame -> A's from the walks' GPS fixes (translation only, as merge.py's first guess)."""
    import georef  # heavy imports, only needed here
    ea, na, za, *_ = georef.fit_translation(wa, pa)
    eb, nb, zb, *_ = georef.fit_translation(wb, pb)
    G = np.eye(4)
    G[:3, 3] = [eb - ea, zb - za, na - nb]
    return G


def test_forest():
    a, b = "walk_20261024_172426", "walk_20261024_170928"
    wa, wb = ROOT / "walks" / a, ROOT / "walks" / b
    pa, pb = poses_of(a), poses_of(b)
    print(f"poses: {a} {'poses.npz' if pa else 'raw'}, {b} {'poses.npz' if pb else 'raw'}")
    init = gps_init(wa, wb, pa, pb)
    print("GPS placement of B in A (m, east/up/south):", np.round(init[:3, 3], 1).tolist())
    loops, summary = visual.visual_loops(wa, poses_a=pa)
    print(f"within {a}: {summary}")
    report_within(loops, visual.keyframes(wa, pa))
    loops, summary = visual.visual_loops(wa, wb, poses_a=pa, poses_b=pb, init=init)
    print(f"{a} vs {b}:")
    report_cross(loops, summary, visual.keyframes(wa, pa), visual.keyframes(wb, pb))


if __name__ == "__main__":
    for test in sys.argv[1:] or ["test_synthetic"]:
        globals()[test]()
    print("OK")
