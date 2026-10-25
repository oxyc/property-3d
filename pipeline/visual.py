"""Visual loop closures: relative pose constraints between RGB keyframes, found by photo feature matching
(SIFT) and made metric with the lidar depth (PnP-RANSAC on keypoints lifted to 3D with frame A's depth).

Conventions. Frame poses are ARKit camera-to-world (+X right, +Y up, -Z forward). OpenCV cameras are
+X right, +Y down, +Z forward, so cv camera = C @ ARKit camera with C = diag(1, -1, -1); ARKit intrinsics
[fx, fy, cx, cy] are already in image pixels (v down) and are used unchanged by OpenCV.

A constraint's T_rel is the pose of camera B expressed in camera A, in ARKit camera axes: T_rel = inv(T_A) @ T_B.
Its information matrix is over the right perturbation T_rel @ Exp(xi), xi = [translation(3), rotation(3)]
in B's ARKit camera axes (metres, radians).

usage (API): kfs = keyframes(walk, poses); pairs = candidate_pairs(kfs, kfs, same_walk=True);
             loops, summary = visual_loops(walk_a, walk_b=None, poses_a=None, poses_b=None)
"""
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from fuse import MAX_DEPTH, MIN_DEPTH, read_depth, read_walk

MATCH_W = 960            # keypoints are found on images downscaled to this width
N_FEATURES = 3000
RATIO = 0.8              # Lowe ratio test
PNP_THRESH = 3.0         # RANSAC reprojection threshold, px at MATCH_W
INLIER_THRESH = 2.5      # inlier threshold after refinement
MIN_INLIERS = 40
MAX_MEDIAN_REPROJ = 2.0
MIN_HULL = 0.10          # inliers must span this fraction of the image (convex hull area), in both images
MIN_DEPTH_CHECKS = 15
DEPTH_AGREE = (0.05, 0.02)   # median |dz| < max(5 cm, 2% of range)
REV_AGREE = (0.05, 0.03, 1.0)  # forward and reverse estimates: max(5 cm, 3% range), degrees
SAME_WALK_MIN_DT = 20.0  # s: closer pairs in one walk are odometry, not loops
C_FLIP = np.diag([1.0, -1.0, -1.0])


# ---------------------------------------------------------------- keyframes and candidate pairs

_META = {}  # walk dir -> meta.json, filled by keyframes() so worker threads never re-read the walk


def _meta(walk):
    walk = Path(walk)
    if walk not in _META:
        _META[walk] = read_walk(walk)[0]
    return _META[walk]


def keyframes(walk, poses=None):
    """RGB keyframes (frame dicts with 'T'). poses: poses.npz path; its transforms replace the recorded ones and
    frames it leaves out (dropped by the optimizer as bad tracking) are skipped."""
    walk = Path(walk)
    meta, frames = read_walk(walk)
    _META[walk] = meta
    # an app killed mid-write leaves empty jpgs
    kfs = [f for f in frames if f["rgb"] and f["tracking"] == "normal" and (walk / "rgb" / f"{f['i']:06d}.jpg").stat().st_size > 0]
    if poses is not None:
        z = np.load(poses)
        corrected = dict(zip(z["i"].tolist(), z["T"]))
        kfs = [dict(f, T=corrected[f["i"]]) for f in kfs if f["i"] in corrected]
    return kfs


def candidate_pairs(kfs_a, kfs_b, same_walk, radius=None, max_angle=None, per_kf=5, min_gap=8.0,
                    max_pairs=3000, seed=0):
    """Index pairs (ia, ib) worth matching, from the current poses. Same walk: within 4 m and 60 deg, at least
    SAME_WALK_MIN_DT apart, each pair once. Cross walk: 8 m and 90 deg (poses may be off by metres and degrees).
    Per frame of A at most per_kf partners, nearest first and at least min_gap s apart (so they are different
    passes, not neighbouring frames of one). Then at most max_pairs, sampled evenly at random."""
    radius = radius or (4.0 if same_walk else 8.0)
    max_angle = max_angle or (60.0 if same_walk else 90.0)
    pa = np.array([f["T"][:3, 3] for f in kfs_a])
    pb = np.array([f["T"][:3, 3] for f in kfs_b])
    fa = np.array([-f["T"][:3, 2] for f in kfs_a])
    fb = np.array([-f["T"][:3, 2] for f in kfs_b])
    tb = np.array([f["t"] for f in kfs_b])
    cos_max = np.cos(np.radians(max_angle))
    tree = cKDTree(pb)
    pairs = []
    for ia, near in enumerate(tree.query_ball_point(pa, radius)):
        near = np.array(near, int)
        if len(near) == 0:
            continue
        ok = fb[near] @ fa[ia] >= cos_max
        if same_walk:
            ok &= (near > ia) & (np.abs(tb[near] - kfs_a[ia]["t"]) >= SAME_WALK_MIN_DT)
        near = near[ok]
        near = near[np.argsort(np.linalg.norm(pb[near] - pa[ia], axis=1))]
        chosen = []
        for ib in near:
            if all(abs(tb[ib] - tb[c]) >= min_gap for c in chosen):
                chosen.append(ib)
            if len(chosen) == per_kf:
                break
        pairs += [(ia, int(ib)) for ib in chosen]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs = [pairs[k] for k in sorted(rng.choice(len(pairs), max_pairs, replace=False))]
    return pairs


# ---------------------------------------------------------------- features and depth lifting

def sample_depth(depth, conf, xy, meta):
    """Bilinear depth at RGB-pixel positions xy (N,2) from the 256x192 map. NaN where any of the four neighbours is
    unconfident / out of range, or they straddle a depth edge (spread > max(5 cm, 6% of range))."""
    dh, dw = depth.shape
    x, y = xy[:, 0] * dw / meta["rgb_w"], xy[:, 1] * dh / meta["rgb_h"]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    inside = (x0 >= 0) & (y0 >= 0) & (x0 < dw - 1) & (y0 < dh - 1)
    x0, y0 = np.clip(x0, 0, dw - 2), np.clip(y0, 0, dh - 2)
    fx, fy = x - x0, y - y0
    nb = np.stack([depth[y0, x0], depth[y0, x0 + 1], depth[y0 + 1, x0], depth[y0 + 1, x0 + 1]])
    cf = np.stack([conf[y0, x0], conf[y0, x0 + 1], conf[y0 + 1, x0], conf[y0 + 1, x0 + 1]])
    valid = (cf >= 1).all(0) & (nb >= MIN_DEPTH).all(0) & (nb <= MAX_DEPTH).all(0) & inside
    spread = nb.max(0) - nb.min(0)
    valid &= spread < np.maximum(0.05, 0.06 * nb.mean(0))
    d = (nb[0] * (1 - fx) * (1 - fy) + nb[1] * fx * (1 - fy) + nb[2] * (1 - fx) * fy + nb[3] * fx * fy)
    return np.where(valid, d, np.nan)


def lift(xy, d, intr):
    """RGB pixels + planar depth -> points in the OpenCV camera frame (+Z forward)."""
    fx, fy, cx, cy = intr
    return np.stack([(xy[:, 0] - cx) / fx * d, (xy[:, 1] - cy) / fy * d, d], 1)


def extract(walk, kf):
    """SIFT features of a keyframe: dict(xy full-res RGB pixels, desc uint8, d lidar depth at each keypoint (NaN
    invalid), scale = matched-res / full-res). The CLAHE'd grey image is what SIFT sees."""
    walk = Path(walk)
    meta = _meta(walk)
    path = str(walk / "rgb" / f"{kf['i']:06d}.jpg")
    img = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_2 if meta["rgb_w"] >= 1.9 * MATCH_W else cv2.IMREAD_GRAYSCALE)
    if img is None:  # truncated / corrupt jpg: no features, the pair is rejected as few_matches
        return dict(xy=np.zeros((0, 2)), desc=np.zeros((0, 128), np.uint8), d=np.zeros(0), scale=1.0, size=(MATCH_W, 1))
    if img.shape[1] > MATCH_W:
        img = cv2.resize(img, (MATCH_W, round(img.shape[0] * MATCH_W / img.shape[1])), interpolation=cv2.INTER_AREA)
    scale = img.shape[1] / meta["rgb_w"]
    img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)
    kp, desc = cv2.SIFT_create(nfeatures=N_FEATURES).detectAndCompute(img, None)
    if desc is None:
        return dict(xy=np.zeros((0, 2)), desc=np.zeros((0, 128), np.uint8), d=np.zeros(0), scale=scale, size=img.shape[::-1])
    xy = np.array([k.pt for k in kp], np.float64) / scale
    depth, conf = read_depth(walk, meta, kf["i"])
    return dict(xy=xy, desc=np.clip(desc, 0, 255).astype(np.uint8), d=sample_depth(depth, conf, xy, meta),
                scale=scale, size=img.shape[::-1])


def mutual_matches(fa, fb):
    """(ia, ib) index arrays: ratio test A->B and B->A is the same nearest neighbour."""
    if len(fa["desc"]) < 2 or len(fb["desc"]) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    da, db = fa["desc"].astype(np.float32), fb["desc"].astype(np.float32)
    knn = bf.knnMatch(da, db, k=2)
    back = np.array([m.trainIdx for m in bf.match(db, da)])
    ia, ib = [], []
    for m in knn:
        if len(m) == 2 and m[0].distance < RATIO * m[1].distance and back[m[0].trainIdx] == m[0].queryIdx:
            ia.append(m[0].queryIdx)
            ib.append(m[0].trainIdx)
    return np.array(ia, int), np.array(ib, int)


# ---------------------------------------------------------------- pose from matches

def _K(intr, s=1.0):
    fx, fy, cx, cy = intr
    return np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1.0]])


def _to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = np.ravel(tvec)
    return T


def _cv_to_rel(X):
    """cv-frame transform X (B <- A points) -> ARKit-axis pose of B in A: C inv(X) C."""
    C = np.eye(4)
    C[:3, :3] = C_FLIP
    return C @ np.linalg.inv(X) @ C


def _rel_to_cv(T_rel):
    return _cv_to_rel(T_rel)  # the map is its own inverse (C inv(.) C is an involution)


def _reproj(T_rel, P, uv, K):
    X = _rel_to_cv(T_rel)
    pc = P @ X[:3, :3].T + X[:3, 3]
    proj = pc @ K.T
    return proj[:, :2] / proj[:, 2:3] - uv


def _hull_fraction(uv, size):
    if len(uv) < 3:
        return 0.0
    try:
        return ConvexHull(uv).volume / (size[0] * size[1])
    except Exception:  # collinear
        return 0.0


def _rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _information(T_rel, P, uv, K, sigma, med_range):
    """6x6 information over T_rel @ Exp(xi), xi = [t, w] in B's ARKit axes. From the reprojection Jacobian (finite
    differences) at pixel noise sigma, plus a covariance floor (1% of range, 0.2 deg): the 2-D residuals know nothing of
    the lidar depth noise that sits in P."""
    n = len(P)
    J = np.zeros((2 * n, 6))
    eps = 1e-5
    for k in range(6):
        xi = np.zeros(6)
        xi[k] = eps
        D = np.eye(4)
        D[:3, :3] = cv2.Rodrigues(xi[3:])[0]
        D[:3, 3] = xi[:3]
        J[:, k] = ((_reproj(T_rel @ D, P, uv, K) - _reproj(T_rel, P, uv, K)) / eps).ravel()
    cov = np.linalg.pinv(J.T @ J / sigma ** 2 + 1e-9 * np.eye(6))
    cov += np.diag([(0.01 * med_range) ** 2] * 3 + [np.radians(0.2) ** 2] * 3)
    return np.linalg.inv(cov)


def _reject(reason, stats):
    return dict(ok=False, reason=reason, T_rel=None, info=None, stats=stats)


def solve_pair(kf_a, fa, kf_b, fb, meta_a, meta_b):
    """Pose of B in A from precomputed features. See match_pair."""
    t0 = time.time()
    stats = {}
    ia, ib = mutual_matches(fa, fb)
    keep = np.isfinite(fa["d"][ia])
    ia, ib = ia[keep], ib[keep]
    stats["n_matches"] = len(ia)
    if len(ia) < 12:
        return _reject("few_matches", stats)
    intr_a, intr_b = kf_a["intrinsics"], kf_b["intrinsics"]
    P = lift(fa["xy"][ia], fa["d"][ia], intr_a)
    uv = fb["xy"][ib] * fb["scale"]  # B keypoints at matched resolution
    Kb = _K(intr_b, fb["scale"])
    ok, rvec, tvec, inl = cv2.solvePnPRansac(P, uv, Kb, None, iterationsCount=1000, reprojectionError=PNP_THRESH,
                                             confidence=0.9999, flags=cv2.SOLVEPNP_AP3P)
    if not ok or inl is None or len(inl) < 6:
        return _reject("pnp_failed", stats)
    mask = np.zeros(len(P), bool)
    mask[inl.ravel()] = True
    for _ in range(2):  # refine on inliers, re-select
        rvec, tvec = cv2.solvePnPRefineLM(P[mask], uv[mask], Kb, None, rvec, tvec)
        err = np.linalg.norm(_reproj(_cv_to_rel(_to_T(rvec, tvec)), P, uv, Kb), axis=1)
        mask = err < INLIER_THRESH
        if mask.sum() < 6:
            return _reject("few_inliers", stats)
    T_rel = _cv_to_rel(_to_T(rvec, tvec))
    err = err[mask]
    n_in = int(mask.sum())
    stats.update(n_inliers=n_in, reproj_med=float(np.median(err)), reproj_rms=float(np.sqrt((err ** 2).mean())))
    if n_in < MIN_INLIERS:
        return _reject("few_inliers", stats)
    if stats["reproj_med"] >= MAX_MEDIAN_REPROJ:
        return _reject("high_reproj", stats)
    uv_a = fa["xy"][ia[mask]] * fa["scale"]
    stats["hull_a"], stats["hull_b"] = _hull_fraction(uv_a, fa["size"]), _hull_fraction(uv[mask], fb["size"])
    if min(stats["hull_a"], stats["hull_b"]) < MIN_HULL:
        return _reject("clustered", stats)

    # depth consistency: A's points moved into B must sit at B's own lidar depth
    Pin = P[mask]
    X = _rel_to_cv(T_rel)
    z_pred = (Pin @ X[:3, :3].T + X[:3, 3])[:, 2]
    stats["median_range"] = float(np.median(Pin[:, 2]))
    d_b = fb["d"][ib[mask]]
    has = np.isfinite(d_b) & (z_pred > 0)
    stats["n_depth_checks"] = int(has.sum())
    if has.sum() < MIN_DEPTH_CHECKS:
        return _reject("no_depth_check", stats)
    dz = np.abs(d_b[has] - z_pred[has])
    ratio = dz / np.maximum(DEPTH_AGREE[0], DEPTH_AGREE[1] * z_pred[has])
    stats["depth_dz_med"] = float(np.median(dz))
    stats["depth_ratio_med"] = float(np.median(ratio))
    if np.median(ratio) >= 1.0:
        return _reject("depth_disagrees", stats)

    # reverse direction: lift B with its own depth, solve against A's pixels, compare the two poses
    Pb = lift(fb["xy"][ib[mask]][has], d_b[has], intr_b)
    uv_ra = fa["xy"][ia[mask]][has] * fa["scale"]
    Ka = _K(intr_a, fa["scale"])
    rv, tv = cv2.Rodrigues(np.linalg.inv(X)[:3, :3])[0], np.linalg.inv(X)[:3, 3].copy()
    rv, tv = cv2.solvePnPRefineLM(Pb, uv_ra, Ka, None, rv, tv)
    T_rev = np.eye(4)
    T_rev[:3, :3], T_rev[:3, 3] = cv2.Rodrigues(rv)[0], np.ravel(tv)
    T_rev = np.linalg.inv(_cv_to_rel(T_rev))  # T_rev maps B -> A points: _cv_to_rel gives A in B, invert for B in A
    dT = np.linalg.inv(T_rel) @ T_rev
    stats["rev_dt"], stats["rev_dr"] = float(np.linalg.norm(dT[:3, 3])), _rot_deg(dT[:3, :3])
    if stats["rev_dt"] > max(REV_AGREE[0], REV_AGREE[1] * stats["median_range"]) or stats["rev_dr"] > REV_AGREE[2]:
        return _reject("asymmetric", stats)

    stats["seconds"] = time.time() - t0
    info = _information(T_rel, Pin, uv[mask], Kb, max(1.0, stats["reproj_rms"]), stats["median_range"])
    return dict(ok=True, reason=None, T_rel=T_rel, info=info, stats=stats)


def match_pair(walk_a, kf_a, walk_b, kf_b, feats_a=None, feats_b=None):
    """Relative pose of keyframe B in keyframe A from photo matches + lidar depth.
    Returns dict(ok, reason, T_rel, info, stats): T_rel = inv(T_A) @ T_B in ARKit camera axes (None if rejected, with
    `reason` one of few_matches, pnp_failed, few_inliers, high_reproj, clustered, no_depth_check, depth_disagrees,
    asymmetric); info = 6x6 information (see module docstring); stats = counts and residuals.
    feats_*: precomputed extract() results (so a keyframe is only processed once across many pairs)."""
    walk_a, walk_b = Path(walk_a), Path(walk_b)
    fa = feats_a if feats_a is not None else extract(walk_a, kf_a)
    fb = feats_b if feats_b is not None else extract(walk_b, kf_b)
    return solve_pair(kf_a, fa, kf_b, fb, _meta(walk_a), _meta(walk_b))


# ---------------------------------------------------------------- driver

def world_alignments(constraints, kfs_a, kfs_b):
    """Per cross-walk constraint, the 4x4 taking walk B's world frame into walk A's that it implies: camera j sits at
    T_a[i] @ T_rel in A's world and at T_b[j] in B's, so G = T_a[i] @ T_rel @ inv(T_b[j]). Use the same kfs lists
    (same poses) the constraints were found with."""
    ta, tb = {f["i"]: f["T"] for f in kfs_a}, {f["i"]: f["T"] for f in kfs_b}
    return np.array([ta[i] @ T_rel @ np.linalg.inv(tb[j]) for _, i, _, j, T_rel, _, _ in constraints])


def _medoid_alignment(Gs):
    """The alignment closest to all the others (yaw weighted as 10 m per radian-ish: a degree is ~17 cm at 10 m)."""
    yaw = np.arctan2(Gs[:, 0, 2] - Gs[:, 2, 0], Gs[:, 0, 0] + Gs[:, 2, 2])
    d = np.abs(np.angle(np.exp(1j * (yaw[:, None] - yaw[None, :])))) * 10 + np.linalg.norm(Gs[:, None, :3, 3] - Gs[None, :, :3, 3], axis=2)
    return Gs[np.argmin(np.sort(d, axis=1)[:, : max(2, len(Gs) // 2)].sum(1))]


def _run_pairs(pairs, walk_a, walk_b, kfs_a, kfs_b, same, feats, workers, out, rejected):
    """Extract features for the pairs' keyframes (cached in feats), match the pairs, append accepted to out."""
    def get(k):
        side, i = k
        return k, extract(walk_a if side == 0 else walk_b, (kfs_a if side == 0 else kfs_b)[i])

    side_b = 0 if same else 1  # one feature set per keyframe within a walk
    need = sorted(({(0, ia) for ia, _ in pairs} | {(side_b, ib) for _, ib in pairs}) - set(feats))
    with ThreadPoolExecutor(workers) as ex:
        for k, f in ex.map(get, need):
            feats[k] = f
    meta_a, meta_b = _meta(walk_a), _meta(walk_b)

    def run(p):
        ia, ib = p
        return p, solve_pair(kfs_a[ia], feats[(0, ia)], kfs_b[ib], feats[(side_b, ib)], meta_a, meta_b)

    with ThreadPoolExecutor(workers) as ex:
        for (ia, ib), r in ex.map(run, pairs):
            if r["ok"]:
                out.append((walk_a.name, kfs_a[ia]["i"], walk_b.name, kfs_b[ib]["i"], r["T_rel"], r["info"], r["stats"]))
            else:
                rejected[r["reason"]] += 1


def visual_loops(walk_a, walk_b=None, poses_a=None, poses_b=None, init=None, max_pairs=3000, workers=None, log=print,
                 **pair_kw):
    """Visual loop closures within walk_a (walk_b None) or between walk_a and walk_b, using the given poses.npz
    (else the recorded transforms). init: 4x4 taking walk_b's world frame into walk_a's (e.g. merge.py's GPS placement);
    only used to choose which pairs to try, the constraints themselves do not depend on it. Each walk is in its own
    ARKit frame (origin where it started), so without init two walks only line up if they started at the same spot.
    If a first pass finds at least 3 cross-walk constraints, the medoid of the alignments they imply replaces init and
    a second pass (the usual same-walk radius) tries the pairs the better placement exposes.
    Returns (constraints, summary): constraints are
    (walk_a, frame_i, walk_b, frame_j, T_rel, info, stats) with T_rel = inv(T_a[i]) @ T_b[j] as measured (see module
    docstring); summary = dict(pairs, accepted, rejected={reason: n}, seconds, extract_seconds, alignment (4x4 or None))."""
    t0 = time.time()
    walk_a = Path(walk_a)
    same = walk_b is None
    walk_b = walk_a if same else Path(walk_b)
    kfs_a = keyframes(walk_a, poses_a)
    kfs_b = kfs_a if same else keyframes(walk_b, poses_b)
    workers = workers or min(8, __import__("os").cpu_count() or 1)
    feats, out, rejected, tried, alignment = {}, [], Counter(), set(), None
    for rnd in range(1 if same else 2):
        placed_b = kfs_b if init is None else [dict(f, T=init @ f["T"]) for f in kfs_b]
        kw = dict(pair_kw)
        if rnd == 1:
            kw.setdefault("radius", 5.0)
            kw.setdefault("max_angle", 70.0)
        pairs = [p for p in candidate_pairs(kfs_a, placed_b, same, max_pairs=max_pairs, **kw) if p not in tried]
        tried.update(pairs)
        log(f"visual: pass {rnd + 1}: {len(kfs_a)} x {len(kfs_b)} keyframes, {len(pairs)} candidate pairs")
        _run_pairs(pairs, walk_a, walk_b, kfs_a, kfs_b, same, feats, workers, out, rejected)
        log(f"visual: {len(out)} accepted so far after {time.time() - t0:.0f}s")
        if same or len(out) < 3:
            break
        alignment = init = _medoid_alignment(world_alignments(out, kfs_a, kfs_b))
    summary = dict(pairs=len(tried), accepted=len(out), rejected=dict(rejected), seconds=time.time() - t0,
                   keyframes_processed=len(feats), alignment=alignment)
    log(f"visual: {len(out)}/{len(tried)} accepted in {summary['seconds']:.0f}s; rejected {dict(rejected)}")
    return out, summary
