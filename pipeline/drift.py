"""How much each kind of pose-graph edge knows. Information matrices are 6x6 in Open3D's order: rotation vector (about x, y = up,
z; radians) then translation (x, y = up, z; metres); only the diagonal is used here.

Odometry edges (consecutive fragments of one walk) say "no correction between these two". ARKit is not that sure: its error
grows with the path walked, so the edge's sigma does too: sigma = max(floor, rate * path between the fragments), separately for
horizontal and vertical translation and for yaw (gravity keeps tilt accurate). The rates are per link, i.e. what one hop may
have drifted; a chain of such links can still bend by the 0.3-0.7 % per metre that visual loops measured on real walks
(calibration: pipeline/test_pipeline.py test_drift, scratch calib_real.py), because independent per-link errors add up as
a random walk and the loops correct the sum.

Loop edges (ICP, photos) carry what their registration knows, but capped: the information Open3D computes from point clouds
counts every point pair as independent evidence (it reaches 1e5 per axis, a tenth of a millimetre), and the photo constraints
add up neighbouring keyframes that share their errors. Neither can know a relative pose better than the fragment's own
reconstruction (5 cm voxels, drift inside the fragment), so both are capped at CAP_SIGMA_T / CAP_SIGMA_R.
"""
import numpy as np

ODOM_RATE_H = 0.02  # m per m walked: horizontal drift one hop may have accumulated (sigma)
ODOM_RATE_V = 0.01  # same, vertical
ODOM_RATE_YAW = 0.1  # degrees per m walked
ODOM_FLOOR_T = 0.01  # m
ODOM_FLOOR_YAW = 0.1  # degrees
TILT_SIGMA = 0.2  # degrees: ARKit's gravity is accurate
CAP_SIGMA_T = 0.03  # m: no loop edge knows a relative translation better than this
CAP_SIGMA_R = 0.3  # degrees, about the camera: ... or a relative yaw better than this


def odometry_info(path):
    """6x6 information of a consecutive-fragment edge across `path` metres of walking."""
    st_h = max(ODOM_FLOOR_T, ODOM_RATE_H * path)
    st_v = max(ODOM_FLOOR_T, ODOM_RATE_V * path)
    sy = np.radians(max(ODOM_FLOOR_YAW, ODOM_RATE_YAW * path))
    sr = np.radians(TILT_SIGMA)
    return np.diag(1 / np.array([sr, sy, sr, st_h, st_v, st_h]) ** 2)


def cap_loop_info(info):
    """The loop information `info` (6x6, about the node origin) scaled down uniformly so that no translation axis is more
    certain than CAP_SIGMA_T. A uniform scale keeps its coupling of rotation and translation (a rotation about the origin is
    a translation at the overlap) intact. Returns (info, scale)."""
    top = float(np.max(np.diag(info)[3:]))
    scale = min(1.0, 1 / CAP_SIGMA_T ** 2 / top) if top > 0 else 1.0
    return info * scale, scale


def cap_visual_info(info):
    """A photo edge's diagonal information with each axis capped (it is built about the overlap, no coupling to keep)."""
    out = info.copy()
    out[np.arange(3), np.arange(3)] = np.minimum(np.diag(info)[:3], 1 / np.radians(CAP_SIGMA_R) ** 2)
    out[np.arange(3, 6), np.arange(3, 6)] = np.minimum(np.diag(info)[3:], 1 / CAP_SIGMA_T ** 2)
    return out


def sigmas(info):
    """(rotation sigmas in degrees, translation sigmas in metres) of the diagonal of an information matrix: for reports."""
    d = np.diag(info)
    with np.errstate(divide="ignore"):
        s = 1 / np.sqrt(d)
    return np.degrees(s[:3]), s[3:]


def summary(infos):
    """Median and range of the implied sigmas over a list of 6x6 informations."""
    if not infos:
        return None
    r, t = zip(*[sigmas(i) for i in infos])
    r, t = np.array(r), np.array(t)
    return {"n": len(infos),
            "sigma_t_m_xyz_median": np.median(t, axis=0).round(4).tolist(), "sigma_t_m_min": float(t.min().round(5)), "sigma_t_m_max": float(t.max().round(4)),
            "sigma_rot_deg_xyz_median": np.median(r, axis=0).round(3).tolist(), "sigma_rot_deg_min": float(r.min().round(5))}
