"""Optional orientation safety box for the peg-insertion env (numpy only).

The real HIL-SERL envs clip the commanded pose to a box in position AND in
roll/pitch/yaw (`franka_env.FrankaEnv.clip_safety_box`, ABS_POSE_LIMIT_*). The
sim env clips only the position, so a policy can twist the tool without limit.
`clip_to_rpy_box` is the orientation half of that box: the deviation of the
desired orientation from the aligned home orientation, as extrinsic x-y-z
angles about the WORLD axes (roll, pitch, and yaw about the bore axis), is
clipped to +/- the given half-widths. Quaternions are scipy-ordered (x, y, z, w).
"""

import numpy as np


def quat_to_mat(q):
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R):
    """Rotation matrix -> unit quaternion (x, y, z, w) with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s,
             (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s,
             (R[0, 2] - R[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s,
             (R[1, 0] - R[0, 1]) / s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0.0 else q


def euler_xyz(R):
    """Extrinsic x-y-z angles (rad) of R = Rz(c) Ry(b) Rx(a), as scipy's
    Rotation.as_euler('xyz')."""
    b = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    a = np.arctan2(R[2, 1], R[2, 2])
    c = np.arctan2(R[1, 0], R[0, 0])
    return np.array([a, b, c])


def mat_from_euler_xyz(e):
    a, b, c = e
    ca, sa, cb, sb, cc, sc = (np.cos(a), np.sin(a), np.cos(b), np.sin(b),
                              np.cos(c), np.sin(c))
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rz @ ry @ rx


def rpy_deviation(quat, home_quat):
    """World-frame roll/pitch/yaw (rad) of `quat` relative to `home_quat`."""
    return euler_xyz(quat_to_mat(quat) @ quat_to_mat(home_quat).T)


def clip_to_rpy_box(quat, home_quat, half_widths):
    """Clip the orientation `quat` to the box of +/- `half_widths` (roll,
    pitch, yaw; rad) around `home_quat`. Returns (quat, clipped); inside the
    box the input is returned unchanged."""
    box = np.asarray(half_widths, dtype=np.float64)
    R_home = quat_to_mat(home_quat)
    e = euler_xyz(quat_to_mat(quat) @ R_home.T)
    c = np.clip(e, -box, box)
    if np.array_equal(c, e):
        return quat, False
    return mat_to_quat(mat_from_euler_xyz(c) @ R_home), True


def parse_rpy_box(value):
    """'roll,pitch,yaw' (rad half-widths) or a 3-sequence -> np.ndarray, or
    None for an empty / missing value (no box)."""
    if value is None:
        return None
    if isinstance(value, str):
        value = [v for v in value.replace(";", ",").split(",") if v.strip()]
        if not value:
            return None
    box = np.asarray([float(v) for v in value], dtype=np.float64)
    if box.shape != (3,) or np.any(box <= 0.0) or np.any(box >= np.pi / 2):
        raise ValueError(f"rpy box needs three half-widths in (0, pi/2) rad "
                         f"(roll, pitch, yaw), got {value!r}")
    return box
