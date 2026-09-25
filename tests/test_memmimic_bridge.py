"""rot6d convention bridge between GMP (rows) and OpenWAM (columns)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "memmimic"))

from robotmq_policy_server import (  # noqa: E402
    gmp10_to_openwam10,
    gmp_rot6d_to_openwam,
    openwam10_to_gmp10,
    openwam_rot6d_to_gmp,
)
from openwam.dataloader.utils.eef import quat_wxyz_to_rot6d  # noqa: E402


def _rot_from_wxyz(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def _quats(n=100):
    rng = np.random.default_rng(0)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.vstack([q, [[0.7071, -0.7071, 0.0, 0.0]]])  # push_cube default orientation


def test_rot6d_conventions_roundtrip():
    q = _quats()
    rows = np.stack([np.concatenate([_rot_from_wxyz(qq)[0], _rot_from_wxyz(qq)[1]]) for qq in q])
    cols = quat_wxyz_to_rot6d(q.astype(np.float32))
    cols_ref = np.stack([np.concatenate([_rot_from_wxyz(qq)[:, 0], _rot_from_wxyz(qq)[:, 1]]) for qq in q])
    assert np.allclose(cols, cols_ref, atol=1e-5)  # OpenWAM reader = first two columns
    assert np.allclose(gmp_rot6d_to_openwam(rows), cols, atol=1e-5)
    assert np.allclose(openwam_rot6d_to_gmp(cols), rows, atol=1e-5)


def test_pose10_roundtrip_keeps_xyz_and_gripper():
    q = _quats(20)
    rows = np.stack([np.concatenate([_rot_from_wxyz(qq)[0], _rot_from_wxyz(qq)[1]]) for qq in q])
    p = np.random.default_rng(1).normal(size=(21, 10))
    p[:, 3:9] = rows
    out = openwam10_to_gmp10(gmp10_to_openwam10(p))
    assert np.allclose(out, p, atol=1e-5)
