"""MemMimic action_repr=delta: increments round-trip and the reader matches the deploy integration."""

from pathlib import Path

import numpy as np
import pytest

from openwam.dataloader.memmimic import delta_to_pose10, pose10_to_delta

PUSH_CUBE = Path("/mnt/cpfs/shared/datasets/memmimic/v1/push_cube")


def test_round_trip_and_first_step_relative_to_state():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(32, 10)).astype(np.float32)
    s = rng.normal(size=10).astype(np.float32)
    d = pose10_to_delta(a, s)
    np.testing.assert_allclose(d[0, :3], a[0, :3] - s[:3], atol=1e-6)
    np.testing.assert_allclose(d[5, :3], a[5, :3] - a[4, :3], atol=1e-6)
    np.testing.assert_array_equal(d[:, 3:], a[:, 3:])  # rot6d / gripper stay absolute
    np.testing.assert_allclose(delta_to_pose10(d, s), a, atol=1e-5)


@pytest.mark.skipif(not PUSH_CUBE.exists(), reason="push_cube dataset not mounted")
def test_reader_delta_window_integrates_back_to_recorded_commands():
    from openwam.dataloader.memmimic import MemMimicDataset, pose10_from_xyz_wxyz
    from openwam.dataloader.utils.normalization import apply_normalization

    ds = MemMimicDataset(str(PUSH_CUBE), action_repr="delta", memory_enabled=False)
    assert Path(ds.normalization_stats_path).name == "openwam_normalization_stats_delta.npy"
    s = ds[3]
    g = ds._episode(s["episode_index"])
    off = s["offset"]
    rec = pose10_from_xyz_wxyz(g["action0_tcp_xyz_wxyz"][off : off + 32], g["action0_gripper_width"][off : off + 32])
    state = pose10_from_xyz_wxyz(g["robot0_tcp_xyz_wxyz"][off : off + 1], g["robot0_gripper_width"][off : off + 1])[0]
    norm_delta = s["action"].numpy()[:, :10]
    # undo min-max with the delta stats, then integrate like the deploy server does
    st = ds._action_stats
    lo, hi = np.asarray(st["min"]), np.asarray(st["max"])
    raw = (norm_delta + 1) / 2 * (hi - lo) + lo
    raw[:, 3:9] = norm_delta[:, 3:9]  # rot6d is identity-normalized
    np.testing.assert_allclose(delta_to_pose10(raw, state)[:, :3], rec[:, :3], atol=2e-4)
    assert np.allclose(apply_normalization(pose10_to_delta(rec, state), st, "min-max"), norm_delta, atol=1e-5)
