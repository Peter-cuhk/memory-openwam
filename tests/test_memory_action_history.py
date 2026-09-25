"""Memory action history (memory.action_history): slot alignment train == deploy, zero-init no-op."""

from pathlib import Path

import numpy as np
import pytest
import torch

from openwam.dataloader.memmimic import memory_action_history
from openwam.model.architectures.utils.memory_tokens import MemoryConfig, MemoryTokens

PUSH_CUBE = Path("/mnt/cpfs/shared/datasets/memmimic/v1/push_cube")


def test_slot_alignment_and_anchor_zero():
    acts = np.arange(64 * 2, dtype=np.float32).reshape(64, 2)  # step index encoded in the values
    out = memory_action_history(acts, [0, 1, 2, 4], 16)
    assert out.shape == (4, 16, 2)
    assert np.all(out[0] == 0)  # anchor: no past actions
    np.testing.assert_array_equal(out[1], acts[0:16])  # slot 1 = steps [0, 16)
    np.testing.assert_array_equal(out[2], acts[16:32])
    np.testing.assert_array_equal(out[3], acts[48:64])  # truncated memory keeps absolute alignment
    with pytest.raises(ValueError):
        memory_action_history(acts, [0, 5], 16)  # slot 5 needs steps up to 80


def test_deploy_executed_chunks_match_reader():
    """Server: one executed 16-step chunk per past request, concatenated == the reader's recorded actions."""
    rng = np.random.default_rng(0)
    recorded = rng.normal(size=(16 * 7, 10)).astype(np.float32)
    executed = [recorded[16 * k : 16 * (k + 1)] for k in range(7)]  # what the server appends per request
    times = [0, 3, 4, 5, 6, 7]
    np.testing.assert_array_equal(
        memory_action_history(np.concatenate(executed), times, 16), memory_action_history(recorded, times, 16)
    )


def test_zero_init_is_noop_and_trainable():
    mem = MemoryTokens(MemoryConfig(enabled=True, gist_tokens=2, action_history=True), dim=32, num_layers=2)
    acts = torch.randn(3, 5, 16, 10)
    out = mem.action_embedding(acts)
    assert out.shape == (3, 5, 32)
    assert torch.count_nonzero(out) == 0
    out.sum().backward()
    assert mem.action_history_mlp[-1].weight.grad is not None
    assert mem.action_history_mlp[-1].weight.grad.abs().sum() > 0


@pytest.mark.skipif(not PUSH_CUBE.exists(), reason="push_cube dataset not mounted")
def test_reader_payload_uses_recorded_commands():
    from openwam.dataloader.memmimic import MemMimicDataset, pose10_from_xyz_wxyz
    from openwam.dataloader.utils.normalization import apply_normalization

    ds = MemMimicDataset(str(PUSH_CUBE), memory_action_history=True, traj_num=2)
    e, offset = ds._index[5]
    c = offset // 16
    g = ds._episode(e)
    payload = ds._memory_payload(e, g, c)
    acts = payload["memory_actions"].numpy()
    assert acts.shape == (len(payload["memory_times"]), 16, 10)
    rec = pose10_from_xyz_wxyz(g["action0_tcp_xyz_wxyz"][:], g["action0_gripper_width"][:])
    rec = apply_normalization(rec, ds._action_stats, ds._normalize_mode)
    np.testing.assert_allclose(acts, memory_action_history(rec[: 16 * c], payload["memory_times"], 16), atol=1e-6)
