"""Per-step action loss weight (MemMimic critical steps)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PUSH_CUBE = Path("/mnt/cpfs/shared/datasets/memmimic/v1/push_cube")


def _loss(step_w):
    from openwam.model.architectures.base import BaseWAMArchitecture

    sched = SimpleNamespace(training_weight=lambda ids: torch.ones(ids.shape[0]))
    pred, target = torch.zeros(2, 4, 3), torch.ones(2, 4, 3)
    target[:, 1] = 3.0  # step 1 has error 9, the others 1
    inputs = {"action_is_pad": torch.zeros(2, 4, 3, dtype=torch.bool)}
    if step_w is not None:
        inputs["action_step_weight"] = step_w
    return BaseWAMArchitecture._compute_action_loss(None, pred, target, torch.zeros(2), sched, inputs, "cpu")


def test_weight_scales_selected_steps():
    base = _loss(None)
    assert torch.isclose(base, torch.tensor((9 + 1 + 1 + 1) / 4))
    w = torch.tensor([[1.0, 5.0, 1.0, 1.0]] * 2)
    assert torch.isclose(_loss(w), torch.tensor((45 + 1 + 1 + 1) / 4))  # emphasised, not renormalized
    assert torch.isclose(_loss(torch.ones(2, 4)), base)


@pytest.mark.skipif(not PUSH_CUBE.exists(), reason="push_cube dataset not mounted")
def test_reader_weights_follow_is_critical():
    from openwam.dataloader.memmimic import MemMimicDataset

    ds = MemMimicDataset(str(PUSH_CUBE), critical_step_weight=5.0, memory_enabled=False, traj_num=1)
    s = ds[1]  # offset 16: covers the first push (critical steps 15..26)
    w = s["action_step_weight"].numpy()
    crit = np.asarray(ds._episode(s["episode_index"])["action0_is_critical"][s["offset"] : s["offset"] + 32]).reshape(-1)
    np.testing.assert_array_equal(w, np.where(crit, 5.0, 1.0))
    assert w.max() == 5.0 and w.shape == (32,)
    assert "action_step_weight" not in MemMimicDataset(str(PUSH_CUBE), memory_enabled=False, traj_num=1)[1]
