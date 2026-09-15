from pathlib import Path

import numpy as np
import pytest
import yaml

from benchmarks.libero.openwam2libero_interface import (
    LIBERO_ACTION_MODE,
    OpenWAMLiberoPolicy,
    libero_action_to_command,
    libero_obs_to_state8,
)


class _Client:
    def __init__(self, representation):
        self.representation = representation

    def ping(self):
        return {"type": "pong", "representation": self.representation}

    def predict(self, payload):
        self.payload = payload
        return {"action": [0.3, -0.4, 0.5, 0.1, -0.2, 0.3, 1]}


def test_config_uses_libero_contract():
    cfg = yaml.safe_load(Path("benchmarks/libero/policy_config.yml").read_text())
    assert cfg["action_mode"] == LIBERO_ACTION_MODE
    assert cfg["state_dim"] == 8


@pytest.mark.parametrize("gripper,command", [(0, 1), (0.5, 1), (0.8, -1), (1, -1)])
def test_native_command_and_binary_gripper(gripper, command):
    raw = np.array([0.3, -0.4, 0.5, 0.8, -0.1, 0.05, gripper], dtype=np.float32)
    action = libero_action_to_command(raw)
    np.testing.assert_array_equal(action[:6], raw[:6])
    assert action[6] == command
    assert raw[6] == pytest.approx(gripper)


def test_command_clipping():
    np.testing.assert_array_equal(
        libero_action_to_command([1.5, -1.5, 0, 2, -2, 0, 0]),
        [1, -1, 0, 1, -1, 0, 1],
    )


def test_policy_sends_state8_without_canonicalizing_quaternion():
    angle = 3.8
    obs = {
        "robot0_eef_pos": [0.1, 0.2, 0.3],
        "robot0_eef_quat": [np.sin(angle / 2), 0, 0, np.cos(angle / 2)],
        "robot0_gripper_qpos": [0.04, -0.039],
        "agentview_image": np.zeros((32, 32, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((32, 32, 3), dtype=np.uint8),
    }
    expected = [0.1, 0.2, 0.3, 3.8, 0, 0, 0.04, -0.039]
    np.testing.assert_allclose(libero_obs_to_state8(obs), expected)
    client = _Client(LIBERO_ACTION_MODE)
    policy = OpenWAMLiberoPolicy(_client=client)
    result = policy.act(obs, "pick the cup")
    np.testing.assert_allclose(client.payload["state"], expected)
    np.testing.assert_allclose(result, [0.3, -0.4, 0.5, 0.1, -0.2, 0.3, -1])


def test_policy_rejects_non_eef_representation():
    with pytest.raises(RuntimeError, match="representation mismatch"):
        OpenWAMLiberoPolicy(_client=_Client("joint"))
