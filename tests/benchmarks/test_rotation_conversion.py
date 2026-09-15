import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.utils.action_conversion import (
    quat_xyzw_to_axis_angle,
    quat_xyzw_to_rot6d,
    rot6d_to_axis_angle,
)


@pytest.mark.parametrize('angle', [0., 1e-4, np.pi - 1e-5, np.pi, np.pi + 1e-5, 3.8, 5.9])
def test_quaternion_branch_conventions_preserve_orientation(angle):
    axis = np.array([1., -2., 3.]) / np.sqrt(14.)
    raw = angle * axis
    quaternion = Rotation.from_rotvec(raw).as_quat()
    original = quaternion.copy()
    preserved = quat_xyzw_to_axis_angle(quaternion, canonical=False)
    shortest = quat_xyzw_to_axis_angle(quaternion)
    np.testing.assert_allclose(preserved, raw, atol=5e-7)
    np.testing.assert_allclose(Rotation.from_rotvec(preserved).as_matrix(), Rotation.from_rotvec(shortest).as_matrix(), atol=5e-7)
    assert np.linalg.norm(shortest) <= np.pi + 1e-6
    np.testing.assert_array_equal(quaternion, original)
    via_rot6d = rot6d_to_axis_angle(quat_xyzw_to_rot6d(quaternion))
    np.testing.assert_allclose(Rotation.from_rotvec(via_rot6d).as_matrix(), Rotation.from_rotvec(raw).as_matrix(), atol=5e-7)


def test_opposite_quaternion_signs_use_explicit_branch_choice():
    quaternion = Rotation.from_rotvec([3.8, 0., 0.]).as_quat()
    np.testing.assert_allclose(quat_xyzw_to_axis_angle(quaternion, canonical=False), [3.8, 0., 0.], atol=5e-7)
    np.testing.assert_allclose(quat_xyzw_to_axis_angle(-quaternion, canonical=False), [3.8 - 2 * np.pi, 0., 0.], atol=5e-7)
    np.testing.assert_allclose(quat_xyzw_to_axis_angle(quaternion), quat_xyzw_to_axis_angle(-quaternion), atol=5e-7)


@pytest.mark.parametrize('w', [-1., 1.])
def test_identity_quaternion_in_both_hemispheres(w):
    np.testing.assert_array_equal(quat_xyzw_to_axis_angle([0., 0., 0., w], canonical=False), np.zeros(3))
