import numpy as np
import pytest

from vla_lab.teaching.adapters import (
    assemble_libero_state,
    camera_hwc_to_chw,
    controller_action,
    denormalize_features,
    normalize_features,
    pack_observation,
)


def test_camera_conversion_and_bgr_correction():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    converted = camera_hwc_to_chw(rgb)
    assert converted.shape == (3, 2, 3)
    np.testing.assert_array_equal(converted[:, -1, -1], [1, 0, 0])
    np.testing.assert_array_equal(camera_hwc_to_chw(rgb[..., ::-1], channel_order="bgr"), converted)


def test_assemble_libero_state_converts_xyzw_quaternion_to_axis_angle():
    raw = {"eef": {"pos": [1, 2, 3], "quat": [0, 0, 0, 1]}, "gripper": {"qpos": [-0.1, 0.1]}}
    state = assemble_libero_state(raw)
    np.testing.assert_allclose(state, [1, 2, 3, 0, 0, 0, -0.1, 0.1], atol=1e-6)


def test_pack_normalize_and_controller_action():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    batch = pack_observation(
        cameras={"wrist": frame, "front": frame},
        state=np.arange(8),
        camera_order=("front", "wrist"),
    )
    assert list(batch) == [
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
    ]
    np.testing.assert_array_equal(normalize_features([[2, 4]], [1, 2], [1, 2]), [[1, 1]])
    physical = denormalize_features([[2, -2, 0, 0, 0, 0, 0]], np.zeros(7), np.ones(7))[0]
    np.testing.assert_array_equal(controller_action(physical), [1, -1, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("frame", [np.zeros((2, 3), np.uint8), np.zeros((2, 3, 4), np.uint8)])
def test_camera_shape_errors_are_clear(frame):
    with pytest.raises(ValueError, match="shape"):
        camera_hwc_to_chw(frame)
