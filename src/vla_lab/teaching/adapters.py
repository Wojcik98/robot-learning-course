"""Small array adapters used by the hands-on VLA course."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def camera_hwc_to_chw(
    image: object, *, channel_order: str = "rgb", rotate_180: bool = True
) -> np.ndarray:
    """Convert one HWC uint8 camera frame to float32 CHW in [0, 1].

    ``channel_order`` describes the incoming frame. Use ``bgr`` to reverse its
    channels. LIBERO's pinned processor also rotates camera frames by 180
    degrees. Set ``rotate_180=False`` only for data that is already rotated.
    """
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("camera frame must have shape (height, width, 3)")
    if frame.dtype != np.uint8:
        raise ValueError("camera frame must use uint8 values")
    if channel_order not in {"rgb", "bgr"}:
        raise ValueError("channel_order must be 'rgb' or 'bgr'")
    if channel_order == "bgr":
        frame = frame[..., ::-1]
    if rotate_180:
        frame = frame[::-1, ::-1]
    return np.ascontiguousarray(frame.transpose(2, 0, 1), dtype=np.float32) / 255.0


def assemble_libero_state(robot_state: Mapping[str, object]) -> np.ndarray:
    """Build the pinned eight-value LIBERO state from raw position fields.

    The quaternion order is x, y, z, w. The returned vector is position,
    axis-angle rotation, then both gripper joint positions.
    """
    try:
        eef = robot_state["eef"]
        gripper = robot_state["gripper"]
        position = np.asarray(eef["pos"], dtype=np.float64)
        quaternion = np.asarray(eef["quat"], dtype=np.float64)
        gripper_position = np.asarray(gripper["qpos"], dtype=np.float64)
    except (KeyError, TypeError) as error:
        raise ValueError("robot_state needs eef.pos, eef.quat, and gripper.qpos") from error
    if position.shape != (3,) or quaternion.shape != (4,) or gripper_position.shape != (2,):
        raise ValueError("LIBERO state needs position (3), xyzw quaternion (4), and gripper (2)")
    if not all(np.isfinite(value).all() for value in (position, quaternion, gripper_position)):
        raise ValueError("LIBERO state fields must be finite")
    norm = np.linalg.norm(quaternion)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("eef.quat must have nonzero length")
    x, y, z, w = quaternion / norm
    vector = np.array([x, y, z], dtype=np.float64)
    vector_norm = np.linalg.norm(vector)
    if vector_norm <= np.finfo(np.float64).eps:
        rotation = np.zeros(3, dtype=np.float64)
    else:
        angle = 2.0 * np.arctan2(vector_norm, w)
        rotation = angle * vector / vector_norm
    return np.concatenate((position, rotation, gripper_position)).astype(np.float32)


def pack_observation(
    *,
    cameras: Mapping[str, object],
    state: object,
    camera_order: tuple[str, ...],
    channel_order: str = "rgb",
) -> dict[str, np.ndarray]:
    """Build the model image mapping and an eight-value state vector."""
    if len(camera_order) != 2 or len(camera_order) != len(set(camera_order)):
        raise ValueError("camera_order must name the two pinned cameras exactly once")
    if set(camera_order) != set(cameras):
        raise ValueError("camera_order must name each camera exactly once")
    image_keys = ("observation.images.image", "observation.images.image2")
    images = {
        image_keys[index]: camera_hwc_to_chw(cameras[name], channel_order=channel_order)
        for index, name in enumerate(camera_order)
    }
    vector = np.asarray(state, dtype=np.float32)
    if vector.shape != (8,) or not np.isfinite(vector).all():
        raise ValueError("state must have shape (8,) and finite values")
    return {**images, "observation.state": vector.copy()}


def normalize_features(values: object, mean: object, std: object) -> np.ndarray:
    """Normalize the final feature axis using training-only statistics."""
    array = np.asarray(values, dtype=np.float32)
    center = np.asarray(mean, dtype=np.float32)
    scale = np.asarray(std, dtype=np.float32)
    if array.ndim < 1 or center.ndim != 1 or scale.shape != center.shape:
        raise ValueError("values need a feature axis and mean/std must be matching vectors")
    if array.shape[-1] != center.size or not np.isfinite(array).all():
        raise ValueError("feature width mismatch or nonfinite values")
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("mean and positive std must be finite")
    return (array - center) / scale


def denormalize_features(values: object, mean: object, std: object) -> np.ndarray:
    """Convert normalized model predictions back to physical feature units."""
    array = np.asarray(values, dtype=np.float32)
    center = np.asarray(mean, dtype=np.float32)
    scale = np.asarray(std, dtype=np.float32)
    if array.ndim < 1 or center.ndim != 1 or scale.shape != center.shape:
        raise ValueError("values need a feature axis and mean/std must be matching vectors")
    if array.shape[-1] != center.size or not np.isfinite(array).all():
        raise ValueError("feature width mismatch or nonfinite values")
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("mean and positive std must be finite")
    return array * scale + center


def controller_action(prediction: object, *, lower: float = -1.0, upper: float = 1.0) -> np.ndarray:
    """Clip one seven-value action that is already in controller units."""
    action = np.asarray(prediction, dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("predicted action must have shape (7,) and finite values")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("action bounds must be finite and lower must be below upper")
    return np.clip(action, lower, upper).astype(np.float32, copy=False)
