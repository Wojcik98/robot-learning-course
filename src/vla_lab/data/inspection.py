"""Check the fixed data split and one decoded training example."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vla_lab.data.splits import partition_sha256, split_episodes


def audit_task_split(configs_root: str | Path, task_id: str) -> dict[str, Any]:
    """Check that the saved statistics use the fixed training episodes."""
    data_root = Path(configs_root) / "data"
    manifest = yaml.safe_load((data_root / "episodes.yaml").read_text())
    task = manifest["tasks"][task_id]
    train_ids = task["train_episode_ids"]
    validation_ids = task["validation_episode_ids"]
    all_ids = train_ids + validation_ids
    if not train_ids or not validation_ids:
        raise ValueError("training and validation episodes are required")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("training and validation episode IDs overlap")
    if task.get("selection_kind") == "nested_train_subset":
        source_train = task["source_train_episode_ids"]
        source_validation = task["source_validation_episode_ids"]
        source = split_episodes(source_train + source_validation)
        if (
            source.sha256 != task["source_split_sha256"]
            or tuple(source_train) != source.train_ids
            or tuple(source_validation) != source.validation_ids
            or train_ids != source_train[: len(train_ids)]
            or validation_ids != source_validation
        ):
            raise ValueError("nested training subset differs from its frozen source split")
        split_hash = partition_sha256(train_ids, validation_ids)
    else:
        expected = split_episodes(all_ids)
        split_hash = expected.sha256
        if (
            tuple(train_ids) != expected.train_ids
            or tuple(validation_ids) != expected.validation_ids
        ):
            raise ValueError("episode split does not match the frozen 80/20 rule")
    if task["split_sha256"] != split_hash:
        raise ValueError("episode split hash does not match the episode IDs")

    stats = json.loads((data_root / "stats" / f"{task_id}.json").read_text())
    revision = manifest["dataset_revision"]
    if stats["dataset_revision"] != revision or stats["split_sha256"] != split_hash:
        raise ValueError("statistics do not match the training split or dataset revision")
    counts = [int(stats["modalities"][key]["count"]) for key in ("action", "observation.state")]
    if counts[0] <= 0 or counts[0] != counts[1]:
        raise ValueError("training state and action counts must match")
    return {
        "task": task_id,
        "dataset_revision": revision,
        "split_sha256": split_hash,
        "training_episodes": len(train_ids),
        "validation_episodes": len(validation_ids),
        "training_frames": counts[0],
    }


def _array(sample: Mapping[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in sample:
        raise ValueError(f"missing {key}")
    value = sample[key]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape != shape or not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"invalid {key} shape or type: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"nonfinite {key}")
    return array


def audit_example(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Check the two-camera, state, and action shapes of a decoded sample."""
    for key in ("observation.images.image", "observation.images.image2"):
        image = _array(sample, key, (3, 256, 256))
        if float(image.min()) < 0 or float(image.max()) > 1:
            raise ValueError(f"{key} values must be in [0, 1]")
    state = _array(sample, "observation.state", (8,))
    actions = _array(sample, "action", (20, 7))
    if float(actions.min()) < -1 or float(actions.max()) > 1:
        raise ValueError("action values must be in [-1, 1]")
    padding = _array(sample, "action_is_pad", (20,))
    if not np.isin(padding, (0, 1)).all():
        raise ValueError("action_is_pad must be a Boolean mask")
    if not np.isfinite(float(sample["timestamp"])):
        raise ValueError("timestamp must be finite")
    return {
        "episode_index": int(sample["episode_index"]),
        "frame_index": int(sample["frame_index"]),
        "timestamp": float(sample["timestamp"]),
        "image_shape": [3, 256, 256],
        "state_shape": list(state.shape),
        "action_shape": list(actions.shape),
        "valid_actions": int(np.count_nonzero(1 - padding)),
        "first_action": actions[0].tolist(),
    }
