"""Check split identity and reject malformed decoded demonstrations."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from vla_lab.data.inspection import audit_example, audit_task_split
from vla_lab.data.splits import split_episodes


def test_task_audit_checks_train_only_statistics(tmp_path: Path) -> None:
    configs = tmp_path / "configs" / "data"
    (configs / "stats").mkdir(parents=True)
    split = split_episodes([1, 2, 3, 4, 5])
    (configs / "episodes.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset_revision": "revision",
                "tasks": {
                    "task": {
                        "train_episode_ids": list(split.train_ids),
                        "validation_episode_ids": list(split.validation_ids),
                        "split_sha256": split.sha256,
                    }
                },
            }
        )
    )
    (configs / "stats" / "task.json").write_text(
        '{"dataset_revision":"revision","split_sha256":"'
        + split.sha256
        + '","modalities":{"action":{"count":8},"observation.state":{"count":8}}}'
    )
    result = audit_task_split(tmp_path / "configs", "task")
    assert result["training_episodes"] == 4
    assert result["validation_episodes"] == 1
    assert result["training_frames"] == 8


def test_example_audit_rejects_invalid_action_or_camera() -> None:
    sample = {
        "observation.images.image": np.zeros((3, 256, 256), dtype=np.float32),
        "observation.images.image2": np.zeros((3, 256, 256), dtype=np.float32),
        "observation.state": np.zeros(8, dtype=np.float32),
        "action": np.zeros((20, 7), dtype=np.float32),
        "action_is_pad": np.zeros(20, dtype=bool),
        "episode_index": 1,
        "frame_index": 2,
        "timestamp": 0.2,
    }
    assert audit_example(sample)["action_shape"] == [20, 7]
    sample["action"][0, 0] = np.nan
    with pytest.raises(ValueError, match="action"):
        audit_example(sample)
    sample["action"][0, 0] = 0
    sample["action"][0, 0] = 1.2
    with pytest.raises(ValueError, match="action"):
        audit_example(sample)
    sample["action"][0, 0] = 0
    sample["observation.images.image2"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="image2"):
        audit_example(sample)
    del sample["observation.images.image2"]
    with pytest.raises(ValueError, match="missing observation.images.image2"):
        audit_example(sample)
