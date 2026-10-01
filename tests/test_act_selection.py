"""Check nested ACT demonstrations and their own statistics."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from vla_lab.data.inspection import audit_task_split
from vla_lab.data.selection import prepare_act_selection
from vla_lab.data.splits import split_episodes
from vla_lab.data.stats import TrainingStats


def _configs(root: Path) -> Path:
    configs = root / "configs"
    data = configs / "data"
    (data / "stats").mkdir(parents=True)
    split = split_episodes([1, 2, 3, 4, 5])
    (data / "episodes.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset_revision": "revision",
                "tasks": {
                    "task": {
                        "episode_count": 5,
                        "train_episode_ids": list(split.train_ids),
                        "validation_episode_ids": list(split.validation_ids),
                        "split_sha256": split.sha256,
                    }
                },
            }
        )
    )
    (data / "stats" / "task.json").write_text("{}")
    return configs


def _rows() -> dict[int, dict[str, np.ndarray]]:
    return {
        episode: {
            "observation.state": np.full((2, 8), episode, dtype=np.float32),
            "action": np.full((2, 7), episode / 10, dtype=np.float32),
        }
        for episode in range(1, 6)
    }


def test_subset_uses_only_selected_episodes_and_keeps_validation_fixed(tmp_path: Path) -> None:
    source = _configs(tmp_path)
    first = prepare_act_selection(
        configs_root=source,
        task_alias="task",
        episode_count=2,
        output_dir=tmp_path / "first",
        episode_rows=_rows(),
    )
    shifted = _rows()
    shifted[5]["observation.state"][:] = 999
    second = prepare_act_selection(
        configs_root=source,
        task_alias="task",
        episode_count=2,
        output_dir=tmp_path / "second",
        episode_rows=shifted,
    )

    manifest = yaml.safe_load((first / "data" / "episodes.yaml").read_text())
    task = manifest["tasks"]["task"]
    assert task["train_episode_ids"] == [1, 2]
    assert task["validation_episode_ids"] == [5]
    assert task["source_train_episode_ids"] == [1, 2, 3, 4]
    assert audit_task_split(first, "task")["training_frames"] == 4
    first_stats = json.loads((first / "data" / "stats" / "task.json").read_text())
    second_stats = json.loads((second / "data" / "stats" / "task.json").read_text())
    assert first_stats == second_stats
    state = TrainingStats.from_dict(first_stats["modalities"]["observation.state"])
    assert state.mean.tolist() == [1.5] * 8
    assert first_stats["modalities"]["action"]["count"] == 4


def test_selection_rejects_count_outside_frozen_train_ids(tmp_path: Path) -> None:
    source = _configs(tmp_path)
    with pytest.raises(ValueError, match="episode_count"):
        prepare_act_selection(
            configs_root=source,
            task_alias="task",
            episode_count=5,
            output_dir=tmp_path / "bad",
            episode_rows=_rows(),
        )
