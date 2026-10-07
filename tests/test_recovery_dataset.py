"""Check alignment and source checks for ACT recovery recordings."""

from pathlib import Path

import numpy as np
import pytest
import torch

from vla_lab.data.recording import RecordedEpisode
from vla_lab.data.recovery_dataset import RecoveryDataset
from vla_lab.experiments.recovery import _timing_provenance
from vla_lab.teaching.data import action_window
from vla_lab.teaching.recording import aligned_transition_rows


def _episode(
    path: Path,
    *,
    episode_id: str,
    task_id: str = "libero_goal:pick_up_the_black_bowl_from_the_plate",
    reset_seed: int = 41,
    length: int = 7,
    offset: float = 0,
    defect: dict | None = None,
) -> Path:
    observations = tuple(
        {"marker": np.asarray([offset + tick], dtype=np.float32)} for tick in range(length + 1)
    )
    actions = np.stack([np.full(7, offset + tick / 10, dtype=np.float32) for tick in range(length)])
    return RecordedEpisode(
        episode_id=episode_id,
        task_id=task_id,
        actions=actions,
        action_ticks=np.arange(length) + (1 if defect else 0),
        observation_ticks=np.arange(length + 1),
        observations=observations,
        defect=defect,
        reset_seed=reset_seed,
    ).save(path)


def _transform(observation):
    """Map fake raw observations to the ACT feature contract."""
    value = float(observation["marker"][0])
    image = torch.full((3, 256, 256), value / 100, dtype=torch.float32)
    return {
        "observation.images.image": image,
        "observation.images.image2": image.clone(),
        "observation.state": torch.full((8,), value, dtype=torch.float32),
    }


def _dataset(
    *paths: Path,
    expected_seeds=(41,),
    transition_rows_fn=None,
    window_fn=None,
) -> RecoveryDataset:
    return RecoveryDataset(
        paths,
        task_id="libero_goal:pick_up_the_black_bowl_from_the_plate",
        expected_reset_seeds=expected_seeds,
        observation_transform=_transform,
        transition_rows_fn=transition_rows_fn,
        window_fn=window_fn,
    )


def test_keeps_every_controller_tick_and_aligns_observation_with_actions(tmp_path: Path) -> None:
    dataset = _dataset(_episode(tmp_path / "episode.npz", episode_id="recovery-1", length=7))

    assert len(dataset) == 7
    first = dataset[0]
    second = dataset[1]
    assert first["observation.state"].tolist() == [0.0] * 8
    assert second["observation.state"].tolist() == [1.0] * 8
    assert first["action"][:4, 0].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert second["action"][:3, 0].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert second["action_is_pad"].tolist() == [False] * 6 + [True] * 14


def test_pads_tail_by_repeating_last_sampled_action(tmp_path: Path) -> None:
    dataset = _dataset(_episode(tmp_path / "episode.npz", episode_id="tail", length=5))
    sample = dataset[4]

    assert sample["action"][:, 0].tolist() == pytest.approx([0.4] * 20)
    assert sample["action_is_pad"].tolist() == [False] + [True] * 19


def test_learner_alignment_and_window_callbacks_drive_recovery_batch(tmp_path: Path) -> None:

    episode = _episode(tmp_path / "learner.npz", episode_id="learner", length=3)
    dataset = _dataset(episode, transition_rows_fn=aligned_transition_rows, window_fn=action_window)
    sample = dataset[2]
    assert sample["observation.state"].tolist() == [2.0] * 8
    assert sample["action"][:, 0].tolist() == pytest.approx([0.2, 0.0, 0.0] + [0.0] * 17)
    assert sample["action_is_pad"].tolist() == [False] + [True] * 19


def test_recovery_timing_uses_control_rate_for_new_recordings():
    report = _timing_provenance(control_hz=20)
    assert report["recorded_control_hz"] == 20
    assert report["training_transition_hz"] == 20
    assert report["source_storage_fps"] == 10
    assert report["saved_transition_stride"] == 1
    assert report["act_action_chunk_duration_s"] == 1.0
    with pytest.raises(ValueError, match="20 Hz"):
        _timing_provenance(control_hz=10)


def test_returns_the_act_sample_shapes_and_dtypes(tmp_path: Path) -> None:
    dataset = _dataset(_episode(tmp_path / "episode.npz", episode_id="shapes", length=3))
    sample = dataset[0]

    assert sample["observation.images.image"].shape == (3, 256, 256)
    assert sample["observation.images.image2"].shape == (3, 256, 256)
    assert sample["observation.state"].shape == (8,)
    assert sample["action"].shape == (20, 7)
    assert sample["action_is_pad"].shape == (20,)
    assert sample["observation.images.image"].dtype == torch.float32
    assert sample["observation.state"].dtype == torch.float32
    assert sample["action"].dtype == torch.float32
    assert sample["action_is_pad"].dtype == torch.bool


def test_action_windows_do_not_cross_episode_boundaries(tmp_path: Path) -> None:
    first = _episode(tmp_path / "first.npz", episode_id="first", length=3, offset=0.1)
    second = _episode(tmp_path / "second.npz", episode_id="second", length=4, offset=0.5)
    dataset = _dataset(first, second)

    last_first = dataset[2]
    first_second = dataset[3]
    assert last_first["action"][:, 0].tolist() == pytest.approx([0.3] * 20)
    assert last_first["action_is_pad"].tolist() == [False] + [True] * 19
    assert first_second["observation.state"].tolist() == [0.5] * 8
    assert first_second["action"][:2, 0].tolist() == pytest.approx([0.5, 0.6])


def test_rejects_wrong_task_seed_and_duplicate_episode_provenance(tmp_path: Path) -> None:
    wrong_task = _episode(
        tmp_path / "wrong-task.npz", episode_id="wrong-task", task_id="another-task"
    )
    with pytest.raises(ValueError, match="task"):
        _dataset(wrong_task)

    wrong_seed = _episode(tmp_path / "wrong-seed.npz", episode_id="wrong-seed", reset_seed=99)
    with pytest.raises(ValueError, match="seed"):
        _dataset(wrong_seed)

    duplicate_a = _episode(tmp_path / "duplicate-a.npz", episode_id="duplicate")
    duplicate_b = _episode(tmp_path / "duplicate-b.npz", episode_id="duplicate")
    with pytest.raises(ValueError, match="episode ID"):
        _dataset(duplicate_a, duplicate_b)


def test_rejects_declared_timing_defects_and_malformed_npz(tmp_path: Path) -> None:
    defective = _episode(
        tmp_path / "defective.npz",
        episode_id="defective",
        defect={
            "label": "synthetic timing defect",
            "kind": "action_tick_offset",
            "offset_ticks": 1,
        },
    )
    with pytest.raises(ValueError, match="timing defect"):
        _dataset(defective)

    valid = _episode(tmp_path / "valid.npz", episode_id="malformed")
    malformed = tmp_path / "malformed.npz"
    with np.load(valid, allow_pickle=False) as saved:
        payload = {key: np.array(saved[key], copy=True) for key in saved.files}
    payload["actions"] = payload["actions"][:, :6]
    np.savez_compressed(malformed, **payload)
    with pytest.raises(ValueError, match="actions"):
        _dataset(malformed)
