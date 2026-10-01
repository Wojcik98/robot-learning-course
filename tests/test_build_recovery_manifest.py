"""Build a checked manifest from accepted keyboard recordings."""

import json
from pathlib import Path

import numpy as np
import pytest

from vla_lab.data.recording import RecordedEpisode
from vla_lab.experiments.build_recovery_manifest import build_keyboard_manifest
from vla_lab.experiments.recovery import load_recovery_manifest


def _recording(path: Path, state_id: int, *, task: str = "bowl_plate") -> Path:
    return RecordedEpisode(
        episode_id=f"human-state-{state_id}",
        task_id=task,
        actions=np.zeros((2, 7), dtype=np.float32),
        action_ticks=np.array([0, 1]),
        observation_ticks=np.array([0, 1, 2]),
        observations=tuple({"robot_state": np.zeros(8)} for _ in range(3)),
        reset_seed=20260930 + state_id,
    ).save(path)


def test_builds_five_recording_manifest_and_keeps_used_states_out_of_assessment(
    tmp_path: Path,
) -> None:
    paths = [_recording(tmp_path / f"state-{state_id}.npz", state_id) for state_id in range(5)]
    output = tmp_path / "manifest.json"

    result = build_keyboard_manifest(
        paths, task_alias="bowl_plate", configs_root="configs", output=output
    )

    assert result == json.loads(output.read_text())
    assert result["source_type"] == "keyboard"
    assert len(result["recordings"]) == 5
    assert result["attempted_state_ids"] == list(range(5))
    assert result["heldout_development_state_ids"] == list(range(5, 20))
    assert all(item["accepted"] for item in result["recordings"])
    assert load_recovery_manifest(output, "configs") == result


def test_rejects_final_state_and_duplicate_or_wrong_task(tmp_path: Path) -> None:
    paths = [_recording(tmp_path / f"state-{state_id}.npz", state_id) for state_id in range(4)]
    final = _recording(tmp_path / "final.npz", 20)
    with pytest.raises(ValueError, match="development"):
        build_keyboard_manifest(
            paths + [final],
            task_alias="bowl_plate",
            configs_root="configs",
            output=tmp_path / "final.json",
        )

    duplicate = _recording(tmp_path / "duplicate.npz", 0)
    with pytest.raises(ValueError, match="repeated"):
        build_keyboard_manifest(
            paths + [duplicate],
            task_alias="bowl_plate",
            configs_root="configs",
            output=tmp_path / "duplicate.json",
        )

    wrong_task = _recording(tmp_path / "wrong-task.npz", 4, task="stove_on")
    with pytest.raises(ValueError, match="task"):
        build_keyboard_manifest(
            paths + [wrong_task],
            task_alias="bowl_plate",
            configs_root="configs",
            output=tmp_path / "wrong.json",
        )
