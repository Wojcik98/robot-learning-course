"""Keep recorded recovery states separate from development assessment."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from vla_lab.data.recording import RecordedEpisode
from vla_lab.experiments.recovery import load_recovery_manifest


def _manifest(tmp_path: Path, *, state_id: int = 0, heldout: list[int] | None = None) -> Path:
    recording = tmp_path / "recording.npz"
    RecordedEpisode(
        episode_id="recovery-0",
        task_id="bowl_plate",
        actions=np.zeros((2, 7), dtype=np.float32),
        action_ticks=np.array([0, 1]),
        observation_ticks=np.array([0, 1, 2]),
        observations=tuple({"state": np.zeros(8)} for _ in range(3)),
        reset_seed=20260930 + state_id,
    ).save(recording)
    data = {
        "schema_version": 1,
        "task_alias": "bowl_plate",
        "source_type": "keyboard",
        "heldout_development_state_ids": heldout or list(range(5, 20)),
        "recordings": [
            {
                "path": str(recording),
                "sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
                "state_id": state_id,
                "reset_seed": 20260930 + state_id,
                "accepted": True,
            }
        ],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data))
    return manifest


def test_recovery_manifest_checks_ids_seeds_and_file_hash(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    result = load_recovery_manifest(manifest, "configs")
    assert result["source_type"] == "keyboard"
    assert result["recordings"][0]["state_id"] == 0

    recording = tmp_path / "recording.npz"
    recording.write_bytes(recording.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_recovery_manifest(manifest, "configs")


def test_recovery_manifest_rejects_final_and_assessment_state_overlap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="development"):
        load_recovery_manifest(_manifest(tmp_path, state_id=20), "configs")
    with pytest.raises(ValueError, match="overlap"):
        load_recovery_manifest(_manifest(tmp_path, heldout=[0, 5, 6]), "configs")
