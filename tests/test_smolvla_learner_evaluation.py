"""Check that learner evaluation uses only declared development states."""

import json
from pathlib import Path

import pytest

from vla_lab.evaluation.smolvla_learner import evaluate_learner_reference
from vla_lab.sim.episode import EpisodeResult

ROOT = Path(__file__).resolve().parents[1]


def _checkpoint(root: Path) -> Path:
    source = root / "checkpoint"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "policy": "smolvla_learner",
                "stage": "c1",
                "step": 100,
                "task_alias": "bowl_plate",
                "model_sha256": "a" * 64,
                "parent_model_sha256": None,
            }
        )
    )
    return source


def test_learner_evaluation_records_explicit_development_states(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    result = evaluate_learner_reference(
        checkpoint=source,
        vlm_dir=tmp_path / "vlm",
        configs_root=ROOT / "configs",
        data_root=tmp_path,
        output_dir=tmp_path / "results",
        state_ids=[0, 1],
        run_one=lambda state_id: EpisodeResult(
            "bowl_plate", state_id, state_id == 0, 30, 0, "success" if state_id == 0 else "timeout"
        ),
    )
    assert result["stage"] == "c1"
    assert result["successes"] == 1
    assert result["state_ids"] == [0, 1]
    assert result["model_sha256"] == "a" * 64


def test_learner_evaluation_rejects_final_state(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="development"):
        evaluate_learner_reference(
            checkpoint=source,
            vlm_dir=tmp_path / "vlm",
            configs_root=ROOT / "configs",
            data_root=tmp_path,
            output_dir=tmp_path / "results",
            state_ids=[20],
            run_one=lambda state_id: EpisodeResult("bowl_plate", state_id, False, 1, 0),
        )
