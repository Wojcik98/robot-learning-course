"""Keep exact state order and failed attempts in evaluation reports."""

import json
from pathlib import Path

import pytest
import yaml

from vla_lab.contracts import EpisodeOutcome
from vla_lab.evaluation.states import evaluate_states, selected_state_ids
from vla_lab.sim.episode import PolicyActionError


def _configs(root: Path) -> Path:
    configs = root / "configs"
    (configs / "eval").mkdir(parents=True)
    (configs / "eval" / "splits.yaml").write_text(
        yaml.safe_dump(
            {
                "tasks": {
                    "bowl_plate": {"development_state_ids": [0, 1, 2], "final_state_ids": [3, 4]}
                }
            }
        )
    )
    return configs


def test_state_selection_rejects_wrap_duplicate_and_early_final(tmp_path: Path) -> None:
    configs = _configs(tmp_path)
    assert selected_state_ids(configs, "bowl_plate", "development", [2, 0]) == [2, 0]
    with pytest.raises(ValueError, match="unique"):
        selected_state_ids(configs, "bowl_plate", "development", [0, 0])
    with pytest.raises(ValueError, match="outside"):
        selected_state_ids(configs, "bowl_plate", "development", [3])
    with pytest.raises(ValueError, match="explicit"):
        selected_state_ids(configs, "bowl_plate", "final", [3])


def test_failed_attempts_stay_in_order_and_denominator(tmp_path: Path) -> None:
    configs = _configs(tmp_path)

    def run_one(state_id: int) -> EpisodeOutcome:
        if state_id == 0:
            raise PolicyActionError("nonfinite action")
        if state_id == 1:
            raise RuntimeError("renderer failed")
        return EpisodeOutcome(
            "bowl_plate",
            state_id,
            "development",
            "act-test",
            "libero-goal-v1",
            True,
            "success",
            {"steps": 7},
            {"video": "state-002.mp4"},
        )

    output = tmp_path / "eval"
    summary = evaluate_states(
        configs_root=configs,
        task_id="bowl_plate",
        partition="development",
        state_ids=[2, 0, 1],
        policy_revision="act-test",
        output_dir=output,
        run_one=run_one,
    )
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert [row["initial_state_id"] for row in rows] == [2, 0, 1]
    assert [row["termination_reason"] for row in rows] == [
        "success",
        "policy_error",
        "infrastructure_error",
    ]
    assert summary["attempts"] == 3
    assert summary["successes"] == 1
    assert summary["success_rate"] == pytest.approx(1 / 3)
