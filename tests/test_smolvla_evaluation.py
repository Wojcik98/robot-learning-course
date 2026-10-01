"""Check the fixed state partition for the published SmolVLA reference."""

import json
from pathlib import Path

import pytest

from vla_lab.evaluation.smolvla import evaluate_published_reference
from vla_lab.sim.episode import EpisodeResult

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_published_evaluation_records_selected_development_states(tmp_path: Path) -> None:
    output = tmp_path / "result"
    seen = []

    def run_one(state_id: int) -> EpisodeResult:
        seen.append(state_id)
        return EpisodeResult("bowl_plate", state_id, state_id == 0, 10, 2)

    summary = evaluate_published_reference(
        tmp_path / "model",
        tmp_path / "vlm",
        "bowl_plate",
        CONFIGS,
        tmp_path,
        output,
        state_ids=[0, 1],
        run_one=run_one,
    )

    assert seen == [0, 1]
    assert summary["successes"] == 1
    assert summary["success_rate"] == 0.5
    assert [
        json.loads(row)["state_id"] for row in (output / "episodes.jsonl").read_text().splitlines()
    ] == [0, 1]


def test_published_evaluation_rejects_final_state(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="development"):
        evaluate_published_reference(
            tmp_path / "model",
            tmp_path / "vlm",
            "bowl_plate",
            CONFIGS,
            tmp_path,
            tmp_path / "result",
            state_ids=[20],
            run_one=lambda state_id: EpisodeResult("bowl_plate", state_id, False, 1, 0),
        )


def test_prompt_change_keeps_the_evaluation_goal(tmp_path: Path) -> None:
    output = tmp_path / "result"
    summary = evaluate_published_reference(
        tmp_path / "model",
        tmp_path / "vlm",
        "bowl_plate",
        CONFIGS,
        tmp_path,
        output,
        state_ids=[0],
        model_prompt="place the bowl on the plate",
        run_one=lambda state_id: EpisodeResult("bowl_plate", state_id, True, 10, 0),
    )
    assert summary["evaluation_task"] == "put the bowl on the plate"
    assert summary["model_prompt"] == "place the bowl on the plate"
    row = json.loads((output / "episodes.jsonl").read_text())
    assert row["evaluation_task"] == summary["evaluation_task"]
    assert row["model_prompt"] == summary["model_prompt"]
