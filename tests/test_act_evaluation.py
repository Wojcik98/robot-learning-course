"""Check that reference evaluation uses development states only."""

import json
from pathlib import Path

import pytest

from vla_lab.evaluation.act import evaluate_act_reference
from vla_lab.sim.episode import EpisodeResult

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_evaluation_saves_each_development_result(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "policy": "act", "task_alias": "bowl_plate"})
    )
    seen = []

    def run_one(state_id: int) -> EpisodeResult:
        seen.append(state_id)
        return EpisodeResult("bowl_plate", state_id, state_id < 11, 75, 2)

    summary = evaluate_act_reference(
        checkpoint,
        CONFIGS,
        tmp_path / "data",
        tmp_path / "results",
        run_one=run_one,
    )

    assert seen == list(range(20))
    assert summary["successes"] == 11
    assert summary["episodes"] == 20
    assert summary["success_rate"] == 0.55
    rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "episodes.jsonl").read_text().splitlines()
    ]
    assert [row["state_id"] for row in rows] == list(range(20))


def test_evaluation_rejects_final_state_during_development(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "policy": "act", "task_alias": "bowl_plate"})
    )
    with pytest.raises(ValueError, match="development"):
        evaluate_act_reference(
            checkpoint,
            CONFIGS,
            tmp_path / "data",
            tmp_path / "results",
            state_ids=[20],
            run_one=lambda state_id: EpisodeResult("bowl_plate", state_id, False, 1, 0),
        )
