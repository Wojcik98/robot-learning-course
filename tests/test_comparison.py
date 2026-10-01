"""Check the frozen final-state recipe and complete attempt recording."""

import json
from pathlib import Path

import pytest
import yaml

from vla_lab.evaluation.comparison import (
    _load_method,
    evaluate_frozen_method,
    load_comparison_config,
)
from vla_lab.sim.episode import EpisodeResult

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
COMPARISON = CONFIGS / "experiments" / "comparison.yaml"


def _changed_config(tmp_path: Path, change: str) -> Path:
    payload = yaml.safe_load(COMPARISON.read_text())
    if change == "omit_state":
        payload["state_ids"].pop()
    elif change == "reorder_state":
        payload["state_ids"][0], payload["state_ids"][1] = (
            payload["state_ids"][1],
            payload["state_ids"][0],
        )
    elif change == "data_split":
        payload["train_split_sha256"] = "0" * 64
    elif change == "prompt":
        payload["task_prompt"] = "a changed prompt"
    elif change == "sampler":
        payload["policies"]["diffusion"]["inference_steps"] = 5
    path = tmp_path / f"{change}.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_comparison_config_freezes_full_final_states_and_selected_recipes() -> None:
    config = load_comparison_config(COMPARISON, CONFIGS)
    assert config["state_ids"] == list(range(20, 50))
    assert config["partition"] == "final"
    assert set(config["policies"]) == {"act", "smolvla_learner", "diffusion"}
    assert config["policies"]["diffusion"]["inference_steps"] == 20


@pytest.mark.parametrize(
    "change", ["omit_state", "reorder_state", "data_split", "prompt", "sampler"]
)
def test_comparison_config_rejects_changes_to_frozen_conditions(
    tmp_path: Path, change: str
) -> None:
    with pytest.raises(ValueError):
        load_comparison_config(_changed_config(tmp_path, change), CONFIGS)


def test_comparison_records_every_final_attempt_in_the_frozen_order(tmp_path: Path) -> None:
    called: list[int] = []

    def run_one(state_id: int) -> EpisodeResult:
        called.append(state_id)
        return EpisodeResult(
            "bowl_plate",
            state_id,
            state_id % 2 == 0,
            20,
            1,
            "success" if state_id % 2 == 0 else "timeout",
        )

    output = tmp_path / "result"
    summary = evaluate_frozen_method(
        method="act",
        config_path=COMPARISON,
        configs_root=CONFIGS,
        data_root=tmp_path,
        output_dir=output,
        run_one=run_one,
    )
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert called == list(range(20, 50))
    assert summary["attempts"] == 30 and summary["successes"] == 15
    assert summary["partition"] == "final"
    assert len({row["policy_revision"] for row in rows}) == 1
    assert [row["metrics"]["seed"] for row in rows] == [20260930 + i for i in range(20, 50)]
    assert [row["success"] for row in rows] == [i % 2 == 0 for i in range(20, 50)]
    with pytest.raises(FileExistsError):
        evaluate_frozen_method(
            method="act",
            config_path=COMPARISON,
            configs_root=CONFIGS,
            data_root=tmp_path,
            output_dir=output,
            run_one=run_one,
        )


def test_comparison_rejects_a_changed_model_before_loading_it(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"changed model")
    recipe = {"checkpoint": str(checkpoint), "model_sha256": "0" * 64}
    with pytest.raises(ValueError, match="model hash"):
        _load_method("act", recipe, "bowl_plate", "split", "revision")


def test_comparison_keeps_infrastructure_errors_and_rejects_the_result(tmp_path: Path) -> None:
    def run_one(state_id: int) -> EpisodeResult:
        if state_id == 20:
            raise RuntimeError("simulator stopped")
        return EpisodeResult("bowl_plate", state_id, False, 300, 0, "timeout")

    output = tmp_path / "result"
    summary = evaluate_frozen_method(
        method="diffusion",
        config_path=COMPARISON,
        configs_root=CONFIGS,
        data_root=tmp_path,
        output_dir=output,
        run_one=run_one,
    )
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert summary["attempts"] == 30
    assert summary["infrastructure_errors"] == 1
    assert not summary["valid_for_comparison"]
    assert rows[0]["termination_reason"] == "infrastructure_error"
    assert rows[0]["metrics"]["seed"] == 20260950
    assert rows[0]["error"] == "RuntimeError: simulator stopped"
    assert len(rows) == 30
