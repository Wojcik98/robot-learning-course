"""Check exact development states and seeds for Diffusion Policy evaluation."""

import json
from pathlib import Path

import pytest

from vla_lab.evaluation.diffusion import evaluate_diffusion_reference
from vla_lab.sim.episode import EpisodeResult

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "policy": "diffusion",
                "task_alias": "bowl_plate",
                "model_sha256": "model-digest",
                "n_obs_steps": 2,
                "horizon": 16,
                "n_action_steps": 8,
            }
        )
    )
    return checkpoint


def test_diffusion_eval_records_each_state_and_noise_seed(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    seen: list[int] = []

    def run_one(state_id: int) -> EpisodeResult:
        seen.append(state_id)
        return EpisodeResult(
            "bowl_plate", state_id, state_id == 1, 12, 0, "success" if state_id == 1 else "timeout"
        )

    output = tmp_path / "result"
    summary = evaluate_diffusion_reference(
        checkpoint=checkpoint,
        configs_root=CONFIGS,
        data_root=tmp_path,
        output_dir=output,
        state_ids=[1, 0],
        seed=12,
        inference_steps=5,
        run_one=run_one,
    )
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert seen == [1, 0]
    assert summary["state_ids"] == [1, 0]
    assert summary["successes"] == 1
    assert summary["inference_steps"] == 5
    assert [row["metrics"]["seed"] for row in rows] == [13, 12]


def test_diffusion_eval_rejects_final_states_and_invalid_sampler(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    options = {
        "checkpoint": checkpoint,
        "configs_root": CONFIGS,
        "data_root": tmp_path,
        "output_dir": tmp_path / "result",
        "run_one": lambda state_id: EpisodeResult("bowl_plate", state_id, False, 1, 0),
    }
    with pytest.raises(ValueError, match="development"):
        evaluate_diffusion_reference(**options, state_ids=[20])
    with pytest.raises(ValueError, match="inference steps"):
        evaluate_diffusion_reference(**options, state_ids=[0], inference_steps=0)
