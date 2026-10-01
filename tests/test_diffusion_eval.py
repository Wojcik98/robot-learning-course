"""Check exact development states and seeds for Diffusion Policy evaluation."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from vla_lab.evaluation.diffusion import evaluate_diffusion_reference
from vla_lab.sim.episode import EpisodeResult

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    split = yaml.safe_load((CONFIGS / "data" / "episodes.yaml").read_text())
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "policy": "diffusion",
                "task_alias": "bowl_plate",
                "model_sha256": "model-digest",
                "dataset_revision": split["dataset_revision"],
                "split_sha256": split["tasks"]["bowl_plate"]["split_sha256"],
                "training_stats_file_sha256": hashlib.sha256(
                    (CONFIGS / "data" / "stats" / "bowl_plate.json").read_bytes()
                ).hexdigest(),
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


def test_diffusion_eval_rejects_checkpoint_for_another_data_split(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["split_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="data split"):
        evaluate_diffusion_reference(
            checkpoint=checkpoint,
            configs_root=CONFIGS,
            data_root=tmp_path,
            output_dir=tmp_path / "result",
            state_ids=[0],
            run_one=lambda state_id: EpisodeResult("bowl_plate", state_id, False, 1, 0),
        )
