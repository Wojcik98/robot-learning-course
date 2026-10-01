"""Check paired three-method reports and reject changed final attempts."""

import json
from pathlib import Path

import pytest

from vla_lab.evaluation.comparison import METHODS, evaluate_frozen_method
from vla_lab.evaluation.comparison_report import build_three_method_report
from vla_lab.sim.episode import EpisodeResult

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
COMPARISON = CONFIGS / "experiments" / "comparison.yaml"


def _results(tmp_path: Path) -> dict[str, Path]:
    directories = {}
    for method in METHODS:
        output = tmp_path / method
        threshold = {"act": 25, "smolvla_learner": 30, "diffusion": 35}[method]

        def run_one(state_id: int, threshold: int = threshold) -> EpisodeResult:
            success = state_id < threshold
            return EpisodeResult(
                "bowl_plate",
                state_id,
                success,
                20 if success else 300,
                1,
                "success" if success else "timeout",
            )

        evaluate_frozen_method(
            method=method,
            config_path=COMPARISON,
            configs_root=CONFIGS,
            data_root=tmp_path,
            output_dir=output,
            run_one=run_one,
        )
        directories[method] = output
    return directories


def test_report_keeps_all_paired_states_and_declares_unequal_recipes(tmp_path: Path) -> None:
    directories = _results(tmp_path)
    report = build_three_method_report(
        config_path=COMPARISON,
        configs_root=CONFIGS,
        result_dirs=directories,
        output=tmp_path / "report.json",
    )
    assert report["state_ids"] == list(range(20, 50))
    assert report["comparison_type"] == "practical_recipe"
    assert {method: row["successes"] for method, row in report["methods"].items()} == {
        "act": 5,
        "smolvla_learner": 10,
        "diffusion": 15,
    }
    pair = report["pairs"]["act_vs_diffusion"]
    assert pair["a_only"] == 0 and pair["b_only"] == 10
    assert pair["paired_states"] == 30
    assert pair["valid_for_policy_comparison"]
    assert report["methods"]["act"]["robot_training_steps"] == 200
    assert report["methods"]["diffusion"]["robot_training_steps"] == 2000
    assert json.loads((tmp_path / "report.json").read_text()) == report


def test_report_rejects_a_changed_seed_or_missing_attempt(tmp_path: Path) -> None:
    directories = _results(tmp_path)
    target = directories["diffusion"] / "episodes.jsonl"
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    rows[0]["metrics"]["seed"] += 1
    target.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="frozen attempt"):
        build_three_method_report(
            config_path=COMPARISON,
            configs_root=CONFIGS,
            result_dirs=directories,
            output=tmp_path / "report.json",
        )

    target.write_text("\n".join(json.dumps(row) for row in rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="attempt count"):
        build_three_method_report(
            config_path=COMPARISON,
            configs_root=CONFIGS,
            result_dirs=directories,
            output=tmp_path / "report.json",
        )
