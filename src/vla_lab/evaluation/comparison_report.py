"""Build a paired report from the three frozen final-state runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from vla_lab.evaluation.comparison import METHODS, load_comparison_config
from vla_lab.evaluation.paired import paired_report, wilson_interval


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_three_method_report(
    *,
    config_path: str | Path,
    configs_root: str | Path,
    result_dirs: dict[str, str | Path],
    output: str | Path,
) -> dict[str, Any]:
    """Reject incomplete results and save counts, intervals, and paired differences."""
    config_file = Path(config_path)
    config = load_comparison_config(config_file, configs_root)
    if set(result_dirs) != set(METHODS):
        raise ValueError("Report needs one result directory for each frozen method")
    config_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()
    all_rows: dict[str, list[dict[str, Any]]] = {}
    methods: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        root = Path(result_dirs[method])
        summary = json.loads((root / "summary.json").read_text())
        rows = _rows(root / "episodes.jsonl")
        recipe = config["policies"][method]
        expected_ids = config["state_ids"]
        if (
            summary["method"] != method
            or summary["comparison_config_sha256"] != config_hash
            or summary["model_sha256"] != recipe["model_sha256"]
            or summary["partition"] != "final"
            or summary["state_ids"] != expected_ids
            or summary["attempts"] != len(expected_ids)
            or summary["policy_errors"] != 0
            or summary["infrastructure_errors"] != 0
            or not summary["valid_for_comparison"]
        ):
            raise ValueError(f"Final {method} summary differs from the frozen recipe or has errors")
        if len(rows) != len(expected_ids):
            raise ValueError(f"Final {method} attempt count differs from the frozen state list")
        for state_id, row in zip(expected_ids, rows, strict=True):
            if (
                row["initial_state_id"] != state_id
                or row["task_id"] != config["task_alias"]
                or row["partition"] != "final"
                or row["policy_revision"] != recipe["model_sha256"]
                or row["success_predicate_version"] != "libero-goal-v1"
                or row["metrics"]["seed"] != config["base_seed"] + state_id
                or not 1 <= row["metrics"]["steps"] <= config["max_steps"]
                or bool(row["success"]) != (row["termination_reason"] == "success")
                or row["error"] is not None
                or row["termination_reason"] in ("policy_error", "infrastructure_error")
            ):
                raise ValueError(f"Final {method} state {state_id} differs from the frozen attempt")
        successes = sum(bool(row["success"]) for row in rows)
        if successes != summary["successes"]:
            raise ValueError(f"Final {method} success count differs from its attempt rows")
        success_steps = [row["metrics"]["steps"] for row in rows if row["success"]]
        total_steps = sum(row["metrics"]["steps"] for row in rows)
        clipped_values = sum(row["metrics"]["clipped_values"] for row in rows)
        methods[method] = {
            "model_sha256": recipe["model_sha256"],
            "successes": successes,
            "attempts": len(rows),
            "success_rate": successes / len(rows),
            "wilson_95": list(wilson_interval(successes, len(rows))),
            "median_success_steps": statistics.median(success_steps) if success_steps else None,
            "total_steps": total_steps,
            "clipped_values": clipped_values,
            "clipped_action_value_fraction": clipped_values / (7 * total_steps),
            "median_episode_elapsed_s": statistics.median(
                row["metrics"]["elapsed_s"] for row in rows
            ),
            "robot_training_steps": recipe["robot_training_steps"],
            "train_episode_count": config["train_episode_count"],
            "pretraining": recipe["pretraining"],
            "result_dir": str(root),
        }
        all_rows[method] = rows

    pairs = {}
    for left, right in (
        ("act", "smolvla_learner"),
        ("act", "diffusion"),
        ("smolvla_learner", "diffusion"),
    ):
        paired = paired_report(all_rows[left], all_rows[right])
        paired["a_wilson_95"] = list(paired["a_wilson_95"])
        paired["b_wilson_95"] = list(paired["b_wilson_95"])
        pairs[f"{left}_vs_{right}"] = paired
    report = {
        "schema_version": 1,
        "comparison_type": "practical_recipe",
        "task_alias": config["task_alias"],
        "partition": "final",
        "state_ids": config["state_ids"],
        "base_seed": config["base_seed"],
        "comparison_config_sha256": config_hash,
        "methods": methods,
        "pairs": pairs,
        "scope": (
            "The three recipes share robot demonstrations, task, cameras, controller, and final states. "
            "They use different model sizes, pretraining, processors, and robot training steps. "
            "The result does not isolate architecture or pretraining effects."
        ),
        "timing_scope": (
            "Episode elapsed time includes simulator startup, policy inference, and simulator steps. "
            "It is not the policy action-generation latency."
        ),
    }
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"Comparison report already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    """Build a checked report from three saved final-state result directories."""
    parser = argparse.ArgumentParser(description="Report the frozen three-method comparison")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/comparison.yaml"))
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    for method in METHODS:
        parser.add_argument(f"--{method.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_three_method_report(
        config_path=args.config,
        configs_root=args.configs_root,
        result_dirs={method: getattr(args, method) for method in METHODS},
        output=args.output,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
