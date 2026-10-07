"""Evaluate an ACT checkpoint on frozen LIBERO development states."""

import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

from vla_lab.policy.bundle import load_act_bundle
from vla_lab.sim.episode import EpisodeResult, open_libero_task, run_act_episode


def evaluate_act_reference(
    checkpoint: str | Path,
    configs_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    state_ids: Sequence[int] | None = None,
    run_one: Callable[[int], EpisodeResult] | None = None,
) -> dict[str, object]:
    """Run and record only the selected development initial states."""
    checkpoint_path = Path(checkpoint)
    manifest = json.loads((checkpoint_path / "manifest.json").read_text())
    if manifest.get("policy") != "act":
        raise ValueError("The checkpoint is not an ACT bundle")
    task_alias = str(manifest["task_alias"])
    split = yaml.safe_load((Path(configs_root) / "eval" / "splits.yaml").read_text())
    try:
        development_ids = split["tasks"][task_alias]["development_state_ids"]
    except KeyError as error:
        raise ValueError(f"Unknown task alias: {task_alias}") from error
    selected = list(development_ids if state_ids is None else state_ids)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Development state IDs must be nonempty and unique")
    if any(state_id not in development_ids for state_id in selected):
        raise ValueError("Evaluation state ID is outside the development partition")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    if run_one is None:
        policy, preprocessor, postprocessor, _ = load_act_bundle(checkpoint_path)

        def run_one(state_id: int) -> EpisodeResult:
            environment = open_libero_task(task_alias, state_id, configs_root, data_root)
            try:
                return run_act_episode(
                    environment,
                    policy,
                    preprocessor,
                    postprocessor,
                    task_alias,
                    state_id,
                    configs_root,
                    reset_seed=20260930 + state_id,
                )
            finally:
                environment.close()

    successes = 0
    with (output / "episodes.jsonl").open("w") as rows:
        for state_id in selected:
            started = time.monotonic()
            result = run_one(state_id)
            if result.task_alias != task_alias or result.state_id != state_id:
                raise ValueError("Episode result identity differs from the selected state")
            successes += int(result.success)
            rows.write(
                json.dumps(
                    {
                        "task_alias": result.task_alias,
                        "state_id": result.state_id,
                        "reset_seed": 20260930 + state_id,
                        "success": result.success,
                        "steps": result.steps,
                        "control_hz": result.control_hz,
                        "simulated_duration_s": result.simulated_duration_s,
                        "inference_wall_s": result.inference_wall_s,
                        "clipped_values": result.clipped_values,
                        "elapsed_s": round(time.monotonic() - started, 3),
                    }
                )
                + "\n"
            )
            rows.flush()
    summary: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path.resolve()),
        "task_alias": task_alias,
        "partition": "development",
        "base_seed": 20260930,
        "state_ids": selected,
        "episodes": len(selected),
        "successes": successes,
        "success_rate": successes / len(selected),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    """Evaluate a saved ACT reference from the terminal."""
    parser = argparse.ArgumentParser(description="Evaluate an ACT bundle on development states")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".local-data"))
    arguments = parser.parse_args(argv)
    summary = evaluate_act_reference(
        arguments.checkpoint, Path("configs"), arguments.data_root, arguments.output
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
