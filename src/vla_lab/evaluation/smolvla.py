"""Evaluate the published SmolVLA checkpoint on LIBERO development states."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
import yaml

from vla_lab.sim.episode import EpisodeResult

MODEL_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


def evaluate_published_reference(
    model_dir: str | Path,
    vlm_dir: str | Path,
    task_alias: str,
    configs_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    state_ids: Sequence[int] | None = None,
    n_action_steps: int = 1,
    seed: int = 20260930,
    run_one: Callable[[int], EpisodeResult] | None = None,
) -> dict[str, object]:
    """Run the published policy on selected development states and save each result."""
    if n_action_steps not in (1, 10):
        raise ValueError("n_action_steps must be 1 or 10 for this comparison")
    root = Path(configs_root)
    split = yaml.safe_load((root / "eval" / "splits.yaml").read_text())
    task_data = yaml.safe_load((root / "data" / "episodes.yaml").read_text())
    try:
        development_ids = split["tasks"][task_alias]["development_state_ids"]
        task_prompt = task_data["tasks"][task_alias]["task_name"]
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
        from vla_lab.policy.smolvla import load_published_smolvla
        from vla_lab.sim.episode import open_libero_task, run_policy_episode

        policy, preprocessor, postprocessor = load_published_smolvla(
            model_dir, vlm_dir, n_action_steps=n_action_steps
        )

        def run_one(state_id: int) -> EpisodeResult:
            environment = open_libero_task(task_alias, state_id, root, data_root)
            try:
                return run_policy_episode(
                    environment,
                    policy,
                    preprocessor,
                    postprocessor,
                    task_alias,
                    state_id,
                    root,
                    task_prompt=task_prompt,
                )
            finally:
                environment.close()

    successes = 0
    with (output / "episodes.jsonl").open("w") as rows:
        for state_id in selected:
            episode_seed = seed + state_id
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(episode_seed)
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
                        "seed": episode_seed,
                        "success": result.success,
                        "steps": result.steps,
                        "clipped_values": result.clipped_values,
                        "elapsed_s": round(time.monotonic() - started, 3),
                    }
                )
                + "\n"
            )
            rows.flush()
    summary: dict[str, object] = {
        "schema_version": 1,
        "policy": "smolvla_published",
        "model_revision": MODEL_REVISION,
        "vlm_revision": VLM_REVISION,
        "model_dir": str(Path(model_dir).resolve()),
        "vlm_dir": str(Path(vlm_dir).resolve()),
        "task_alias": task_alias,
        "task_prompt": task_prompt,
        "partition": "development",
        "n_action_steps": n_action_steps,
        "base_seed": seed,
        "state_ids": selected,
        "episodes": len(selected),
        "successes": successes,
        "success_rate": successes / len(selected),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    """Evaluate the pinned published SmolVLA files from the terminal."""
    parser = argparse.ArgumentParser(description="Evaluate published SmolVLA on development states")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vlm-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-ids", type=int, nargs="+")
    parser.add_argument("--n-action-steps", type=int, choices=(1, 10), default=1)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--data-root", type=Path, default=Path(".local-data"))
    args = parser.parse_args(argv)
    summary = evaluate_published_reference(
        args.model_dir,
        args.vlm_dir,
        args.task,
        Path("configs"),
        args.data_root,
        args.output,
        state_ids=args.state_ids,
        n_action_steps=args.n_action_steps,
        seed=args.seed,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
