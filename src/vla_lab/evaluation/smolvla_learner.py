"""Evaluate a trained SmolVLA learner on frozen development states."""

import argparse
import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
import yaml

from vla_lab.contracts import EpisodeOutcome
from vla_lab.evaluation.states import evaluate_states, selected_state_ids
from vla_lab.policy.smolvla_learner import load_learner_bundle
from vla_lab.sim.episode import EpisodeResult, open_libero_task, run_policy_episode
from vla_lab.types import JSONValue


def evaluate_learner_reference(
    *,
    checkpoint: str | Path,
    vlm_dir: str | Path,
    configs_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    state_ids: Sequence[int] | None = None,
    seed: int = 20260930,
    run_one: Callable[[int], EpisodeResult] | None = None,
) -> dict[str, JSONValue]:
    """Record one C1 or C2 attempt for each selected development state."""
    source = Path(checkpoint).expanduser().resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("policy") != "smolvla_learner" or manifest.get("stage") not in (
        "c1",
        "c2",
    ):
        raise ValueError("Checkpoint is not a trained SmolVLA learner stage")
    task_alias = str(manifest["task_alias"])
    configs = Path(configs_root)
    partitions = yaml.safe_load((configs / "eval" / "splits.yaml").read_text())
    default_ids = partitions["tasks"][task_alias]["development_state_ids"]
    selected = selected_state_ids(
        configs, task_alias, "development", default_ids if state_ids is None else state_ids
    )
    task_data = yaml.safe_load((configs / "data" / "episodes.yaml").read_text())
    task_prompt = task_data["tasks"][task_alias]["task_name"]
    if run_one is None:
        policy, preprocessor, postprocessor, loaded = load_learner_bundle(source, vlm_dir)
        if loaded != manifest:
            raise ValueError("Learner manifest changed while the policy was loaded")
        policy.eval()

        def run_one(state_id: int) -> EpisodeResult:
            environment = open_libero_task(task_alias, state_id, configs, data_root)
            try:
                return run_policy_episode(
                    environment,
                    policy,
                    preprocessor,
                    postprocessor,
                    task_alias,
                    state_id,
                    configs,
                    task_prompt=task_prompt,
                    reset_seed=seed + state_id,
                )
            finally:
                environment.close()

    model_hash = str(manifest["model_sha256"])

    def run_record(state_id: int) -> EpisodeOutcome:
        episode_seed = seed + state_id
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)
        result = run_one(state_id)
        return EpisodeOutcome(
            task_id=result.task_alias,
            initial_state_id=result.state_id,
            partition="development",
            policy_revision=model_hash,
            success_predicate_version="libero-goal-v1",
            success=result.success,
            termination_reason=result.termination_reason,
            metrics={
                "steps": result.steps,
                "control_hz": result.control_hz,
                "simulated_duration_s": result.simulated_duration_s,
                "inference_wall_s": result.inference_wall_s,
                "clipped_values": result.clipped_values,
                "seed": episode_seed,
            },
            artifact_paths={},
        )

    summary = evaluate_states(
        configs_root=configs,
        task_id=task_alias,
        partition="development",
        state_ids=selected,
        policy_revision=model_hash,
        output_dir=output_dir,
        run_one=run_record,
    )
    summary.update(
        {
            "policy": "smolvla_learner",
            "stage": manifest["stage"],
            "step": manifest["step"],
            "checkpoint": str(source),
            "model_sha256": model_hash,
            "parent_model_sha256": manifest["parent_model_sha256"],
            "vlm_sha256": manifest.get("vlm_sha256"),
            "task_prompt": task_prompt,
            "base_seed": seed,
        }
    )
    (Path(output_dir) / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    """Evaluate a saved learner stage from the terminal."""
    parser = argparse.ArgumentParser(description="Evaluate a trained SmolVLA learner")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--vlm-dir", type=Path, default=Path(".local-data/models/smolvlm2-500m-video-instruct")
    )
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--data-root", type=Path, default=Path(".local-data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-ids", type=int, nargs="+")
    args = parser.parse_args(argv)
    result = evaluate_learner_reference(
        checkpoint=args.checkpoint,
        vlm_dir=args.vlm_dir,
        configs_root=args.configs_root,
        data_root=args.data_root,
        output_dir=args.output,
        state_ids=args.state_ids,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
