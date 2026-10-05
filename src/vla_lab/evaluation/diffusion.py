"""Evaluate a saved Diffusion Policy on frozen development states."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
import yaml

from vla_lab.contracts import EpisodeOutcome
from vla_lab.evaluation.states import evaluate_states, selected_state_ids
from vla_lab.sim.episode import EpisodeResult
from vla_lab.types import JSONValue


def evaluate_diffusion_reference(
    *,
    checkpoint: str | Path,
    configs_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    state_ids: Sequence[int] | None = None,
    seed: int = 20260930,
    inference_steps: int | None = None,
    run_one: Callable[[int], EpisodeResult] | None = None,
) -> dict[str, JSONValue]:
    """Run one attempt per development state with a fixed noise and reset seed."""
    source = Path(checkpoint).expanduser().resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("policy") != "diffusion":
        raise ValueError("Checkpoint is not a Diffusion Policy bundle")
    task_alias = str(manifest["task_alias"])
    configs = Path(configs_root)
    data_split = yaml.safe_load((configs / "data" / "episodes.yaml").read_text())
    if (
        manifest["dataset_revision"] != data_split["dataset_revision"]
        or manifest["split_sha256"] != data_split["tasks"][task_alias]["split_sha256"]
    ):
        raise ValueError("Diffusion checkpoint data split differs from the frozen course split")
    stats_file = configs / "data" / "stats" / f"{task_alias}.json"
    if (
        hashlib.sha256(stats_file.read_bytes()).hexdigest()
        != manifest["training_stats_file_sha256"]
    ):
        raise ValueError("Diffusion checkpoint statistics differ from the frozen course statistics")
    partitions = yaml.safe_load((configs / "eval" / "splits.yaml").read_text())
    development_ids = partitions["tasks"][task_alias]["development_state_ids"]
    selected = selected_state_ids(
        configs,
        task_alias,
        "development",
        development_ids if state_ids is None else state_ids,
    )
    if inference_steps is not None and not 1 <= inference_steps <= 100:
        raise ValueError("Diffusion inference steps must be from 1 to 100")
    sample_times_ms: list[float] = []
    if run_one is None:
        from vla_lab.sim.episode import open_libero_task, run_policy_episode
        from vla_lab.training.diffusion import load_diffusion_bundle

        policy, preprocessor, postprocessor, loaded = load_diffusion_bundle(source)
        if loaded != manifest:
            raise ValueError("Diffusion manifest changed while the policy was loaded")
        if inference_steps is not None:
            policy.config.num_inference_steps = inference_steps
            policy.diffusion.num_inference_steps = inference_steps
        inference_steps = policy.diffusion.num_inference_steps
        policy.eval()
        generate_actions = policy.diffusion.generate_actions

        def timed_generate_actions(
            batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None
        ) -> torch.Tensor:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            actions = generate_actions(batch, noise=noise)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            sample_times_ms.append((time.perf_counter() - start) * 1000)
            return actions

        policy.diffusion.generate_actions = timed_generate_actions

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
                    reset_seed=seed + state_id,
                )
            finally:
                environment.close()

    model_hash = str(manifest["model_sha256"])

    def run_record(state_id: int) -> EpisodeOutcome:
        episode_seed = seed + state_id
        sample_times_ms.clear()
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)
        result = run_one(state_id)
        metrics = {
            "steps": result.steps,
            "clipped_values": result.clipped_values,
            "seed": episode_seed,
            "sampler_calls": len(sample_times_ms),
        }
        if sample_times_ms:
            metrics["sampler_mean_ms"] = float(np.mean(sample_times_ms))
            metrics["sampler_p50_ms"] = float(np.median(sample_times_ms))
        return EpisodeOutcome(
            task_id=result.task_alias,
            initial_state_id=result.state_id,
            partition="development",
            policy_revision=model_hash,
            success_predicate_version="libero-goal-v1",
            success=result.success,
            termination_reason=result.termination_reason,
            metrics=metrics,
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
            "policy": "diffusion",
            "checkpoint": str(source),
            "model_sha256": model_hash,
            "task_alias": task_alias,
            "base_seed": seed,
            "inference_steps": inference_steps,
            "n_obs_steps": manifest["n_obs_steps"],
            "horizon": manifest["horizon"],
            "n_action_steps": manifest["n_action_steps"],
        }
    )
    (Path(output_dir) / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    """Evaluate one saved Diffusion Policy from the terminal."""
    parser = argparse.ArgumentParser(description="Evaluate a saved Diffusion Policy")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--data-root", type=Path, default=Path(".local-data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-ids", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--inference-steps", type=int)
    args = parser.parse_args(argv)
    result = evaluate_diffusion_reference(
        checkpoint=args.checkpoint,
        configs_root=args.configs_root,
        data_root=args.data_root,
        output_dir=args.output,
        state_ids=args.state_ids,
        seed=args.seed,
        inference_steps=args.inference_steps,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
