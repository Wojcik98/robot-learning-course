"""Evaluate frozen policy recipes on the reserved LIBERO final states."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vla_lab.teaching.records import ComparisonConfig, PolicyRecipe

if TYPE_CHECKING:
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies.diffusion import DiffusionPolicy
    from lerobot.policies.smolvla import SmolVLAPolicy

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import yaml

from vla_lab.contracts import EpisodeOutcome
from vla_lab.evaluation.states import evaluate_states, selected_state_ids
from vla_lab.sim.episode import EpisodeResult
from vla_lab.teaching.interfaces import ActionProcessor, ObservationProcessor
from vla_lab.types import JSONValue

METHODS = ("act", "smolvla_learner", "diffusion")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_comparison_config(path: str | Path, configs_root: str | Path) -> ComparisonConfig:
    """Read one fixed final recipe and reject changes to its states or data split."""
    config = yaml.safe_load(Path(path).read_text())
    if (
        config.get("schema_version") != 1
        or config.get("comparison_type") != "practical_recipe"
        or config.get("partition") != "final"
    ):
        raise ValueError("Comparison needs a schema-1 final-state config")
    configs = Path(configs_root)
    task_alias = str(config["task_alias"])
    split = yaml.safe_load((configs / "eval" / "splits.yaml").read_text())
    final_ids = split["tasks"][task_alias]["final_state_ids"]
    selected = selected_state_ids(
        configs, task_alias, "final", config["state_ids"], allow_final=True
    )
    if selected != final_ids:
        raise ValueError("Comparison must use the full frozen final-state list in order")
    episodes = yaml.safe_load((configs / "data" / "episodes.yaml").read_text())
    task = episodes["tasks"][task_alias]
    if (
        config["dataset_revision"] != episodes["dataset_revision"]
        or config["train_split_sha256"] != task["split_sha256"]
        or config["train_episode_count"] != len(task["train_episode_ids"])
        or config["task_prompt"] != task["task_name"]
    ):
        raise ValueError("Comparison data split or task prompt differs from the frozen data")
    if config["base_seed"] != 20260930 or config["max_steps"] != 300:
        raise ValueError("Comparison reset seed or timeout differs from the frozen recipe")
    if set(config["policies"]) != set(METHODS):
        raise ValueError("Comparison needs ACT, learner SmolVLA, and Diffusion Policy")
    for method, recipe in config["policies"].items():
        if not isinstance(recipe["model_sha256"], str) or len(recipe["model_sha256"]) != 64:
            raise ValueError(f"Invalid model hash for {method}")
        if recipe["robot_training_steps"] < 1 or recipe["n_action_steps"] < 1:
            raise ValueError(f"Invalid training or execution steps for {method}")
    if config["policies"]["diffusion"]["inference_steps"] != 20:
        raise ValueError("The selected Diffusion sampler must use 20 steps")
    return config


def _load_method(
    method: str, recipe: PolicyRecipe, task_alias: str, split_hash: str, revision: str
) -> tuple[ACTPolicy | DiffusionPolicy | SmolVLAPolicy, ObservationProcessor, ActionProcessor]:
    """Load one policy after its saved model and training split pass checks."""
    source = Path(recipe["checkpoint"])
    if _sha256(source / "model.safetensors") != recipe["model_sha256"]:
        raise ValueError(f"Frozen {method} model hash differs from the local checkpoint")
    manifest = json.loads((source / "manifest.json").read_text())
    if (
        manifest["task_alias"] != task_alias
        or manifest["split_sha256"] != split_hash
        or manifest["dataset_revision"] != revision
        or manifest["n_action_steps"] != recipe["n_action_steps"]
    ):
        raise ValueError(f"Frozen {method} checkpoint differs from its data or action recipe")
    if method in ("act", "diffusion"):
        run = json.loads((source.parent / "run.json").read_text())
        if (
            run["steps"] != recipe["robot_training_steps"]
            or run["task_alias"] != task_alias
            or run["split_sha256"] != split_hash
        ):
            raise ValueError(f"Frozen {method} training run differs from its recipe")
    if method == "act":
        from vla_lab.policy.bundle import load_act_bundle

        if manifest["chunk_size"] != recipe["chunk_size"]:
            raise ValueError("Frozen ACT chunk size differs from its checkpoint")
        policy, preprocessor, postprocessor, _ = load_act_bundle(source)
    elif method == "smolvla_learner":
        from vla_lab.policy.smolvla_learner import load_learner_bundle

        if (
            manifest["stage"] != "c2"
            or manifest["step"] != recipe["robot_training_steps"]
            or manifest["chunk_size"] != recipe["chunk_size"]
            or manifest["vlm_sha256"] != recipe["vlm_sha256"]
        ):
            raise ValueError("Frozen SmolVLA lineage differs from its checkpoint")
        policy, preprocessor, postprocessor, _ = load_learner_bundle(
            source, Path(recipe["vlm_dir"])
        )
    elif method == "diffusion":
        from vla_lab.training.diffusion import load_diffusion_bundle

        if (
            manifest["n_obs_steps"] != recipe["n_obs_steps"]
            or manifest["horizon"] != recipe["horizon"]
        ):
            raise ValueError("Frozen Diffusion horizons differ from its checkpoint")
        policy, preprocessor, postprocessor, _ = load_diffusion_bundle(source)
        policy.config.num_inference_steps = recipe["inference_steps"]
        policy.diffusion.num_inference_steps = recipe["inference_steps"]
    else:
        raise ValueError(f"Unknown comparison method: {method}")
    if policy.config.n_action_steps != recipe["n_action_steps"]:
        raise ValueError(f"Loaded {method} execution horizon differs from its recipe")
    if method == "diffusion" and (
        policy.config.n_obs_steps != recipe["n_obs_steps"]
        or policy.config.horizon != recipe["horizon"]
        or policy.diffusion.num_inference_steps != recipe["inference_steps"]
    ):
        raise ValueError("Loaded Diffusion sampler differs from its recipe")
    policy.eval()
    return policy, preprocessor, postprocessor


def evaluate_frozen_method(
    *,
    method: str,
    config_path: str | Path,
    configs_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    run_one: Callable[[int], EpisodeResult] | None = None,
) -> dict[str, JSONValue]:
    """Record one attempt on every final state for one committed recipe."""
    config = load_comparison_config(config_path, configs_root)
    if method not in METHODS:
        raise ValueError(f"Unknown comparison method: {method}")
    recipe = config["policies"][method]
    task_alias = config["task_alias"]
    configs = Path(configs_root)
    seed = int(config["base_seed"])
    if run_one is None:
        from vla_lab.sim.episode import open_libero_task, run_policy_episode

        policy, preprocessor, postprocessor = _load_method(
            method,
            recipe,
            task_alias,
            config["train_split_sha256"],
            config["dataset_revision"],
        )

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
                    partition="final",
                    max_steps=config["max_steps"],
                    task_prompt=config["task_prompt"] if method == "smolvla_learner" else None,
                    reset_seed=seed + state_id,
                )
            finally:
                active_error = sys.exc_info()[0]
                try:
                    environment.close()
                except Exception:
                    if active_error is None:
                        raise

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
            partition="final",
            policy_revision=recipe["model_sha256"],
            success_predicate_version="libero-goal-v1",
            success=result.success,
            termination_reason=result.termination_reason,
            metrics={
                "steps": result.steps,
                "clipped_values": result.clipped_values,
                "seed": episode_seed,
            },
            artifact_paths={},
        )

    summary = evaluate_states(
        configs_root=configs,
        task_id=task_alias,
        partition="final",
        state_ids=config["state_ids"],
        policy_revision=recipe["model_sha256"],
        output_dir=output_dir,
        run_one=run_record,
        allow_final=True,
        error_metrics=lambda state_id: {"seed": seed + state_id},
    )
    summary.update(
        {
            "method": method,
            "model_sha256": recipe["model_sha256"],
            "checkpoint": recipe["checkpoint"],
            "base_seed": seed,
            "max_steps": config["max_steps"],
            "comparison_config_sha256": _sha256(Path(config_path)),
            "recipe": recipe,
            "comparison_type": config["comparison_type"],
            "valid_for_comparison": (
                summary["policy_errors"] == 0 and summary["infrastructure_errors"] == 0
            ),
        }
    )
    (Path(output_dir) / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    """Evaluate one selected policy on the full frozen final-state list."""
    parser = argparse.ArgumentParser(description="Evaluate one frozen final-state policy")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/comparison.yaml"))
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--data-root", type=Path, default=Path(".local-data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_frozen_method(
        method=args.method,
        config_path=args.config,
        configs_root=args.configs_root,
        data_root=args.data_root,
        output_dir=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    if not result["valid_for_comparison"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
