"""Save policy-generated recovery examples on development states only."""

import argparse
import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import yaml
from lerobot.envs.utils import preprocess_observation
from lerobot.processor import LiberoProcessorStep

from vla_lab.data.recording import record_action_episode
from vla_lab.policy.bundle import load_act_bundle
from vla_lab.sim.episode import _single_env_batch, open_libero_task
from vla_lab.teaching.interfaces import EpisodeEnvironment, LiberoObservation
from vla_lab.types import JSONValue


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_development_states(
    configs_root: str | Path, task_alias: str, state_ids: list[int]
) -> list[int]:
    """Reject any recording state outside the frozen development partition."""
    task = yaml.safe_load((Path(configs_root) / "eval" / "splits.yaml").read_text())["tasks"][
        task_alias
    ]
    if not state_ids or len(set(state_ids)) != len(state_ids):
        raise ValueError("Recovery candidates need unique development states")
    if any(state_id not in task["development_state_ids"] for state_id in state_ids):
        raise ValueError("Recovery candidate is outside the development partition")
    return state_ids


class _TrackedEnvironment[ObservationT]:
    def __init__(self, environment: EpisodeEnvironment[ObservationT]) -> None:
        self.environment = environment
        self.success = False

    def reset(self, seed: int | None = None) -> tuple[ObservationT, Mapping[str, JSONValue]]:
        self.success = False
        return self.environment.reset(seed=seed)

    def step(
        self, action: np.ndarray
    ) -> tuple[ObservationT, float, bool, bool, Mapping[str, JSONValue]]:
        result = self.environment.step(action)
        self.success = bool(result[4].get("is_success", False))
        return result


def collect_reference_recovery(
    *,
    checkpoint: str | Path,
    configs_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    task_alias: str = "bowl_plate",
    candidate_ids: list[int] | None = None,
    target_successes: int = 5,
) -> dict[str, JSONValue]:
    """Record a bounded set of successful ACT rollouts as labelled examples."""

    configs = Path(configs_root)
    candidates = candidate_development_states(
        configs, task_alias, list(range(10)) if candidate_ids is None else candidate_ids
    )
    if not 1 <= target_successes <= 10:
        raise ValueError("Reference collection needs 1 to 10 successful episodes")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Reference recovery output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    policy, preprocessor, postprocessor, bundle = load_act_bundle(checkpoint)
    if bundle["task_alias"] != task_alias:
        raise ValueError("Reference teacher checkpoint belongs to another task")
    policy.eval()
    libero_processor = LiberoProcessorStep()
    accepted: list[dict[str, JSONValue]] = []
    rejected: list[dict[str, JSONValue]] = []
    attempted: list[int] = []
    for state_id in candidates:
        if len(accepted) >= target_successes:
            break
        attempted.append(state_id)
        reset_seed = 20260930 + state_id
        random.seed(reset_seed)
        np.random.seed(reset_seed)
        torch.manual_seed(reset_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(reset_seed)
        policy.reset()
        environment = open_libero_task(task_alias, state_id, configs, data_root)
        tracked = _TrackedEnvironment(environment)

        def action_at_tick(_tick: int, observation: LiberoObservation) -> np.ndarray:
            processed = libero_processor.observation(
                preprocess_observation(_single_env_batch(observation))
            )
            batch = preprocessor(processed)
            with torch.no_grad():
                selected = policy.select_action(batch)
                physical = postprocessor(selected)
            return np.asarray(physical, dtype=np.float32).reshape(-1)

        try:
            episode = record_action_episode(
                tracked,
                action_at_tick,
                task_id=task_alias,
                episode_id=f"reference-act-{task_alias}-state-{state_id}",
                reset_seed=reset_seed,
                max_steps=300,
            )
        finally:
            environment.close()
        kind = "accepted" if tracked.success else "rejected"
        path = output / kind / f"state-{state_id:02d}.npz"
        episode.save(path)
        item = {
            "path": str(path),
            "sha256": _sha256(path),
            "state_id": state_id,
            "reset_seed": reset_seed,
            "accepted": tracked.success,
            "action_count": len(episode.actions),
        }
        (accepted if tracked.success else rejected).append(item)
    development = yaml.safe_load((configs / "eval" / "splits.yaml").read_text())["tasks"][
        task_alias
    ]["development_state_ids"]
    manifest = {
        "schema_version": 1,
        "task_alias": task_alias,
        "source_type": "policy_generated",
        "source_checkpoint": str(checkpoint),
        "source_model_sha256": _sha256(Path(checkpoint) / "model.safetensors"),
        "candidate_state_ids": candidates,
        "attempted_state_ids": attempted,
        "heldout_development_state_ids": [
            state_id for state_id in development if state_id not in attempted
        ],
        "recordings": accepted,
        "rejected_recordings": rejected,
        "provenance_limit": "These examples came from ACT, not a human keyboard operator.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    """Collect local reference episodes without opening final states."""
    parser = argparse.ArgumentParser(description="Collect ACT reference recovery episodes")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="bowl_plate")
    parser.add_argument("--candidate-ids", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--target-successes", type=int, default=5)
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--data-root", type=Path, default=Path(".local-data"))
    args = parser.parse_args(argv)
    result = collect_reference_recovery(
        checkpoint=args.checkpoint,
        configs_root=args.configs_root,
        data_root=args.data_root,
        output_dir=args.output,
        task_alias=args.task,
        candidate_ids=args.candidate_ids,
        target_successes=args.target_successes,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
