"""Run one ACT episode with the pinned LIBERO observation path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from lerobot.envs.utils import preprocess_observation
from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline

from .setup import prepare_local_libero, select_local_assets


@dataclass(frozen=True)
class EpisodeResult:
    """The outcome of one fixed simulator initial state."""

    task_alias: str
    state_id: int
    success: bool
    steps: int
    clipped_values: int


def _task_split(task_alias: str, configs_root: str | Path) -> dict[str, Any]:
    manifest = yaml.safe_load((Path(configs_root) / "eval" / "splits.yaml").read_text())
    try:
        return manifest["tasks"][task_alias]
    except KeyError as error:
        raise ValueError(f"Unknown task alias: {task_alias}") from error


def _single_env_batch(observation: dict[str, Any]) -> dict[str, Any]:
    """Add one batch axis to each array from a direct LIBERO environment."""

    def add_axis(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: add_axis(item) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            return np.expand_dims(value, axis=0)
        return value

    return add_axis(observation)


def open_libero_task(
    task_alias: str, state_id: int, configs_root: str | Path, data_root: str | Path
) -> Any:
    """Open one pinned LIBERO task at one initial state."""
    task = _task_split(task_alias, configs_root)
    if state_id < 0 or state_id >= task["state_count"]:
        raise ValueError(f"Initial state {state_id} is outside the task state range")
    prepare_local_libero(data_root)
    select_local_assets(data_root)
    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[task["suite"]]()
    return LiberoEnv(
        task_suite=suite,
        task_id=task["task_id"],
        task_suite_name=task["suite"],
        camera_name="agentview_image,robot0_eye_in_hand_image",
        obs_type="pixels_agent_pos",
        observation_width=256,
        observation_height=256,
        init_states=True,
        episode_index=state_id,
        n_envs=1,
        num_steps_wait=10,
        control_freq=20,
        control_mode="relative",
        hard_reset=True,
    )


def run_act_episode(
    environment: Any,
    policy: Any,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    task_alias: str,
    state_id: int,
    configs_root: str | Path,
    *,
    partition: str = "development",
    max_steps: int = 300,
    on_frame: Callable[[np.ndarray], None] | None = None,
) -> EpisodeResult:
    """Run ACT on a declared state partition and use LeRobot transforms."""
    task = _task_split(task_alias, configs_root)
    if partition not in ("development", "final"):
        raise ValueError(f"Unknown state partition: {partition}")
    if state_id not in task[f"{partition}_state_ids"]:
        raise ValueError(f"State {state_id} is not in the {partition} partition")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    observation, _ = environment.reset()
    policy.reset()
    if on_frame is not None:
        on_frame(np.ascontiguousarray(observation["pixels"]["image"][::-1, ::-1]))
    libero_processor = LiberoProcessorStep()
    clipped_values = 0
    success = False
    for step in range(1, max_steps + 1):
        processed = libero_processor.observation(
            preprocess_observation(_single_env_batch(observation))
        )
        batch = preprocessor(processed)
        with torch.no_grad():
            normalized_action = policy.select_action(batch)
            physical_action = postprocessor(normalized_action)
        action = np.asarray(physical_action, dtype=np.float32).reshape(-1)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("ACT returned an invalid seven-value action")
        clipped_values += int(np.count_nonzero((action < -1.0) | (action > 1.0)))
        action = np.clip(action, -1.0, 1.0)
        observation, _, terminated, truncated, info = environment.step(action)
        if on_frame is not None:
            on_frame(np.ascontiguousarray(observation["pixels"]["image"][::-1, ::-1]))
        success = bool(info.get("is_success", False))
        if success or terminated or truncated:
            break
    return EpisodeResult(task_alias, state_id, success, step, clipped_values)
