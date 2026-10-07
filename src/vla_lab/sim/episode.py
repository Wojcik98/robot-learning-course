"""Run one ACT episode with the pinned LIBERO observation path."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import torch
import yaml
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.processor import LiberoProcessorStep
from libero.libero import benchmark
from numpy.typing import NDArray

from vla_lab.teaching.interfaces import (
    ActionPolicy,
    ActionProcessor,
    EpisodeEnvironment,
    LiberoObservation,
    ManagedEnvironment,
    ObservationPolicy,
    ObservationProcessor,
    ProcessorObservation,
    RawObservationValue,
)
from vla_lab.teaching.records import SimulationTask
from vla_lab.timing import LIBERO_CONTROL_HZ, require_libero_rate

from .setup import prepare_local_libero, select_local_assets


@dataclass(frozen=True)
class EpisodeResult:
    """The outcome of one fixed simulator initial state."""

    task_alias: str
    state_id: int
    success: bool
    steps: int
    clipped_values: int
    termination_reason: str = "unknown"
    control_hz: int = LIBERO_CONTROL_HZ
    inference_wall_s: float = 0.0

    @property
    def simulated_duration_s(self) -> float:
        """Return policy control time; reset settling time is excluded."""
        return self.steps / self.control_hz


class PolicyActionError(ValueError):
    """A policy sent an action that cannot enter the controller."""


def _task_split(task_alias: str, configs_root: str | Path) -> SimulationTask:
    manifest = yaml.safe_load((Path(configs_root) / "eval" / "splits.yaml").read_text())
    try:
        return manifest["tasks"][task_alias]
    except KeyError as error:
        raise ValueError(f"Unknown task alias: {task_alias}") from error


def _single_env_batch(observation: LiberoObservation) -> dict[str, RawObservationValue]:
    """Add one batch axis to each array from a direct LIBERO environment."""

    def add_axis(value: RawObservationValue) -> RawObservationValue:
        if isinstance(value, dict):
            return {key: add_axis(item) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            return np.expand_dims(value, axis=0)
        return value

    return cast(
        dict[str, RawObservationValue], add_axis(cast(dict[str, RawObservationValue], observation))
    )


def open_libero_task(
    task_alias: str,
    state_id: int,
    configs_root: str | Path,
    data_root: str | Path,
    *,
    control_hz: int = LIBERO_CONTROL_HZ,
) -> ManagedEnvironment[LiberoObservation]:
    """Open one pinned LIBERO task at one initial state."""
    require_libero_rate(control_hz)
    task = _task_split(task_alias, configs_root)
    if state_id < 0 or state_id >= task["state_count"]:
        raise ValueError(f"Initial state {state_id} is outside the task state range")
    prepare_local_libero(data_root)
    select_local_assets(data_root)

    suite = benchmark.get_benchmark_dict()[task["suite"]]()
    environment = LiberoEnv(
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
        control_freq=control_hz,
        control_mode="relative",
        hard_reset=True,
    )

    return cast(ManagedEnvironment[LiberoObservation], environment)


def run_policy_episode(
    environment: EpisodeEnvironment[LiberoObservation],
    policy: ActionPolicy | ObservationPolicy,
    preprocessor: ObservationProcessor,
    postprocessor: ActionProcessor,
    task_alias: str,
    state_id: int,
    configs_root: str | Path,
    *,
    partition: str = "development",
    max_steps: int = 300,
    on_frame: Callable[[np.ndarray], None] | None = None,
    task_prompt: str | None = None,
    reset_seed: int | None = None,
    control_hz: int = LIBERO_CONTROL_HZ,
) -> EpisodeResult:
    """Run one policy on a declared state partition."""
    require_libero_rate(control_hz)
    task = _task_split(task_alias, configs_root)
    if partition not in ("development", "final"):
        raise ValueError(f"Unknown state partition: {partition}")
    if state_id not in (
        task["development_state_ids"] if partition == "development" else task["final_state_ids"]
    ):
        raise ValueError(f"State {state_id} is not in the {partition} partition")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    observation, _ = (
        environment.reset(seed=reset_seed) if reset_seed is not None else environment.reset()
    )
    policy.reset()
    if on_frame is not None:
        on_frame(np.ascontiguousarray(observation["pixels"]["image"][::-1, ::-1]))
    libero_processor = LiberoProcessorStep()
    # LeRobot method annotations omit the task text kept by its processors.
    select_action = cast(Callable[[ProcessorObservation], torch.Tensor], policy.select_action)
    inference_wall_s = 0.0
    clipped_values = 0
    success = False
    termination_reason = "timeout"
    for step in range(1, max_steps + 1):  # noqa: B007 - saved in the episode result
        processed = libero_processor.observation(
            cast(
                Callable[[dict[str, RawObservationValue]], ProcessorObservation],
                preprocess_observation,
            )(_single_env_batch(observation))
        )
        if task_prompt is not None:
            processed["task"] = task_prompt
        batch = preprocessor(cast(ProcessorObservation, processed))
        prediction_started = perf_counter()
        with torch.no_grad():
            normalized_action = select_action(batch)
            physical_action = postprocessor(normalized_action)
        if physical_action.is_cuda:
            torch.cuda.synchronize(physical_action.device)
        inference_wall_s += perf_counter() - prediction_started
        action: NDArray[np.float32] = np.asarray(physical_action, dtype=np.float32).reshape(-1)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise PolicyActionError("Policy returned an invalid seven-value action")
        clipped_values += int(np.count_nonzero((action < -1.0) | (action > 1.0)))
        action = np.clip(action, -1.0, 1.0)
        observation, _, terminated, truncated, info = environment.step(action)
        if on_frame is not None:
            on_frame(np.ascontiguousarray(observation["pixels"]["image"][::-1, ::-1]))
        success = bool(info.get("is_success", False))
        if success or terminated or truncated:
            termination_reason = (
                "success" if success else "terminated" if terminated else "truncated"
            )
            break
    return EpisodeResult(
        task_alias,
        state_id,
        success,
        step,
        clipped_values,
        termination_reason,
        control_hz,
        inference_wall_s,
    )


def run_act_episode(
    environment: EpisodeEnvironment[LiberoObservation],
    policy: ActionPolicy | ObservationPolicy,
    preprocessor: ObservationProcessor,
    postprocessor: ActionProcessor,
    task_alias: str,
    state_id: int,
    configs_root: str | Path,
    *,
    partition: str = "development",
    max_steps: int = 300,
    on_frame: Callable[[np.ndarray], None] | None = None,
    reset_seed: int | None = None,
    control_hz: int = LIBERO_CONTROL_HZ,
) -> EpisodeResult:
    """Run ACT with the shared LIBERO observation path."""
    return run_policy_episode(
        environment,
        policy,
        preprocessor,
        postprocessor,
        task_alias,
        state_id,
        configs_root,
        partition=partition,
        max_steps=max_steps,
        on_frame=on_frame,
        reset_seed=reset_seed,
        control_hz=control_hz,
    )
