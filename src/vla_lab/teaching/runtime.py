"""Small runtime pieces used by the policy execution lessons."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from vla_lab.teaching.interfaces import (
    ActionT,
    ActionValue,
    BatchT,
    EpisodeEnvironment,
    LiberoObservation,
    ObservationProcessor,
    ObservationT,
    PolicyT,
    ProcessorObservation,
    ResettableEnvironment,
    ResettablePolicy,
    TensorBatch,
)

type RawObservationValue = np.ndarray | dict[str, "RawObservationValue"] | int | float | bool | None


@dataclass(frozen=True)
class RolloutResult:
    """Summary of one bounded episode."""

    success: bool
    steps: int
    policy_calls: int
    clipped_values: int
    termination_reason: str
    actions: tuple[tuple[float, ...], ...]


def prepare_libero_observation(
    observation: LiberoObservation,
    preprocessor: ObservationProcessor,
    *,
    task_prompt: str | None = None,
) -> TensorBatch:
    """Apply the pinned observation path to one direct LIBERO observation.

    LIBERO returns unbatched NumPy images and state. This helper adds a batch
    axis, applies LeRobot's LIBERO field mapping, adds an optional task prompt,
    and runs the saved model preprocessor.
    """
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor import LiberoProcessorStep

    def add_batch_axis(value: RawObservationValue) -> RawObservationValue:
        if isinstance(value, dict):
            return {key: add_batch_axis(item) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            return np.expand_dims(value, axis=0)
        return value

    # The library annotation omits the nested robot state.
    preprocess = cast(Callable[[RawObservationValue], ProcessorObservation], preprocess_observation)
    prepared = LiberoProcessorStep().observation(
        preprocess(add_batch_axis(cast(dict[str, RawObservationValue], observation)))
    )
    if task_prompt is not None:
        prepared["task"] = task_prompt
    processed_batch = preprocessor(cast(ProcessorObservation, prepared))
    return {key: value for key, value in processed_batch.items() if isinstance(value, torch.Tensor)}


def reset_environment(
    environment: ResettableEnvironment[ObservationT],
    policy: ResettablePolicy,
    *,
    seed: int | None = None,
) -> ObservationT:
    """Reset the environment, then clear the policy's action queue.

    The environment follows the Gymnasium ``(observation, info)`` reset API.
    A seed of ``None`` uses the environment's saved random generator state.
    """
    reset_value = environment.reset(seed=seed) if seed is not None else environment.reset()
    if not isinstance(reset_value, tuple) or len(reset_value) != 2:
        raise TypeError("environment.reset() must return (observation, info)")
    observation, _info = reset_value
    policy.reset()
    return observation


def select_chunk_action(action_chunk: ActionT, offset: int) -> ActionT:
    """Select one action from a one-environment chunk with shape ``[1, T, D]``.

    ``offset`` is a zero-based index. The return value has shape ``[D]`` and
    keeps the input type (Tensor or NumPy array).
    """
    if offset < 0:
        raise ValueError("action offset must be zero or greater")
    if action_chunk.ndim != 3 or action_chunk.shape[0] != 1:
        raise ValueError("action chunk must have shape [1, time, action_dim]")
    if offset >= action_chunk.shape[1]:
        raise IndexError("action offset is outside the predicted chunk")
    return action_chunk[0, offset]


def run_chunked_episode(
    environment: EpisodeEnvironment[ObservationT],
    policy: PolicyT,
    prepare_observation: Callable[[ObservationT], BatchT],
    predict_chunk: Callable[[PolicyT, BatchT], ActionT],
    action_to_controller: Callable[[ActionT], ActionValue],
    *,
    max_steps: int,
    execute_steps: int,
    seed: int | None = None,
    on_observation: Callable[[ObservationT], None] | None = None,
) -> RolloutResult:
    """Run policy action chunks through environment observations and steps.

    The policy prediction must have shape ``[1, T, 7]``. The preparation and
    action callbacks own model-specific image/state processing. The controller
    receives one finite seven-value action, clipped to ``[-1, 1]``.
    """
    if max_steps < 1 or execute_steps < 1:
        raise ValueError("max_steps and execute_steps must be positive")
    observation = reset_environment(environment, policy, seed=seed)
    if on_observation is not None:
        on_observation(observation)
    calls = 0
    clipped = 0
    success = False
    reason = "timeout"
    action_log: list[tuple[float, ...]] = []
    steps = 0
    while steps < max_steps:
        batch = prepare_observation(observation)
        with torch.no_grad():
            chunk = predict_chunk(policy, batch)
        calls += 1
        if not isinstance(chunk, (torch.Tensor, np.ndarray)):
            raise TypeError("predict_chunk must return a Tensor or NumPy array")
        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[2] != 7:
            raise ValueError("predict_chunk must return shape [1, time, 7]")
        count = min(execute_steps, int(chunk.shape[1]), max_steps - steps)
        if count < 1:
            raise ValueError("predict_chunk returned an empty action chunk")
        for offset in range(count):
            selected = select_chunk_action(chunk, offset)
            converted = action_to_controller(selected)
            if isinstance(converted, torch.Tensor):
                converted = converted.detach().cpu().numpy()
            raw = np.asarray(converted, dtype=np.float32).reshape(-1)
            if raw.shape != (7,) or not np.isfinite(raw).all():
                raise ValueError("controller action must contain seven finite values")
            clipped += int(np.count_nonzero((raw < -1.0) | (raw > 1.0)))
            action = np.clip(raw, -1.0, 1.0)
            action_log.append(tuple(float(value) for value in action))
            observation, _reward, terminated, truncated, info = environment.step(action)
            steps += 1
            if on_observation is not None:
                on_observation(observation)
            success = bool(info.get("is_success", False))
            if success or terminated or truncated:
                reason = "success" if success else "terminated" if terminated else "truncated"
                break
        if success or terminated or truncated:
            break
    return RolloutResult(
        success=success,
        steps=steps,
        policy_calls=calls,
        clipped_values=clipped,
        termination_reason=reason,
        actions=tuple(action_log),
    )
