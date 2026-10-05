"""Request full SmolVLA chunks for the virtual-tick experiment."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from vla_lab.teaching.interfaces import (
    ActionProcessor,
    LiberoObservation,
    ObservationProcessor,
    ProcessorObservation,
    RawObservationValue,
    RTCChunkOperation,
    RTCOptions,
)

if TYPE_CHECKING:
    from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy

from typing import cast

import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation
from lerobot.processor import LiberoProcessorStep

from vla_lab.sim.episode import _single_env_batch

RTC_MODES = frozenset({"rtc_hard", "rtc_soft"})
RUNTIME_MODES = frozenset({"synchronous", "naive_async", "temporal_ensemble", *RTC_MODES})


def prepare_policy_batch(
    observation: LiberoObservation, task_prompt: str, preprocessor: ObservationProcessor
) -> ProcessorObservation:
    """Use the same LIBERO image and state path as ordinary evaluation."""
    processed = LiberoProcessorStep().observation(
        cast(
            Callable[[dict[str, RawObservationValue]], ProcessorObservation], preprocess_observation
        )(_single_env_batch(observation))
    )
    processed["task"] = task_prompt
    return preprocessor(cast(ProcessorObservation, processed))


class SmolVLAChunkPredictor:
    """Use fixed noise and return normalized and controller action chunks."""

    def __init__(
        self,
        policy: SmolVLAPolicy,
        *,
        preprocessor: ObservationProcessor,
        postprocessor: ActionProcessor,
        task_prompt: str,
        mode: str,
        stride: int,
        delay: int,
        noise_seed: int,
    ) -> None:
        if mode not in RUNTIME_MODES:
            raise ValueError(f"Unknown runtime mode: {mode}")
        if stride < 1 or delay < 0 or noise_seed < 0:
            raise ValueError("Stride, delay, and noise seed must be valid")
        self.config = cast("SmolVLAConfig", policy.config)
        if mode in RTC_MODES and not delay <= stride <= self.config.chunk_size - delay:
            raise ValueError("RTC needs delay <= stride <= horizon - delay")
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.task_prompt = task_prompt
        self.mode = mode
        self.stride = stride
        self.delay = delay
        self.noise_seed = noise_seed

    def __call__(
        self,
        observation: LiberoObservation,
        observation_tick: int,
        previous_normalized_chunk: np.ndarray | None,
        previous_origin: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample once with a request-specific noise seed."""
        if observation_tick < 0:
            raise ValueError("Observation tick must be nonnegative")
        batch = prepare_policy_batch(observation, self.task_prompt, self.preprocessor)
        device = torch.device(self.config.device or "cpu")
        generator = torch.Generator(device=device)
        generator.manual_seed(self.noise_seed + observation_tick)
        noise = torch.randn(
            (1, self.config.chunk_size, self.config.max_action_dim),
            generator=generator,
            device=device,
        )
        kwargs: RTCOptions = {}
        if self.mode in RTC_MODES:
            if previous_normalized_chunk is not None:
                previous = np.asarray(previous_normalized_chunk, dtype=np.float32)
                if previous.ndim != 2 or previous.shape[1] != 7:
                    raise ValueError("Previous normalized actions need shape (steps, 7)")
                if previous_origin is None or previous_origin > observation_tick:
                    raise ValueError("Previous chunk origin is invalid")
                kwargs["prev_chunk_left_over"] = torch.from_numpy(previous.copy())[None].to(device)
            kwargs["inference_delay"] = self.delay
            kwargs["execution_horizon"] = self.stride
        with torch.no_grad():
            # LeRobot annotations omit the task text retained in this batch.
            predict_chunk = cast(RTCChunkOperation, self.policy.predict_action_chunk)
            normalized = predict_chunk(batch, noise=noise, **kwargs)
            physical = self.postprocessor(normalized)
        normalized_np = np.asarray(normalized.detach().cpu(), dtype=np.float32)
        physical_np = np.asarray(physical.detach().cpu(), dtype=np.float32)
        expected = (1, self.config.chunk_size, 7)
        if normalized_np.shape != expected or physical_np.shape != expected:
            raise ValueError(f"SmolVLA chunk needs shape {expected}")
        if not np.isfinite(normalized_np).all() or not np.isfinite(physical_np).all():
            raise ValueError("SmolVLA chunk has a nonfinite action")
        return normalized_np[0].copy(), np.clip(physical_np[0], -1.0, 1.0)
