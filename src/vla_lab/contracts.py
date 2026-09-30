"""Small validated packets shared by policies, the simulator, and timing labs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SCHEMA_VERSION = 1
CAMERA_ORDER = ("agentview_image", "robot0_eye_in_hand_image")


@dataclass(frozen=True)
class ObservationPacket:
    """One simulator observation with its exact camera and reset identity."""

    episode_id: str
    reset_counter: int
    sequence: int
    observation_tick: int
    capture_time_s: float
    cameras: dict[str, np.ndarray]
    state: np.ndarray
    state_schema: tuple[str, ...]
    task_id: str
    instruction: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported observation schema version")
        if not self.episode_id or not self.task_id:
            raise ValueError("Observation needs an episode and task ID")
        if min(self.reset_counter, self.sequence, self.observation_tick) < 0:
            raise ValueError("Observation counters and tick must not be negative")
        if not math.isfinite(self.capture_time_s) or self.capture_time_s < 0:
            raise ValueError("Observation capture time is invalid")
        if tuple(self.cameras) != CAMERA_ORDER:
            raise ValueError(f"Observation camera order must be {CAMERA_ORDER}")
        for name, image in self.cameras.items():
            if (
                not isinstance(image, np.ndarray)
                or image.shape != (256, 256, 3)
                or image.dtype != np.uint8
            ):
                raise ValueError(f"Observation camera {name} must be 256x256 uint8 RGB")
        if not isinstance(self.state, np.ndarray) or self.state.shape != (8,):
            raise ValueError("Observation state must have eight values")
        if len(self.state_schema) != self.state.size or len(set(self.state_schema)) != len(
            self.state_schema
        ):
            raise ValueError("Observation state schema does not match its vector")
        if not np.isfinite(self.state).all():
            raise ValueError("Observation state has a nonfinite value")
        if self.instruction is not None and not self.instruction.strip():
            raise ValueError("Instruction must be nonempty when supplied")


@dataclass(frozen=True)
class ActionSchema:
    """The simulator's seven-value relative controller input."""

    frame: str
    units: str
    rotation_convention: str
    delta: bool
    scaling: str
    gripper: str
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    padding_mask: tuple[bool, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported action schema version")
        if (
            not self.frame
            or not self.units
            or not self.rotation_convention
            or not self.scaling
            or not self.gripper
        ):
            raise ValueError(
                "Action schema needs frame, units, rotation, scale, and gripper meaning"
            )
        if len(self.lower) != 7 or len(self.upper) != 7 or len(self.padding_mask) != 7:
            raise ValueError("Action schema needs seven bounds and mask values")
        if any(
            not math.isfinite(low) or not math.isfinite(high) or low >= high
            for low, high in zip(self.lower, self.upper, strict=True)
        ):
            raise ValueError("Action schema bounds are invalid")

    @staticmethod
    def libero_relative() -> ActionSchema:
        """Declare the pinned LIBERO 20 Hz relative action convention."""
        return ActionSchema(
            frame="LIBERO end-effector controller",
            units="normalized controller input",
            rotation_convention="controller-native three-value rotation delta",
            delta=True,
            scaling="LIBERO controller applies physical scale",
            gripper="signed controller command; -1 is used during reset",
            lower=(-1.0,) * 7,
            upper=(1.0,) * 7,
            padding_mask=(True,) * 7,
        )


@dataclass(frozen=True)
class PolicyRequest:
    """One request with a fixed trained horizon and explicit runtime options."""

    observation: ObservationPacket
    policy_id: str
    request_id: str
    trained_horizon: int
    n_action_steps: int
    history_requested: bool = False
    temporal_ensemble: bool = False
    sampler_guidance: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported policy request schema version")
        if not self.policy_id or not self.request_id:
            raise ValueError("Policy and request IDs are required")
        if self.trained_horizon < 1 or not 1 <= self.n_action_steps <= self.trained_horizon:
            raise ValueError("Requested action steps exceed the trained horizon")


@dataclass(frozen=True)
class PolicyCapabilities:
    """Options that a concrete policy supports."""

    language: bool
    observation_history: bool
    temporal_ensembling: bool
    sampler_guidance: bool
    trained_horizon: int

    def validate(self, request: PolicyRequest) -> None:
        """Reject a control that this policy or checkpoint does not support."""
        if request.trained_horizon != self.trained_horizon:
            raise ValueError("Request differs from the checkpoint's trained horizon")
        if request.observation.instruction is not None and not self.language:
            raise ValueError("This policy does not support language control")
        if request.history_requested and not self.observation_history:
            raise ValueError("This policy does not support observation history")
        if request.temporal_ensemble and not self.temporal_ensembling:
            raise ValueError("This policy does not support temporal ensembling")
        if request.sampler_guidance and not self.sampler_guidance:
            raise ValueError("This policy does not support sampler guidance")


@dataclass(frozen=True)
class ActionChunk:
    """Actions aligned to the observation tick that made the request."""

    episode_id: str
    reset_counter: int
    request_id: str
    observation_tick: int
    availability_tick: int
    actions: np.ndarray
    schema: ActionSchema
    controller_period_ms: int
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.schema.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported action chunk schema version")
        if not self.episode_id or not self.request_id or self.reset_counter < 0:
            raise ValueError("Action chunk identity is invalid")
        if self.observation_tick < 0 or self.availability_tick < self.observation_tick:
            raise ValueError("Action chunk availability tick is invalid")
        if self.controller_period_ms != 50:
            raise ValueError("Action chunk must use the 50 ms LIBERO controller period")
        if (
            not isinstance(self.actions, np.ndarray)
            or self.actions.ndim != 2
            or self.actions.shape[1] != 7
            or self.actions.shape[0] < 1
        ):
            raise ValueError("Action chunk must have shape (horizon, 7)")
        if self.availability_tick >= self.observation_tick + self.actions.shape[0]:
            raise ValueError("Action chunk is available after its horizon")
        if not np.isfinite(self.actions).all():
            raise ValueError("Action chunk must have finite values")
        lower = np.asarray(self.schema.lower)
        upper = np.asarray(self.schema.upper)
        if np.any(self.actions < lower) or np.any(self.actions > upper):
            raise ValueError("Action chunk exceeds controller bounds")


@dataclass(frozen=True)
class EpisodeOutcome:
    """One explicit simulator state result for paired evaluation."""

    task_id: str
    initial_state_id: int
    partition: str
    policy_revision: str
    success_predicate_version: str
    success: bool
    termination_reason: str
    metrics: dict[str, float | int]
    artifact_paths: dict[str, str]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported episode result schema version")
        if not self.task_id or self.initial_state_id < 0 or not self.policy_revision:
            raise ValueError("Episode result identity is invalid")
        if self.partition not in ("development", "final"):
            raise ValueError("Episode result partition is invalid")
        if not self.success_predicate_version:
            raise ValueError("Episode result needs a success predicate version")
        if self.termination_reason not in (
            "success",
            "timeout",
            "policy_error",
            "infrastructure_error",
            "terminated",
            "truncated",
        ):
            raise ValueError("Episode result termination reason is invalid")
        if self.success != (self.termination_reason == "success"):
            raise ValueError("Episode success and termination reason differ")
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("Episode result has a nonfinite metric")
