"""Field types for data and result records used by the course."""

from __future__ import annotations

from typing import NotRequired, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from vla_lab.data.stats import TrainingStats

EpisodeRow = TypedDict(
    "EpisodeRow",
    {
        "episode_index": int,
        "frame_index": int,
        "action": NotRequired[ArrayLike],
        "observation.state": NotRequired[ArrayLike],
    },
)


class EpisodeArrays(TypedDict):
    episode_index: int
    frame_indices: NDArray[np.int64]
    actions: NDArray[np.float32]
    states: NDArray[np.float32]


ImageSample = TypedDict(
    "ImageSample",
    {
        "start": int,
        "observation.images.image": NDArray[np.float32],
        "observation.images.image2": NDArray[np.float32],
    },
)


class TrainingEpisode(TypedDict):
    actions: ArrayLike
    states: ArrayLike
    samples: list[ImageSample]
    frame_count: NotRequired[int]


class ACTBatchData(TypedDict):
    train: list[dict[str, Tensor]]
    validation: list[dict[str, Tensor]]
    state_stats: TrainingStats
    action_stats: TrainingStats
    training_episode_ids: list[int]
    validation_episode_ids: list[int]


class ActBatchMetadata(TypedDict):
    task_alias: str
    dataset_revision: str
    split_sha256: str
    train_episode_ids: list[int]
    validation_episode_ids: list[int]
    seed: int
    chunk_size: int
    batch_size: int
    train_rows: int
    validation_rows: int
    normalization_source: str
    device: str


class CaseResult(TypedDict):
    case_id: str | int
    state_id: int
    reset_seed: int
    task_id: str
    success: bool
    partition: NotRequired[str]
    prompt: NotRequired[str]
    action_steps: NotRequired[int]
    policy_revision: NotRequired[str]


class PairedSummary(TypedDict):
    cases: int
    case_ids: list[str | int]
    left_successes: int
    right_successes: int
    left_policy_revision: str
    right_policy_revision: str
    left_success_rate: float
    right_success_rate: float
    left_only: int
    right_only: int
    paired_difference: float
    failed_attempts_included: bool


class LatencySummary(TypedDict):
    count: int
    mean_s: float | None
    p95_s: float | None
    max_s: float | None


class LatencySamples(LatencySummary):
    samples_s: list[float]


class TimingAttempt(TypedDict):
    state_id: int
    reset_seed: int
    action_selection_api: LatencySamples
    action_generation: LatencySamples
    timing_available: bool


class TimingReport(TypedDict):
    scope: dict[str, str]
    attempts: list[TimingAttempt]
    action_selection_api: LatencySummary
    action_generation: LatencySummary


class TrialConfig(TypedDict, total=False):
    layers: int
    heads: int
    pos: bool
    mask: bool
    mem_pos: bool
    k: int
    arm: str
    lr: float
    kl_weight: float
    schedule: str
    cond: str
    film_scale: bool
    dims: str
    kernel: int
    steps: int
    clip: bool
    trainable: str
    time_prior: str


class TrialResult(TypedDict):
    config: TrialConfig
    status: str


class AlignedTransition[ObservationT](TypedDict):
    tick: int
    observation: ObservationT
    action: NDArray[np.float32]


class TransitionAlignment(TypedDict):
    aligned: bool
    offset_ticks: list[int]
    action_hz: int
    recorded_dataset_hz: int
    dataset_stride_ticks: int


class SimulationTask(TypedDict):
    suite: str
    task_id: int
    state_count: int
    states_sha256: str
    development_state_ids: list[int]
    final_state_ids: list[int]


class RecordingDefect(TypedDict):
    label: str
    kind: str
    offset_ticks: int
    source_history: str


class PolicyRecipe(TypedDict):
    checkpoint: str
    model_sha256: str
    robot_training_steps: int
    n_action_steps: int
    chunk_size: NotRequired[int]
    vlm_dir: NotRequired[str]
    vlm_sha256: NotRequired[str]
    inference_steps: NotRequired[int]
    n_obs_steps: NotRequired[int]
    horizon: NotRequired[int]


class ComparisonConfig(TypedDict):
    schema_version: int
    comparison_type: str
    partition: str
    task_alias: str
    task_prompt: str
    state_ids: list[int]
    base_seed: int
    max_steps: int
    dataset_revision: str
    train_split_sha256: str
    train_episode_count: int
    policies: dict[str, PolicyRecipe]


class ChunkTrial(TypedDict):
    execute_steps: int
    status: str
    state: int
    seed: int
    success: NotRequired[bool]
    steps: NotRequired[int]
    policy_calls: NotRequired[int]
    max_action_change: NotRequired[float]
    termination: NotRequired[str]
    error_stage: NotRequired[str]
    error: NotRequired[str]
    close_error: NotRequired[str]


class InterventionCase(TypedDict):
    case_id: str | int
    success: bool
    state_id: NotRequired[int]
    reset_seed: NotRequired[int]
    task_id: NotRequired[str]
    prompt: NotRequired[str]


class InterventionSummary(TypedDict):
    cases: int
    baseline_successes: int
    changed_data_successes: int
    gains: list[str | int]
    losses: list[str | int]
    failed_attempts_included: bool
