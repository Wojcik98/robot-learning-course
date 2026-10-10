"""Record types for data and results used by the course.

The module uses two kinds of record:

* A **dataclass** holds values that course code builds and reads in Python. Its
  fields are attributes (``episode.actions``), so an editor can list them.
* A **TypedDict** describes a plain ``dict`` whose shape is set somewhere else:
  a row returned by LeRobot, a JSON or YAML file on disk, or a sparse table of
  experiment settings. The value stays a ``dict``, so it moves to and from
  ``json``/``yaml`` without conversion. Each TypedDict docstring says where its
  shape comes from.

Model batches stay ``dict[str, Tensor]`` because LeRobot policies read them by
feature name. :class:`FeatureKey` lists those names.

Shape symbols used below: ``L`` is the number of frames in one episode, ``B``
the batch size, ``T`` the action horizon (target slots per example), ``H`` and
``W`` the image height and width in pixels.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import NotRequired, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from vla_lab.data.stats import TrainingStats


class FeatureKey(StrEnum):
    """LeRobot feature names: keys of dataset rows and of model batches.

    A member is also a ``str``, so ``batch[FeatureKey.STATE]`` and
    ``batch["observation.state"]`` read the same entry.
    """

    IMAGE = "observation.images.image"
    """Agent-view (third-person) camera, ``float32`` CHW in ``[0, 1]``."""
    IMAGE2 = "observation.images.image2"
    """Wrist (eye-in-hand) camera, ``float32`` CHW in ``[0, 1]``."""
    STATE = "observation.state"
    """Eight robot state values; see :class:`TrainingEpisode`."""
    ACTION = "action"
    """Seven controller command values; see :class:`TrainingEpisode`."""
    ACTION_IS_PAD = "action_is_pad"
    """Boolean mask over target slots; ``True`` marks a padded, absent action."""


CAMERA_KEYS = (FeatureKey.IMAGE, FeatureKey.IMAGE2)
"""The two camera features, in the order the ACT checkpoints expect."""


class Partition(StrEnum):
    """Which group of simulator initial states an evaluation used.

    Development states may be used while choosing a method. Final states are
    opened only once, after every recipe is frozen.
    """

    DEVELOPMENT = "development"
    FINAL = "final"


# ---------------------------------------------------------------------------
# Demonstration data (lessons 3 and 4)
# ---------------------------------------------------------------------------

# The functional form is required: "observation.state" is not a valid Python
# attribute name, so it cannot be written as a class field.
EpisodeRow = TypedDict(
    "EpisodeRow",
    {
        "episode_index": int,
        "frame_index": int,
        "action": NotRequired[ArrayLike],
        "observation.state": NotRequired[ArrayLike],
    },
)
EpisodeRow.__doc__ = """One raw row of the LeRobot dataset, as returned by ``hf_dataset[i]``.

LeRobot sets this shape, so it stays a ``dict``. A row holds one frame of one
episode.

* ``episode_index`` - ID of the demonstration the row belongs to.
* ``frame_index`` - position of the row inside its episode, from 0.
* ``action`` - seven controller command values applied at this frame.
* ``observation.state`` - eight robot state values measured at this frame.
"""


@dataclass(frozen=True)
class EpisodeArrays:
    """The numeric columns of one complete episode, ordered by frame.

    Built from that episode's :data:`EpisodeRow` values by
    ``extract_episode_arrays``.

    Attributes:
        episode_index: ID of the demonstration.
        frame_indices: ``(L,)`` ``int64`` frame numbers, increasing.
        actions: ``(L, 7)`` actions; row ``i`` was applied at frame ``i``.
        states: ``(L, 8)`` robot states; row ``i`` was measured at frame ``i``.
    """

    episode_index: int
    frame_indices: NDArray[np.int64]
    actions: NDArray[np.float32]
    states: NDArray[np.float32]


@dataclass(frozen=True)
class CameraFrame:
    """Both camera images decoded at one frame of an episode.

    Each image is ``float32`` with shape ``(3, H, W)`` (channels first) and
    values in ``[0, 1]``. Images are already oriented for the model; do not
    rotate or rescale them again.

    Attributes:
        image: Agent-view camera; batch key :attr:`FeatureKey.IMAGE`.
        image2: Wrist camera; batch key :attr:`FeatureKey.IMAGE2`.
    """

    image: NDArray[np.float32]
    image2: NDArray[np.float32]


@dataclass(frozen=True)
class TrainingEpisode:
    """One demonstration prepared for ACT training.

    The numeric columns cover **every** frame, so action windows stop at the
    true end of the episode and statistics use all frames. Decoding video is
    slow, so ``images`` covers only **a few** frames. A training example can
    start only at a frame that has decoded images.

    Attributes:
        actions: ``(L, 7)`` relative controller commands, one row per frame:
            three translation channels, three rotation channels, one gripper
            channel. Values are bounded controller units, not metres or radians.
        states: ``(L, 8)`` robot states, one row per frame: end-effector
            position (3 values, metres), axis-angle orientation (3 values,
            radians), two gripper finger positions (metres).
        images: Maps a frame index ``s`` (``0 <= s < L``) to the camera pair
            decoded at that frame. Frames without an entry have no images.
    """

    actions: NDArray[np.float32]
    states: NDArray[np.float32]
    images: dict[int, CameraFrame]

    def __post_init__(self) -> None:
        if len(self.actions) != len(self.states):
            raise ValueError("actions and states must have one row per frame")
        if any(not 0 <= frame < len(self.actions) for frame in self.images):
            raise ValueError("image frame indices must lie inside the episode")

    @property
    def frame_count(self) -> int:
        """Number of frames ``L`` in the episode."""
        return len(self.actions)


@dataclass(frozen=True)
class ACTBatchData:
    """Training and validation batches for ACT, plus the statistics behind them.

    Each batch maps a :class:`FeatureKey` to a tensor:

    * camera keys - ``(B, 3, H, W)`` ``float32`` images in ``[0, 1]``;
    * ``observation.state`` - ``(B, 8)`` normalized current state;
    * ``action`` - ``(B, T, 7)`` normalized target actions;
    * ``action_is_pad`` - ``(B, T)`` ``bool``, ``True`` for padded slots.

    Attributes:
        train: Batches from training episodes.
        validation: Batches from validation episodes, normalized with the
            training statistics.
        state_stats: Mean and standard deviation of states, fitted on
            training frames only.
        action_stats: Mean and standard deviation of actions, fitted on
            training frames only.
        training_episode_ids: Sorted IDs of the training episodes.
        validation_episode_ids: Sorted IDs of the validation episodes.
    """

    train: list[dict[str, Tensor]]
    validation: list[dict[str, Tensor]]
    state_stats: TrainingStats
    action_stats: TrainingStats
    training_episode_ids: list[int]
    validation_episode_ids: list[int]


class ActBatchMetadata(TypedDict):
    """Provenance of a saved ACT batch file; written next to it as JSON.

    * ``task_alias`` - short task name, e.g. ``"bowl_plate"``.
    * ``dataset_revision`` - pinned LeRobot dataset revision.
    * ``split_sha256`` - hash of the frozen train/validation split.
    * ``train_episode_ids``, ``validation_episode_ids`` - episodes used.
    * ``seed`` - random seed for batch order.
    * ``chunk_size`` - action horizon ``T``.
    * ``batch_size`` - examples per batch ``B``.
    * ``train_rows``, ``validation_rows`` - number of examples per split.
    * ``normalization_source`` - where the statistics came from.
    * ``device`` - device the batch tensors were created on.
    """

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


# ---------------------------------------------------------------------------
# Evaluation results (lesson 22)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class CaseResult:
    """The outcome of one policy rollout on one evaluation case.

    A case is one simulator initial state with one reset seed. Two methods are
    compared only on identical cases; the optional fields carry that identity.

    Attributes:
        case_id: Unique case name, e.g. ``"state-20-seed-20260950"``.
        state_id: Index of the simulator initial state.
        reset_seed: Seed passed to the environment reset.
        task_id: Task name.
        success: ``True`` only if the task was completed.
        partition: Development or final states.
        prompt: Language instruction given to the policy, if any.
        action_steps: Actions executed from each predicted chunk.
        policy_revision: Identifier of the checkpoint that produced the result.
    """

    case_id: str | int
    state_id: int
    reset_seed: int
    task_id: str
    success: bool
    partition: Partition | None = None
    prompt: str | None = None
    action_steps: int | None = None
    policy_revision: str | None = None


@dataclass(frozen=True)
class PairedSummary:
    """Two policies compared on the same cases; failed rollouts are counted.

    "Left" and "right" are the two arguments of ``paired_summary``.

    Attributes:
        cases: Number of paired cases ``n``.
        case_ids: The cases, in the requested order.
        left_successes, right_successes: Successful rollouts per policy.
        left_policy_revision, right_policy_revision: Checkpoint identifiers.
        left_success_rate, right_success_rate: Successes divided by ``n``.
        left_only: Cases where only the left policy succeeded.
        right_only: Cases where only the right policy succeeded.
        paired_difference: ``(left_successes - right_successes) / n``.
        failed_attempts_included: Always ``True``; failures are never dropped.
    """

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
    """Summary of measured call times; part of ``inference-timing.json``.

    All times are seconds. The time values are ``None`` when ``count`` is 0.

    * ``count`` - number of measured calls.
    * ``mean_s`` - mean call time.
    * ``p95_s`` - 95th percentile call time.
    * ``max_s`` - slowest call.
    """

    count: int
    mean_s: float | None
    p95_s: float | None
    max_s: float | None


class LatencySamples(LatencySummary):
    """A :class:`LatencySummary` together with every measured time.

    * ``samples_s`` - each call time in seconds, in call order.
    """

    samples_s: list[float]


class TimingAttempt(TypedDict):
    """Call times for one rollout; one entry of ``inference-timing.json``.

    * ``state_id``, ``reset_seed`` - the evaluated case.
    * ``action_selection_api`` - times of ``policy.select_action``, which
      often returns a queued action without running the model.
    * ``action_generation`` - times of the model call that predicts a chunk.
    * ``timing_available`` - ``False`` if no call was measured.
    """

    state_id: int
    reset_seed: int
    action_selection_api: LatencySamples
    action_generation: LatencySamples
    timing_available: bool


class TimingReport(TypedDict):
    """Contents of ``inference-timing.json`` for one method.

    * ``scope`` - plain-language description of what each timer measures.
    * ``attempts`` - one :class:`TimingAttempt` per rollout.
    * ``action_selection_api``, ``action_generation`` - summaries over all
      calls of all rollouts.
    """

    scope: dict[str, str]
    attempts: list[TimingAttempt]
    action_selection_api: LatencySummary
    action_generation: LatencySummary


# ---------------------------------------------------------------------------
# Architecture trial tables (lessons 6 to 8)
# ---------------------------------------------------------------------------


class AssemblyArm(StrEnum):
    """Model variants compared in lesson 8; each removes one input path."""

    FULL = "full"
    """Your hooks unchanged: both cameras with positions, state, and latent."""
    NO_CAM_POS = "no_cam_pos"
    """Camera cells get no positional embedding."""
    ONE_CAMERA = "one_camera"
    """Only the first (agent-view) camera."""
    STATE_ONLY = "state_only"
    """No camera tokens; the encoder sees only the latent and the state."""
    CAMERAS_ONLY = "cameras_only"
    """The robot state is replaced by zeros."""


class TrialConfig(TypedDict, total=False):
    """Settings of one small architecture trial; every key is optional.

    Each lesson varies a different subset, so a trial lists only the keys it
    sets. The dict is spread into a pandas table row, where unset keys stay
    blank instead of showing ``None``.

    * ``layers`` - number of transformer layers.
    * ``heads`` - attention heads per layer (lesson 6).
    * ``pos`` - add positional encodings to the encoder input (lesson 6).
    * ``mask`` - pass the padding mask to attention (lessons 6 and 7).
    * ``mem_pos`` - add learned positions to decoder memory (lesson 7).
    * ``k`` - use only the newest ``k`` history states, 1 to 8 (lesson 7).
    * ``arm`` - model variant (lesson 8).
    * ``lr`` - learning rate (lesson 8).
    * ``kl_weight`` - weight of the CVAE KL loss term (lesson 8).
    """

    layers: int
    heads: int
    pos: bool
    mask: bool
    mem_pos: bool
    k: int
    arm: AssemblyArm
    lr: float
    kl_weight: float


class TrialResult(TypedDict):
    """One row of a trial table.

    Measured values (losses, times) are added as extra keys next to these.

    * ``config`` - the trial settings.
    * ``status`` - ``"ok"``, or ``"failed: <error>"`` with the error text.
    """

    config: TrialConfig
    status: str


# ---------------------------------------------------------------------------
# Recording quality and data interventions (lesson 21)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignedTransition[ObservationT]:
    """One recorded step: the observation seen, then the action applied.

    Attributes:
        tick: Control tick at which the action was applied; one tick is one
            0.05 s controller step.
        observation: The observation recorded at the same tick, before the
            action.
        action: ``(7,)`` ``float32`` controller command.
    """

    tick: int
    observation: ObservationT
    action: NDArray[np.float32]


@dataclass(frozen=True)
class TransitionAlignment:
    """Whether each action in a recording uses the tick of its observation.

    For ``N`` actions there are ``N + 1`` observations; action ``i`` should
    carry the tick of observation ``i``.

    Attributes:
        aligned: ``True`` if every action tick equals its observation tick.
        offset_ticks: Each distinct ``action_tick - observation_tick``,
            sorted. ``[0]`` for a clean recording.
        control_hz: Controller rate (20 Hz in LIBERO).
        source_storage_fps: Frame rate label stored in the source dataset;
            it is not the control rate.
        simulated_duration_s: ``N / control_hz`` seconds.
    """

    aligned: bool
    offset_ticks: list[int]
    control_hz: int
    source_storage_fps: int
    simulated_duration_s: float


@dataclass(frozen=True, kw_only=True)
class InterventionCase:
    """The outcome of one development case before or after a data change.

    The identity fields are optional; when either run sets one, both runs must
    set it to the same value.

    Attributes:
        case_id: Unique case name.
        success: ``True`` only if the task was completed.
        state_id: Index of the simulator initial state.
        reset_seed: Seed passed to the environment reset.
        task_id: Task name.
        prompt: Language instruction, if any.
    """

    case_id: str | int
    success: bool
    state_id: int | None = None
    reset_seed: int | None = None
    task_id: str | None = None
    prompt: str | None = None


@dataclass(frozen=True)
class InterventionSummary:
    """Effect of a data change, compared on the same development cases.

    Attributes:
        cases: Number of compared cases.
        baseline_successes: Successes before the change.
        changed_data_successes: Successes after the change.
        gains: Cases that fail before and succeed after.
        losses: Cases that succeed before and fail after.
        failed_attempts_included: Always ``True``; failures are never dropped.
    """

    cases: int
    baseline_successes: int
    changed_data_successes: int
    gains: list[str | int]
    losses: list[str | int]
    failed_attempts_included: bool


class RecordingDefect(TypedDict):
    """Label of a deliberate timing defect; saved in a recording's metadata.

    * ``label`` - always ``"synthetic timing defect"``.
    * ``kind`` - always ``"action_tick_offset"``.
    * ``offset_ticks`` - how many ticks every action tick was shifted.
    * ``source_history`` - free text on where the defect came from.
    """

    label: str
    kind: str
    offset_ticks: int
    source_history: str


# ---------------------------------------------------------------------------
# Configuration files
# ---------------------------------------------------------------------------


class SimulationTask(TypedDict):
    """One task entry of ``configs/eval/splits.yaml``.

    * ``suite`` - LIBERO suite name.
    * ``task_id`` - task index inside the suite.
    * ``state_count`` - number of stored initial states.
    * ``states_sha256`` - hash of the initial-state file.
    * ``development_state_ids`` - states usable while choosing methods.
    * ``final_state_ids`` - states reserved for the final comparison.
    """

    suite: str
    task_id: int
    state_count: int
    states_sha256: str
    development_state_ids: list[int]
    final_state_ids: list[int]


class PolicyRecipe(TypedDict):
    """One frozen policy in a comparison YAML file.

    * ``checkpoint`` - directory of the saved policy.
    * ``model_sha256`` - hash of its ``model.safetensors``.
    * ``robot_training_steps`` - optimizer steps on robot data.
    * ``n_action_steps`` - actions executed from each predicted chunk.
    * ``chunk_size`` - actions predicted per call (ACT and SmolVLA).
    * ``vlm_dir``, ``vlm_sha256`` - vision-language backbone (SmolVLA).
    * ``inference_steps`` - denoising steps (Diffusion Policy).
    * ``n_obs_steps`` - observation frames per input (Diffusion Policy).
    * ``horizon`` - predicted action steps (Diffusion Policy).
    """

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
    """Contents of a comparison YAML file in ``configs/experiments``.

    * ``schema_version`` - file format version, currently 1.
    * ``comparison_type`` - currently ``"practical_recipe"``.
    * ``partition`` - currently ``"final"``.
    * ``task_alias``, ``task_prompt`` - task name and its instruction.
    * ``state_ids`` - evaluated initial states, in order.
    * ``base_seed`` - reset seed of state ``i`` is ``base_seed + i``.
    * ``max_steps`` - control ticks before a rollout times out.
    * ``dataset_revision``, ``train_split_sha256``, ``train_episode_count`` -
      the training data every policy must have used.
    * ``policies`` - method name to its :class:`PolicyRecipe`.
    """

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
