"""Shared contracts for the small examples and the real policy paths."""

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict, TypeVar, Unpack

import numpy as np
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

from vla_lab.data.stats import TrainingStats
from vla_lab.teaching.records import AlignedTransition
from vla_lab.types import JSONValue

if TYPE_CHECKING:
    from vla_lab.teaching.diffusion import DDPMSchedule

type ACTOutput = tuple[Tensor, tuple[Tensor | None, Tensor | None]]

type TensorBatch = dict[str, Tensor]
type ActionValue = Tensor | NDArray[np.float32]
type ProcessorObservation = dict[str, Tensor | str | list[str]]
ObservationT = TypeVar("ObservationT")
Observation_co = TypeVar("Observation_co", covariant=True)
PolicyT = TypeVar("PolicyT", bound="ResettablePolicy")
ModelT = TypeVar("ModelT", bound=nn.Module)
Model_contra = TypeVar("Model_contra", bound=nn.Module, contravariant=True)
BatchT = TypeVar("BatchT")
ActionT = TypeVar("ActionT", Tensor, NDArray[np.float32])
type FlowSampler = Callable[[Callable[[Tensor, Tensor], Tensor], Tensor, int], Tensor]


class ResettablePolicy(Protocol):
    def reset(self) -> None: ...


class ActionChunkPolicy(ResettablePolicy, Protocol):
    def predict_action_chunk(self, batch: TensorBatch) -> Tensor: ...


class ActionPolicy(ResettablePolicy, Protocol):
    def select_action(self, batch: TensorBatch) -> Tensor: ...


class ObservationPolicy(ResettablePolicy, Protocol):
    def select_action(self, batch: ProcessorObservation) -> Tensor: ...


class ACTPredictionModel(Protocol):
    def forward(self, batch: Mapping[str, Tensor]) -> ACTOutput: ...

    def __call__(self, batch: Mapping[str, Tensor], /) -> ACTOutput: ...


class DiffusionTrainingModel(Protocol):
    @property
    def schedule(self) -> "DDPMSchedule": ...

    def __call__(
        self, noisy_actions: Tensor, timesteps: Tensor, batch: Mapping[str, Tensor], /
    ) -> Tensor: ...


class NoiseScheduler(Protocol):
    @property
    def betas(self) -> Tensor: ...

    @property
    def alphas_cumprod(self) -> Tensor: ...


class ResettableEnvironment(Protocol[Observation_co]):
    def reset(self, seed: int | None = None) -> tuple[Observation_co, Mapping[str, JSONValue]]: ...


class EpisodeEnvironment(ResettableEnvironment[Observation_co], Protocol[Observation_co]):
    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[Observation_co, float, bool, bool, Mapping[str, JSONValue]]: ...


class ObservationProcessor(Protocol):
    def __call__(self, observation: ProcessorObservation, /) -> ProcessorObservation: ...


class ActionProcessor(Protocol):
    def __call__(self, action: Tensor, /) -> Tensor: ...


class LossCallback(Protocol[Model_contra]):
    def __call__(
        self,
        model: Model_contra,
        batch: Mapping[str, Tensor],
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, Mapping[str, Tensor | float]]: ...


class PromptTokenizer(Protocol):
    def __call__(
        self,
        texts: list[str],
        *,
        max_length: int,
        truncation: bool,
        padding: str,
        padding_side: str,
        return_tensors: str,
    ) -> Mapping[str, Tensor]: ...


class VLAInputConfig(Protocol):
    @property
    def resize_imgs_with_padding(self) -> tuple[int, int] | None: ...

    @property
    def image_features(self) -> Mapping[str, object]: ...

    @property
    def tokenizer_max_length(self) -> int: ...

    @property
    def max_state_dim(self) -> int: ...


class VLAInputs(TypedDict):
    images: list[Tensor]
    img_masks: list[Tensor]
    lang_tokens: Tensor
    lang_masks: Tensor
    state: Tensor
    actions: NotRequired[Tensor]


class EndEffectorState(TypedDict):
    pos: ArrayLike
    quat: ArrayLike


class GripperState(TypedDict):
    qpos: ArrayLike


class RobotState(TypedDict):
    eef: EndEffectorState
    gripper: GripperState


class CameraImages(TypedDict):
    image: NDArray[np.uint8]
    image2: NDArray[np.uint8]


class LiberoObservation(TypedDict):
    pixels: CameraImages
    robot_state: RobotState


class WindowFunction(Protocol):
    def __call__(
        self,
        episode_actions: ArrayLike,
        *,
        start: int,
        horizon: int,
        action_dim: int | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]: ...


class StatisticsFunction(Protocol):
    def __call__(
        self,
        training_episodes: Mapping[int, ArrayLike],
        *,
        validation_episodes: Mapping[int, ArrayLike] | None = None,
    ) -> TrainingStats: ...


class ManagedEnvironment(EpisodeEnvironment[Observation_co], Protocol[Observation_co]):
    def close(self) -> None: ...


type RawObservationValue = (
    NDArray[np.generic] | int | float | bool | Mapping[str, "RawObservationValue"]
)
type RawObservation = dict[str, RawObservationValue]
type RecordedObservation = dict[str, NDArray[np.generic] | "RecordedObservation"]


class TransitionRowsFunction(Protocol):
    def __call__(
        self,
        observations: Sequence[RecordedObservation],
        actions: ArrayLike,
        observation_ticks: Sequence[int] | NDArray[np.int64],
        action_ticks: Sequence[int] | NDArray[np.int64],
    ) -> list[AlignedTransition[RecordedObservation]]: ...


class RTCOptions(TypedDict, total=False):
    prev_chunk_left_over: Tensor
    inference_delay: int
    execution_horizon: int


class RTCChunkOperation(Protocol):
    def __call__(
        self,
        batch: ProcessorObservation,
        noise: Tensor | None = None,
        **kwargs: Unpack[RTCOptions],
    ) -> Tensor: ...
