"""Fit and store statistics from training episodes only."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray


class StatsPayload(TypedDict):
    mean: list[float]
    std: list[float]
    sha256: str


def _canonical_payload(mean: list[float], std: list[float]) -> bytes:
    return json.dumps({"mean": mean, "std": std}, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class TrainingStats:
    """Per-feature mean and standard deviation from training samples."""

    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    sha256: str

    @classmethod
    def fit[EpisodeId](
        cls,
        training_episodes: Mapping[EpisodeId, ArrayLike],
        *,
        validation_episodes: Mapping[EpisodeId, ArrayLike] | None = None,
    ) -> "TrainingStats":
        """Fit population statistics; validation samples do not enter the fit."""
        del validation_episodes
        if not training_episodes:
            raise ValueError("at least one training episode is required")
        arrays = [np.asarray(value, dtype=np.float64) for value in training_episodes.values()]
        if any(array.ndim < 1 for array in arrays):
            raise ValueError("each episode must have a feature dimension")
        feature_sizes = {array.shape[-1] for array in arrays}
        if len(feature_sizes) != 1:
            raise ValueError("all episodes must have the same feature dimension")
        samples = np.concatenate([array.reshape(-1, array.shape[-1]) for array in arrays])
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError("training samples must be non-empty and finite")
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        std[std == 0] = 1.0
        mean = mean.astype(np.float64)
        std = std.astype(np.float64)
        digest = hashlib.sha256(_canonical_payload(mean.tolist(), std.tolist())).hexdigest()
        return cls(mean=mean, std=std, sha256=digest)

    def normalize(self, values: ArrayLike) -> NDArray[np.float64]:
        """Normalize values with these training statistics."""
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def denormalize(self, values: ArrayLike) -> NDArray[np.float64]:
        """Convert normalized values back to the input scale."""
        return np.asarray(values, dtype=np.float64) * self.std + self.mean

    def to_dict(self) -> StatsPayload:
        """Return checkpoint-safe statistics and their hash."""
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "sha256": self.sha256}

    @classmethod
    def from_dict(cls, payload: StatsPayload) -> "TrainingStats":
        """Load checkpoint statistics and verify their content hash."""
        mean = np.asarray(payload["mean"], dtype=np.float64)
        std = np.asarray(payload["std"], dtype=np.float64)
        if mean.ndim != 1 or std.shape != mean.shape:
            raise ValueError("mean and std must be one-dimensional arrays of equal size")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("mean and std must be finite and std must be positive")
        digest = hashlib.sha256(_canonical_payload(mean.tolist(), std.tolist())).hexdigest()
        if payload.get("sha256") != digest:
            raise ValueError("statistics hash does not match its values")
        return cls(mean=mean, std=std, sha256=digest)
