"""Deterministic episode-level data splits."""

import hashlib
import json
from collections.abc import Hashable, Iterable
from dataclasses import dataclass


def partition_sha256(train_ids: Iterable[Hashable], validation_ids: Iterable[Hashable]) -> str:
    """Hash two ordered lists of complete episode IDs."""
    content = {"train_ids": list(train_ids), "validation_ids": list(validation_ids)}
    try:
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise TypeError("episode IDs must be JSON serializable") from error
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EpisodeSplit:
    """A fixed split of complete episodes and its content hash."""

    train_ids: tuple[Hashable, ...]
    validation_ids: tuple[Hashable, ...]
    sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe manifest."""
        return {
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "sha256": self.sha256,
        }


def split_episodes(episode_ids: Iterable[Hashable], train_fraction: float = 0.8) -> EpisodeSplit:
    """Split complete episode IDs in input order with at least one validation ID."""
    ids = tuple(episode_ids)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be greater than 0 and less than 1")
    try:
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate episode IDs are not allowed")
    except TypeError as error:
        raise TypeError("episode IDs must be hashable") from error
    if not ids:
        raise ValueError("at least one episode ID is required")

    train_count = min(len(ids) - 1, int(len(ids) * train_fraction))
    train_ids = ids[:train_count]
    validation_ids = ids[train_count:]
    return EpisodeSplit(train_ids, validation_ids, partition_sha256(train_ids, validation_ids))
