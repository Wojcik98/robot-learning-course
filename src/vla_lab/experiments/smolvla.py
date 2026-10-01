"""Use fixed local settings for the learner SmolVLA stages."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from vla_lab.evaluation.states import selected_state_ids


@dataclass(frozen=True)
class SmolVLATrainConfig:
    """The declared learner stages and development states."""

    task_alias: str
    seed: int
    batch_size: int
    device: str
    c1_steps: int
    c2_steps: int
    learning_rate: float
    weight_decay: float
    state_ids: tuple[int, ...]
    config_sha256: str


def load_smolvla_train_config(path: str | Path, configs_root: str | Path) -> SmolVLATrainConfig:
    """Read the two stages and reject final simulator states."""
    content = Path(path).read_bytes()
    payload = yaml.safe_load(content)
    if (
        payload.get("schema_version") != 1
        or payload.get("policy") != "smolvla_learner"
        or payload.get("profile") != "local"
    ):
        raise ValueError("SmolVLA learner needs schema version 1 and the local profile")
    task_alias = str(payload["task_alias"])
    state_ids = tuple(
        selected_state_ids(
            configs_root, task_alias, "development", payload["development_state_ids"]
        )
    )
    c1_steps = int(payload["c1_steps"])
    c2_steps = int(payload["c2_steps"])
    batch_size = int(payload["batch_size"])
    learning_rate = float(payload["learning_rate"])
    weight_decay = float(payload["weight_decay"])
    if not 0 < c1_steps < c2_steps:
        raise ValueError("c1_steps must be positive and less than c2_steps")
    if batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("batch size and learning rate must be positive")
    return SmolVLATrainConfig(
        task_alias=task_alias,
        seed=int(payload["seed"]),
        batch_size=batch_size,
        device=str(payload["device"]),
        c1_steps=c1_steps,
        c2_steps=c2_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        state_ids=state_ids,
        config_sha256=hashlib.sha256(content).hexdigest(),
    )
