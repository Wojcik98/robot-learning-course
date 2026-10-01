"""ACT policy settings for the pinned LIBERO tasks."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act import ACTConfig, make_act_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline

from vla_lab.data.stats import TrainingStats


def make_act_config(*, device: str = "cpu", chunk_size: int = 20) -> ACTConfig:
    """Set the LIBERO camera, state, action, and small local model shapes."""
    if chunk_size < 1:
        raise ValueError("ACT chunk size must be positive")
    return ACTConfig(
        device=device,
        push_to_hub=False,
        input_features={
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.images.image2": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        chunk_size=chunk_size,
        n_action_steps=min(10, chunk_size),
        dim_model=256,
        n_heads=4,
        dim_feedforward=1024,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        pretrained_backbone_weights=None,
    )


def make_act_processors(
    config: ACTConfig, task_alias: str, configs_root: str | Path
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Load verified training statistics into LeRobot's ACT pipelines."""
    root = Path(configs_root)
    split_manifest = yaml.safe_load((root / "data" / "episodes.yaml").read_text())
    try:
        split = split_manifest["tasks"][task_alias]
    except KeyError as error:
        raise ValueError(f"Unknown task alias: {task_alias}") from error
    stats_path = root / "data" / "stats" / f"{task_alias}.json"
    payload = json.loads(stats_path.read_text())
    if payload["split_sha256"] != split["split_sha256"]:
        raise ValueError(
            "Training statistics split hash differs from the frozen episode split hash"
        )
    if payload["dataset_revision"] != split_manifest["dataset_revision"]:
        raise ValueError("Training statistics dataset revision differs from the split revision")

    stats: dict[str, dict[str, torch.Tensor]] = {}
    for key, dimension in (("observation.state", 8), ("action", 7)):
        modality = payload["modalities"][key]
        verified = TrainingStats.from_dict(modality)
        if verified.mean.shape != (dimension,):
            raise ValueError(f"Invalid training statistics shape for {key}")
        stats[key] = {
            "mean": torch.tensor(verified.mean, dtype=torch.float32),
            "std": torch.tensor(verified.std, dtype=torch.float32),
        }
    return make_act_pre_post_processors(config, dataset_stats=stats)
