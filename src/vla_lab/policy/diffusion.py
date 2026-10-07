"""Diffusion Policy settings for the pinned LIBERO demonstrations."""

import json
from pathlib import Path

import torch
import yaml
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion import DiffusionConfig, make_diffusion_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline

from vla_lab.data.stats import TrainingStats

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_diffusion_config(*, device: str = "cpu") -> DiffusionConfig:
    """Set two cameras, two observations, and a 16-step action horizon."""
    return DiffusionConfig(
        device=device,
        push_to_hub=False,
        input_features={
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.images.image2": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        drop_n_last_frames=7,
        resize_shape=(128, 128),
        crop_is_random=False,
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        down_dims=(128, 256, 512),
        num_train_timesteps=100,
        num_inference_steps=20,
        do_mask_loss_for_padding=True,
    )


def make_diffusion_processors(
    config: DiffusionConfig, task_alias: str, configs_root: str | Path
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Use frozen training extrema and fixed ImageNet channel statistics."""
    root = Path(configs_root)
    split_manifest = yaml.safe_load((root / "data" / "episodes.yaml").read_text())
    try:
        split = split_manifest["tasks"][task_alias]
    except KeyError as error:
        raise ValueError(f"Unknown task alias: {task_alias}") from error
    payload = json.loads((root / "data" / "stats" / f"{task_alias}.json").read_text())
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
        minimum = torch.tensor(modality["min"], dtype=torch.float32)
        maximum = torch.tensor(modality["max"], dtype=torch.float32)
        if minimum.shape != (dimension,) or maximum.shape != (dimension,):
            raise ValueError(f"Invalid training extrema shape for {key}")
        if not torch.isfinite(minimum).all() or not torch.isfinite(maximum).all():
            raise ValueError(f"Nonfinite training extrema for {key}")
        if not torch.all(minimum < maximum):
            raise ValueError(f"Training extrema must increase for {key}")
        stats[key] = {"min": minimum, "max": maximum}
    for key in ("observation.images.image", "observation.images.image2"):
        stats[key] = {
            "mean": torch.tensor(IMAGENET_MEAN, dtype=torch.float32),
            "std": torch.tensor(IMAGENET_STD, dtype=torch.float32),
        }
    return make_diffusion_pre_post_processors(config, dataset_stats=stats)
