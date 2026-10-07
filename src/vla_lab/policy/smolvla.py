"""Load the pinned published SmolVLA reference from local files."""

from pathlib import Path

from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline, load_pretrained_policy_processors


def load_published_smolvla(
    model_dir: str | Path,
    vlm_dir: str | Path,
    *,
    device: str = "cuda",
    n_action_steps: int = 1,
) -> tuple[SmolVLAPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Use the saved policy weights and saved normalization with a local tokenizer."""
    if not 1 <= n_action_steps <= 50:
        raise ValueError("n_action_steps must be from 1 to 50")
    model_path = Path(model_dir).resolve()
    vlm_path = Path(vlm_dir).resolve()
    for required in (
        model_path / "model.safetensors",
        model_path / "policy_preprocessor.json",
        model_path / "policy_postprocessor.json",
        vlm_path / "config.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Required SmolVLA file is missing: {required}")

    config = SmolVLAConfig.from_pretrained(str(model_path))
    if config.chunk_size != 50:
        raise ValueError("Published SmolVLA chunk size differs from the pinned recipe")
    config.vlm_model_name = str(vlm_path)
    config.load_vlm_weights = False
    config.push_to_hub = False
    config.device = device
    config.n_action_steps = n_action_steps
    policy = (
        SmolVLAPolicy.from_pretrained(
            str(model_path), config=config, strict=True, local_files_only=True
        )
        .eval()
        .to(device)
    )
    preprocessor, postprocessor = load_pretrained_policy_processors(
        str(model_path),
        preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": str(vlm_path)}},
    )
    return policy, preprocessor, postprocessor
