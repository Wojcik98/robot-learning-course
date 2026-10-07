"""Fetch and verify the pinned public SmolVLA model files."""

import hashlib
import re
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

from vla_lab.types import JSONValue

MODEL_FILES = {
    "published_reference": (
        "smolvla-libero",
        (
            "model.safetensors",
            "config.json",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
            "policy_postprocessor_step_1_unnormalizer_processor.safetensors",
        ),
    ),
    "learner_start": (
        "smolvlm2-500m-video-instruct",
        ("model.safetensors", "config.json", "tokenizer.json", "processor_config.json"),
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_pinned_models(
    workspace: str | Path, data_root: str | Path, *, offline: bool = False
) -> dict[str, JSONValue]:
    """Download exact Hub commits or check the files that are already local."""
    root = Path(workspace).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    manifest = yaml.safe_load((root / "manifests" / "smolvla_reference.yaml").read_text())
    cache = data / "hf" / "hub"
    report: dict[str, JSONValue] = {"offline": offline, "models": {}}
    for key, (name, required) in MODEL_FILES.items():
        source = manifest[key]
        revision = str(source["revision"])
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError(f"{key} needs a pinned 40-character Hub commit")
        destination = data / "models" / name
        if not offline:
            destination.mkdir(parents=True, exist_ok=True)
            cache.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=str(source["repo"]),
                revision=revision,
                local_dir=destination,
                cache_dir=cache,
                max_workers=4,
            )
        missing = [filename for filename in required if not (destination / filename).is_file()]
        if missing:
            raise FileNotFoundError(f"Pinned {key} files are missing: {', '.join(missing)}")
        model_hash = _sha256(destination / "model.safetensors")
        if model_hash != source["model_sha256"]:
            raise ValueError(f"Pinned {key} model weights differ from the manifest hash")
        report["models"][key] = {
            "repo": source["repo"],
            "revision": revision,
            "directory": str(destination),
            "model_sha256": model_hash,
            "required_files": list(required),
        }
    return report
