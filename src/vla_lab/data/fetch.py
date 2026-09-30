"""Fetch pinned public LIBERO files for the first lab and training tasks."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download


def dataset_patterns(scope: str) -> list[str]:
    """Select one demo frame or the data and camera pairs for training."""
    sample = "videos/observation.images.image/chunk-000/file-000.mp4"
    if scope == "demo":
        return [sample]
    if scope != "training":
        raise ValueError("Data scope must be demo or training")
    patterns = ["data/**", "meta/**"]
    for camera in ("image", "image2"):
        for index in (0, *range(10, 19)):
            patterns.append(f"videos/observation.images.{camera}/chunk-000/file-{index:03d}.mp4")
    return patterns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_local_data(
    workspace: str | Path,
    data_root: str | Path,
    *,
    scope: str,
    offline: bool = False,
) -> dict[str, Any]:
    """Download pinned public assets without moving the user's login store."""
    root = Path(workspace).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    patterns = dataset_patterns(scope)
    data.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(data).free / 2**30
    if free_gib < (1 if scope == "demo" else 2):
        raise RuntimeError(f"Not enough free space for {scope} data: {free_gib:.1f} GiB")
    core = yaml.safe_load((root / "manifests" / "core.yaml").read_text())
    dataset = core["dataset"]
    assets = core["libero_assets"]
    cache = data / "hf" / "hub"
    paths = {
        "dataset": data / "hub" / "lerobot-libero",
        "assets": data / "libero-assets",
    }
    if not offline:
        snapshot_download(
            repo_id=dataset["repository"],
            repo_type="dataset",
            revision=dataset["revision"],
            local_dir=paths["dataset"],
            cache_dir=cache,
            allow_patterns=patterns,
            max_workers=4,
        )
        snapshot_download(
            repo_id=assets["repository"],
            repo_type="dataset",
            revision=assets["revision"],
            local_dir=paths["assets"],
            cache_dir=cache,
            max_workers=4,
        )
    sample_path = dataset_patterns("demo")[0]
    sample = paths["dataset"] / sample_path
    expected = core["dataset"]["sample_files"][sample_path]
    if not sample.is_file() or _sha256(sample) != expected:
        raise ValueError("Pinned AV1 sample file is missing or changed")
    wall = paths["assets"] / "wall.xml"
    if not wall.is_file() or _sha256(wall) != assets["wall_xml_sha256"]:
        raise ValueError("Pinned LIBERO simulator assets are missing or changed")
    if scope == "training":
        for path, digest in dataset["sample_files"].items():
            if (
                not (paths["dataset"] / path).is_file()
                or _sha256(paths["dataset"] / path) != digest
            ):
                raise ValueError(f"Pinned training sample is missing or changed: {path}")
        for path in patterns:
            if path.endswith(".mp4") and not (paths["dataset"] / path).is_file():
                raise ValueError(f"Selected training video is missing: {path}")
        for metadata in (
            "meta/info.json",
            "meta/tasks.parquet",
            "meta/episodes/chunk-000/file-000.parquet",
        ):
            if not (paths["dataset"] / metadata).is_file():
                raise ValueError(f"Training metadata is missing: {metadata}")
    return {
        "scope": scope,
        "dataset_revision": dataset["revision"],
        "asset_revision": assets["revision"],
        "data_root": str(data),
        "sample_sha256": expected,
        "offline": offline,
    }
