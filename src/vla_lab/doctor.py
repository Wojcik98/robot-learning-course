"""Check the local files and runtime needed for the first ACT lab."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch
import yaml
from torchcodec.decoders import VideoDecoder

from vla_lab.types import JSONValue


def _result(name: str, ok: bool, detail: str) -> dict[str, str | bool]:
    return {"name": name, "ok": ok, "detail": detail}


def check_act_references(
    workspace: Path, manifest: dict[str, JSONValue]
) -> list[dict[str, str | bool]]:
    """Check each complete ACT bundle against its recorded file hashes."""
    results = []
    for task, item in manifest["tasks"].items():
        root = workspace / item["checkpoint_path"]
        for filename, expected in item["checkpoint_files_sha256"].items():
            path = root / filename
            if not path.is_file():
                results.append(_result(f"act:{task}:{filename}", False, f"missing: {path}"))
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            results.append(
                _result(
                    f"act:{task}:{filename}",
                    actual == expected,
                    "hash matches" if actual == expected else f"hash differs: {path}",
                )
            )
    return results


def check_data_files(data_root: Path) -> list[dict[str, str | bool]]:
    """Check pinned simulator assets and decode one real dataset frame."""
    assets = data_root / "libero-assets" / "wall.xml"
    results = [_result("libero-assets", assets.is_file(), str(assets))]
    video_root = data_root / "hub" / "lerobot-libero" / "videos"
    sample = next(video_root.rglob("*.mp4"), None) if video_root.is_dir() else None
    if sample is None:
        results.append(_result("av1-frame", False, f"No dataset video in {video_root}"))
    else:
        try:
            frame = VideoDecoder(str(sample))[0].data
            results.append(_result("av1-frame", tuple(frame.shape) == (3, 256, 256), str(sample)))
        except Exception as error:  # noqa: BLE001
            results.append(_result("av1-frame", False, f"{type(error).__name__}: {error}"))
    return results


def check_local_runtime(workspace: Path, data_root: Path) -> list[dict[str, str | bool]]:
    """Check disk space, video encoding, CUDA, and required local files."""
    workspace = workspace.resolve()
    data_root = data_root.resolve()
    results: list[dict[str, str | bool]] = []
    data_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(data_root).free / 2**30
    results.append(_result("free-disk", free_gib >= 5, f"{free_gib:.1f} GiB free"))
    results.append(_result("ffmpeg", shutil.which("ffmpeg") is not None, "MP4 encoder command"))
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is not available")
        device_name = torch.cuda.get_device_name(0)
        tensor = torch.ones((16, 16), device="cuda")
        if not torch.isfinite(tensor @ tensor).all().item():
            raise RuntimeError("CUDA matrix result is not finite")
        torch.cuda.synchronize()
        results.append(_result("cuda", True, device_name))
    except Exception as error:  # noqa: BLE001
        results.append(_result("cuda", False, f"{type(error).__name__}: {error}"))
    results.extend(check_data_files(data_root))
    path = workspace / "manifests" / "act_references.yaml"
    if not path.is_file():
        results.append(_result("act-manifest", False, f"missing: {path}"))
    else:
        results.extend(check_act_references(workspace, yaml.safe_load(path.read_text())))
    return results


def check_live_renderer(workspace: Path, data_root: Path) -> dict[str, str | bool]:
    """Reset one development state and inspect a rendered camera image."""
    from vla_lab.sim.episode import open_libero_task

    environment = None
    try:
        environment = open_libero_task("bowl_plate", 0, workspace / "configs", data_root)
        observation, _ = environment.reset()
        image = observation["pixels"]["image"]
        if tuple(image.shape) != (256, 256, 3):
            raise ValueError(f"Unexpected camera shape: {image.shape}")
        if int(image.max()) == int(image.min()):
            raise ValueError("Rendered image is constant")
        return _result("egl-reset", True, f"camera shape {image.shape}")
    except Exception as error:  # noqa: BLE001
        return _result("egl-reset", False, f"{type(error).__name__}: {error}")
    finally:
        if environment is not None:
            environment.close()


def doctor_report(workspace: Path, data_root: Path, *, live: bool = False) -> dict[str, JSONValue]:
    """Return a structured check report for a local course launch."""
    checks = check_local_runtime(workspace, data_root)
    if live and all(item["ok"] for item in checks):
        checks.append(check_live_renderer(workspace, data_root))
    return {
        "schema_version": 1,
        "profile": "local",
        "ok": all(x["ok"] for x in checks),
        "checks": checks,
    }


def report_json(report: dict[str, JSONValue]) -> str:
    """Format a doctor report for scripts and notebook tools."""
    return json.dumps(report, indent=2, sort_keys=True)
