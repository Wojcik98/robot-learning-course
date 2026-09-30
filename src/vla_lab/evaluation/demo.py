"""Run one live ACT demonstration and save its video and run record."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from vla_lab.policy.bundle import load_act_bundle
from vla_lab.sim.episode import open_libero_task, run_act_episode


def encode_rgb_video(frames: Sequence[np.ndarray], output: str | Path, *, fps: int = 20) -> Path:
    """Encode equally sized RGB frames as an MP4 file at the control rate."""
    if fps < 1 or not frames:
        raise ValueError("Video needs frames and a positive frame rate")
    height, width, channels = frames[0].shape
    if channels != 3 or width % 2 or height % 2:
        raise ValueError("Video frames must have three channels and even dimensions")
    if any(frame.shape != (height, width, 3) for frame in frames):
        raise ValueError("Video frames must all have the same size")
    if any(frame.dtype != np.uint8 for frame in frames):
        raise ValueError("Video frames must use uint8 RGB values")
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"Video output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ],
        input=b"".join(np.ascontiguousarray(frame).tobytes() for frame in frames),
        check=True,
        capture_output=True,
    )
    return target


def run_act_demo(
    *,
    task_alias: str,
    checkpoint: str | Path,
    state_id: int,
    configs_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Reset the selected development state, run ACT, and save a live video."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Demo output directory is not empty: {output}")
    checkpoint_path = Path(checkpoint)
    policy, preprocessor, postprocessor, bundle = load_act_bundle(checkpoint_path)
    if bundle["task_alias"] != task_alias:
        raise ValueError("ACT checkpoint task differs from the selected simulator task")
    output.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    environment = open_libero_task(task_alias, state_id, configs_root, data_root)
    try:
        started = time.monotonic()
        result = run_act_episode(
            environment,
            policy.eval(),
            preprocessor,
            postprocessor,
            task_alias,
            state_id,
            configs_root,
            on_frame=frames.append,
        )
        elapsed = time.monotonic() - started
    finally:
        environment.close()
    video = encode_rgb_video(frames, output / "rollout.mp4", fps=20)
    model_hash = hashlib.sha256((checkpoint_path / "model.safetensors").read_bytes()).hexdigest()
    record: dict[str, object] = {
        "schema_version": 1,
        "policy": "act",
        "task_alias": task_alias,
        "state_id": state_id,
        "state_partition": "development",
        "success": result.success,
        "steps": result.steps,
        "clipped_values": result.clipped_values,
        "elapsed_s": round(elapsed, 3),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_model_sha256": model_hash,
        "checkpoint_manifest": bundle,
        "controller_hz": 20,
        "control_mode": "relative",
        "video": str(video.resolve()),
        "video_frames": len(frames),
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
    }
    (output / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record
