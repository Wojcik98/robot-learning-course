"""Run one live ACT demonstration and save its video and run record."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from vla_lab.evaluation.errors import EvaluationRunError
from vla_lab.policy.bundle import load_act_bundle
from vla_lab.sim.episode import PolicyActionError, open_libero_task, run_act_episode


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
    environment = None
    failure: Exception | None = None
    result = None
    started = time.monotonic()
    try:
        environment = open_libero_task(task_alias, state_id, configs_root, data_root)
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
    except Exception as error:  # noqa: BLE001
        failure = error
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception as error:  # noqa: BLE001
                if failure is None:
                    failure = error
    elapsed = time.monotonic() - started
    video = encode_rgb_video(frames, output / "rollout.mp4", fps=20) if frames else None
    model_hash = hashlib.sha256((checkpoint_path / "model.safetensors").read_bytes()).hexdigest()
    record: dict[str, object] = {
        "schema_version": 1,
        "policy": "act",
        "task_alias": task_alias,
        "state_id": state_id,
        "state_partition": "development",
        "success": result.success if result is not None and failure is None else False,
        "termination_reason": (
            "policy_error"
            if isinstance(failure, PolicyActionError)
            else "infrastructure_error"
            if failure is not None
            else result.termination_reason
        ),
        "steps": result.steps if result is not None else None,
        "clipped_values": result.clipped_values if result is not None else None,
        "elapsed_s": round(elapsed, 3),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_model_sha256": model_hash,
        "checkpoint_manifest": bundle,
        "controller_hz": 20,
        "control_mode": "relative",
        "video": str(video.resolve()) if video is not None else None,
        "video_frames": len(frames),
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest()
        if video is not None
        else None,
        "error": f"{type(failure).__name__}: {failure}" if failure is not None else None,
    }
    (output / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if failure is not None:
        artifacts = {"run": str((output / "run.json").resolve())}
        if video is not None:
            artifacts["video"] = str(video.resolve())
        raise EvaluationRunError(
            f"Live ACT rollout failed: {failure}",
            policy_fault=isinstance(failure, PolicyActionError),
            artifact_paths=artifacts,
        ) from failure
    return record
