"""Check that a live ACT rollout can produce a standard video file."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vla_lab.evaluation.demo import encode_rgb_video


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg tools are required for video checks",
)
def test_rgb_video_uses_the_control_frame_rate(tmp_path: Path) -> None:
    frames = [np.full((32, 32, 3), index * 120, dtype=np.uint8) for index in range(2)]
    output = tmp_path / "rollout.mp4"

    encode_rgb_video(frames, output, fps=20)

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    assert (stream["width"], stream["height"]) == (32, 32)
    assert stream["avg_frame_rate"] == "20/1"
    assert stream["nb_frames"] == "2"


def test_rgb_video_rejects_inconsistent_frame_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same size"):
        encode_rgb_video(
            [np.zeros((32, 32, 3), dtype=np.uint8), np.zeros((16, 32, 3), dtype=np.uint8)],
            tmp_path / "bad.mp4",
        )
