"""Check that a live ACT rollout can produce a standard video file."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vla_lab.evaluation.demo import encode_rgb_video, run_act_demo
from vla_lab.evaluation.errors import EvaluationRunError
from vla_lab.sim.episode import EpisodeResult, PolicyActionError


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


def test_policy_error_keeps_partial_video_and_run_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")

    class FakePolicy:
        def eval(self) -> "FakePolicy":
            return self

    class FakeEnvironment:
        def close(self) -> None:
            pass

    def fail_after_frame(*args: object, **kwargs: object) -> None:
        kwargs["on_frame"](np.zeros((32, 32, 3), dtype=np.uint8))
        raise PolicyActionError("nonfinite action")

    monkeypatch.setattr(
        "vla_lab.evaluation.demo.load_act_bundle",
        lambda _: (FakePolicy(), None, None, {"task_alias": "bowl_plate"}),
    )
    monkeypatch.setattr(
        "vla_lab.evaluation.demo.open_libero_task", lambda *args, **kwargs: FakeEnvironment()
    )
    monkeypatch.setattr("vla_lab.evaluation.demo.run_act_episode", fail_after_frame)
    with pytest.raises(EvaluationRunError) as caught:
        run_act_demo(
            task_alias="bowl_plate",
            checkpoint=checkpoint,
            state_id=0,
            configs_root=tmp_path,
            data_root=tmp_path,
            output_dir=tmp_path / "result",
        )
    assert caught.value.policy_fault
    assert Path(caught.value.artifact_paths["video"]).is_file()
    record = json.loads((tmp_path / "result" / "run.json").read_text())
    assert record["termination_reason"] == "policy_error"
    assert record["video_frames"] == 1


def test_success_run_record_keeps_termination_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")

    class FakePolicy:
        def eval(self) -> "FakePolicy":
            return self

    class FakeEnvironment:
        def close(self) -> None:
            pass

    def succeed(*args: object, **kwargs: object) -> EpisodeResult:
        assert kwargs["reset_seed"] == 20260930
        kwargs["on_frame"](np.zeros((32, 32, 3), dtype=np.uint8))
        return EpisodeResult("bowl_plate", 0, True, 1, 0, "success")

    monkeypatch.setattr(
        "vla_lab.evaluation.demo.load_act_bundle",
        lambda _: (FakePolicy(), None, None, {"task_alias": "bowl_plate"}),
    )
    monkeypatch.setattr(
        "vla_lab.evaluation.demo.open_libero_task", lambda *args, **kwargs: FakeEnvironment()
    )
    monkeypatch.setattr("vla_lab.evaluation.demo.run_act_episode", succeed)
    record = run_act_demo(
        task_alias="bowl_plate",
        checkpoint=checkpoint,
        state_id=0,
        configs_root=tmp_path,
        data_root=tmp_path,
        output_dir=tmp_path / "result",
    )
    assert record["success"]
    assert record["termination_reason"] == "success"
    assert record["error"] is None
