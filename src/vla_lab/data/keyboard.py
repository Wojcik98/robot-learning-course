"""Record one bounded keyboard episode from a development state."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vla_lab.data.recording import CONTROL_PERIOD_MS, RecordedEpisode, record_episode


@dataclass(frozen=True)
class KeyboardEvent:
    """Keys for one tick, with optional save or quit request."""

    keys: frozenset[str] = frozenset()
    save: bool = False
    quit: bool = False


@dataclass(frozen=True)
class KeyboardRecordingResult:
    """Episode data and the path written when the user saves."""

    episode: RecordedEpisode
    path: Path | None


def validate_development_state(task_alias: str, state_id: int, configs_root: str | Path) -> int:
    """Reject any state that is outside the declared development partition."""
    manifest_path = Path(configs_root) / "eval" / "splits.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    try:
        task = manifest["tasks"][task_alias]
        development_ids = task["development_state_ids"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unknown task alias: {task_alias}") from error
    if state_id not in development_ids:
        raise ValueError(f"State {state_id} is not in the development partition")
    return int(task["task_id"])


class OpenCVKeyboardInterface:
    """Show each camera frame and read one key event for a 50 ms tick."""

    def __init__(self, window_name: str = "VLA keyboard recording") -> None:
        import cv2

        self._cv2 = cv2
        self._window_name = window_name
        self._cv2.namedWindow(window_name, self._cv2.WINDOW_NORMAL)

    def tick(self, frame: np.ndarray, delay_ms: int = CONTROL_PERIOD_MS) -> KeyboardEvent:
        """Display a frame and return keys pressed during this tick."""
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Camera frame must have three color channels")
        camera = np.ascontiguousarray(image[::-1, ::-1, ::-1])
        view = np.zeros((max(camera.shape[0], 256), camera.shape[1] + 260, 3), dtype=np.uint8)
        view[: camera.shape[0], : camera.shape[1]] = camera
        for line, label in enumerate(
            (
                "W/S: x   A/D: y",
                "R/F: z",
                "U/O: roll",
                "J/L: pitch",
                "I/K: yaw",
                "[: open   ]: close",
                "X: save and stop",
                "Q: discard and stop",
            ),
            start=1,
        ):
            self._cv2.putText(
                view,
                label,
                (camera.shape[1] + 8, 12 + 27 * line),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                self._cv2.LINE_AA,
            )
        self._cv2.imshow(self._window_name, view)
        keycode = self._cv2.waitKeyEx(delay_ms)
        key = keycode & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return KeyboardEvent(quit=True)
        if key in (ord("x"), ord("X")):
            return KeyboardEvent(save=True)
        keys = {
            ord("["): "open",
            ord("]"): "close",
        }
        if key in keys:
            return KeyboardEvent(keys=frozenset({keys[key]}))
        try:
            char = chr(key).lower()
        except ValueError:
            return KeyboardEvent()
        allowed = "wsdarfuojlik"
        return KeyboardEvent(keys=frozenset({char}) if char in allowed else frozenset())

    def close(self) -> None:
        """Close the display window."""
        self._cv2.destroyWindow(self._window_name)


class _StopAwareEnvironment:
    """Mark the simulator step after a save or quit key as truncated."""

    def __init__(self, environment: Any, stop_requested: Callable[[], bool]) -> None:
        self._environment = environment
        self._stop_requested = stop_requested

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        return self._environment.reset(*args, **kwargs)

    def step(self, action: np.ndarray) -> Any:
        observation, reward, terminated, truncated, info = self._environment.step(action)
        if self._stop_requested():
            truncated = True
        return observation, reward, terminated, truncated, info


def _camera_frame(observation: Mapping[str, Any]) -> np.ndarray:
    """Read the agent camera image from a LIBERO observation."""
    try:
        return np.asarray(observation["pixels"]["image"])
    except (KeyError, TypeError) as error:
        raise ValueError("Observation does not contain pixels.image") from error


def run_keyboard_recording(
    environment: Any,
    interface: Any,
    *,
    configs_root: str | Path,
    task_alias: str,
    state_id: int,
    output_path: str | Path,
    episode_id: str,
    max_steps: int = 300,
) -> KeyboardRecordingResult:
    """Record, save on request, and close the simulator and input interface."""
    validate_development_state(task_alias, state_id, configs_root)
    last_event = KeyboardEvent()

    def stop_requested() -> bool:
        return last_event.save or last_event.quit

    def keys_at_tick(_tick: int, observation: Mapping[str, Any]) -> frozenset[str]:
        nonlocal last_event
        last_event = interface.tick(_camera_frame(observation), CONTROL_PERIOD_MS)
        return last_event.keys

    try:
        episode = record_episode(
            _StopAwareEnvironment(environment, stop_requested),
            keys_at_tick,
            task_id=task_alias,
            max_steps=max_steps,
            episode_id=episode_id,
            reset_seed=20260930 + state_id,
        )
        if last_event.quit:
            return KeyboardRecordingResult(episode, None)
        path = episode.save(output_path)
        return KeyboardRecordingResult(episode, path)
    finally:
        try:
            environment.close()
        finally:
            interface.close()


def main(argv: list[str] | None = None) -> int:
    """Open a development state and record keyboard actions."""
    parser = argparse.ArgumentParser(
        description=(
            "Record LIBERO keyboard actions on a development state. "
            "W/S move x, A/D move y, R/F move z, U/O roll, J/L pitch, "
            "I/K yaw, [ opens and ] closes the gripper. X saves and stops; Q quits."
        )
    )
    parser.add_argument("--task", default="bowl_plate", help="task alias from the state split")
    parser.add_argument("--state", type=int, default=0, help="development state ID")
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--data-root", type=Path, default=Path(".local-data"))
    parser.add_argument("--output", type=Path, help="NPZ output path")
    parser.add_argument("--max-steps", type=int, default=300, help="maximum 50 ms ticks")
    args = parser.parse_args(argv)

    # Validate the partition before importing or opening the simulator.
    validate_development_state(args.task, args.state, args.configs_root)
    from vla_lab.sim.episode import open_libero_task

    interface = OpenCVKeyboardInterface()
    try:
        environment = open_libero_task(
            args.task, args.state, args.configs_root, args.data_root.expanduser()
        )
    except Exception:
        interface.close()
        raise
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path(".local-data/keyboard-recordings") / (
        f"{args.task}-state-{args.state}-{stamp}.npz"
    )
    result = run_keyboard_recording(
        environment,
        interface,
        configs_root=args.configs_root,
        task_alias=args.task,
        state_id=args.state,
        output_path=output,
        episode_id=f"{args.task}-state-{args.state}-{stamp}",
        max_steps=args.max_steps,
    )
    if result.path is None:
        print("Recording discarded.")
    else:
        print(f"Saved {len(result.episode.actions)} ticks to {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
