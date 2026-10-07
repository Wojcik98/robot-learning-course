"""Record one bounded keyboard episode from a development state."""

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Protocol

import cv2
import numpy as np
import yaml

from vla_lab.data.recording import CONTROL_PERIOD_MS, RecordedEpisode, record_episode
from vla_lab.sim.episode import open_libero_task
from vla_lab.teaching.interfaces import EpisodeEnvironment, LiberoObservation, ManagedEnvironment
from vla_lab.types import JSONValue


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


class KeyboardControlState:
    """Keep a short motion pulse and one persistent gripper command."""

    _MOTION_LABELS: ClassVar[dict[str, str]] = {
        "w": "W X+",
        "s": "S X-",
        "a": "A Y-",
        "d": "D Y+",
        "r": "R Z+",
        "f": "F Z-",
        "u": "U ROLL+",
        "o": "O ROLL-",
        "j": "J PITCH+",
        "l": "L PITCH-",
        "i": "I YAW+",
        "k": "K YAW-",
    }

    def __init__(self, *, motion_ticks: int = 4) -> None:
        if motion_ticks < 1:
            raise ValueError("Motion pulse needs at least one controller tick")
        self.motion_ticks = motion_ticks
        self._motion_key: str | None = None
        self._motion_ticks_left = 0
        self._gripper_closed = False
        self.last_key_label = "NONE"

    @property
    def gripper_label(self) -> str:
        """Show the command sent to the gripper on each tick."""
        return "CLOSE" if self._gripper_closed else "OPEN"

    @property
    def motion_label(self) -> str:
        """Show the arm action that the next tick will use."""
        return self._MOTION_LABELS[self._motion_key] if self._motion_ticks_left else "HOLD"

    def interpret(self, keycode: int) -> KeyboardEvent:
        """Convert one OpenCV key event into one visible controller command."""
        key = chr(keycode & 0xFF).lower() if keycode >= 0 else ""
        if key in ("q", "\x1b"):
            self.last_key_label = "QUIT"
            return KeyboardEvent(quit=True)
        if key == "x":
            self.last_key_label = "SAVE"
            return KeyboardEvent(save=True)
        if key in self._MOTION_LABELS:
            self._motion_key = key
            self._motion_ticks_left = self.motion_ticks
            self.last_key_label = key.upper()
        elif key in ("1", "["):
            self._gripper_closed = False
            self.last_key_label = "OPEN"
        elif key in ("2", "]"):
            self._gripper_closed = True
            self.last_key_label = "CLOSE"
        elif key == " ":
            self._gripper_closed = not self._gripper_closed
            self.last_key_label = "SPACE"
        keys = {"close" if self._gripper_closed else "open"}
        if self._motion_ticks_left:
            assert self._motion_key is not None
            keys.add(self._motion_key)
            self._motion_ticks_left -= 1
        return KeyboardEvent(keys=frozenset(keys))


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
    """Show a camera and an action panel for each 50 ms controller tick."""

    def __init__(self, window_name: str = "VLA keyboard recording") -> None:

        self._cv2 = cv2
        self._window_name = window_name
        self._controls = KeyboardControlState()
        self._cv2.namedWindow(window_name, self._cv2.WINDOW_NORMAL)
        self._cv2.resizeWindow(window_name, 760, 400)

    def tick(self, frame: np.ndarray, delay_ms: int = CONTROL_PERIOD_MS) -> KeyboardEvent:
        """Display a frame and return keys pressed during this tick."""
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Camera frame must have three color channels")
        camera = np.ascontiguousarray(image[::-1, ::-1, ::-1])
        camera = self._cv2.resize(camera, (360, 360))
        view = np.zeros((400, 760, 3), dtype=np.uint8)
        view[:360, :360] = camera
        labels = (
            ("CLICK WINDOW. PRESS W TO MOVE.", 28),
            ("MOVE ARM: TAP OR HOLD", 61),
            ("W/S: X + / X -", 87),
            ("A/D: Y - / Y +", 113),
            ("R/F: Z + / Z -", 139),
            ("ROTATE ARM", 172),
            ("U/O: ROLL  J/L: PITCH", 198),
            ("I/K: YAW", 224),
            ("GRIPPER", 256),
            ("SPACE: SWITCH GRIPPER", 282),
            ("1: OPEN  2: CLOSE", 308),
            ("X: SAVE  Q: DISCARD", 335),
            (f"LAST KEY: {self._controls.last_key_label}", 361),
            (
                f"ARM: {self._controls.motion_label}  GRIP: {self._controls.gripper_label}",
                387,
            ),
        )
        for label, y in labels:
            color = (0, 220, 255) if y in (28, 387) else (255, 255, 255)
            self._cv2.putText(
                view,
                label,
                (372, y),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                self._cv2.LINE_AA,
            )
        self._cv2.imshow(self._window_name, view)
        keycode = self._cv2.waitKeyEx(delay_ms)
        return self._controls.interpret(keycode)

    def close(self) -> None:
        """Close the display window."""
        self._cv2.destroyWindow(self._window_name)


class KeyboardInterface(Protocol):
    def tick(self, frame: np.ndarray, period_ms: int) -> KeyboardEvent: ...

    def close(self) -> None: ...


class _StopAwareEnvironment[ObservationT]:
    """Mark the simulator step after a save or quit key as truncated."""

    def __init__(
        self, environment: EpisodeEnvironment[ObservationT], stop_requested: Callable[[], bool]
    ) -> None:
        self._environment = environment
        self._stop_requested = stop_requested

    def reset(self, seed: int | None = None) -> tuple[ObservationT, Mapping[str, JSONValue]]:
        return self._environment.reset(seed=seed)

    def step(
        self, action: np.ndarray
    ) -> tuple[ObservationT, float, bool, bool, Mapping[str, JSONValue]]:
        observation, reward, terminated, truncated, info = self._environment.step(action)
        if self._stop_requested():
            truncated = True
        return observation, reward, terminated, truncated, info


def _camera_frame(observation: LiberoObservation) -> np.ndarray:
    """Read the agent camera image from a LIBERO observation."""
    try:
        return np.asarray(observation["pixels"]["image"])
    except (KeyError, TypeError) as error:
        raise ValueError("Observation does not contain pixels.image") from error


def run_keyboard_recording(
    environment: ManagedEnvironment[LiberoObservation],
    interface: KeyboardInterface,
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

    def keys_at_tick(_tick: int, observation: LiberoObservation) -> frozenset[str]:
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
            "Record LIBERO keyboard actions on a development state. Click the window first. "
            "W/S move X, A/D move Y, R/F move Z. U/O, J/L, and I/K rotate. "
            "Space toggles the gripper; 1 opens and 2 closes it. X saves; Q quits."
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
