"""Clock values for LIBERO control and source dataset row lookup."""

LIBERO_CONTROL_HZ = 20
SOURCE_STORAGE_FPS = 10


def require_libero_rate(control_hz: int) -> None:
    """Reject a rate that differs from the checkpoint and simulator contract."""
    if isinstance(control_hz, bool) or control_hz != LIBERO_CONTROL_HZ:
        raise ValueError("LIBERO policies require 20 Hz control (one action per 50 ms tick)")


def frame_offset_seconds(offsets: list[int], storage_fps: int = SOURCE_STORAGE_FPS) -> list[float]:
    """Convert row offsets to the timestamp labels used by LeRobot lookup.

    These labels select adjacent saved frames. They do not set simulator time,
    action repetition, or the physical duration of the source demonstration.
    """
    if storage_fps <= 0:
        raise ValueError("storage_fps must be positive")
    return [offset / storage_fps for offset in offsets]
