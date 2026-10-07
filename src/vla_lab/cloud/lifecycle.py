"""Stop one paid instance even when its job or export fails."""

import subprocess
import time
from collections.abc import Callable
from typing import Protocol

STOPPED_STATES = {"stopped", "off", "suspended"}


class StopProvider(Protocol):
    """Provider operations needed to verify one stop."""

    def state(self, name: str) -> str: ...

    def stop(self, name: str) -> None: ...


def stop_with_retries(
    provider: StopProvider,
    name: str,
    *,
    delays_s: tuple[int, ...] = (0, 15, 30, 60),
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    """Request provider stop and check state after each bounded retry."""
    states = []
    for delay in delays_s:
        if delay:
            sleep(delay)
        try:
            state = provider.state(name)
            states.append(state)
            if state in STOPPED_STATES:
                return states
            provider.stop(name)
            state = provider.state(name)
            states.append(state)
            if state in STOPPED_STATES:
                return states
        except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as error:
            states.append(f"error:{type(error).__name__}")
    raise RuntimeError(
        f"Brev stop is not verified for {name}. Run: brev stop {name}. States: {states}"
    )


def finish_paid_session(
    provider: StopProvider,
    name: str,
    *,
    checkpoint: Callable[[], None] | None = None,
    terminate: Callable[[], None] | None = None,
    export: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Try to save work, then stop the provider even if saving fails."""
    errors = []
    for action, operation in (
        ("checkpoint", checkpoint),
        ("terminate", terminate),
        ("export", export),
    ):
        if operation is None:
            continue
        try:
            operation()
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"{action}:{type(error).__name__}: {error}")
    states = stop_with_retries(provider, name, sleep=sleep)
    return {"instance": name, "provider_states": states, "errors": errors, "stopped": True}
