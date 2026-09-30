"""Check provider stop retries and stop after failed export."""

import pytest

from vla_lab.cloud.lifecycle import finish_paid_session, stop_with_retries


class FakeProvider:
    def __init__(self, states: list[str]):
        self.states = iter(states)
        self.stop_calls = 0

    def state(self, name: str) -> str:
        assert name == "course-probe"
        return next(self.states)

    def stop(self, name: str) -> None:
        assert name == "course-probe"
        self.stop_calls += 1


def test_stop_retries_until_provider_confirms_stopped() -> None:
    provider = FakeProvider(["running", "stopping", "stopped"])
    waits = []
    states = stop_with_retries(provider, "course-probe", sleep=waits.append)
    assert states[-1] == "stopped"
    assert provider.stop_calls == 1
    assert waits == [15]


def test_export_failure_still_stops_provider() -> None:
    provider = FakeProvider(["running", "stopped"])

    def failed_export() -> None:
        raise OSError("copy failed")

    result = finish_paid_session(
        provider, "course-probe", export=failed_export, sleep=lambda _: None
    )
    assert result["stopped"]
    assert result["errors"] == ["export:OSError: copy failed"]
    assert provider.stop_calls == 1


def test_failed_stop_reports_exact_instance_command() -> None:
    provider = FakeProvider(["running"] * 8)
    with pytest.raises(RuntimeError, match="brev stop course-probe"):
        stop_with_retries(provider, "course-probe", sleep=lambda _: None)
    assert provider.stop_calls == 4
