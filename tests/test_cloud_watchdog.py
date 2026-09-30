"""Check that a saved lease stops its exact instance at the deadline."""

from vla_lab.cloud.watchdog import watchdog_once


class Provider:
    def __init__(self) -> None:
        self.state_value = "running"
        self.stopped = []

    def state(self, name: str) -> str:
        assert name == "course-probe"
        return self.state_value

    def stop(self, name: str) -> None:
        self.stopped.append(name)
        self.state_value = "stopped"


def test_watchdog_stops_only_the_leased_name_at_deadline() -> None:
    provider = Provider()
    lease = {"instance_name": "course-probe", "hard_deadline_at": 1000}
    assert watchdog_once(provider, lease, now=999)["phase"] == "watching"
    assert provider.stopped == []
    result = watchdog_once(provider, lease, now=1000)
    assert result["phase"] == "stopped"
    assert provider.stopped == ["course-probe"]
