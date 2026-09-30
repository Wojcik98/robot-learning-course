"""Check the first learner command without opening a simulator."""

from typer.testing import CliRunner

from vla_lab.cli import app


def test_demo_help_shows_task_and_state_controls() -> None:
    result = CliRunner().invoke(app, ["demo", "--help"])
    assert result.exit_code == 0
    assert "--task" in result.stdout
    assert "--state-id" in result.stdout
    assert "--checkpoint" in result.stdout


def test_demo_rejects_cloud_profile_before_starting_any_job() -> None:
    result = CliRunner().invoke(app, ["demo", "--profile", "cloud", "--output", "unused"])
    assert result.exit_code != 0
    assert "Cloud demo profile is not ready" in result.output
