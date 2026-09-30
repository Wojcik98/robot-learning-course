"""Terminal commands for the robot learning course."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from vla_lab.doctor import doctor_report, report_json
from vla_lab.evaluation.demo import run_act_demo
from vla_lab.jobs.manager import JobManager
from vla_lab.policy.transport import install_act_references, pack_act_references

app = typer.Typer(help="Run the pinned robot learning labs.", no_args_is_help=True)
references_app = typer.Typer(help="Move verified ACT checkpoints between machines.")
jobs_app = typer.Typer(help="Submit and inspect detached local jobs.")
app.add_typer(references_app, name="references")
app.add_typer(jobs_app, name="jobs")
DEFAULT_WORKSPACE = Path(".")


@app.callback()
def main() -> None:
    """Select a course command."""


@app.command()
def doctor(
    workspace: Annotated[
        Path, typer.Option(help="Course workspace directory.")
    ] = DEFAULT_WORKSPACE,
    data_root: Annotated[
        Path | None, typer.Option(help="Local LIBERO asset and data directory.")
    ] = None,
    live: Annotated[
        bool, typer.Option(help="Reset LIBERO with EGL and inspect a camera image.")
    ] = False,
) -> None:
    """Check local GPU, data, ACT bundles, and optional live rendering."""
    root = workspace.expanduser().resolve()
    data = (data_root or root / ".local-data").expanduser().resolve()
    report = doctor_report(root, data, live=live)
    typer.echo(report_json(report))
    if not report["ok"]:
        raise typer.Exit(code=1)


@references_app.command("pack")
def references_pack(
    output: Annotated[Path, typer.Option(help="New ZIP archive path.")],
    workspace: Annotated[
        Path, typer.Option(help="Course workspace directory.")
    ] = DEFAULT_WORKSPACE,
) -> None:
    """Pack both verified ACT reference checkpoints for transfer."""
    typer.echo(
        json.dumps(pack_act_references(workspace.resolve(), output.resolve()), sort_keys=True)
    )


@references_app.command("install")
def references_install(
    archive: Annotated[Path, typer.Option(help="Transferred ACT ZIP archive path.")],
    workspace: Annotated[
        Path, typer.Option(help="Course workspace directory.")
    ] = DEFAULT_WORKSPACE,
) -> None:
    """Verify the archive and restore both ACT checkpoints."""
    typer.echo(
        json.dumps(install_act_references(workspace.resolve(), archive.resolve()), sort_keys=True)
    )


def _jobs(workspace: Path) -> JobManager:
    return JobManager(workspace.expanduser().resolve() / ".local-data" / "jobs")


@jobs_app.command(
    "submit", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def jobs_submit(
    context: typer.Context,
    config: Annotated[Path, typer.Option(help="Resolved YAML or JSON job configuration.")],
    purpose: Annotated[str, typer.Option(help="Purpose of this run.")],
    seed: Annotated[int, typer.Option(help="Random seed for this run.")],
    repeat: Annotated[bool, typer.Option(help="Start a new run after a prior result.")] = False,
    workspace: Annotated[
        Path, typer.Option(help="Course workspace directory.")
    ] = DEFAULT_WORKSPACE,
) -> None:
    """Run the argument list after `--` outside the notebook kernel."""
    command = list(context.args)
    if command and command[0] == "--":
        command.pop(0)
    resolved = yaml.safe_load(config.read_text())
    if not isinstance(resolved, dict):
        raise typer.BadParameter("Job config must be a mapping", param_hint="--config")
    report = _jobs(workspace).submit(
        config=resolved, purpose=purpose, seed=seed, command=command, repeat=repeat
    )
    typer.echo(json.dumps(report, sort_keys=True))


@jobs_app.command("status")
def jobs_status(
    run_id: str,
    workspace: Annotated[
        Path, typer.Option(help="Course workspace directory.")
    ] = DEFAULT_WORKSPACE,
) -> None:
    """Reconnect to a run after a notebook or terminal closes."""
    typer.echo(json.dumps(_jobs(workspace).status(run_id), sort_keys=True))


@jobs_app.command("cancel")
def jobs_cancel(
    run_id: str,
    workspace: Annotated[
        Path, typer.Option(help="Course workspace directory.")
    ] = DEFAULT_WORKSPACE,
) -> None:
    """Stop one verified local worker process group."""
    typer.echo(json.dumps(_jobs(workspace).cancel(run_id), sort_keys=True))


@jobs_app.command("logs")
def jobs_logs(
    run_id: str,
    lines: Annotated[int, typer.Option(help="Number of last log lines.")] = 40,
    workspace: Annotated[
        Path, typer.Option(help="Course workspace directory.")
    ] = DEFAULT_WORKSPACE,
) -> None:
    """Show saved output from a detached job."""
    typer.echo(_jobs(workspace).log_tail(run_id, lines=lines))


@jobs_app.command("artifacts")
def jobs_artifacts(
    run_id: str,
    output: Annotated[Path | None, typer.Option(help="New directory for copied artifacts.")] = None,
    max_bytes: Annotated[int, typer.Option(help="Maximum transfer size in bytes.")] = 2**30,
    workspace: Annotated[
        Path, typer.Option(help="Course workspace directory.")
    ] = DEFAULT_WORKSPACE,
) -> None:
    """List or copy bounded files saved by one run."""
    manager = _jobs(workspace)
    files = (
        manager.artifacts(run_id)
        if output is None
        else manager.retrieve(run_id, output, max_bytes=max_bytes)
    )
    typer.echo(json.dumps(files))


@app.command()
def demo(
    output: Annotated[Path, typer.Option(help="New directory for the live run and video.")],
    task: Annotated[
        str, typer.Option(help="Task alias from configs/eval/splits.yaml.")
    ] = "bowl_plate",
    state_id: Annotated[int, typer.Option(help="Development initial state ID, from 0 to 19.")] = 0,
    checkpoint: Annotated[Path | None, typer.Option(help="ACT bundle directory.")] = None,
    profile: Annotated[str, typer.Option(help="Execution profile. Only local is ready.")] = "local",
    workspace: Annotated[
        Path, typer.Option(help="Course workspace directory.")
    ] = DEFAULT_WORKSPACE,
    data_root: Annotated[
        Path | None, typer.Option(help="Local LIBERO asset and data directory.")
    ] = None,
) -> None:
    """Reset one LIBERO task, run ACT, and save a live video."""
    if profile != "local":
        raise typer.BadParameter("Cloud demo profile is not ready", param_hint="--profile")
    root = workspace.expanduser().resolve()
    if checkpoint is None:
        manifest = yaml.safe_load((root / "manifests" / "act_references.yaml").read_text())
        try:
            checkpoint = root / manifest["tasks"][task]["checkpoint_path"]
        except KeyError as error:
            raise typer.BadParameter(
                f"Unknown ACT task alias: {task}", param_hint="--task"
            ) from error
    elif not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if not output.is_absolute():
        output = root / output
    if data_root is None:
        data_root = root / ".local-data"
    elif not data_root.is_absolute():
        data_root = root / data_root
    record = run_act_demo(
        task_alias=task,
        checkpoint=checkpoint,
        state_id=state_id,
        configs_root=root / "configs",
        data_root=data_root,
        output_dir=output,
    )
    typer.echo(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    app()
