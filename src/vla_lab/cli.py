"""Terminal commands for the robot learning course."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from vla_lab.evaluation.demo import run_act_demo

app = typer.Typer(help="Run the pinned robot learning labs.", no_args_is_help=True)
DEFAULT_WORKSPACE = Path(".")


@app.callback()
def main() -> None:
    """Select a course command."""


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
