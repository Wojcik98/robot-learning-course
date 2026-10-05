"""Check course notebooks without running them unless ``--execute`` is set."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import warnings
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import uuid4

import nbformat

from vla_lab.types import JSONValue

WORKSPACE_MARKER = Path("manifests/act_references.yaml")
STUB_PATTERN = re.compile(r"\braise\s+NotImplementedError\b")
EXERCISE_PATTERN = re.compile(r"\bexercise\s+\d+(?:[A-Za-z]?(?:\.\d+)*)?\b", re.IGNORECASE)
MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
CELL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def find_workspace(start: str | Path | None = None) -> Path:
    """Find the repository root from a directory inside the checkout."""
    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / WORKSPACE_MARKER).is_file():
            return candidate
    script_root = Path(__file__).resolve().parents[1]
    if (script_root / WORKSPACE_MARKER).is_file():
        return script_root
    raise FileNotFoundError(f"Could not find {WORKSPACE_MARKER} above {current}")


def discover_notebooks(paths: list[str | Path], workspace: Path, caller_dir: Path) -> list[Path]:
    """Resolve files and directories to a sorted list of notebooks."""
    found: set[Path] = set()
    for raw_path in paths:
        raw = Path(raw_path).expanduser()
        if raw.is_absolute():
            candidate = raw
        else:
            from_caller = caller_dir / raw
            candidate = from_caller if from_caller.exists() else workspace / raw
        candidate = candidate.resolve()
        if candidate.is_dir():
            found.update(
                path.resolve()
                for path in candidate.rglob("*.ipynb")
                if ".ipynb_checkpoints" not in path.parts
            )
        elif candidate.is_file() and candidate.suffix == ".ipynb":
            found.add(candidate)
    return sorted(found)


def _local_markdown_targets(source: str) -> Iterator[str]:
    """Yield local paths from ordinary inline Markdown links and images."""
    for match in MARKDOWN_LINK_PATTERN.finditer(source):
        destination = match.group(1).strip()
        if destination.startswith("<"):
            end = destination.find(">")
            destination = destination[1:end] if end >= 0 else destination[1:]
        else:
            destination = destination.split(maxsplit=1)[0]
        destination = unquote(destination.strip())
        parsed = urlsplit(destination)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        yield parsed.path


def _notebook_class(path: Path) -> str:
    if "-reference" in path.stem:
        return "reference"
    if path.parent.name == "01-playground":
        return "playground"
    return "learner"


def _first_stub(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode | None:
    for cell in notebook.cells:
        if cell.cell_type == "code" and STUB_PATTERN.search(cell.source):
            return cell
    return None


def _check_stub_context(
    notebook: nbformat.NotebookNode, stub_cell: nbformat.NotebookNode
) -> str | None:
    stub_index = notebook.cells.index(stub_cell)
    preceding_markdown = "\n".join(
        cell.source for cell in notebook.cells[:stub_index] if cell.cell_type == "markdown"
    )
    if not EXERCISE_PATTERN.search(preceding_markdown):
        return "first NotImplementedError is not under an exercise heading"
    return None


def _kernel_python(path: Path) -> str:
    """Prefer the repository venv so execution uses the pinned project packages."""
    try:
        workspace = find_workspace(path)
    except FileNotFoundError:
        return sys.executable
    candidate = workspace / ".venv" / "bin" / "python"
    return str(candidate) if candidate.is_file() else sys.executable


def _static_checks(path: Path, workspace: Path) -> tuple[object | None, list[str]]:
    errors: list[str] = []
    try:
        raw_notebook = json.loads(path.read_text(encoding="utf-8"))
        raw_cells = raw_notebook.get("cells", [])
        raw_ids: set[str] = set()
        for index, cell in enumerate(raw_cells):
            cell_id = cell.get("id", "")
            if not CELL_ID_PATTERN.fullmatch(cell_id):
                errors.append(f"cell {index}: invalid or missing cell id {cell_id!r}")
            if cell_id in raw_ids:
                errors.append(f"cell {index}: duplicate cell id {cell_id!r}")
            raw_ids.add(cell_id)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=nbformat.validator.DuplicateCellId)
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
    except (OSError, UnicodeError, ValueError, nbformat.validator.NotebookValidationError) as error:
        return None, errors + [f"invalid notebook: {type(error).__name__}: {error}"]

    for index, cell in enumerate(notebook.cells):
        cell_id = cell.get("id", "")
        if cell.cell_type == "code":
            if cell.get("execution_count") is not None:
                errors.append(f"cell {cell_id}: execution count must be empty")
            if cell.get("outputs"):
                errors.append(f"cell {cell_id}: saved code output must be empty")
            try:
                compile(cell.source, f"{path}#{cell_id}", "exec")
            except (SyntaxError, ValueError) as error:
                errors.append(f"cell {cell_id}: code syntax error: {error}")
        elif cell.cell_type == "markdown":
            for target in _local_markdown_targets(cell.source):
                linked_path = (path.parent / target).resolve()
                if not linked_path.exists():
                    errors.append(f"cell {cell_id}: local Markdown link does not exist: {target}")

    stub = _first_stub(notebook) if _notebook_class(path) == "learner" else None
    if stub is not None:
        context_error = _check_stub_context(notebook, stub)
        if context_error:
            errors.append(f"cell {stub.id}: {context_error}")
    return notebook, errors


def _execute_notebook(
    notebook: nbformat.NotebookNode, path: Path, timeout: int
) -> tuple[str, str | None, str | None, str | None]:
    """Execute in a fresh kernel that uses this validator's Python executable."""
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    kernel_name = f"course-validator-{uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix="course-kernel-") as temp_dir:
        spec_dir = Path(temp_dir) / kernel_name
        spec_dir.mkdir()
        kernel_spec = {
            "argv": [_kernel_python(path), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Course validator",
            "language": "python",
        }
        (spec_dir / "kernel.json").write_text(json.dumps(kernel_spec), encoding="utf-8")
        spec_manager = KernelSpecManager(kernel_dirs=[temp_dir])
        manager = AsyncKernelManager(kernel_name=kernel_name, kernel_spec_manager=spec_manager)
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=timeout,
            resources={"metadata": {"path": str(path.parent)}},
            allow_errors=False,
        )
        environment = os.environ.copy()
        environment.update(
            OMP_NUM_THREADS="2",
            MPLBACKEND="Agg",
            CUDA_VISIBLE_DEVICES="",
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        try:
            client.execute(cwd=str(path.parent), env=environment, cleanup_kc=True)
        except CellExecutionError as error:
            failed_cell = next(
                (
                    cell
                    for cell in reversed(notebook.cells)
                    if cell.cell_type == "code"
                    and any(output.output_type == "error" for output in cell.get("outputs", []))
                ),
                None,
            )
            error_name = error.ename
            failed_id = failed_cell.id if failed_cell is not None else None
            return "failed", error_name, failed_id, str(error.evalue)
        except Exception as error:  # noqa: BLE001 - preserve all kernel startup and execution failures in the report
            return "failed", type(error).__name__, None, str(error)
    return "passed", None, None, None


def audit_notebook(
    path: str | Path,
    *,
    workspace: str | Path | None = None,
    execute: bool = False,
    timeout: int = 180,
) -> dict[str, JSONValue]:
    """Run static checks and, when requested, fresh-kernel checks on one notebook."""
    notebook_path = Path(path).resolve()
    root = Path(workspace).resolve() if workspace is not None else find_workspace(notebook_path)
    started = time.monotonic()
    notebook, errors = _static_checks(notebook_path, root)
    row = {
        "path": str(notebook_path),
        "status": "failed" if errors else "passed",
        "errors": errors,
        "execution": "not-requested",
    }
    if execute and notebook is not None and not errors:
        kind = _notebook_class(notebook_path)
        stub = _first_stub(notebook) if kind == "learner" else None
        if kind == "learner" and stub is None:
            row["errors"].append("learner notebook has no NotImplementedError exercise stub")
            row["status"] = "failed"
        else:
            execution, error_name, failed_id, error_message = _execute_notebook(
                notebook, notebook_path, timeout
            )
            row["execution_error"] = error_name
            row["execution_cell_id"] = failed_id
            row["execution_message"] = error_message
            cell_order = {cell.id: index for index, cell in enumerate(notebook.cells)}
            expected_stub = (
                kind == "learner"
                and error_name == "NotImplementedError"
                and failed_id in cell_order
                and cell_order[failed_id] >= cell_order[stub.id]
            )
            if expected_stub:
                row["status"] = "expected-stub"
                row["execution"] = "expected-stub"
                row["stub_cell_id"] = stub.id
            elif execution == "passed" and kind in {"reference", "playground"}:
                row["status"] = "passed"
                row["execution"] = "passed"
            else:
                row["status"] = "failed"
                row["execution"] = "failed"
                row["errors"].append(
                    f"execution ended as {error_name or 'success'} at cell {failed_id or 'none'}; "
                    f"expected {'a NotImplementedError at the first exercise' if kind == 'learner' else 'success'}"
                )
    row["seconds"] = round(time.monotonic() - started, 2)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", help="notebook files or directories (default: tutorials)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute each notebook in a fresh local Python kernel",
    )
    parser.add_argument(
        "--timeout", type=int, default=180, help="seconds allowed per notebook cell"
    )
    parser.add_argument(
        "--report",
        default=".local-data/course-redesign/validator-report.json",
        help="write a JSON report to this path",
    )
    args = parser.parse_args(argv)
    caller_dir = Path.cwd().resolve()
    workspace = find_workspace(caller_dir)
    requested = args.paths or ["tutorials"]
    paths = discover_notebooks(requested, workspace, caller_dir)
    if not paths:
        parser.error("no notebook files found in the requested paths")
    report_path = Path(args.report).expanduser()
    if not report_path.is_absolute():
        report_path = caller_dir / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        audit_notebook(path, workspace=workspace, execute=args.execute, timeout=args.timeout)
        for path in paths
    ]
    report = {
        "workspace": str(workspace),
        "ok": all(row["status"] in {"passed", "expected-stub"} for row in rows),
        "execute": args.execute,
        "notebooks": rows,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
