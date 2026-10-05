"""Run one explicitly selected course job in memory and save its GPU resource record.

This tool does not save the executed notebook. It requires an explicit flag, uses offline
model/data settings, and records the source notebook hash plus PyTorch CUDA allocator peaks.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path
from uuid import uuid4

import nbformat

GPU_ALLOCATOR_SCOPE = (
    "PyTorch CUDA peak allocated/reserved bytes for this kernel process after reset, across the "
    "full notebook run; excludes non-PyTorch allocations and other processes."
)
RESOURCE_PREFIX = "COURSE_MEASURE_JSON="


def enable_flag(notebook, flag: str) -> None:
    """Change the one exact ``NAME = False`` code assignment to ``NAME = True``."""
    if not flag.isidentifier():
        raise ValueError(f"flag must be a Python identifier: {flag!r}")
    pattern = re.compile(rf"(?m)^(?P<indent>[ \t]*){re.escape(flag)}[ \t]*=[ \t]*False[ \t]*$")
    matches = [
        (cell, match)
        for cell in notebook.cells
        if cell.cell_type == "code"
        for match in pattern.finditer(cell.source)
    ]
    if not matches:
        raise ValueError(f"flag assignment {flag} = False was not found")
    if len(matches) != 1:
        raise ValueError(
            f"flag assignment {flag} = False must occur exactly once; found {len(matches)}"
        )
    cell, match = matches[0]
    cell.source = (
        cell.source[: match.start()]
        + f"{match.group('indent')}{flag} = True"
        + cell.source[match.end() :]
    )


def replace_once(notebook, old: str, new: str) -> None:
    """Replace one exact source string across code cells; reject zero or multiple matches."""
    if not old:
        raise ValueError("replacement OLD text must not be empty")
    matches = [
        (cell, cell.source.count(old))
        for cell in notebook.cells
        if cell.cell_type == "code" and old in cell.source
    ]
    count = sum(hits for _, hits in matches)
    if count != 1:
        raise ValueError(
            f"replacement OLD text must match exactly once in code cells; found {count}"
        )
    cell = next(cell for cell, hits in matches if hits)
    cell.source = cell.source.replace(old, new, 1)


def _validate_assertion(assertion: str | None) -> None:
    if assertion is None:
        return
    try:
        parsed = ast.parse(assertion)
    except SyntaxError as error:
        raise ValueError(f"--assertion is not valid Python: {error}") from error
    if not parsed.body or not all(isinstance(statement, ast.Assert) for statement in parsed.body):
        raise ValueError("--assertion must contain one or more Python assert statements")


def _apply_edits(notebook, flag: str, replacements: list[tuple[str, str]]) -> None:
    for old, new in replacements:
        replace_once(notebook, old, new)
    enable_flag(notebook, flag)


def _find_workspace(notebook_path: Path) -> Path:
    marker = Path("manifests/act_references.yaml")
    for candidate in (notebook_path.parent, *notebook_path.parent.parents):
        if (candidate / marker).is_file():
            return candidate
    script_workspace = Path(__file__).resolve().parents[1]
    if (script_workspace / marker).is_file():
        return script_workspace
    raise FileNotFoundError(f"Could not find workspace marker {marker} above {notebook_path}")


def _kernel_python(workspace: Path) -> str:
    candidate = workspace / ".venv" / "bin" / "python"
    return str(candidate) if candidate.is_file() else sys.executable


def _measurement_cells(workload: str, flag: str, assertion: str | None):
    prelude = """import json as _course_json
import time as _course_time
import torch as _course_torch

if not _course_torch.cuda.is_available():
    raise RuntimeError("This measurement requires a CUDA GPU; no notebook cells were run.")
_course_torch.cuda.init()
_course_torch.cuda.synchronize()
_course_torch.cuda.reset_peak_memory_stats()
_course_started = _course_time.perf_counter()
"""
    cells = [nbformat.v4.new_code_cell(prelude)]
    if assertion is not None:
        cells.append(nbformat.v4.new_code_cell(assertion))
    summary = f"""_course_torch.cuda.synchronize()
_course_report = {{
    "workload": {json.dumps(workload)},
    "flag": {json.dumps(flag)},
    "notebook_elapsed_seconds": _course_time.perf_counter() - _course_started,
    "gpu": {{
        "device_name": _course_torch.cuda.get_device_name(),
        "peak_allocated_mib": _course_torch.cuda.max_memory_allocated() / (1024 ** 2),
        "peak_reserved_mib": _course_torch.cuda.max_memory_reserved() / (1024 ** 2),
    }},
}}
print({json.dumps(RESOURCE_PREFIX)} + _course_json.dumps(_course_report, sort_keys=True))
"""
    cells.append(nbformat.v4.new_code_cell(summary))
    return cells


def _kernel_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="2",
        MPLBACKEND="Agg",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        DIFFUSERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    return environment


def _failed_cell(notebook):
    return next(
        (
            cell
            for cell in reversed(notebook.cells)
            if cell.cell_type == "code"
            and any(output.output_type == "error" for output in cell.get("outputs", []))
        ),
        None,
    )


def _resource_streams(notebook) -> list[dict]:
    found = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.output_type == "stream" and RESOURCE_PREFIX in output.text:
                for line in output.text.splitlines():
                    if line.startswith(RESOURCE_PREFIX):
                        found.append(json.loads(line[len(RESOURCE_PREFIX) :]))
    return found


def measure_notebook(
    notebook_path: str | Path,
    *,
    flag: str,
    replacements: list[tuple[str, str]],
    workload: str,
    report_path: str | Path,
    assertion: str | None = None,
    timeout: int = 1800,
) -> dict:
    """Execute an explicitly enabled notebook in a fresh GPU kernel and write a JSON record."""
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    path = Path(notebook_path).expanduser().resolve()
    if path.suffix != ".ipynb" or not path.is_file():
        raise ValueError(f"notebook must be an existing .ipynb file: {path}")
    if timeout < 1:
        raise ValueError("timeout must be at least one second")
    _validate_assertion(assertion)
    source_bytes = path.read_bytes()
    notebook_hash = hashlib.sha256(source_bytes).hexdigest()
    notebook = nbformat.reads(source_bytes.decode("utf-8"), as_version=4)
    _apply_edits(notebook, flag, replacements)
    measurement = _measurement_cells(workload, flag, assertion)
    notebook.cells = measurement[:1] + notebook.cells + measurement[1:]

    workspace = _find_workspace(path)
    kernel_name = f"course-measure-{uuid4().hex}"
    started = time.perf_counter()
    result = {
        "notebook": str(path),
        "notebook_sha256": notebook_hash,
        "workload": workload,
        "settings": {
            "flag_enabled": flag,
            "replacements": [{"old": old, "new": new} for old, new in replacements],
            "assertion": assertion,
            "timeout_seconds": timeout,
            "offline_environment": True,
            "omp_num_threads": 2,
            "mpl_backend": "Agg",
        },
        "gpu_allocator_scope": GPU_ALLOCATOR_SCOPE,
        "status": "failed",
        "execution_error": None,
        "execution_traceback": None,
        "execution_cell_id": None,
        "gpu_allocator_measurement_status": "not-recorded",
    }

    with tempfile.TemporaryDirectory(prefix="course-measure-kernel-") as temp_dir:
        spec_dir = Path(temp_dir) / kernel_name
        spec_dir.mkdir()
        kernel_spec = {
            "argv": [
                _kernel_python(workspace),
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ],
            "display_name": "Course measurement",
            "language": "python",
        }
        (spec_dir / "kernel.json").write_text(json.dumps(kernel_spec), encoding="utf-8")
        manager = AsyncKernelManager(
            kernel_name=kernel_name,
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temp_dir]),
        )
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=timeout,
            resources={"metadata": {"path": str(path.parent)}},
            allow_errors=False,
        )
        try:
            client.execute(cwd=str(path.parent), env=_kernel_environment(), cleanup_kc=True)
        except CellExecutionError as error:
            failed = _failed_cell(notebook)
            result["execution_error"] = error.ename
            result["execution_message"] = str(error.evalue)
            trace = getattr(error, "traceback", None)
            result["execution_traceback"] = "\n".join(trace) if isinstance(trace, list) else trace
            result["execution_cell_id"] = failed.id if failed is not None else None
        except Exception as error:  # noqa: BLE001 - preserve kernel startup and execution failures
            result["execution_error"] = type(error).__name__
            result["execution_message"] = str(error)
            result["execution_traceback"] = traceback.format_exc()
            failed = _failed_cell(notebook)
            result["execution_cell_id"] = failed.id if failed is not None else None
        else:
            result["status"] = "passed"

    streams = _resource_streams(notebook)
    result["notebook_streams"] = [
        {"cell_id": cell.id, "name": output.name, "text": output.text}
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "stream"
    ]
    result["resource_measurement"] = streams[-1] if streams else None
    if result["resource_measurement"] is not None:
        result["gpu_allocator_measurement_status"] = "recorded"
    if result["status"] == "passed" and result["resource_measurement"] is None:
        result["status"] = "failed"
        result["execution_error"] = "MissingResourceRecord"
        result["execution_message"] = "The notebook finished without its CUDA resource record."
    result["wall_seconds"] = round(time.perf_counter() - started, 2)

    destination = Path(report_path).expanduser()
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one explicitly selected real course job in a fresh CUDA kernel. "
            "This is not a CPU validator: execution requires a CUDA GPU, the named False flag "
            "must exist, and report output is saved without writing the executed notebook. "
            "Hugging Face and Diffusers downloads are disabled."
        )
    )
    parser.add_argument("notebook", help="notebook file whose actual cells will run in memory")
    parser.add_argument(
        "--flag",
        required=True,
        help="named notebook flag; its unique `= False` is enabled in memory",
    )
    parser.add_argument(
        "--replace",
        action="append",
        nargs=2,
        metavar=("OLD", "NEW"),
        default=[],
        help="exact code-cell source replacement; OLD must match once (repeat for bounded subset changes)",
    )
    parser.add_argument(
        "--assertion",
        help="optional Python assert statement appended after the notebook, for example a failed-trial-row check",
    )
    parser.add_argument(
        "--workload",
        required=True,
        help="short description of the job and its trial/subset settings",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="seconds allowed per notebook cell (default: 1800)",
    )
    parser.add_argument(
        "--report", required=True, help="JSON report path; the executed notebook is never saved"
    )
    args = parser.parse_args(argv)
    try:
        result = measure_notebook(
            args.notebook,
            flag=args.flag,
            replacements=[tuple(pair) for pair in args.replace],
            workload=args.workload,
            report_path=args.report,
            assertion=args.assertion,
            timeout=args.timeout,
        )
    except (OSError, ValueError, nbformat.NBFormatError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "resource_measurement"}, indent=2
        )
    )
    if result["resource_measurement"] is not None:
        print(json.dumps(result["resource_measurement"], indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
