#!/usr/bin/env python
"""Check course types, notebook signatures, and rejected policy interfaces."""

from __future__ import annotations

import ast
import builtins
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKED_MODULES = (
    "src/vla_lab/teaching",
    "src/vla_lab/sim/episode.py",
    "src/vla_lab/runtime/smolvla_chunks.py",
    "src/vla_lab/runtime/experiment.py",
    "src/vla_lab/data/recording.py",
    "src/vla_lab/data/recovery_dataset.py",
    "tests/typecheck/policy_contracts.py",
)


def check_signatures(code: str, label: str) -> dict[str, str]:
    """Require typed function arguments and returns without Any."""
    signatures = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Name) and node.id == "Any":
            raise ValueError(f"{label}: Any is not permitted")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = (
            node.args.posonlyargs
            + node.args.args
            + node.args.kwonlyargs
            + [arg for arg in (node.args.vararg, node.args.kwarg) if arg]
        )
        if node.returns is None or any(
            arg.annotation is None for arg in arguments if arg.arg not in {"self", "cls"}
        ):
            raise ValueError(f"{label}: incomplete signature for {node.name}")
        declaration = ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=[ast.Pass()],
            decorator_list=[],
            returns=node.returns,
            type_params=node.type_params,
        )
        signatures[node.name] = ast.dump(declaration, include_attributes=False)
    return signatures


def check_notebook_imports(code: str, label: str) -> None:
    """Require a definition or import for each name used in an annotation."""
    defined = set(dir(builtins))
    used = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            defined.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        if isinstance(node, ast.TypeVar):
            defined.add(node.name)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        annotations = []
        if isinstance(node, ast.arg) and node.annotation:
            annotations.append(node.annotation)
        if isinstance(node, ast.FunctionDef) and node.returns:
            annotations.append(node.returns)
        if isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        for annotation in annotations:
            used.update(item.id for item in ast.walk(annotation) if isinstance(item, ast.Name))
    missing = used - defined
    if missing:
        raise ValueError(f"{label}: annotation names are not defined: {sorted(missing)}")


def notebook_code(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def main() -> int:
    """Run type checks without changing notebook files or saved outputs."""
    for folder in ("src", "tools"):
        for path in (ROOT / folder).rglob("*.py"):
            check_signatures(path.read_text(), str(path.relative_to(ROOT)))
    notebooks = sorted(
        path
        for path in (ROOT / "tutorials").rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )
    for path in notebooks:
        code = notebook_code(path)
        check_notebook_imports(code, str(path.relative_to(ROOT)))
        signatures = check_signatures(code, str(path.relative_to(ROOT)))
        if path.stem.endswith("-reference") or path.stem == "index":
            continue
        reference = path.with_name(path.stem + "-reference.ipynb")
        other = check_signatures(notebook_code(reference), str(reference.relative_to(ROOT)))
        for name in signatures.keys() & other.keys():
            if signatures[name] != other[name]:
                raise ValueError(f"{path}: learner and reference types differ for {name}")
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--python-executable",
        sys.executable,
        "--config-file",
        str(ROOT / "tools/mypy_course.ini"),
        "--cache-dir",
        str(ROOT / ".local-data/mypy-cache"),
    ]
    with tempfile.TemporaryDirectory(prefix="vla-notebook-types-") as temporary:
        paths = []
        for path in notebooks:
            if path.parent.name == "02-run-policy":
                exported = Path(temporary) / (path.stem.replace("-", "_") + ".py")
                exported.write_text(notebook_code(path))
                paths.append(str(exported))
        result = subprocess.run(command + list(CHECKED_MODULES) + paths, cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    negative = subprocess.run(
        command + ["tests/typecheck/policy_rejections.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    errors = re.findall(r"policy_rejections.py:\d+: error:.*\[(.*?)\]", negative.stdout)
    if negative.returncode != 1 or sorted(errors) != ["arg-type"] * 4 + ["dict-item"]:
        print(negative.stdout, end="")
        print(negative.stderr, end="", file=sys.stderr)
        raise ValueError("Expected four interface errors and one tensor batch error")
    print(f"Checked {len(notebooks)} notebooks; all five invalid policy calls were rejected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
