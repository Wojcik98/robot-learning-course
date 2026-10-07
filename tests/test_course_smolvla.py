"""Checks for the SmolVLA course lesson pairs and experiment tags."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LESSONS = (
    (16, "smolvla-execution", "execution"),
    (17, "smolvla-training", "training"),
    (18, "smolvla-conditioning", "conditioning"),
    (19, "smolvla-flow", "flow"),
    (20, "smolvla-adaptation", "adaptation"),
)


def _notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def test_smolvla_course_has_five_paired_lessons_with_metadata():
    for number, slug, _ in LESSONS:
        for role in ("learner", "reference"):
            filename = (
                f"{number}-{slug}.ipynb"
                if role == "learner"
                else f"{number}-{slug}-reference.ipynb"
            )
            path = ROOT / "tutorials" / f"{number}-{slug}" / filename
            notebook = _notebook(path)
            course = notebook["metadata"]["course"]
            assert course["lesson"] == number
            assert course["role"] == role
            assert course["title"]


def test_real_experiment_cells_have_slugs_and_dependency_cells_are_tagged():
    for number, slug, experiment in LESSONS:
        for role in ("learner", "reference"):
            filename = (
                f"{number}-{slug}.ipynb"
                if role == "learner"
                else f"{number}-{slug}-reference.ipynb"
            )
            path = ROOT / "tutorials" / f"{number}-{slug}" / filename
            notebook = _notebook(path)
            tagged = [
                cell
                for cell in notebook["cells"]
                if cell.get("metadata", {}).get("course", {}).get("experiment") == experiment
            ]
            assert tagged, f"{path} has no {experiment} experiment"
            assert all("requires" in cell["metadata"]["course"] for cell in tagged)


def test_notebooks_have_no_run_flags_future_annotations_or_lazy_imports():
    for number, slug, _ in LESSONS:
        for role in ("learner", "reference"):
            filename = (
                f"{number}-{slug}.ipynb"
                if role == "learner"
                else f"{number}-{slug}-reference.ipynb"
            )
            path = ROOT / "tutorials" / f"{number}-{slug}" / filename
            notebook = _notebook(path)
            code = "\n".join(
                "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
            )
            assert "from __future__ import annotations" not in code
            assert "RUN_EXPERIMENT" not in code
            assert "RUN_REAL" not in code
            assert "import " not in "\n".join(
                line for line in code.splitlines() if line.startswith("    ")
            )


def test_notebook_functions_have_parameter_and_return_types():
    for number, slug, _ in LESSONS:
        for role in ("learner", "reference"):
            filename = (
                f"{number}-{slug}.ipynb"
                if role == "learner"
                else f"{number}-{slug}-reference.ipynb"
            )
            notebook = _notebook(ROOT / "tutorials" / f"{number}-{slug}" / filename)
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                tree = ast.parse("".join(cell["source"]))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        assert node.returns is not None, (
                            f"{filename}: {node.name} has no return type"
                        )
                        assert all(arg.annotation is not None for arg in node.args.args), (
                            f"{filename}: {node.name} has an untyped parameter"
                        )
