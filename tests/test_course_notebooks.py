"""Tests for the course notebook validator."""

import json
from pathlib import Path

import nbformat

from tools.check_course import audit_notebook, find_workspace, main


def write_notebook(path: Path, cells: list[nbformat.NotebookNode]) -> Path:
    notebook = nbformat.v4.new_notebook(cells=cells)
    nbformat.write(notebook, path)
    return path


def test_find_workspace_from_nested_directory(tmp_path):
    root = tmp_path / "workspace"
    nested = root / "tutorials" / "07-attention"
    (root / "manifests").mkdir(parents=True)
    (root / "manifests" / "act_references.yaml").write_text("schema_version: 1\n")
    nested.mkdir(parents=True)

    assert find_workspace(nested) == root


def test_static_audit_checks_ids_syntax_outputs_and_local_links(tmp_path):
    target = tmp_path / "linked.md"
    target.write_text("local target\n")
    notebook = write_notebook(
        tmp_path / "unit.ipynb",
        [
            nbformat.v4.new_markdown_cell("[local](linked.md)"),
            nbformat.v4.new_code_cell("value = 1"),
        ],
    )

    assert audit_notebook(notebook, workspace=tmp_path)["status"] == "passed"

    payload = json.loads(notebook.read_text())
    payload["cells"][1]["id"] = payload["cells"][0]["id"]
    notebook.write_text(json.dumps(payload))
    row = audit_notebook(notebook, workspace=tmp_path)
    assert row["status"] == "failed"
    assert any("cell id" in error.lower() for error in row["errors"])

    broken = json.loads(notebook.read_text())
    broken["cells"][1]["id"] = "unique-cell-id"
    broken["cells"][1]["source"] = "if True print('syntax error')"
    broken["cells"][0]["source"] = "[missing](missing.md)"
    notebook.write_text(json.dumps(broken))
    row = audit_notebook(notebook, workspace=tmp_path)
    assert row["status"] == "failed"
    assert any("syntax" in error.lower() for error in row["errors"])
    assert any("missing.md" in error for error in row["errors"])


def test_static_audit_rejects_saved_code_outputs(tmp_path):
    notebook = write_notebook(
        tmp_path / "unit.ipynb",
        [
            nbformat.v4.new_code_cell(
                "1 + 1",
                outputs=[nbformat.v4.new_output("execute_result", data={}, execution_count=1)],
                execution_count=1,
            )
        ],
    )

    row = audit_notebook(notebook, workspace=tmp_path)
    assert row["status"] == "failed"
    assert any(
        "output" in error.lower() or "execution count" in error.lower() for error in row["errors"]
    )


def test_execution_requires_expected_learner_stub_in_exercise_context(tmp_path):
    notebook = write_notebook(
        tmp_path / "02-run-policy.ipynb",
        [
            nbformat.v4.new_markdown_cell(
                "## Exercise 1 — Reset both parts\n\n**Contract:** Return the first observation.\n\n**Method:** Reset the environment."
            ),
            nbformat.v4.new_code_cell(
                "def reset_episode():\n    raise NotImplementedError('Exercise 1')",
                id="exercise-stub",
            ),
            nbformat.v4.new_code_cell("reset_episode()", id="exercise-check"),
        ],
    )
    row = audit_notebook(notebook, workspace=tmp_path, execute=True, timeout=30)
    assert row["status"] == "expected-stub"
    assert row["stub_cell_id"] == "exercise-stub"
    assert row["execution_error"] == "NotImplementedError"
    assert row["execution_cell_id"] == "exercise-check"

    early_error = nbformat.read(notebook, as_version=4)
    early_error.cells.insert(0, nbformat.v4.new_code_cell("raise ValueError('before exercise')"))
    nbformat.write(early_error, notebook)
    row = audit_notebook(notebook, workspace=tmp_path, execute=True, timeout=30)
    assert row["status"] == "failed"
    assert row["execution_error"] == "ValueError"


def test_reference_notebook_must_execute_and_report_is_written(tmp_path):
    path = write_notebook(
        tmp_path / "unit-reference.ipynb", [nbformat.v4.new_code_cell("value = 3")]
    )
    report = tmp_path / "reports" / "course.json"
    code = main(["--execute", "--report", str(report), str(path)])

    assert code == 0
    payload = json.loads(report.read_text())
    assert payload["ok"] is True
    assert payload["notebooks"][0]["status"] == "passed"


def test_cli_defaults_to_course_folder_from_nested_cwd(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    course = root / "tutorials"
    nested = course / "07-attention"
    (root / "manifests").mkdir(parents=True)
    (root / "manifests" / "act_references.yaml").write_text("schema_version: 1\n")
    nested.mkdir(parents=True)
    write_notebook(
        nested / "07-attention-reference.ipynb", [nbformat.v4.new_code_cell("value = 1")]
    )
    monkeypatch.chdir(nested)
    report = tmp_path / "course.json"

    code = main(["--report", str(report)])

    assert code == 0
    payload = json.loads(report.read_text())
    assert len(payload["notebooks"]) == 1
    assert Path(payload["notebooks"][0]["path"]) == nested / "07-attention-reference.ipynb"
