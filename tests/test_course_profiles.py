"""Check notebook experiment selection and dependency execution."""

import nbformat
import pytest

from tools import check_course


def experiment(source, name, requires=()):
    return nbformat.v4.new_code_cell(
        source, metadata={"course": {"experiment": name, "requires": list(requires)}}
    )


def test_quick_profile_reports_skips_and_does_not_execute_live_cells(tmp_path):
    path = tmp_path / "unit-reference.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell("value = 7"),
                experiment("raise RuntimeError('requires simulator')", "rollout"),
                nbformat.v4.new_code_cell("assert value == 7"),
            ]
        ),
        path,
    )
    result = check_course.audit_notebook(path, workspace=tmp_path, execute=True, profile="quick")
    assert result["status"] == "passed"
    assert result["skipped_experiments"] == ["rollout"]
    assert result["selected_experiments"] == []


def test_full_selection_executes_dependencies_without_changing_source(tmp_path):
    path = tmp_path / "unit-reference.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell("values = []"),
                experiment("values.append(3)", "data"),
                experiment("values.append(4)", "train", ["data"]),
                experiment("assert values == [3, 4]", "evaluate", ["train"]),
                experiment("raise RuntimeError('unselected')", "other"),
            ]
        ),
        path,
    )
    before = path.read_bytes()
    result = check_course.audit_notebook(
        path, workspace=tmp_path, execute=True, profile="full", experiment="evaluate"
    )
    assert result["status"] == "passed", result
    assert result["selected_experiments"] == ["data", "evaluate", "train"]
    assert result["skipped_experiments"] == ["other"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("requires", [["missing"], ["job"]])
def test_invalid_dependencies_fail_before_execution(tmp_path, requires):
    path = tmp_path / "unit-reference.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[experiment("pass", "job", requires)]), path)
    result = check_course.audit_notebook(path, workspace=tmp_path, profile="full")
    assert result["status"] == "failed"
    assert result["errors"]


def test_unknown_experiment_is_not_a_successful_empty_run(tmp_path):
    path = tmp_path / "unit-reference.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[experiment("pass", "job")]), path)
    result = check_course.audit_notebook(
        path, workspace=tmp_path, profile="full", experiment="typo"
    )
    assert result["status"] == "failed"
