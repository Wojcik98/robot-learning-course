from __future__ import annotations

import json

import nbformat
import pytest

from tools import measure_course
from tools.measure_course import RESOURCE_PREFIX, enable_flag, replace_once


def test_enable_flag_changes_only_named_false_assignment():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("RUN_JOB = False"),
            nbformat.v4.new_code_cell("OTHER = False\nRUN_JOB = False\n"),
        ]
    )

    enable_flag(notebook, "RUN_JOB")

    assert notebook.cells[0].source == "RUN_JOB = False"
    assert notebook.cells[1].source == "OTHER = False\nRUN_JOB = True\n"


@pytest.mark.parametrize(
    "source, message",
    [
        ("RUN_JOB = True", "False"),
        ("x = 1", "not found"),
        ("RUN_JOB = False\nRUN_JOB = False", "once"),
    ],
)
def test_enable_flag_rejects_missing_true_or_duplicate_assignments(source, message):
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])

    with pytest.raises(ValueError, match=message):
        enable_flag(notebook, "RUN_JOB")


def test_replace_once_requires_one_exact_match_across_code_cells():
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell("replace me"), nbformat.v4.new_code_cell("steps = 5")]
    )

    replace_once(notebook, "steps = 5", "steps = 2")

    assert notebook.cells[0].source == "replace me"
    assert notebook.cells[1].source == "steps = 2"


@pytest.mark.parametrize("sources", [["steps = 5"], ["steps = 5\nsteps = 5"]])
def test_replace_once_rejects_missing_or_duplicate_text(sources):
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(source) for source in sources]
    )

    with pytest.raises(ValueError, match="exactly once"):
        replace_once(notebook, "steps = 9", "steps = 2") if sources == [
            "steps = 5"
        ] else replace_once(notebook, "steps = 5", "steps = 2")


def _cpu_measurement_cells(workload, flag, assertion):
    prelude = """import json as _course_json
import time as _course_time
_course_started = _course_time.perf_counter()
"""
    suffix = f"""{assertion or "assert True"}
assert course_order == ["notebook body"], "measurement suffix ran before the notebook body"
_course_report = {{
    "workload": {json.dumps(workload)},
    "flag": {json.dumps(flag)},
    "body_value": body_value,
    "body_ran": course_order == ["notebook body"],
    "elapsed_seconds": _course_time.perf_counter() - _course_started,
}}
print({json.dumps(RESOURCE_PREFIX)} + _course_json.dumps(_course_report, sort_keys=True))
"""
    return [nbformat.v4.new_code_cell(prelude), nbformat.v4.new_code_cell(suffix)]


def _measurement_notebook(path):
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                'RUN_JOB = False\nbody_value = 37\ncourse_order = ["notebook body"]'
            )
        ]
    )
    nbformat.write(notebook, path)
    return path


def test_measure_notebook_runs_body_before_assertion_and_records_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(measure_course, "_measurement_cells", _cpu_measurement_cells)
    notebook = _measurement_notebook(tmp_path / "unit.ipynb")
    report_path = tmp_path / "success-report.json"

    result = measure_course.measure_notebook(
        notebook,
        flag="RUN_JOB",
        replacements=[],
        workload="cpu regression",
        report_path=report_path,
        assertion="assert body_value == 37",
        timeout=30,
    )

    assert result["status"] == "passed"
    assert result["gpu_allocator_measurement_status"] == "recorded"
    assert result["resource_measurement"] == {
        "workload": "cpu regression",
        "flag": "RUN_JOB",
        "body_value": 37,
        "body_ran": True,
        "elapsed_seconds": result["resource_measurement"]["elapsed_seconds"],
    }
    assert result["resource_measurement"]["elapsed_seconds"] >= 0
    assert json.loads(report_path.read_text()) == result


def test_measure_notebook_failing_assertion_cannot_pass_or_record_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(measure_course, "_measurement_cells", _cpu_measurement_cells)
    notebook = _measurement_notebook(tmp_path / "unit-fails.ipynb")
    report_path = tmp_path / "failed-report.json"

    result = measure_course.measure_notebook(
        notebook,
        flag="RUN_JOB",
        replacements=[],
        workload="cpu assertion failure",
        report_path=report_path,
        assertion="assert body_value == 0",
        timeout=30,
    )

    assert result["status"] == "failed"
    assert result["execution_error"] == "AssertionError"
    assert result["gpu_allocator_measurement_status"] == "not-recorded"
    assert result["resource_measurement"] is None
    assert json.loads(report_path.read_text()) == result
