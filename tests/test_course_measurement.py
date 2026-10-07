"""Check selected experiment measurement and failure records."""

import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from types import ModuleType

import nbformat

from tools import measure_course
from tools.measure_course import RESOURCE_PREFIX


def _cpu_measurement_cells(workload, experiment, assertion):
    prelude = """import json as _course_json
import time as _course_time
_course_started = _course_time.perf_counter()
"""
    suffix = f"""{assertion or "assert True"}
assert course_order == ["notebook body"], "measurement suffix ran before the notebook body"
_course_report = {{
    "workload": {json.dumps(workload)},
    "experiment": {json.dumps(experiment)},
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
                'body_value = 37\ncourse_order = ["notebook body"]',
                metadata={"course": {"experiment": "job"}},
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
        experiment="job",
        workload="cpu regression",
        report_path=report_path,
        assertion="assert body_value == 37",
        timeout=30,
    )

    assert result["status"] == "passed"
    assert result["gpu_allocator_measurement_status"] == "recorded"
    assert result["resource_measurement"] == {
        "workload": "cpu regression",
        "experiment": "job",
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
        experiment="job",
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


def test_generated_measurement_cells_keep_peak_across_notebook_resets(monkeypatch):
    class FakeCuda:
        def __init__(self):
            self.allocated = 0
            self.reserved = 0
            self.reset_calls = 0

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def init():
            return None

        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def get_device_name():
            return "fake CUDA"

        def reset_peak_memory_stats(self, *_args, **_kwargs):
            self.allocated = 0
            self.reserved = 0
            self.reset_calls += 1

        def max_memory_allocated(self):
            return self.allocated

        def max_memory_reserved(self):
            return self.reserved

    fake_torch = ModuleType("torch")
    fake_torch.cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    prelude, summary = measure_course._measurement_cells("test workload", "job", None)
    namespace = {}
    original_reset = fake_torch.cuda.reset_peak_memory_stats
    exec(prelude.source, namespace)
    wrapped_reset = fake_torch.cuda.reset_peak_memory_stats
    assert wrapped_reset is not original_reset

    fake_torch.cuda.allocated, fake_torch.cuda.reserved = 120, 220
    wrapped_reset()
    assert (fake_torch.cuda.allocated, fake_torch.cuda.reserved) == (0, 0)
    fake_torch.cuda.allocated, fake_torch.cuda.reserved = 80, 340
    wrapped_reset()
    assert (fake_torch.cuda.allocated, fake_torch.cuda.reserved) == (0, 0)
    fake_torch.cuda.allocated, fake_torch.cuda.reserved = 90, 300

    output = StringIO()
    with redirect_stdout(output):
        exec(summary.source, namespace)

    report_line = next(
        line for line in output.getvalue().splitlines() if line.startswith(RESOURCE_PREFIX)
    )
    report = json.loads(report_line[len(RESOURCE_PREFIX) :])
    assert report["gpu"]["peak_allocated_mib"] == 120 / 1024**2
    assert report["gpu"]["peak_reserved_mib"] == 340 / 1024**2
    restored_reset = fake_torch.cuda.reset_peak_memory_stats
    assert restored_reset.__func__ is original_reset.__func__
    assert restored_reset.__self__ is original_reset.__self__
