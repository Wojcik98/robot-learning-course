"""Check that course tools stop their kernels on success and failure."""

import jupyter_client
import nbformat
import pytest
from jupyter_core.utils import run_sync

from tools import check_course, measure_course


@pytest.mark.parametrize("tool", ["validator", "measurement"])
@pytest.mark.parametrize("fails", [False, True])
def test_course_tool_stops_kernel(tmp_path, monkeypatch, tool, fails):

    managers = []
    original = jupyter_client.AsyncKernelManager

    def tracked_manager(*args, **kwargs):
        manager = original(*args, **kwargs)
        managers.append(manager)
        return manager

    monkeypatch.setattr(check_course, "AsyncKernelManager", tracked_manager)
    monkeypatch.setattr(measure_course, "AsyncKernelManager", tracked_manager)
    path = tmp_path / "unit-reference.ipynb"
    source = 'raise ValueError("exercise failure")' if fails else "value = 1"
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(source, metadata={"course": {"experiment": "job"}})]
    )
    nbformat.write(notebook, path)
    try:
        if tool == "validator":
            check_course._execute_notebook(notebook, path, timeout=30)
        else:
            monkeypatch.setattr(measure_course, "_find_workspace", lambda _: tmp_path)
            monkeypatch.setattr(
                measure_course,
                "_measurement_cells",
                lambda *args: [
                    nbformat.v4.new_code_cell("pass"),
                    nbformat.v4.new_code_cell(
                        f'print({measure_course.RESOURCE_PREFIX!r} + "{{}}")'
                    ),
                ],
            )
            measure_course.measure_notebook(
                path,
                experiment="job",
                workload="CPU kernel cleanup",
                report_path=tmp_path / "report.json",
                timeout=30,
            )
        assert len(managers) == 1
        assert not managers[0].has_kernel, "The course tool left its kernel running"
    finally:
        # Stop any leaked kernel even when this regression check fails.
        for manager in managers:
            if manager.has_kernel:
                run_sync(manager.shutdown_kernel)(now=True)
