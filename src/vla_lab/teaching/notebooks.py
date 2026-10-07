"""Select notebook experiments without changing their code."""

from copy import deepcopy

import nbformat


def select_cells(
    notebook: nbformat.NotebookNode,
    *,
    profile: str = "quick",
    experiment: str | None = None,
) -> tuple[nbformat.NotebookNode, list[str], list[str]]:
    """Select core cells and complete experiment groups in source order.

    Quick runs contain only core cells. Full runs contain all experiments,
    or the named experiment and its dependencies. Dependencies must occur
    before their consumers; cycles and unknown names are errors.
    """
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be quick or full")
    if experiment is not None and profile != "full":
        raise ValueError("select an experiment with the full profile")
    dependencies: dict[str, set[str]] = {}
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    for index, cell in enumerate(notebook.cells):
        course = cell.metadata.get("course", {})
        name = course.get("experiment")
        if name is None:
            if course.get("requires"):
                raise ValueError(f"cell {cell.id}: dependencies require an experiment name")
            continue
        if not isinstance(name, str) or not name.strip():
            raise ValueError("experiment names must be nonempty strings")
        requires = course.get("requires", [])
        if not isinstance(requires, list) or not all(isinstance(item, str) for item in requires):
            raise ValueError(f"{name}: requires must contain experiment names")
        dependencies.setdefault(name, set()).update(requires)
        first.setdefault(name, index)
        last[name] = index
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> set[str]:
        if name not in dependencies:
            raise ValueError(f"unknown experiment: {name}")
        if name in visiting:
            raise ValueError(f"experiment dependency cycle at {name}")
        visiting.add(name)
        closure = {name}
        for dependency in sorted(dependencies[name]):
            closure.update(visit(dependency))
            if last[dependency] >= first[name]:
                raise ValueError(f"{dependency} must finish before {name}")
        visiting.remove(name)
        visited.add(name)
        return closure

    for name in dependencies:
        if name not in visited:
            visit(name)
    if experiment is not None and experiment not in dependencies:
        raise ValueError(f"unknown experiment: {experiment}")
    selected = (
        set()
        if profile == "quick"
        else (visit(experiment) if experiment is not None else set(dependencies))
    )
    result = deepcopy(notebook)
    result.cells = [
        cell
        for cell in result.cells
        if cell.metadata.get("course", {}).get("experiment") in selected | {None}
    ]
    return result, sorted(selected), sorted(set(dependencies) - selected)
