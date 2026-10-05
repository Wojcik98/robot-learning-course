import ast
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
LABS = sorted((REPOSITORY / "tutorials").glob("[0-9]*/*.ipynb"))


@pytest.mark.parametrize("notebook", LABS, ids=lambda path: path.parent.name)
@pytest.mark.parametrize("folder", [".", "tutorials", "lab"])
def test_lab_workspace_setup_finds_repository(notebook, folder, monkeypatch):
    start = notebook.parent if folder == "lab" else REPOSITORY / folder
    monkeypatch.chdir(start)
    nodes = []
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if (
                isinstance(node, ast.FunctionDef) and node.name in {"find_root", "find_workspace"}
            ) or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in {"WORKSPACE", "workspace"}
                    for target in node.targets
                )
            ):
                nodes.append(node)
    assert nodes, f"No workspace setup found in {notebook}"
    namespace = {"Path": Path}

    # This runs only the repository-root discovery code extracted from notebooks.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(notebook), "exec"), namespace)  # noqa: S102

    workspace = namespace.get("WORKSPACE", namespace.get("workspace"))
    assert workspace == REPOSITORY
    assert (workspace / "manifests/act_references.yaml").is_file()
    assert Path.cwd() == start
