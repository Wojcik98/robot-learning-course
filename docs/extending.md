# Add a lesson

This guide shows how to add one lesson that reuses an existing policy and evaluator. You change no method code. You add a config, a fixture reference, a notebook, an exercise, a solution, and an index entry. Run every command from the repository root.

## 1. Rules

1. Reuse a policy from `src/vla_lab/policy/`. Do not edit it.
2. Reuse an evaluator from `src/vla_lab/evaluation/`. Do not edit it.
3. Take state IDs from `configs/eval/splits.yaml`. Do not choose new task states.
4. Use development IDs 0 to 19. Final IDs 20 to 49 need a frozen recipe.
5. Read saved results from `artifacts/reference/` first. Run a live step only as an optional cell.
6. Keep notebook outputs small. Store large files under `.local-data/`.
7. Write in short sentences. Use the same word for the same thing.

## 2. Files for one lesson

Use `NN-name` for the lesson folder. `NN` is the next free lesson number.

| Purpose | Path |
|---|---|
| Notebook | `tutorials/NN-name/NN-name.ipynb` |
| Guide | `tutorials/NN-name/README.md` |
| Exercise | `tutorials/NN-name/exercise.md` |
| Solution | `solutions/NN-name.md` |
| Config (if needed) | `configs/experiments/name.yaml` |
| Index entry | one line in `tutorials/index.ipynb` |
| Fixture | An existing file under `artifacts/reference/`. Add a new compact file only if no existing file fits. |

Copy the structure of [Lab 9](../tutorials/09-runtime/README.md) or [Lab 11](../tutorials/11-deployment/README.md). Each guide names its inputs, compute, cloud cost, and limits.

## 3. Steps

1. Pick the existing parts: policy, evaluator, config, fixture, and states.
2. Write the dry run in section 4. Run it. Fix every failure before you write the notebook.
3. Write the notebook. Start with a markdown cell that gives the goal, inputs, compute, cloud cost, and limit. Find the repository root. Load tracked records. Check hashes. Add assertions for each claim.
4. Mark any GPU or cloud cell as optional. Gate it with a flag that is `False` by default.
5. Write the exercise. Make each answer checkable from a saved file or a printed value.
6. Write the solution with the exact expected values.
7. Add the index entry. Put the lesson link, one sentence, and a learner-hour estimate in `tutorials/index.ipynb`.
8. Run the notebook from a fresh kernel (section 6).
9. Report the exact result. Do not state a time or a cost that you did not measure.

## 4. Concrete dry run: one example lesson

The example lesson is "Re-read one ACT result". It is a worked example. It is not part of the course. It reuses:

| Part | Existing component |
|---|---|
| Policy | ACT, `bowl_plate` checkpoint. Hash `e14e1ef8866623882b92eeaad8deda1c22acbf798e9bef3d27d700eb64216107` in `manifests/act_references.yaml`. |
| Evaluator | `vla_lab.evaluation.states.selected_state_ids` and the `robot-lab eval` command. |
| Config | `configs/eval/splits.yaml`, `bowl_plate` development IDs. |
| Fixture | `artifacts/reference/act/bowl_plate_seeded_dev_all/episodes.jsonl`. |
| States | Development IDs 0, 1, 2, 3, 4. These IDs already exist. The example selects no new state. |
| Notebook style | [Lab 11](../tutorials/11-deployment/11-deployment.ipynb). |
| Solution style | [Lab 11 solution](../solutions/11-deployment.md). |

### 4.1 Check the states and the fixture

This command runs on CPU. It starts no simulator. It writes no file.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from vla_lab.evaluation.states import selected_state_ids

ids = selected_state_ids("configs", "bowl_plate", "development", [0, 1, 2, 3, 4])
path = Path("artifacts/reference/act/bowl_plate_seeded_dev_all/episodes.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines()]
rows = [row for row in rows if row["state_id"] in ids]
assert [row["reset_seed"] for row in rows] == [20260930 + i for i in ids]
print(ids, sum(row["success"] for row in rows), "of", len(rows))
PY
```

Expected output:

```text
[0, 1, 2, 3, 4] 5 of 5
```

This output was checked on 2026-10-01 with the files in this repository.

### 4.2 Check that the evaluator rejects a final state

```bash
.venv/bin/python -c "
from vla_lab.evaluation.states import selected_state_ids
selected_state_ids('configs', 'bowl_plate', 'final', [20])
"
```

Expected result: a `ValueError` with the message `Final states need an explicit final-evaluation flag`. The lesson must not pass a final ID.

### 4.3 Check the checkpoint hash

If the checkpoint exists on your host:

```bash
sha256sum .local-data/runs/act-bowl-plate-probe-200/checkpoint-000200/model.safetensors
```

Compare the value with the one in `manifests/act_references.yaml`. If the file is absent, the notebook must print `absent` and continue. It must not fail.

### 4.4 Check the index links

```bash
.venv/bin/python - <<'PY'
import json, re
from pathlib import Path

index = json.loads(Path("tutorials/index.ipynb").read_text())
links = re.findall(r"\]\(([^)#]+\.ipynb)\)", "".join(index["cells"][0]["source"]))
missing = [link for link in links if not (Path("tutorials") / link).is_file()]
print(len(links), "links; missing:", missing)
PY
```

Run it again after you add the index entry. The list of missing files must be empty.

### 4.5 Optional live step

To run the same five states live, use the existing evaluator on a GPU host. Use a new output path:

```bash
.venv/bin/robot-lab eval \
  --checkpoint .local-data/runs/act-bowl-plate-probe-200/checkpoint-000200 \
  --output .local-data/runs/example-lesson-eval \
  --state-id 0 --state-id 1 --state-id 2 --state-id 3 --state-id 4
```

The saved fixture shows 5 of 5 successes for these states. A new run can differ. State this in the lesson. Keep the live step optional.

## 5. What the new lesson must say

- Name the compute and the cloud cost. Write US$0 only if no paid job runs.
- Name what the lesson does not prove.
- Keep failed attempts in every denominator.
- Name the partition. Do not tune on final states.

## 6. Run the notebook from a fresh kernel

```bash
.venv/bin/python - <<'PY'
import nbformat
from nbclient import NotebookClient

path = "tutorials/NN-name/NN-name.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
NotebookClient(notebook, timeout=300, kernel_name="python3",
               resources={"metadata": {"path": "tutorials/NN-name"}}).execute()
errors = [o for c in notebook.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
print("errors:", len(errors))
PY
```

This command executes the notebook in memory. It does not change the file. Replace `NN-name` with your folder.

Then run the tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q
```

## 7. Limits of this guide

- Lab 11 is a completed example of reuse. Its [notebook](../tutorials/11-deployment/11-deployment.ipynb) reads the ACT checkpoint hash from `manifests/act_references.yaml`, uses `vla_lab.evaluation.paired`, reads `configs/experiments/comparison.yaml` and the saved ACT and recovery fixtures, and has an [exercise](../tutorials/11-deployment/exercise.md), a [solution](../solutions/11-deployment.md), and an entry in the course index. Its [fresh-kernel check](../artifacts/reference/deployment/lab11_notebook.json) records nine executed code cells and zero errors. No policy or evaluator code changed for that lesson.
- The dry-run example in section 4 is a plan. It has no separate notebook in the course.
- The dry run checks states, fixture rows, and index links. It does not run a policy or a simulator.
- The index link check reads only the first index cell.
- A new method or a new task needs new code and new tests. This guide does not cover that work.
