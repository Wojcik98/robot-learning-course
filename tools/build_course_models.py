#!/usr/bin/env python
"""Build the learner and reference notebooks for course units 7-12 (model internals).

    .venv/bin/python tools/build_course_models.py                 # write all notebooks
    .venv/bin/python tools/build_course_models.py --unit 07-attention
    .venv/bin/python tools/build_course_models.py --check         # also validate structure
    .venv/bin/python tools/build_course_models.py --execute-reference [--unit ...]
    .venv/bin/python tools/build_course_models.py --execute-learner   [--unit ...]

Each unit is written once as a list of items and rendered twice:

* the *learner* notebook has a blank, named ``NotImplementedError`` stub for every exercise;
* the *reference* notebook has the real solution source of the same function.

Reference solutions are taken from the tested package modules with ``inspect.getsource``, so a
notebook answer and the package function cannot drift apart. Notebooks are written without
outputs. Executed copies live only in memory (``--execute-*``).
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
COURSE = ROOT / "tutorials"

# --- Notebook items ------------------------------------------------------------------------------


@dataclass
class Md:
    text: str


@dataclass
class Code:
    """A code cell that is identical in both notebooks."""

    source: str


@dataclass
class Pair:
    """A code cell that differs: a blank for the learner, a worked version for the reference."""

    learner: str
    reference: str


@dataclass
class Exercise:
    key: str
    title: str
    minutes: str
    purpose: str
    contract: str
    method: str
    pointers: str
    hint: str
    names: list[str]  # package function names whose source is the reference answer
    stub: str  # learner code with NotImplementedError("Complete Exercise <key> ...")
    check: str
    use_intro: str
    use: str
    module: str = ""  # module that holds the reference functions
    extra_solution: str = ""  # extra source appended to the reference cell


@dataclass
class Lesson:
    folder: str
    slug: str
    title: str
    items: list = field(default_factory=list)


def tidy(text: str) -> str:
    return textwrap.dedent(text).strip("\n")


def source_of(module_name: str, names: list[str]) -> str:
    module = __import__(module_name, fromlist=["*"])
    return "\n\n\n".join(inspect.getsource(getattr(module, name)).rstrip() for name in names)


SETUP = """
from pathlib import Path
import os
import sys


def find_root():
    for path in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (path / "manifests" / "act_references.yaml").is_file():
            return path
    raise FileNotFoundError("Start Jupyter inside the repository tree.")


WORKSPACE = find_root()
if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))

# Set cache paths and offline mode before any data or model library is imported.
os.environ["HF_HUB_CACHE"] = str(WORKSPACE / ".local-data/hf/hub")
os.environ["HF_DATASETS_CACHE"] = str(WORKSPACE / ".local-data/hf/datasets")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("TORCH_HOME", str(WORKSPACE / ".local-data/torch"))
CONFIGS, DATA = WORKSPACE / "configs", WORKSPACE / ".local-data"
{imports}
print("Workspace:", WORKSPACE)
"""


def setup_cell(imports: str) -> Code:
    return Code(
        "# Supplied: workspace, offline caches, imports\n"
        + tidy(SETUP).replace("{imports}", tidy(imports))
    )


# --- Rendering ------------------------------------------------------------------------------------


def _cell(kind: str, source: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def _exercise_cells(ex: Exercise, reference: bool) -> list[dict]:
    header = (
        f"## Exercise {ex.key} — {ex.title} (about {ex.minutes} min)\n\n"
        f"**Purpose.** {ex.purpose}\n\n"
        f"**Contract (shapes, units, masks).**\n\n{tidy(ex.contract)}\n\n"
        f"**Method and source.** {tidy(ex.method)}\n\n"
        f"**Pointers.** {tidy(ex.pointers)}\n\n"
        f"<details><summary>Detailed hint</summary>\n\n{tidy(ex.hint)}\n\n</details>"
    )
    if reference:
        solution = source_of(ex.module, ex.names) if ex.names else ""
        if ex.extra_solution:
            solution += "\n\n\n" + tidy(ex.extra_solution)
        code = (
            f"# Reference solution for Exercise {ex.key}\nfrom __future__ import annotations\n\n"
            + solution
        )
        header += "\n\n*This is the reference notebook: the next cell is a complete solution.*"
    else:
        code = "# Implement: Exercise " + ex.key + "\n" + tidy(ex.stub)
    cells = [
        _cell("markdown", header),
        _cell("code", code),
        _cell("code", "# Check: Exercise " + ex.key + "\n" + tidy(ex.check)),
        _cell("markdown", f"**Use your function.** {tidy(ex.use_intro)}"),
        _cell("code", "# Use your function: Exercise " + ex.key + "\n" + tidy(ex.use)),
    ]
    return cells


def render(lesson: Lesson, reference: bool) -> dict:
    cells: list[dict] = []
    in_experiment = False
    for item in lesson.items:
        if isinstance(item, Md):
            markdown = tidy(item.text)
            heading = next(
                (line.strip() for line in markdown.splitlines() if line.startswith("## ")), ""
            )
            if in_experiment and heading and not heading.startswith("## Experiment —"):
                in_experiment = False
            cells.append(_cell("markdown", markdown))
            if heading.startswith("## Experiment —"):
                cells.append(
                    _cell(
                        "code",
                        "# Larger training experiment: run only when you opt in.\nRUN_EXPERIMENT = False",
                    )
                )
                in_experiment = True
        elif isinstance(item, Code):
            source = tidy(item.source)
            if in_experiment:
                source = "if RUN_EXPERIMENT:\n" + textwrap.indent(source, "    ")
            cells.append(_cell("code", source))
        elif isinstance(item, Pair):
            source = tidy(item.reference if reference else item.learner)
            if in_experiment:
                source = "if RUN_EXPERIMENT:\n" + textwrap.indent(source, "    ")
            cells.append(_cell("code", source))
        elif isinstance(item, Exercise):
            cells.extend(_exercise_cells(item, reference))
        else:
            raise TypeError(item)
    for index, cell in enumerate(cells):
        cell["id"] = f"{lesson.slug[:12]}-{index:03d}".replace(".", "-")
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_paths(lesson: Lesson) -> tuple[Path, Path]:
    folder = COURSE / lesson.folder
    return folder / f"{lesson.slug}.ipynb", folder / f"{lesson.slug}-reference.ipynb"


# --- Validation -----------------------------------------------------------------------------------


def _params(source: str, name: str) -> list[str]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            arguments = node.args
            return [a.arg for a in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)]
    raise KeyError(name)


def validate(lesson: Lesson) -> list[str]:
    """Return a list of problems; an empty list means the lesson is well formed."""
    problems: list[str] = []
    learner_path, reference_path = notebook_paths(lesson)
    for path, reference in ((learner_path, False), (reference_path, True)):
        notebook = json.loads(path.read_text())
        text = json.dumps(notebook)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code" and (
                cell["outputs"] or cell["execution_count"] is not None
            ):
                problems.append(f"{path.name}: saved output or execution count")
        if "find_root" not in text or "WORKSPACE" not in text:
            problems.append(f"{path.name}: missing workspace discovery")
        if "manifests" not in text or "HF_HUB_OFFLINE" not in text:
            problems.append(f"{path.name}: missing manifest discovery or offline setup")
        if "http" in text and "download" in text.lower() and "no download" not in text.lower():
            pass
    exercises = [item for item in lesson.items if isinstance(item, Exercise)]
    learner = json.loads(learner_path.read_text())
    reference = json.loads(reference_path.read_text())
    learner_code = ["".join(c["source"]) for c in learner["cells"] if c["cell_type"] == "code"]
    reference_code = ["".join(c["source"]) for c in reference["cells"] if c["cell_type"] == "code"]
    learner_markdown = "\n".join(
        "".join(c["source"]) for c in learner["cells"] if c["cell_type"] == "markdown"
    )
    for ex in exercises:
        if f"Complete Exercise {ex.key}" not in ex.stub:
            problems.append(
                f"{ex.key}: stub must raise NotImplementedError('Complete Exercise {ex.key} ...')"
            )
        if "NotImplementedError" not in ex.stub:
            problems.append(f"{ex.key}: stub has no NotImplementedError")
        if "<details>" not in learner_markdown or ex.key not in learner_markdown:
            problems.append(f"{ex.key}: missing collapsed hint")
        if "Use your function" not in "\n".join(learner_code):
            problems.append(f"{ex.key}: missing use cell")
        solution = next(
            (
                c
                for c in reference_code
                if c.startswith(f"# Reference solution for Exercise {ex.key}\n")
            ),
            "",
        )
        solution = solution.replace("from __future__ import annotations\n", "")
        if "NotImplementedError" in solution:
            problems.append(f"{ex.key}: reference has a stub")
        for name in ex.names:
            tree = ast.parse(solution)
            defined = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
            if not defined:
                problems.append(f"{ex.key}: reference does not define {name}")
                continue
            body = [
                n
                for n in defined[0].body
                if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
            ]
            if len(body) < 2 and not isinstance(body[0], (ast.Return, ast.For, ast.If)):
                problems.append(f"{ex.key}: reference {name} is not a real implementation")
            if (
                len(body) == 1
                and isinstance(body[0], ast.Return)
                and isinstance(body[0].value, ast.Call)
            ):
                callee = ast.unparse(body[0].value.func)
                if callee.startswith(("vla_lab.", "teaching.")):
                    problems.append(f"{ex.key}: reference {name} only calls the package")
            if ex.names and f"def {name}(" not in ex.stub:
                problems.append(f"{ex.key}: stub does not define {name}")
            elif _params(tidy(ex.stub), name) != _params(solution, name):
                problems.append(f"{ex.key}: stub and solution parameters differ for {name}")
        if any(
            line.strip().startswith(("from vla_lab", "import vla_lab"))
            and any(n in line for n in ex.names)
            for line in solution.splitlines()
        ):
            problems.append(f"{ex.key}: reference imports the answer instead of defining it")
    ids = [ex.key for ex in exercises]
    if len(ids) != len(set(ids)):
        problems.append("duplicate exercise keys")
    return problems


# --- Execution ------------------------------------------------------------------------------------


def execute(
    path: Path,
    *,
    timeout: int = 900,
    enable: tuple[str, ...] = (),
    show: bool = False,
    replace: tuple[str, ...] = (),
) -> tuple[bool, str, float]:
    """Run a notebook in a fresh kernel without saving it. Returns ``(ok, error text, seconds)``.

    ``enable`` switches opt-in flags on in memory (``RUN_X = False`` becomes ``RUN_X = True``);
    ``replace`` holds ``"old::new"`` text substitutions for bounded validation runs.
    """
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        for flag in enable:
            cell.source = cell.source.replace(f"{flag} = False", f"{flag} = True")
        for pair in replace:
            old, new = pair.split("::", 1)
            cell.source = cell.source.replace(old, new)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    started = time.perf_counter()
    try:
        client.execute()
    except CellExecutionError as error:
        return False, str(error), time.perf_counter() - started
    finally:
        if show:  # print text outputs for review; the notebook file is never rewritten
            for cell in notebook.cells:
                for output in cell.get("outputs", []):
                    text = output.get("text") or output.get("data", {}).get("text/plain", "")
                    if text:
                        print(
                            "   |",
                            cell.source.splitlines()[0][:60],
                            "->",
                            str(text).strip()[:900].replace("\n", "\n   |   "),
                        )
    return True, "", time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--unit", action="append", help="folder name such as 07-attention (repeatable)"
    )
    parser.add_argument("--check", action="store_true", help="validate the written notebooks")
    parser.add_argument("--execute-reference", action="store_true")
    parser.add_argument("--execute-learner", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        help='in-memory text substitution "old::new" for bounded runs',
    )
    parser.add_argument(
        "--show", action="store_true", help="print text outputs of the executed reference notebook"
    )
    parser.add_argument(
        "--enable",
        action="append",
        default=[],
        help="opt-in flag to switch on, e.g. RUN_REAL_ACT_ENCODER",
    )
    args = parser.parse_args(argv)

    selected = [lesson for lesson in lessons() if not args.unit or lesson.folder in args.unit]
    if not selected:
        print("no matching unit")
        return 2
    status = 0
    for lesson in selected:
        learner_path, reference_path = notebook_paths(lesson)
        learner_path.parent.mkdir(parents=True, exist_ok=True)
        for path, reference in ((learner_path, False), (reference_path, True)):
            path.write_text(
                json.dumps(render(lesson, reference), indent=1, ensure_ascii=False) + "\n"
            )
        print(f"wrote {learner_path.relative_to(ROOT)} and {reference_path.name}")
        if args.check or args.execute_reference or args.execute_learner:
            problems = validate(lesson)
            for problem in problems:
                print("  PROBLEM:", problem)
            status |= bool(problems)
        if args.execute_reference:
            ok, error, seconds = execute(
                reference_path,
                timeout=args.timeout,
                enable=tuple(args.enable),
                show=args.show,
                replace=tuple(args.replace),
            )
            print(f"  reference execution: {'ok' if ok else 'FAILED'} in {seconds:.0f} s")
            if not ok:
                print(error[-2500:])
                status = 1
        if args.execute_learner:
            first = next(item for item in lesson.items if isinstance(item, Exercise))
            ok, error, seconds = execute(learner_path, timeout=args.timeout)
            expected = f"Complete Exercise {first.key}"
            if ok or expected not in error:
                print(f"  learner notebook did not stop at Exercise {first.key}:\n{error[-1500:]}")
                status = 1
            else:
                print(
                    f"  learner notebook stops at Exercise {first.key} as intended ({seconds:.0f} s)"
                )
    return status


# =====================================================================================================
# Shared pieces
# =====================================================================================================

CLOSE_HELPER = r'''
def close(actual, expected, message, atol=1e-5):
    """Compare two tensors and say what failed."""
    assert actual.shape == expected.shape, f"{message}: shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    worst = (actual.detach() - expected.detach()).abs().max().item()
    assert worst <= atol, f"{message}: largest difference {worst:.2e} is above {atol:.0e}"
'''

PYTORCH_NOTES_ATTENTION = """
## PyTorch notes

- A tensor has a **shape**, such as `(B, T, D)` = batch, positions, features. Most mistakes are shape mistakes. Print `x.shape` often.
- **Broadcasting** lets `(B, 1, 1, Tk)` combine with `(B, H, Tq, Tk)`. A size-1 axis repeats.
- `x.masked_fill(mask, value)` writes `value` where `mask` is `True`. `softmax(dim=-1)` normalizes over the last axis.
- `transpose(a, b)` swaps two axes without copying. After `transpose`, use `reshape`, not `view`.
- `loss.backward()` fills `.grad` on every tensor created with `requires_grad=True`. Padding must receive zero gradient.
- `module.train()` enables dropout. `module.eval()` disables it. The checks below use `eval()` or zero dropout.
"""

REAL_WINDOW_NOTE = """
**Real data used in this unit.** The cells below read real LIBERO `bowl_plate` demonstrations from local files
(no download starts here). Each *window* is 16 consecutive actions (7 values each). A window that runs past the end of an
episode repeats the last action and sets `is_pad=True` for those steps. The source data is recorded at 10 Hz;
nothing is resampled to the simulator's 20 Hz controller. If the files are missing, the cell tells you the one setup command.
"""


def lesson_07() -> Lesson:
    L = Lesson("07-attention", "07-attention", "Attention by hand")
    L.items += [
        Md(r"""
        # Unit 7 — Attention by hand

        **Your result:** a Transformer encoder layer written with plain tensor operations. It will match the encoder blocks inside the installed ACT model, for both outputs and gradients.

        Five short exercises (about 10–20 minutes each) build it from the inside out: scaled attention with a padding mask, heads, multi-head attention, positions, then the layer. A larger experiment (30–60 minutes) uses your layer to learn from real, padded action chunks. These times are estimates, not measurements.

        **Sources.** The equations follow [Attention Is All You Need](https://arxiv.org/abs/1706.03762), sections 3.1–3.5: encoder layer (3.1), scaled dot-product and multi-head attention (3.2), feed-forward (3.3), positional encoding (3.5). ACT uses this block in its [paper](https://arxiv.org/abs/2304.13705); its code is in `lerobot/policies/act/modeling_act.py` at the pinned revision `e0d50211ef236143ae867228662b7dfaba554f02`.

        **You need** Python and NumPy. PyTorch is explained as it appears. Run from any folder inside this repository. Full answers are in [the reference notebook](07-attention-reference.ipynb). Three things stay separate: *your code* (cells you write), the *package reference* (`vla_lab.teaching.attention`, used here only as an oracle in checks), and the *installed LeRobot ACT blocks* (the pinned upstream code).
        """),
        setup_cell("""
            import math
            import torch
            import torch.nn.functional as F
            from torch import nn
            import matplotlib.pyplot as plt

            import vla_lab.teaching.attention as ref  # package reference: used by checks only
            from lerobot.policies.act.configuration_act import ACTConfig
            from lerobot.policies.act.modeling_act import ACTEncoder, ACTEncoderLayer, create_sinusoidal_pos_embedding
            from vla_lab.teaching.diffusion import load_state_action_windows

            torch.manual_seed(0)
            print("PyTorch:", torch.__version__)
        """),
        Code("# Supplied: a helper that the checks use\n" + tidy(CLOSE_HELPER)),
        Md(PYTORCH_NOTES_ATTENTION),
        Md(REAL_WINDOW_NOTE),
        Code(r"""
            # Supplied: real padded action windows (train and validation episodes are kept apart)
            def read_windows(split):
                return load_state_action_windows(
                    CONFIGS, DATA, "bowl_plate", split,
                    horizon=16, n_obs_steps=1, action_start=0, drop_n_last_frames=0,
                )

            TRAIN, VALID = read_windows("train"), read_windows("validation")
            padded_counts = VALID["action_is_pad"].sum(1)
            EXAMPLE = int((padded_counts == 6).nonzero()[0])   # a window whose last 6 steps are padding
            example_actions = VALID["action"][EXAMPLE]          # (16, 7), scaled to [-1, 1]
            example_pad = VALID["action_is_pad"][EXAMPLE]       # (16,), True for padding
            print("train windows:", len(TRAIN["action"]), "validation windows:", len(VALID["action"]))
            print("example window padding:", example_pad.int().tolist())
        """),
    ]

    L.items.append(
        Exercise(
            key="7.1",
            title="Scaled dot-product attention with a padding mask",
            minutes="15–20",
            purpose="Write the one function that every Transformer block calls. It decides how much each query reads from each key.",
            contract="""
        - `query`: `(B, H, Tq, Dh)`; `key`, `value`: `(B, H, Tk, Dh)`. `B` batch, `H` heads, `T*` positions, `Dh` features per head.
        - `key_padding_mask`: `(B, Tk)` bool or `None`. **`True` means padding**: that key must get weight exactly 0.
        - Return `(output, weights)`: `output (B, H, Tq, Dh)` and `weights (B, H, Tq, Tk)`. Each row of `weights` sums to 1 over the real keys.
        - A query row whose keys are all padding returns zero weights and zero output. It must not contain `nan`, and its gradient must be finite.
        - `dropout_p`: probability of dropping attention weights in training. You may ignore it here; later modules pass 0 in evaluation.
        """,
            method="`Attention(Q, K, V) = softmax(Q Kᵀ / √Dh) V` — [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762), section 3.2.1. Padded keys are blocked before the softmax.",
            pointers="Get all query–key scores with one matrix product. Scale them. Remove padded keys. Normalize over the key axis (the last one). Then take the weighted sum of values. Do not call torch's attention helpers: the check uses them as an oracle, and they do not return the weights you need.",
            hint="""
        `scores = query @ key.transpose(-2, -1) / math.sqrt(Dh)` has shape `(B, H, Tq, Tk)`.
        `blocked = key_padding_mask[:, None, None, :]` broadcasts over heads and queries.
        Fill blocked scores with `torch.finfo(scores.dtype).min` (a large negative but finite number), then `softmax(dim=-1)`.
        A row that is entirely blocked now has equal weights; set blocked positions of `weights` to 0 with `masked_fill(blocked, 0.0)` so that row becomes all zero.
        Optional dropout: `F.dropout(weights, p=dropout_p, training=True)` on the weights, just before multiplying by `value`; return the weights without dropout.
        """,
            names=["scaled_dot_product_attention"],
            module="vla_lab.teaching.attention",
            stub="""
        def scaled_dot_product_attention(query, key, value, key_padding_mask=None, dropout_p=0.0):
            # query (B,H,Tq,Dh); key,value (B,H,Tk,Dh); key_padding_mask (B,Tk) with True = padding.
            # Return (output (B,H,Tq,Dh), weights (B,H,Tq,Tk)).
            raise NotImplementedError("Complete Exercise 7.1: scaled_dot_product_attention")
        """,
            check="""
        torch.manual_seed(1)
        B_, H_, Tq, Tk, Dh = 3, 4, 5, 6, 8
        q, k, v = (torch.randn(B_, H_, n, Dh, requires_grad=True) for n in (Tq, Tk, Tk))
        pad = torch.zeros(B_, Tk, dtype=torch.bool)
        pad[1, 4:] = True
        pad[2, 2:] = True
        out, weights = scaled_dot_product_attention(q, k, v, pad)
        assert out.shape == (B_, H_, Tq, Dh) and weights.shape == (B_, H_, Tq, Tk), "Return (output, weights) with the documented shapes."
        allowed = (~pad)[:, None, None, :].expand(B_, H_, Tq, Tk)
        close(out, F.scaled_dot_product_attention(q, k, v, attn_mask=allowed), "output differs from the oracle")
        close(weights.sum(-1), torch.ones(B_, H_, Tq), "each weight row must sum to 1")
        assert weights.masked_select(pad[:, None, None, :].expand_as(weights)).abs().sum() == 0, "Padded keys must get weight 0."
        no_mask, _ = scaled_dot_product_attention(q, k, v)
        close(no_mask, F.scaled_dot_product_attention(q, k, v), "output without a mask differs from the oracle")

        out.square().sum().backward()
        q2, k2, v2 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        F.scaled_dot_product_attention(q2, k2, v2, attn_mask=allowed).square().sum().backward()
        for name, mine, theirs in (("query", q, q2), ("key", k, k2), ("value", v, v2)):
            close(mine.grad, theirs.grad, f"gradient of {name}", atol=1e-4)
        assert k.grad[1, :, 4:].abs().sum() == 0 and v.grad[2, :, 2:].abs().sum() == 0, "Padded keys and values must receive zero gradient."

        changed = v.detach().clone()
        changed[pad[:, None, :, None].expand_as(changed)] = 50.0
        close(scaled_dot_product_attention(q.detach(), k.detach(), changed, pad)[0], out.detach(), "changing padded values must not change the output")

        all_pad = torch.tensor([[True] * Tk, [False] * Tk, [False] * Tk])
        qa, ka, va = (torch.randn(3, H_, n, Dh, requires_grad=True) for n in (Tq, Tk, Tk))
        out_a, w_a = scaled_dot_product_attention(qa, ka, va, all_pad)
        out_a.sum().backward()
        assert torch.isfinite(out_a).all() and out_a[0].abs().sum() == 0 and w_a[0].abs().sum() == 0, "A row of only padding must give zero weights and output."
        assert torch.isfinite(qa.grad).all() and torch.isfinite(ka.grad).all(), "Gradients must stay finite for an all-padding row."
        print("Exercise 7.1 passed")
        """,
            use_intro="Attend over a real action window. Tokens are a fixed random projection of the 16 real actions (no training yet), so only the *mask* is meaningful here: padded columns must be black.",
            use="""
        torch.manual_seed(3)
        projection = torch.randn(7, 16)
        tokens = (example_actions @ projection)[None, None]        # (1, 1, 16, 16): one batch row, one head
        out, weights = scaled_dot_product_attention(tokens, tokens, tokens, example_pad[None])
        plt.imshow(weights[0, 0], cmap="viridis")
        plt.xlabel("key position"); plt.ylabel("query position")
        plt.title("Your attention weights; padded keys have weight 0")
        plt.colorbar(); plt.show()
        print("weight on padded keys:", float(weights[0, 0][:, example_pad].sum()))
        """,
        )
    )

    L.items.append(
        Exercise(
            key="7.2",
            title="Split and merge attention heads",
            minutes="10",
            purpose="Several heads let a layer look at several relations at once. Each head works on its own slice of the features.",
            contract="""
        - `split_heads(x, n_heads)`: `x (B, T, D)` → `(B, H, T, D // H)`. Head `h` owns features `h*Dh : (h+1)*Dh`. Raise `ValueError` if `D` is not divisible by `H`.
        - `merge_heads(x)`: `x (B, H, T, Dh)` → `(B, T, H * Dh)`. It exactly undoes `split_heads`.
        """,
            method="Multi-head attention splits the model width into `h` parts of size `d_model / h` — [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762), section 3.2.2.",
            pointers="First reshape the last axis into `(H, Dh)`. Then move the head axis in front of the position axis. Merging reverses both steps in the opposite order.",
            hint="""
        `x.reshape(B, T, H, Dh).transpose(1, 2)` splits. `x.transpose(1, 2).reshape(B, T, H * Dh)` merges.
        `transpose` returns a non-contiguous view, so use `reshape` (not `view`) when merging.
        """,
            names=["split_heads", "merge_heads"],
            module="vla_lab.teaching.attention",
            stub="""
        def split_heads(x, n_heads):
            # (B, T, D) -> (B, H, T, D // H). Raise ValueError when D % H != 0.
            raise NotImplementedError("Complete Exercise 7.2: split_heads")


        def merge_heads(x):
            # (B, H, T, Dh) -> (B, T, H * Dh).
            raise NotImplementedError("Complete Exercise 7.2: merge_heads")
        """,
            check="""
        x = torch.arange(2 * 5 * 12, dtype=torch.float32).reshape(2, 5, 12)
        heads = split_heads(x, 3)
        assert heads.shape == (2, 3, 5, 4), "Expected (B, H, T, D // H)."
        assert torch.equal(heads[1, 2, 3], x[1, 3, 8:12]), "Head 2 must own features 8..11."
        assert torch.equal(heads[0, 0, :, 1], x[0, :, 1]), "Head 0, feature 1 is feature 1 of every position."
        merged = merge_heads(heads)
        assert merged.shape == x.shape and torch.equal(merged, x), "merge_heads must undo split_heads."
        assert torch.equal(merge_heads(split_heads(x, 1)), x) and torch.equal(merge_heads(split_heads(x, 12)), x)
        try:
            split_heads(x, 5)
        except ValueError:
            pass
        else:
            raise AssertionError("split_heads must raise ValueError when 12 is not divisible by 5.")
        print("Exercise 7.2 passed")
        """,
            use_intro="Split the example window's tokens into four heads and look at what each head receives.",
            use="""
        tokens = (example_actions @ projection)[None]                  # (1, 16, 16)
        four = split_heads(tokens, 4)                                  # (1, 4, 16, 4)
        print("split shape:", tuple(four.shape))
        print("head 2 sees features 8..11 of position 0:", four[0, 2, 0].tolist() == tokens[0, 0, 8:12].tolist())
        print("merge restores the tokens:", bool(torch.equal(merge_heads(four), tokens)))
        """,
        )
    )

    L.items.append(
        Exercise(
            key="7.3",
            title="Multi-head attention",
            minutes="15",
            purpose="Combine projections, heads and your Exercise 7.1 function into the layer that real models use.",
            contract="""
        - `attention` is a module with `q_proj`, `k_proj`, `v_proj`, `out_proj` (each `nn.Linear(D, D)`), `n_heads` (int), `dropout` (float) and the `training` flag.
        - `query (B, Tq, D)`, `key (B, Tk, D)`, `value (B, Tk, D)`; `key_padding_mask (B, Tk)` with `True` = padding, or `None`.
        - `attend` is a function with the contract of Exercise 7.1. **Call it; do not repeat its algebra here.**
        - Pass `dropout_p = attention.dropout if attention.training else 0.0` to `attend`.
        - Return `(output (B, Tq, D), weights (B, H, Tq, Tk))`.
        """,
            method="`MultiHead(Q, K, V) = Concat(head_1, …, head_h) Wᴼ`, `head_i = Attention(Q Wᵢ^Q, K Wᵢ^K, V Wᵢ^V)` — section 3.2.2. One wide linear layer followed by a split is the same as `h` small matrices.",
            pointers="Project the three inputs. Split each into heads. Call `attend`. Merge the heads. Apply the output projection. The query and the key/value inputs may have different lengths (cross-attention).",
            hint="""
        ```python
        heads = attention.n_heads
        q = split_heads(attention.q_proj(query), heads)
        k = split_heads(attention.k_proj(key), heads)
        v = split_heads(attention.v_proj(value), heads)
        dropout_p = attention.dropout if attention.training else 0.0
        out, weights = attend(q, k, v, key_padding_mask=key_padding_mask, dropout_p=dropout_p)
        return attention.out_proj(merge_heads(out)), weights
        ```
        Only the last line of the exercise is "new": the rest is the plumbing from Exercises 7.1 and 7.2.
        """,
            names=["multi_head_forward"],
            module="vla_lab.teaching.attention",
            stub="""
        def multi_head_forward(attention, query, key, value, key_padding_mask=None, attend=scaled_dot_product_attention):
            # Use split_heads, attend (Exercise 7.1) and merge_heads. Return (output, weights).
            raise NotImplementedError("Complete Exercise 7.3: multi_head_forward")
        """,
            check="""
        D_, H_ = 16, 4
        calls = []

        def counting_attend(*args, **kwargs):
            calls.append(tuple(args[0].shape))
            return scaled_dot_product_attention(*args, **kwargs)

        torch.manual_seed(2)
        oracle = nn.MultiheadAttention(D_, H_, dropout=0.0, batch_first=True)
        mine = ref.MultiHeadAttention(D_, H_, forward_fn=multi_head_forward, attend_fn=counting_attend)
        ref.load_lerobot_attention(mine, oracle)        # copies the oracle's weights into your module's four layers
        x = torch.randn(3, 5, D_, requires_grad=True)
        memory = torch.randn(3, 7, D_)
        pad = torch.zeros(3, 7, dtype=torch.bool)
        pad[1, 5:] = True
        out = mine(x, memory, memory, pad)
        assert calls == [(3, H_, 5, D_ // H_)], f"Your function must call attend once with head-split queries; saw {calls}."
        expected, _ = oracle(x, memory, memory, key_padding_mask=pad)
        close(out, expected, "cross-attention output differs from torch")
        out.square().sum().backward()
        x2 = x.detach().clone().requires_grad_(True)
        oracle(x2, memory, memory, key_padding_mask=pad)[0].square().sum().backward()
        close(x.grad, x2.grad, "gradient of the query input", atol=1e-4)
        packed = torch.cat([mine.q_proj.weight.grad, mine.k_proj.weight.grad, mine.v_proj.weight.grad])
        close(packed, oracle.in_proj_weight.grad, "gradient of the q/k/v projections", atol=1e-4)
        assert mine.last_weights.shape == (3, H_, 5, 7), "weights must be (B, H, Tq, Tk)."
        assert mine.last_weights[1, :, :, 5:].abs().sum() == 0, "padded memory keys must get weight 0."
        self_out = mine(x, x, x)
        close(self_out, oracle(x, x, x)[0], "self-attention output differs from torch")
        print("Exercise 7.3 passed")
        """,
            use_intro="Run your module on the real window with and without the padding mask, and see which positions change.",
            use="""
        torch.manual_seed(4)
        layer = ref.MultiHeadAttention(16, 4, forward_fn=multi_head_forward, attend_fn=scaled_dot_product_attention)
        tokens = (example_actions @ projection)[None]
        with torch.no_grad():
            without_mask = layer(tokens, tokens, tokens)
            weights_open = layer.last_weights.clone()
            with_mask = layer(tokens, tokens, tokens, example_pad[None])
            weights_masked = layer.last_weights.clone()
        difference = (with_mask - without_mask).abs().sum(-1)[0]
        print("output change at the first real positions:", [round(float(d), 3) for d in difference[~example_pad]][:6], "...")
        print("attention weight on padded keys, no mask:", round(float(weights_open[..., example_pad].sum()), 3))
        print("attention weight on padded keys, with mask:", float(weights_masked[..., example_pad].sum()))
        """,
        )
    )

    L.items.append(
        Exercise(
            key="7.4",
            title="Sinusoidal positional encoding",
            minutes="10–15",
            purpose="Attention does not know the order of tokens. Add a fixed code that tells each position where it is.",
            contract="""
        - `sinusoidal_positions(num_positions, dimension)` returns a `float32` tensor `(num_positions, dimension)`; `dimension` is even.
        - Row `p`, column `j`: angle `= p / 10000 ** (2 * (j // 2) / dimension)`. Even columns use `sin(angle)`; odd columns use `cos(angle)`.
        - Positions count from 0, so row 0 is `[0, 1, 0, 1, …]`.
        """,
            method="`PE(pos, 2i) = sin(pos / 10000^(2i/d))`, `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))` — [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762), section 3.5. ACT builds the same table for its posterior encoder.",
            pointers="Build a column of positions and a row of column indices, divide, then fill the even and odd columns separately. Compute in `float64` and convert to `float32` at the end for the best match.",
            hint="""
        ```python
        positions = torch.arange(num_positions, dtype=torch.float64)[:, None]
        columns = torch.arange(dimension, dtype=torch.float64)[None, :]
        angles = positions / 10000 ** (2 * (columns // 2) / dimension)
        table = torch.zeros_like(angles)
        table[:, 0::2] = angles[:, 0::2].sin()
        table[:, 1::2] = angles[:, 1::2].cos()
        return table.float()
        ```
        """,
            names=["sinusoidal_positions"],
            module="vla_lab.teaching.attention",
            stub="""
        def sinusoidal_positions(num_positions, dimension):
            # Return a float32 tensor (num_positions, dimension): sin on even columns, cos on odd columns.
            raise NotImplementedError("Complete Exercise 7.4: sinusoidal_positions")
        """,
            check="""
        table = sinusoidal_positions(21, 32)
        assert table.shape == (21, 32) and table.dtype == torch.float32, "Expected a float32 (num_positions, dimension) table."
        assert torch.equal(table[0, 0::2], torch.zeros(16)) and torch.equal(table[0, 1::2], torch.ones(16)), "Position 0 must be [0, 1, 0, 1, ...]."
        close(table[3, 0], torch.tensor(math.sin(3.0)), "row 3, column 0 is sin(3)", atol=1e-6)
        close(table[3, 5], torch.tensor(math.cos(3 / 10000 ** (2 * 2 / 32))), "row 3, column 5 is cos(3 / 10000^(4/32))", atol=1e-6)
        close(table, create_sinusoidal_pos_embedding(21, 32), "table differs from the pinned ACT table", atol=1e-6)
        print("Exercise 7.4 passed")
        """,
            use_intro="Look at the table. Nearby positions get similar codes; the dot product of two rows shows it.",
            use="""
        table = sinusoidal_positions(20, 32)
        similarity = table @ table.T
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        axes[0].imshow(table, aspect="auto", cmap="coolwarm"); axes[0].set_title("encoding (position x feature)")
        axes[1].imshow(similarity, cmap="viridis"); axes[1].set_title("similarity between positions")
        plt.tight_layout(); plt.show()
        """,
        )
    )

    L.items.append(
        Exercise(
            key="7.5",
            title="Encoder layer: residual, norm, feed-forward",
            minutes="15–20",
            purpose="Wrap attention with the two residual blocks that make deep Transformers trainable. This is the block ACT stacks four times.",
            contract="""
        - `layer` is a module with `self_attn` (callable `(query, key, value, key_padding_mask) -> (B, T, D)`), `linear1`, `linear2`, `norm1`, `norm2`, `dropout`, `dropout1`, `dropout2`, `activation` and the bool `pre_norm`.
        - `x (B, T, D)`; `pos_embed` is `None` or broadcastable to `(B, T, D)`; `key_padding_mask (B, T)` with `True` = padding.
        - **Position rule:** add `pos_embed` to the *query and key only*, never to the value.
        - Return `(B, T, D)`.
        - Post-norm (`pre_norm=False`): `x = norm1(x + attn(x))`, then `x = norm2(x + ff(x))`.
        - Pre-norm (`pre_norm=True`): `x = x + attn(norm1(x))`, then `x = x + ff(norm2(x))`.
        - `ff(x) = linear2(dropout(activation(linear1(x))))`. Apply `dropout1` to the attention output and `dropout2` to the feed-forward output before each residual add.
        """,
            method="Encoder layer — [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762), sections 3.1 and 3.3. ACT's `ACTEncoderLayer` (pinned) adds the position to the query and key only and uses post-norm by default.",
            pointers="Keep the input in a `skip` variable before each sub-layer. Decide where the norm goes from `layer.pre_norm`. Call `layer.self_attn(q, k, value, key_padding_mask)` with the position added to `q` and `k`.",
            hint="""
        Post-norm pattern (default):
        ```python
        skip = x
        qk = x if pos_embed is None else x + pos_embed
        x = layer.self_attn(qk, qk, x, key_padding_mask)
        x = layer.norm1(skip + layer.dropout1(x))
        skip = x
        x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = layer.norm2(skip + layer.dropout2(x))
        ```
        Pre-norm moves each `norm` in front of its sub-layer and removes it after the add. Write both branches with `if layer.pre_norm:`.
        """,
            names=["encoder_layer_forward"],
            module="vla_lab.teaching.attention",
            stub="""
        def encoder_layer_forward(layer, x, pos_embed=None, key_padding_mask=None):
            # Attention then feed-forward, each with a residual connection and a norm. Position on q and k only.
            raise NotImplementedError("Complete Exercise 7.5: encoder_layer_forward")
        """,
            check="""
        D_, H_, FF_ = 16, 4, 32
        config = lambda pre: ACTConfig(dim_model=D_, n_heads=H_, dim_feedforward=FF_, dropout=0.0, pre_norm=pre,
                                       chunk_size=4, n_action_steps=4, input_features={}, output_features={})
        pad = torch.zeros(3, 6, dtype=torch.bool)
        pad[1, 4:] = True
        pad[2, 2:] = True
        valid = ~pad[..., None]
        for pre in (False, True):
            torch.manual_seed(5)
            oracle = ACTEncoderLayer(config(pre)).eval()
            mine = ref.EncoderLayer(D_, H_, FF_, pre_norm=pre, forward_fn=encoder_layer_forward,
                                    attention_forward_fn=multi_head_forward, attend_fn=scaled_dot_product_attention).eval()
            ref.load_lerobot_encoder_layer(mine, oracle)
            x = torch.randn(3, 6, D_, requires_grad=True)
            pos = torch.randn(1, 6, D_)
            out = mine(x, pos, pad)
            expected = oracle(x.transpose(0, 1), pos_embed=pos.expand(3, -1, -1).transpose(0, 1), key_padding_mask=pad).transpose(0, 1)
            close(out * valid, expected * valid, f"layer output (pre_norm={pre}) differs from ACTEncoderLayer")
            (out * valid).square().sum().backward()
            x2 = x.detach().clone().requires_grad_(True)
            (oracle(x2.transpose(0, 1), pos_embed=pos.expand(3, -1, -1).transpose(0, 1), key_padding_mask=pad).transpose(0, 1) * valid).square().sum().backward()
            close(x.grad, x2.grad, f"input gradient (pre_norm={pre})", atol=1e-4)
            close(mine.linear1.weight.grad, oracle.linear1.weight.grad, f"feed-forward gradient (pre_norm={pre})", atol=1e-4)
            assert not torch.allclose(mine(x, None, pad) * valid, out.detach() * valid, atol=1e-3), "The position must change the result."
            moved = x.detach().clone()
            moved[pad] = 100.0
            close(mine(moved, pos, pad) * valid, out.detach() * valid, f"padded values must not change real outputs (pre_norm={pre})", atol=1e-4)
        print("Exercise 7.5 passed")
        """,
            use_intro="Wire all your functions into a two-layer encoder and compare it, output and gradient, with the installed ACT encoder stack on real action tokens.",
            use="""
        D_, H_, FF_ = 16, 4, 32
        config = ACTConfig(dim_model=D_, n_heads=H_, dim_feedforward=FF_, n_encoder_layers=2, dropout=0.0,
                           chunk_size=4, n_action_steps=4, input_features={}, output_features={})
        torch.manual_seed(6)
        pinned = ACTEncoder(config).eval()
        yours = ref.TransformerEncoder(2, D_, H_, FF_, forward_fn=encoder_layer_forward,
                                       attention_forward_fn=multi_head_forward, attend_fn=scaled_dot_product_attention).eval()
        ref.load_lerobot_encoder(yours, pinned)
        batch_tokens = torch.stack([(VALID["action"][i] @ projection) for i in range(8)]).requires_grad_(True)
        batch_pad = VALID["action_is_pad"][:8]
        positions = sinusoidal_positions(16, D_)[None]
        out = yours(batch_tokens, positions, batch_pad)
        theirs_input = batch_tokens.detach().clone().requires_grad_(True)
        theirs = pinned(theirs_input.transpose(0, 1), pos_embed=positions.expand(8, -1, -1).transpose(0, 1), key_padding_mask=batch_pad).transpose(0, 1)
        real = ~batch_pad[..., None]
        (out * real).sum().backward(); (theirs * real).sum().backward()
        print("largest output difference at real positions:", float(((out - theirs).abs() * real).max()))
        print("largest input-gradient difference:", float((batch_tokens.grad - theirs_input.grad).abs().max()))
        """,
        )
    )
    return L


EXPERIMENT_TABLE_HELPER = r'''
import time
import pandas as pd

def show_trials(trials, columns):
    """Print every trial that was run, including failed ones, and return a table."""
    table = pd.DataFrame([{**t["config"], **{c: t.get(c) for c in columns}, "status": t["status"]} for t in trials])
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return table
'''


def lesson_07_tail(L: Lesson) -> Lesson:
    L.items += [
        Md(r"""
        ## Experiment — where does the padding mask matter? (30–60 min)

        **Question.** Does the key padding mask change what an encoder learns from real padded action chunks, and does it change how the model reacts to padded values?

        **Task (real data).** Input: a 16-step action window from a real demonstration; padded steps repeat the last real action. Target: the mean of the *real* (not padded) actions, 7 values. The pooled output uses only real positions in every run, so any difference comes from attention, not from pooling.

        **Fixed for every trial (do not change).** Windows: `TRAIN` and `VALID` above (validation episodes are never used for updates). Seed `0` for weights, seed `1` for batch draws. **150 updates**, batch 64, AdamW with learning rate `2e-3`. Model width 32. Your own functions from Exercises 7.1–7.5 are used inside every encoder layer.

        **Allowed changes (per trial).** `mask` (pass the padding mask to attention or not), `pos` (add sinusoidal positions or not), `layers` (1–3), `heads` (1, 2 or 4).

        **Measures.** `val_mse_padded`: mean squared error on validation windows that contain padding. `val_mse_all`: all validation windows. `pad_sensitivity`: mean absolute change of the output when the padded steps are replaced by random values (a robust model gives about 0).

        **Prediction first.** Before running anything, write two sentences: which arm do you expect to be worst on `pad_sensitivity`, and will `val_mse_all` change much? Edit the next cell.

        **Keep failed trials.** Every trial is appended to `TRIALS`. Do not delete or overwrite rows; a result that does not improve is a valid result.
        """),
        Code(r"""
            # Write your prediction here (one cell, before you run the trials).
            PREDICTION = "TODO: replace this text with your prediction."
            print(PREDICTION)
        """),
        Code("# Supplied: trial table helper\n" + tidy(EXPERIMENT_TABLE_HELPER)),
        Code(r'''
            # Supplied: the experiment harness. It calls YOUR functions through the encoder hooks.
            STEPS, BATCH, LR, SEED_WEIGHTS, SEED_BATCH, WIDTH = 150, 64, 2e-3, 0, 1, 32


            def masked_mean(values, is_pad):
                keep = (~is_pad).unsqueeze(-1).float()
                return (values * keep).sum(1) / keep.sum(1).clamp_min(1)


            class ChunkEncoder(nn.Module):
                """Linear input layer -> your encoder layers -> mean over real positions -> linear output."""

                def __init__(self, layers=2, heads=4, pos=True):
                    super().__init__()
                    self.inp = nn.Linear(7, WIDTH)
                    self.encoder = ref.TransformerEncoder(
                        layers, WIDTH, heads, 2 * WIDTH,
                        forward_fn=encoder_layer_forward,
                        attention_forward_fn=multi_head_forward,
                        attend_fn=scaled_dot_product_attention,
                    )
                    self.out = nn.Linear(WIDTH, 7)
                    self.register_buffer("positions", sinusoidal_positions(16, WIDTH)[None] if pos else torch.zeros(1, 16, WIDTH))

                def forward(self, actions, is_pad, use_mask=True):
                    hidden = self.encoder(self.inp(actions), self.positions, is_pad if use_mask else None)
                    return self.out(masked_mean(hidden, is_pad))


            BASELINE = {"mask": True, "pos": True, "layers": 2, "heads": 4}
            TRIALS = []


            def run_trial(config):
                started = time.perf_counter()
                torch.manual_seed(SEED_WEIGHTS)
                model = ChunkEncoder(config["layers"], config["heads"], config["pos"])
                optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
                draws = torch.Generator().manual_seed(SEED_BATCH)
                for _ in range(STEPS):
                    pick = torch.randint(0, len(TRAIN["action"]), (BATCH,), generator=draws)
                    actions, pad = TRAIN["action"][pick], TRAIN["action_is_pad"][pick]
                    loss = (model(actions, pad, config["mask"]) - masked_mean(actions, pad)).square().mean()
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                model.eval()
                with torch.no_grad():
                    has_pad = VALID["action_is_pad"].any(1)
                    a, p = VALID["action"][has_pad], VALID["action_is_pad"][has_pad]
                    out = model(a, p, config["mask"])
                    noise = torch.randn(a.shape, generator=torch.Generator().manual_seed(5)) * 3
                    shaken = torch.where(p.unsqueeze(-1), noise, a)
                    all_a, all_p = VALID["action"], VALID["action_is_pad"]
                    return {
                        "val_mse_padded": (out - masked_mean(a, p)).square().mean().item(),
                        "val_mse_all": (model(all_a, all_p, config["mask"]) - masked_mean(all_a, all_p)).square().mean().item(),
                        "pad_sensitivity": (model(shaken, p, config["mask"]) - out).abs().mean().item(),
                        "seconds": time.perf_counter() - started,
                    }


            def record(config):
                """Run one trial and append it to TRIALS. Errors are kept as failed trials."""
                config = {**BASELINE, **config}
                try:
                    TRIALS.append({"config": config, "status": "ok", **run_trial(config)})
                except Exception as error:  # a failed trial is evidence too
                    TRIALS.append({"config": config, "status": f"failed: {type(error).__name__}: {error}"})


            record(BASELINE)
            COLUMNS = ["val_mse_padded", "val_mse_all", "pad_sensitivity", "seconds"]
            show_trials(TRIALS, COLUMNS);
        '''),
        Md(
            "**Your trials.** Add at least two trials that change one setting each. List them in `MY_TRIALS`, then run the next cell. You may run more than once; every run adds rows."
        ),
        Pair(
            learner="""
            # Implement: choose your trials (each dict changes only the settings you name).
            MY_TRIALS = [
                # {"mask": False},
            ]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
            reference="""
            # Reference choice of trials: one change per trial.
            MY_TRIALS = [{"mask": False}, {"pos": False}, {"layers": 1, "heads": 1}, {"layers": 3}]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
        ),
        Code(r"""
            # Check: the protocol was followed.
            assert TRIALS[0]["config"] == BASELINE, "The first trial must be the baseline."
            assert len(TRIALS) >= 3, "Run the baseline and at least two trials of your own."
            assert all(t["status"] == "ok" or t["status"].startswith("failed") for t in TRIALS)
            assert all(set(t["config"]) == set(BASELINE) for t in TRIALS), "Only mask, pos, layers and heads may change."
            ok = [t for t in TRIALS if t["status"] == "ok"]
            assert all(torch.isfinite(torch.tensor([t[c] for c in COLUMNS])).all() for t in ok), "Every measure must be finite."
            print(f"{len(TRIALS)} trials kept ({len(ok)} ok). Budget: {STEPS} updates, batch {BATCH}, seeds {SEED_WEIGHTS}/{SEED_BATCH}.")
        """),
        Code(r"""
            # Use your function: plot every trial (nothing is hidden or averaged away)
            labels = [", ".join(f"{k}={v}" for k, v in t["config"].items() if v != BASELINE[k]) or "baseline" for t in TRIALS if t["status"] == "ok"]
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            for axis, column in zip(axes, ["val_mse_padded", "pad_sensitivity"]):
                axis.bar(range(len(labels)), [t[column] for t in TRIALS if t["status"] == "ok"])
                axis.set_xticks(range(len(labels))); axis.set_xticklabels(labels, rotation=30, ha="right"); axis.set_title(column)
            plt.tight_layout(); plt.show()
        """),
        Md("""
        **Explain (2–4 sentences, edit this cell).** Which trial had the largest `pad_sensitivity`, and why does the mask change it? Did your prediction hold? One limit of this evidence: the padded steps in training always repeat the last real action. What would you test next?

        *Optional extension:* repeat the best two arms with a different batch seed (`SEED_BATCH`) and report the spread. Keep both sets of rows.
        """),
        Md("""
        ## Optional: your layers on the real ACT posterior encoder (opt-in)

        This section loads the **prepared** ACT checkpoint `act-bowl-plate-probe-200` from local files and copies its posterior-encoder weights into *your* layers. It then feeds real, padded action chunks to both encoders. This shows that your code reproduces a real trained block. It is not a training run and it says nothing about robot success. The workload is 8 video frames. No download starts. Set `RUN_REAL_ACT_ENCODER = True` to run it.
        """),
        Code(r"""
            RUN_REAL_ACT_ENCODER = False

            if RUN_REAL_ACT_ENCODER:
                from lerobot.policies.act import ACTPolicy
                from vla_lab.teaching.act_model import load_act_samples

                checkpoint = DATA / "runs/act-bowl-plate-probe-200/checkpoint-000200"
                if not (checkpoint / "model.safetensors").is_file():
                    raise FileNotFoundError(f"Prepared ACT checkpoint missing: {checkpoint}")
                config = ACTConfig.from_pretrained(checkpoint, local_files_only=True)
                config.device = "cpu"
                pinned = ACTPolicy.from_pretrained(checkpoint, config=config, strict=True, local_files_only=True).model.eval()
                width = config.dim_model
                yours = ref.TransformerEncoder(
                    config.n_vae_encoder_layers, width, config.n_heads, config.dim_feedforward,
                    forward_fn=encoder_layer_forward, attention_forward_fn=multi_head_forward,
                    attend_fn=scaled_dot_product_attention,
                ).eval()
                ref.load_lerobot_encoder(yours, pinned.vae_encoder)
                samples = load_act_samples(CONFIGS, DATA, "bowl_plate", "validation", count=8, chunk_size=config.chunk_size, image_size=32)
                with torch.no_grad():
                    cls = pinned.vae_encoder_cls_embed.weight[None].expand(8, -1, -1)
                    tokens = torch.cat([cls, pinned.vae_encoder_robot_state_input_proj(samples["observation.state"])[:, None],
                                        pinned.vae_encoder_action_input_proj(samples["action"])], dim=1)
                    mask = torch.cat([torch.zeros(8, 2, dtype=torch.bool), samples["action_is_pad"]], dim=1)
                    pos = pinned.vae_encoder_pos_enc
                    theirs = pinned.vae_encoder(tokens.transpose(0, 1), pos_embed=pos.transpose(0, 1), key_padding_mask=mask).transpose(0, 1)
                    mine = yours(tokens, pos, mask)
                    real = ~mask[..., None]
                    print("padded steps in these chunks:", int(samples["action_is_pad"].sum()))
                    print("largest difference at real positions:", float(((mine - theirs).abs() * real).max()))
                    shaken = tokens.clone(); shaken[:, 2:][samples["action_is_pad"]] = 25.0
                    print("change when padded tokens are replaced:", float(((yours(shaken, pos, mask) - mine).abs() * real).max()))
            else:
                print("Real ACT encoder check is off. Set RUN_REAL_ACT_ENCODER=True to run it.")
        """),
        Md("""
        ## What you built

        - `scaled_dot_product_attention`, `split_heads`/`merge_heads`, `multi_head_forward`, `sinusoidal_positions` and `encoder_layer_forward`, checked for output **and** gradient against torch and the pinned ACT blocks.
        - A padding rule you tested three ways: weights, gradients and downstream predictions.

        **Next:** Unit 8A adds the decoder: self-attention, cross-attention, feed-forward. From Unit 8A on you can import the package versions of these five functions (`vla_lab.teaching.attention`); your notebook answers are not needed to continue.
        """),
    ]
    return L


def lesson_08a() -> Lesson:
    L = Lesson("08a-decoder", "08a-decoder", "The decoder")
    L.items += [
        Md(r"""
        # Unit 8A — The decoder: self-attention, cross-attention, feed-forward

        **Your result:** a Transformer decoder layer that reads an encoder's memory. It will match the decoder inside the installed ACT model, output and gradient. Then you use it to predict a chunk of 16 future robot actions from a short history of real robot states.

        Two short exercises (about 15–20 minutes each) and one larger experiment (30–60 minutes). These times are estimates, not measurements.

        **Sources.** [Attention Is All You Need](https://arxiv.org/abs/1706.03762), sections 3.1 (decoder stack), 3.2.3 (where queries, keys and values come from), 3.3 (feed-forward) and 3.5 (positions). ACT's decoder uses *learned action queries* and reads the encoder's tokens by cross-attention — see the [ACT paper](https://arxiv.org/abs/2304.13705), appendix on architecture, and `ACTDecoderLayer` in the pinned `lerobot/policies/act/modeling_act.py`. ACT's decoder has no causal mask: all 20 actions are predicted at once.

        **Where this unit sits.** In Unit 7 you wrote attention, heads, positions and the encoder layer. **Here we use the package versions of those five functions** (`vla_lab.teaching.attention`), so you can start without your Unit 7 notebook. Your new Unit 8A functions are wired into the real decoder through its hooks.
        """),
        setup_cell("""
            import math
            import time
            import torch
            import torch.nn.functional as F
            from torch import nn
            import matplotlib.pyplot as plt

            import vla_lab.teaching.attention as ref  # package reference: used by checks only
            from vla_lab.teaching.attention import (  # your Unit 7 functions, as package versions
                MultiHeadAttention, TransformerDecoder, multi_head_forward, scaled_dot_product_attention,
                sinusoidal_positions, load_lerobot_decoder, load_lerobot_decoder_layer,
            )
            from vla_lab.teaching.training import masked_action_l1  # package version from the training unit
            from vla_lab.teaching.diffusion import load_state_action_windows
            from lerobot.policies.act.configuration_act import ACTConfig
            from lerobot.policies.act.modeling_act import ACTDecoder, ACTDecoderLayer

            torch.manual_seed(0)
            print("PyTorch:", torch.__version__)
        """),
        Code("# Supplied: a helper that the checks use\n" + tidy(CLOSE_HELPER)),
        Md("""
        ## Background: encoder memory, queries, and cross-attention

        An **encoder** turns the input into a list of *memory* tokens. A **decoder** starts with a list of *query* tokens. In ACT the queries start as zeros: one per future step. Their only information is a learned **query position** vector (one row per future step). In each decoder layer:

        1. **Self-attention** lets the queries look at each other.
        2. **Cross-attention** lets each query read the memory. Query = decoder token (+ query position). Key = memory token (+ memory position). Value = memory token, with no position.
        3. A **feed-forward** block processes every token alone.

        Each block adds its result to its input (residual) and normalizes (`LayerNorm`). `LayerNorm` rescales each token's features to zero mean and unit variance, then applies a learned scale and shift.
        """),
        Md(
            REAL_WINDOW_NOTE.replace(
                "16 consecutive actions (7 values each)",
                "8 consecutive past robot states (8 values each) and the next 16 actions (7 values each)",
            ).replace(
                "A window that runs past the end of an\nepisode repeats the last action and sets `is_pad=True` for those steps.",
                "A history that starts before the episode repeats the first state and sets `state_is_pad=True`; a window that runs past the end repeats the last action and sets `action_is_pad=True`.",
            )
        ),
        Code(r"""
            # Supplied: real windows. 8 states of history; actions start at the current frame.
            def read_windows(split):
                return load_state_action_windows(
                    CONFIGS, DATA, "bowl_plate", split,
                    horizon=16, n_obs_steps=8, action_start=0, drop_n_last_frames=0,
                )

            TRAIN, VALID = read_windows("train"), read_windows("validation")
            EXAMPLE = int((VALID["state_is_pad"].sum(1) == 3).nonzero()[0])  # a window with 3 padded history states
            print("train windows:", len(TRAIN["action"]), "validation windows:", len(VALID["action"]))
            print("states:", tuple(VALID["observation.state"].shape[1:]), "actions:", tuple(VALID["action"].shape[1:]))
            print("example history padding:", VALID["state_is_pad"][EXAMPLE].int().tolist())
        """),
    ]
    L.items.append(
        Exercise(
            key="8A.1",
            title="Attention with positions on the query and key only",
            minutes="10–15",
            purpose="Decoder layers use attention two ways: tokens attend to themselves, and tokens attend to encoder memory. One small function handles both and applies the position rule correctly.",
            contract="""
        - `attention` is a callable `attention(query, key, value, key_padding_mask) -> (B, Tq, D)` (a `MultiHeadAttention` module).
        - `query_input (B, Tq, D)` and `key_value_input (B, Tk, D)`.
        - `query_pos` broadcasts to `(B, Tq, D)`; `key_pos` broadcasts to `(B, Tk, D)`. Either may be `None` (no position).
        - **Query** = `query_input + query_pos`. **Key** = `key_value_input + key_pos`. **Value** = `key_value_input`, with *no* position.
        - `key_padding_mask (B, Tk)` with `True` = padding is passed through unchanged. Return the attention output `(B, Tq, D)`.
        """,
            method="Positions help queries find the right keys, but copying them into the values would add them to the output. This is the placement used by DETR and ACT; the pinned `ACTDecoderLayer` does the same. [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762), sections 3.2.3 and 3.5, describe the three uses of attention.",
            pointers="Build the three tensors first, handling `None`. Then make one call to `attention`. Self-attention is the case where both inputs are the same tensor.",
            hint="""
        ```python
        query = query_input if query_pos is None else query_input + query_pos
        key = key_value_input if key_pos is None else key_value_input + key_pos
        return attention(query, key, key_value_input, key_padding_mask)
        ```
        """,
            names=["attend_with_positions"],
            module="vla_lab.teaching.attention",
            stub="""
        def attend_with_positions(attention, query_input, key_value_input, query_pos=None, key_pos=None, key_padding_mask=None):
            # query = query_input + query_pos; key = key_value_input + key_pos; value = key_value_input.
            raise NotImplementedError("Complete Exercise 8A.1: attend_with_positions")
        """,
            check='''
        class Recorder:
            """A stand-in attention module that records what it was called with."""
            def __call__(self, query, key, value, key_padding_mask=None):
                self.args = (query, key, value, key_padding_mask)
                return query

        recorder = Recorder()
        x, memory = torch.randn(2, 4, 8), torch.randn(2, 6, 8)
        query_pos, key_pos = torch.randn(1, 4, 8), torch.randn(1, 6, 8)
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[0, 4:] = True
        attend_with_positions(recorder, x, memory, query_pos, key_pos, mask)
        q, k, v, passed = recorder.args
        close(q, x + query_pos, "query must be query_input + query_pos")
        close(k, memory + key_pos, "key must be key_value_input + key_pos")
        assert torch.equal(v, memory), "The value must be key_value_input with no position."
        assert passed is mask or torch.equal(passed, mask), "Pass key_padding_mask through unchanged."
        attend_with_positions(recorder, x, x)
        q, k, v, passed = recorder.args
        assert torch.equal(q, x) and torch.equal(k, x) and torch.equal(v, x) and passed is None, "With no positions and no mask, query, key and value are the input."
        torch.manual_seed(1)
        attention = MultiHeadAttention(8, 2)
        close(attend_with_positions(attention, x, memory, query_pos, key_pos, mask),
              attention(x + query_pos, memory + key_pos, memory, mask), "result with a real attention module")
        print("Exercise 8A.1 passed")
        ''',
            use_intro="Self-attention with a query position (as the decoder queries do), then cross-attention to a real state memory. Look at which memory tokens the queries read.",
            use="""
        torch.manual_seed(2)
        W = 16
        read_states = torch.nn.Linear(8, W)
        memory = read_states(VALID["observation.state"][EXAMPLE][None])        # (1, 8, W): eight state tokens
        history_pad = VALID["state_is_pad"][EXAMPLE][None]
        queries = torch.zeros(1, 16, W)                                         # ACT-style zero queries
        query_pos = torch.randn(1, 16, W)                                       # stand-in for learned query positions
        memory_pos = sinusoidal_positions(8, W)[None]
        cross = MultiHeadAttention(W, 4)
        with torch.no_grad():
            result = attend_with_positions(cross, queries, memory, query_pos, memory_pos, history_pad)
        weights = cross.last_weights[0].mean(0)                                 # average over heads: (16 queries, 8 memory tokens)
        plt.imshow(weights, cmap="viridis"); plt.colorbar()
        plt.xlabel("memory token (state history, oldest first)"); plt.ylabel("query (future step)")
        plt.title("Cross-attention weights (random weights). Padded history gets 0"); plt.show()
        print("weight on padded history tokens:", float(weights[:, history_pad[0]].sum()))
        """,
        )
    )
    L.items.append(
        Exercise(
            key="8A.2",
            title="Decoder layer: self-attention, cross-attention, feed-forward",
            minutes="15–20",
            purpose="Assemble the three blocks of a decoder layer with residual connections and norms. This is the layer ACT uses to turn learned queries into actions.",
            contract="""
        - `layer` has `self_attn`, `cross_attn` (callables like Exercise 8A.1's `attention`), `linear1`, `linear2`, `norm1`, `norm2`, `norm3`, `dropout`, `dropout1`, `dropout2`, `dropout3`, `activation` and the bool `pre_norm`.
        - `x (B, Tq, D)` queries; `memory (B, Tm, D)`; `query_pos` broadcasts to `(B, Tq, D)` or is `None`; `memory_pos` broadcasts to `(B, Tm, D)` or is `None`; `memory_key_padding_mask (B, Tm)` with `True` = padding, or `None`.
        - Self-attention: `attend_with_positions(layer.self_attn, x, x, query_pos, query_pos)`; no mask. Cross-attention: query from `x`, key/value from `memory`, with `query_pos`, `memory_pos` and `memory_key_padding_mask`. **Call your Exercise 8A.1 function for both.**
        - Post-norm (`pre_norm=False`): `x = norm1(x + dropout1(self_attn))`; `x = norm2(x + dropout2(cross_attn))`; `x = norm3(x + dropout3(ff))`, with `ff(x) = linear2(dropout(activation(linear1(x))))`.
        - Pre-norm: each norm goes before its block (`norm1` before self-attention, `norm2` before cross-attention, `norm3` before the feed-forward) and none after the add.
        - Return `(B, Tq, D)`.
        """,
            method="Decoder layer: [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762), section 3.1 (three sub-layers, each with a residual connection and layer normalization). ACT's `ACTDecoderLayer` (pinned) is this layer without a causal mask.",
            pointers="Copy the structure of the encoder layer from Unit 7 and insert the cross-attention block between self-attention and the feed-forward block. Keep a `skip` tensor before each block.",
            hint="""
        Post-norm version:
        ```python
        skip = x
        x = attend_with_positions(layer.self_attn, x, x, query_pos, query_pos)
        x = layer.norm1(skip + layer.dropout1(x))
        skip = x
        x = attend_with_positions(layer.cross_attn, x, memory, query_pos, memory_pos, memory_key_padding_mask)
        x = layer.norm2(skip + layer.dropout2(x))
        skip = x
        x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        return layer.norm3(skip + layer.dropout3(x))
        ```
        For pre-norm, apply `norm1/2/3` to `x` just before each block instead and drop the norm after each add.
        """,
            names=["decoder_layer_forward"],
            module="vla_lab.teaching.attention",
            stub="""
        def decoder_layer_forward(layer, x, memory, query_pos=None, memory_pos=None, memory_key_padding_mask=None):
            # Self-attention, cross-attention, feed-forward: each with a residual connection and a norm.
            raise NotImplementedError("Complete Exercise 8A.2: decoder_layer_forward")
        """,
            check="""
        D_, H_, FF_ = 16, 4, 32
        config = lambda pre: ACTConfig(dim_model=D_, n_heads=H_, dim_feedforward=FF_, dropout=0.0, pre_norm=pre,
                                       chunk_size=4, n_action_steps=4, input_features={}, output_features={})
        for pre in (False, True):
            torch.manual_seed(3)
            oracle = ACTDecoderLayer(config(pre)).eval()
            mine = ref.DecoderLayer(D_, H_, FF_, pre_norm=pre, forward_fn=decoder_layer_forward).eval()
            load_lerobot_decoder_layer(mine, oracle)
            x = torch.randn(3, 4, D_, requires_grad=True)
            memory = torch.randn(3, 7, D_, requires_grad=True)
            query_pos, memory_pos = torch.randn(1, 4, D_), torch.randn(1, 7, D_)
            out = mine(x, memory, query_pos, memory_pos)
            expected = oracle(x.transpose(0, 1), memory.transpose(0, 1), decoder_pos_embed=query_pos.transpose(0, 1),
                              encoder_pos_embed=memory_pos.transpose(0, 1)).transpose(0, 1)
            close(out, expected, f"layer output (pre_norm={pre}) differs from ACTDecoderLayer")
            out.square().sum().backward()
            x2, memory2 = (t.detach().clone().requires_grad_(True) for t in (x, memory))
            oracle(x2.transpose(0, 1), memory2.transpose(0, 1), decoder_pos_embed=query_pos.transpose(0, 1),
                   encoder_pos_embed=memory_pos.transpose(0, 1)).transpose(0, 1).square().sum().backward()
            close(x.grad, x2.grad, f"gradient of the queries (pre_norm={pre})", atol=1e-4)
            close(memory.grad, memory2.grad, f"gradient of the memory (pre_norm={pre})", atol=1e-4)
            close(mine.cross_attn.k_proj.weight.grad, oracle.multihead_attn.in_proj_weight.grad[D_:2 * D_], "cross-attention key gradient", atol=1e-4)
        # memory padding: the pinned layer has no mask, so check the behaviour directly.
        mask = torch.zeros(3, 7, dtype=torch.bool)
        mask[:, 5:] = True
        base = mine(x.detach(), memory.detach(), query_pos, memory_pos, mask)
        changed = memory.detach().clone()
        changed[:, 5:] = 40.0
        close(mine(x.detach(), changed, query_pos, memory_pos, mask), base, "padded memory must not change the output", atol=1e-4)
        assert not torch.allclose(mine(x.detach(), memory.detach(), query_pos, memory_pos), base, atol=1e-4), "The mask must change the result when padded memory is real data."
        print("Exercise 8A.2 passed")
        """,
            use_intro="Stack two of your layers into a real decoder and run ACT-style zero queries against a real state memory. Compare with the installed ACT decoder stack.",
            use="""
        D_, H_, FF_ = 16, 4, 32
        torch.manual_seed(4)
        pinned = ACTDecoder(ACTConfig(dim_model=D_, n_heads=H_, dim_feedforward=FF_, n_decoder_layers=2, dropout=0.0,
                                      chunk_size=16, n_action_steps=16, input_features={}, output_features={})).eval()
        yours = TransformerDecoder(2, D_, H_, FF_, forward_fn=decoder_layer_forward).eval()
        load_lerobot_decoder(yours, pinned)
        embed = torch.nn.Linear(8, D_)
        memory = embed(VALID["observation.state"][:6])                          # (6, 8, D): real state histories
        query_pos, memory_pos = torch.randn(1, 16, D_), sinusoidal_positions(8, D_)[None]
        queries = torch.zeros(6, 16, D_)
        with torch.no_grad():
            out = yours(queries, memory, query_pos, memory_pos)
            theirs = pinned(queries.transpose(0, 1), memory.transpose(0, 1), decoder_pos_embed=query_pos.transpose(0, 1),
                            encoder_pos_embed=memory_pos.transpose(0, 1)).transpose(0, 1)
        print("largest difference from the installed ACT decoder:", float((out - theirs).abs().max()))
        """,
        )
    )
    L.items += [
        Md(r"""
        ## Experiment — what does the decoder read? (30–60 min)

        **Question.** A decoder predicts 16 future actions from a history of real robot states. Which parts of its input matter: the length of the history, the positions on the memory, or the memory padding mask?

        **Task (real data).** Memory: up to 8 robot states (oldest first), each projected to 32 features. Queries: 16 zero vectors plus learned query positions (one row per future step, as in ACT). Output: 16 actions of 7 values in `[-1, 1]` (min–max scaled with training extrema). Loss: masked L1 over real (non-padded) actions — the function from the training unit. States before the episode start are padded copies of the first state.

        **Fixed for every trial.** The windows above (validation episodes are never used for updates); seed `0` for weights, seed `1` for batch draws; **200 updates**, batch 64, AdamW `1e-3`; width 32, 4 heads. Your Exercise 8A.1 and 8A.2 functions run inside every decoder layer.

        **Allowed changes (per trial).** `k` (use only the newest `k` states, 1–8), `mem_pos` (learned memory positions on or off), `mask` (pass the history padding mask to cross-attention or not), `layers` (1 or 2).

        **Measures.** `val_l1`: masked L1 on all validation windows. `val_l1_padhist`: the same on windows whose history contains padding. `pad_sensitivity`: mean change of the output when the padded history states are replaced by random values.

        **Prediction first.** Write two sentences in the next cell before you run anything: which setting will hurt `val_l1` most, and which will change `pad_sensitivity`?

        **Keep failed trials.** Every trial is appended to `TRIALS`; never delete rows. No improvement is a valid result.
        """),
        Code('PREDICTION = "TODO: replace this text with your prediction."\nprint(PREDICTION)'),
        Code("# Supplied: trial table helper\n" + tidy(EXPERIMENT_TABLE_HELPER)),
        Code(r"""
            # Supplied: harness. Your decoder_layer_forward and attend_with_positions are in the real path.
            STEPS, BATCH, LR, SEED_WEIGHTS, SEED_BATCH, WIDTH = 200, 64, 1e-3, 0, 1, 32


            class HistoryDecoder(nn.Module):
                def __init__(self, layers=1, mem_pos=True):
                    super().__init__()
                    self.inp = nn.Linear(8, WIDTH)
                    self.mem_pos = nn.Embedding(8, WIDTH) if mem_pos else None
                    self.query_pos = nn.Embedding(16, WIDTH)
                    self.decoder = TransformerDecoder(layers, WIDTH, 4, 2 * WIDTH, forward_fn=decoder_layer_forward)
                    self.out = nn.Linear(WIDTH, 7)

                def forward(self, states, state_pad, k=8, use_mask=True):
                    states, state_pad = states[:, -k:], state_pad[:, -k:]
                    memory_pos = self.mem_pos.weight[None, -k:] if self.mem_pos is not None else None
                    queries = torch.zeros(states.shape[0], 16, WIDTH)
                    hidden = self.decoder(queries, self.inp(states), self.query_pos.weight[None], memory_pos,
                                          state_pad if use_mask else None)
                    return self.out(hidden)


            BASELINE = {"k": 8, "mem_pos": True, "mask": True, "layers": 1}
            TRIALS = []


            def run_trial(config):
                started = time.perf_counter()
                torch.manual_seed(SEED_WEIGHTS)
                model = HistoryDecoder(config["layers"], config["mem_pos"])
                optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
                draws = torch.Generator().manual_seed(SEED_BATCH)
                for _ in range(STEPS):
                    pick = torch.randint(0, len(TRAIN["action"]), (BATCH,), generator=draws)
                    prediction = model(TRAIN["observation.state"][pick], TRAIN["state_is_pad"][pick], config["k"], config["mask"])
                    loss = masked_action_l1(prediction, TRAIN["action"][pick], TRAIN["action_is_pad"][pick])
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                model.eval()
                with torch.no_grad():
                    S, P, A, AP = (VALID[key] for key in ("observation.state", "state_is_pad", "action", "action_is_pad"))
                    has = P[:, -config["k"]:].any(1)
                    shaken = torch.where(P[has].unsqueeze(-1), torch.randn(S[has].shape, generator=torch.Generator().manual_seed(5)) * 3, S[has])
                    base = model(S[has], P[has], config["k"], config["mask"])
                    return {
                        "val_l1": masked_action_l1(model(S, P, config["k"], config["mask"]), A, AP).item(),
                        "val_l1_padhist": masked_action_l1(base, A[has], AP[has]).item() if has.any() else float("nan"),
                        "pad_sensitivity": (model(shaken, P[has], config["k"], config["mask"]) - base).abs().mean().item() if has.any() else 0.0,
                        "seconds": time.perf_counter() - started,
                    }


            def record(config):
                config = {**BASELINE, **config}
                try:
                    TRIALS.append({"config": config, "status": "ok", **run_trial(config)})
                except Exception as error:  # a failed trial is evidence too
                    TRIALS.append({"config": config, "status": f"failed: {type(error).__name__}: {error}"})


            record(BASELINE)
            COLUMNS = ["val_l1", "val_l1_padhist", "pad_sensitivity", "seconds"]
            show_trials(TRIALS, COLUMNS);
        """),
        Md(
            "**Your trials.** Add at least two trials that each change one setting. `k=1` has no padded history, so its `pad_sensitivity` is 0 by construction: read it with that in mind."
        ),
        Pair(
            learner="""
            # Implement: choose your trials.
            MY_TRIALS = [
                # {"k": 1},
            ]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
            reference="""
            # Reference choice of trials: one change per trial.
            MY_TRIALS = [{"k": 1}, {"k": 4}, {"mem_pos": False}, {"mask": False}, {"layers": 2}]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
        ),
        Code(r"""
            # Check: the protocol was followed.
            assert TRIALS[0]["config"] == BASELINE, "The first trial must be the baseline."
            assert len(TRIALS) >= 3, "Run the baseline and at least two trials of your own."
            assert all(set(t["config"]) == set(BASELINE) for t in TRIALS), "Only k, mem_pos, mask and layers may change."
            ok = [t for t in TRIALS if t["status"] == "ok"]
            assert all(t["status"] == "ok" or t["status"].startswith("failed") for t in TRIALS)
            assert all(torch.isfinite(torch.tensor([t["val_l1"], t["pad_sensitivity"]])).all() for t in ok), "Measures must be finite."
            print(f"{len(TRIALS)} trials kept ({len(ok)} ok). Budget: {STEPS} updates, batch {BATCH}, seeds {SEED_WEIGHTS}/{SEED_BATCH}.")
        """),
        Code(r"""
            # Use your function: plot every trial
            ok = [t for t in TRIALS if t["status"] == "ok"]
            labels = [", ".join(f"{k}={v}" for k, v in t["config"].items() if v != BASELINE[k]) or "baseline" for t in ok]
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
            for axis, column in zip(axes, ["val_l1", "val_l1_padhist", "pad_sensitivity"]):
                axis.bar(range(len(ok)), [t[column] for t in ok])
                axis.set_xticks(range(len(ok))); axis.set_xticklabels(labels, rotation=30, ha="right"); axis.set_title(column)
            plt.tight_layout(); plt.show()
        """),
        Md("""
        **Explain (2–4 sentences, edit this cell).** Which setting changed `val_l1` the most? Which one changed only `pad_sensitivity`? Did your prediction hold? Remember that the target is a future action chunk that also depends on the camera images, which this model never sees, so a large remaining error is expected.

        *Optional extension:* replace a trial's decoder layers by the package `DecoderLayer` default (no learner hooks) and confirm the table is identical for the baseline. Keep all rows.
        """),
        Md("""
        ## Optional: your decoder layer with real ACT weights (opt-in)

        This copies the **prepared** ACT checkpoint's decoder weights (`act-bowl-plate-probe-200`) into a decoder built from *your* `decoder_layer_forward`, then compares it with the installed ACT decoder on a real state memory. The memory has only the latent and state tokens (no camera tokens), so the outputs are not the policy's actions; the check shows that your layer reproduces a real trained decoder. The workload is a few validation frames. Set `RUN_REAL_ACT_DECODER = True` to run it.
        """),
        Code(r"""
            RUN_REAL_ACT_DECODER = False

            if RUN_REAL_ACT_DECODER:
                from lerobot.policies.act import ACTPolicy
                from vla_lab.teaching.act_model import load_act_samples

                checkpoint = DATA / "runs/act-bowl-plate-probe-200/checkpoint-000200"
                if not (checkpoint / "model.safetensors").is_file():
                    raise FileNotFoundError(f"Prepared ACT checkpoint missing: {checkpoint}")
                config = ACTConfig.from_pretrained(checkpoint, local_files_only=True)
                config.device = "cpu"
                pinned = ACTPolicy.from_pretrained(checkpoint, config=config, strict=True, local_files_only=True).model.eval()
                yours = TransformerDecoder(config.n_decoder_layers, config.dim_model, config.n_heads, config.dim_feedforward,
                                           forward_fn=decoder_layer_forward).eval()
                load_lerobot_decoder(yours, pinned.decoder)
                samples = load_act_samples(CONFIGS, DATA, "bowl_plate", "validation", count=4, chunk_size=config.chunk_size, image_size=32)
                with torch.no_grad():
                    latent = pinned.encoder_latent_input_proj(torch.zeros(4, config.latent_dim))
                    state = pinned.encoder_robot_state_input_proj(samples["observation.state"])
                    memory = torch.stack([latent, state], dim=1)                          # (4, 2, D)
                    memory_pos = pinned.encoder_1d_feature_pos_embed.weight[None]          # (1, 2, D)
                    query_pos = pinned.decoder_pos_embed.weight[None]                      # (1, T, D)
                    queries = torch.zeros(4, config.chunk_size, config.dim_model)
                    mine = yours(queries, memory, query_pos, memory_pos)
                    theirs = pinned.decoder(queries.transpose(0, 1), memory.transpose(0, 1), decoder_pos_embed=query_pos.transpose(0, 1),
                                            encoder_pos_embed=memory_pos.transpose(0, 1)).transpose(0, 1)
                    print("largest difference from the installed ACT decoder:", float((mine - theirs).abs().max()))
                    print("decoded action shape:", tuple(pinned.action_head(mine).shape))
            else:
                print("Real ACT decoder check is off. Set RUN_REAL_ACT_DECODER=True to run it.")
        """),
        Md("""
        ## What you built

        - `attend_with_positions` and `decoder_layer_forward`, checked for output and gradient against the pinned ACT decoder layer, plus a behaviour check for memory padding (which the pinned layer does not have).
        - Evidence from real data about which decoder inputs matter, with every trial kept.

        **Next:** Unit 8B assembles the whole ACT model: camera, state and latent tokens, the posterior encoder, action queries and the head. It uses the package versions of the Unit 7 and 8A functions.
        """),
    ]
    return L


ACT_FIXTURE = r'''
IMAGES = ("observation.images.image", "observation.images.image2")


def small_lerobot_act(seed=11):
    """A small *real* pinned ACT module with random weights (64x64 images, 20-step chunk)."""
    torch.manual_seed(seed)
    config = ACTConfig(
        input_features={
            IMAGES[0]: PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            IMAGES[1]: PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        chunk_size=20, n_action_steps=20, dim_model=32, n_heads=4, dim_feedforward=64,
        n_encoder_layers=2, n_decoder_layers=1, n_vae_encoder_layers=2, latent_dim=6,
        dropout=0.0, pretrained_backbone_weights=None, push_to_hub=False, device="cpu",
    )
    model = ACT(config)
    for parameter in model.parameters():       # distinct vectors make wrong wiring visible
        if parameter.dim() == 1:
            nn.init.normal_(parameter, std=0.1)
    return model


def algorithm_check_batch(batch=3, chunk=20, seed=0):
    """Random numbers in the right shapes. Clearly labelled: an algorithm check, not data."""
    g = torch.Generator().manual_seed(seed)
    pad = torch.zeros(batch, chunk, dtype=torch.bool)
    pad[1, 14:] = True
    pad[2, 3:] = True
    return {
        "observation.state": torch.randn(batch, 8, generator=g),
        IMAGES[0]: torch.rand(batch, 3, 64, 64, generator=g),
        IMAGES[1]: torch.rand(batch, 3, 64, 64, generator=g),
        "action": torch.randn(batch, chunk, 7, generator=g),
        "action_is_pad": pad,
    }


REFERENCE = small_lerobot_act()                      # the installed LeRobot ACT (reference weights)
LEARNER = act_from_lerobot(REFERENCE)                # the learner module holding *reference weights*
CHECK_BATCH = algorithm_check_batch()
print("learner parameters:", sum(p.numel() for p in LEARNER.parameters()))
'''


def lesson_08b() -> Lesson:
    L = Lesson("08b-act", "08b-act", "Assemble ACT")
    L.items += [
        Md(r"""
        # Unit 8B — Assemble ACT from its parts

        **Your result:** an Action Chunking Transformer built from your functions. It reproduces the installed LeRobot ACT *exactly* when it holds the same weights, and it trains with the loss callback from the training unit.

        Five short exercises (10–20 minutes each) and a larger experiment (30–60 minutes). These are estimates, not measurements.

        **Sources.** [ACT paper](https://arxiv.org/abs/2304.13705) (action chunks, the CVAE posterior encoder, the Transformer encoder-decoder with learned action queries) and the code in the pinned `lerobot/policies/act/modeling_act.py` (revision `e0d50211ef236143ae867228662b7dfaba554f02`), read as `ACT.forward`. The blocks come from [Attention Is All You Need](https://arxiv.org/abs/1706.03762). **The raw model returns `(actions, (mu, logvar))` and does not compute a loss.** The installed `ACTPolicy.forward` computes its own loss inside; this unit bypasses it on purpose and uses the loss callback from the training unit.

        **Supplied here:** the CNN backbone (a ResNet-18 that turns each camera image into a feature map), the Transformer encoder and decoder blocks (package versions of your Unit 7 and 8A answers), the data reader, and the weight converter. **You write** the five functions that make ACT *ACT*: 2D camera positions, the posterior encoder, the latent choice (including zero-latent inference), the encoder tokens, and the action decoder.

        Full answers are in [the reference notebook](08b-act-reference.ipynb).
        """),
        setup_cell("""
            import copy
            from functools import partial
            import math
            import time
            import torch
            import yaml
            from torch import nn
            import matplotlib.pyplot as plt

            import vla_lab.teaching.act_model as ref  # package reference: used by checks only
            from vla_lab.teaching.act_model import (
                ACTHooks, ACTModel, ACTSettings, act_from_lerobot, load_act_samples, load_lerobot_act,
                make_act_backbone, verify_act_parity, convert_lerobot_act_state_dict,
            )
            from vla_lab.teaching.cvae import reparameterize  # from the training unit (package version)
            from vla_lab.teaching.training import act_loss_fn, optimizer_update, train_loop, masked_action_l1
            from lerobot.configs import FeatureType, PolicyFeature
            from lerobot.policies.act.configuration_act import ACTConfig
            from lerobot.policies.act.modeling_act import ACT, ACTSinusoidalPositionEmbedding2d

            torch.manual_seed(0)
            print("PyTorch:", torch.__version__)
        """),
        Code("# Supplied: a helper that the checks use\n" + tidy(CLOSE_HELPER)),
        Md("""
        ## The parts of ACT

        ```
        training only:   [cls, state, action_1 … action_T] ─ posterior encoder ─▶ (mu, logvar) ─▶ sample z
        inference:                                                              z = 0
        encoder tokens:  [ z, state, camera-1 cells…, camera-2 cells… ]  (+ positions) ─ encoder ─▶ memory
        decoder:         T zero queries (+ learned query positions) ─ cross-attend to memory ─▶ T actions
        ```

        - A **camera** goes through the CNN backbone. Its feature map `(C, h, w)` becomes `h*w` tokens of width `D`, each with a fixed 2D sinusoidal position.
        - The **posterior encoder** exists only to *train* the latent `z` (a conditional VAE). It sees the future actions, so it can encode "which style of demonstration this was". Padded actions are masked.
        - **At inference there are no future actions.** The latent is set to zero, the mean of the prior.
        - The **decoder** has no input except learned query positions. Each of the `T` queries asks the memory for one future action.

        ## Three kinds of weights — keep them apart

        | Name | What it is | Where it comes from |
        |---|---|---|
        | **Reference weights in the learner** | LeRobot ACT weights copied into *your* module by the converter | `load_lerobot_act`; proves your code computes the same function |
        | **Learner trained weights** | weights produced by *your* updates in this notebook | the experiment below |
        | **Prepared policy results** | numbers saved from the LeRobot policy (e.g. simulator successes) | `manifests/act_references.yaml`; says nothing about your module |
        """),
        Code(
            "# Supplied: the CNN backbone and a first look\n"
            + tidy(r"""
            backbone, channels = make_act_backbone("resnet18")
            frame = torch.rand(1, 3, 256, 256)
            print("feature map for one 256x256 camera:", tuple(backbone(frame)["feature_map"].shape), "(channels, rows, columns) after the batch axis")
        """)
        ),
        Code(
            "# Supplied: a small real LeRobot ACT, its learner twin, and a batch for algorithm checks\n"
            + tidy(ACT_FIXTURE)
        ),
    ]

    L.items.append(
        Exercise(
            key="8B.1",
            title="2D positions for camera features",
            minutes="15",
            purpose="Camera tokens come from a grid. The encoder needs to know each token's row and column; this is a fixed sinusoidal code, half for the row and half for the column.",
            contract="""
        - `camera_position_embedding_2d(height, width, dim)` returns a `float32` tensor `(1, dim, height, width)`; `dim` is even.
        - Let `d = dim // 2`. Channels `0 … d-1` encode the **row**, channels `d … dim-1` encode the **column**.
        - Row index `r` runs `1 … height` (not from 0). Scale it: `r / (height + 1e-6) * 2π`. The column index is scaled the same way with `width`.
        - For channel `j` inside a half: angle `= scaled_index / 10000 ** (2 * (j // 2) / d)`. Even `j` uses `sin(angle)`, odd `j` uses `cos(angle)`, so sines and cosines are interleaved.
        """,
            method="2D sinusoidal positions as in DETR; ACT uses them for its image tokens ([ACT paper](https://arxiv.org/abs/2304.13705)). The 1D version is Exercise 7.4 ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762), section 3.5). The pinned class is `ACTSinusoidalPositionEmbedding2d`.",
            pointers="Make a row code of shape `(height, d)` and a column code of shape `(width, d)` first. Then broadcast them to `(height, width, d)` each, concatenate the row half first, and move channels to the front.",
            hint="""
        ```python
        per_axis = dim // 2
        rows = torch.arange(1, height + 1, dtype=torch.float32)
        columns = torch.arange(1, width + 1, dtype=torch.float32)
        rows = rows / (rows[-1] + 1e-6) * 2 * math.pi        # same for columns
        inverse = 10000 ** (2 * (torch.arange(per_axis) // 2) / per_axis)

        def encode(positions):
            angles = positions[:, None] / inverse
            return torch.stack((angles[:, 0::2].sin(), angles[:, 1::2].cos()), dim=-1).flatten(1)

        row_code = encode(rows)[:, None, :].expand(height, width, per_axis)
        column_code = encode(columns)[None, :, :].expand(height, width, per_axis)
        return torch.cat((row_code, column_code), dim=-1).permute(2, 0, 1).unsqueeze(0)
        ```
        `torch.stack((sines, cosines), dim=-1).flatten(1)` interleaves the two.
        """,
            names=["camera_position_embedding_2d"],
            module="vla_lab.teaching.act_model",
            stub="""
        def camera_position_embedding_2d(height, width, dim):
            # Return (1, dim, height, width): row code in the first half of channels, column code in the second.
            raise NotImplementedError("Complete Exercise 8B.1: camera_position_embedding_2d")
        """,
            check="""
        for height, width, dim in ((2, 2, 32), (3, 5, 32), (8, 8, 64)):
            expected = ACTSinusoidalPositionEmbedding2d(dim // 2)(torch.zeros(1, 4, height, width))
            mine = camera_position_embedding_2d(height, width, dim)
            assert mine.shape == (1, dim, height, width), f"Expected (1, {dim}, {height}, {width}), got {tuple(mine.shape)}."
            close(mine, expected, f"map for {height}x{width}, dim {dim} differs from ACTSinusoidalPositionEmbedding2d", atol=1e-6)
        m = camera_position_embedding_2d(4, 6, 16)
        assert torch.equal(m[0, :8, 1, 0], m[0, :8, 1, 5]), "Row channels must not depend on the column."
        assert torch.equal(m[0, 8:, 0, 2], m[0, 8:, 3, 2]), "Column channels must not depend on the row."
        print("Exercise 8B.1 passed")
        """,
            use_intro="Show the position code on the feature grid of one real camera image (the encoder sees exactly this grid).",
            use="""
        rows, columns = backbone(frame)["feature_map"].shape[-2:]
        codes = camera_position_embedding_2d(rows, columns, 256)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for axis, channel, title in zip(axes, (0, 64, 128), ("row code, channel 0", "row code, channel 64", "column code, channel 128")):
            axis.imshow(codes[0, channel], cmap="coolwarm"); axis.set_title(title)
        plt.tight_layout(); plt.show()
        print("tokens per camera:", rows * columns)
        """,
        )
    )

    L.items.append(
        Exercise(
            key="8B.2",
            title="The posterior encoder (CLS, state, actions, padding)",
            minutes="15–20",
            purpose="During training the model sees the future actions and encodes them into a latent code. This function turns `(state, actions, padding)` into the mean and log-variance of that latent.",
            contract="""
        - `model` has: `vae_cls_embed` (`nn.Embedding` with one row, width `D`), `vae_state_proj` (`Linear(S, D)`), `vae_action_proj` (`Linear(A, D)`), `vae_encoder` (callable `vae_encoder(tokens, pos_embed, key_padding_mask) -> (B, N, D)`), `vae_pos_embed` (fixed table `(1, T + 2, D)`) and `vae_latent_proj` (`Linear(D, 2 * L)`).
        - `state (B, S)`, `actions (B, T, A)`, `action_is_pad (B, T)` with `True` = padding.
        - Token sequence: `[cls, state, action_1 … action_T]`, shape `(B, T + 2, D)`. The key padding mask is `[False, False, *action_is_pad]`, shape `(B, T + 2)`. Pass `model.vae_pos_embed` as the position.
        - Read the encoder output at position 0 (the CLS token), apply `vae_latent_proj`, and split the `2 * L` values: the first `L` are `mu`, the last `L` are `logvar` (= log of the variance, `log σ²`).
        - Return `(mu, logvar)`, each `(B, L)`.
        """,
            method="The posterior `q(z | state, actions)` is a BERT-style Transformer encoder whose CLS output gives a diagonal Gaussian — [ACT paper](https://arxiv.org/abs/2304.13705), section on the CVAE encoder. Pinned code: the `if self.config.use_vae and ACTION in batch and self.training` block of `ACT.forward`.",
            pointers="Build the three kinds of tokens first and concatenate them on the sequence axis. Build the mask from `action_is_pad` plus two `False` columns on the left. Take only position 0 of the encoder output.",
            hint="""
        ```python
        batch = state.shape[0]
        cls_token = model.vae_cls_embed.weight[None].expand(batch, -1, -1)           # (B, 1, D)
        tokens = torch.cat([cls_token, model.vae_state_proj(state)[:, None], model.vae_action_proj(actions)], dim=1)
        no_pad = torch.zeros(batch, 2, dtype=torch.bool, device=state.device)
        mask = torch.cat([no_pad, action_is_pad.to(torch.bool)], dim=1)
        cls_out = model.vae_encoder(tokens, model.vae_pos_embed, mask)[:, 0]
        mu, logvar = model.vae_latent_proj(cls_out).chunk(2, dim=-1)
        return mu, logvar
        ```
        """,
            names=["encode_posterior"],
            module="vla_lab.teaching.act_model",
            stub="""
        def encode_posterior(model, state, actions, action_is_pad):
            # Tokens [cls, state, actions]; mask [False, False, *action_is_pad]; read the CLS output; split into mu, logvar.
            raise NotImplementedError("Complete Exercise 8B.2: encode_posterior")
        """,
            check="""
        # The module passed in is the learner twin of a real pinned ACT, so a correct function reproduces the pinned posterior.
        model = copy.deepcopy(LEARNER).train()
        batch = CHECK_BATCH
        mu, logvar = encode_posterior(model, batch["observation.state"], batch["action"], batch["action_is_pad"])
        assert mu.shape == (3, 6) and logvar.shape == (3, 6), "mu and logvar must be (B, latent_dim)."
        reference = copy.deepcopy(REFERENCE).train()
        _, (ref_mu, ref_logvar) = reference({**batch, "observation.images": [batch[k] for k in IMAGES]})
        close(mu, ref_mu, "mu differs from the pinned ACT posterior")
        close(logvar, ref_logvar, "logvar differs from the pinned ACT posterior")
        shaken = batch["action"].clone()
        shaken[batch["action_is_pad"]] = 60.0
        mu2, logvar2 = encode_posterior(model, batch["observation.state"], shaken, batch["action_is_pad"])
        close(mu2, mu, "changing padded actions must not change mu", atol=1e-4)
        close(logvar2, logvar, "changing padded actions must not change logvar", atol=1e-4)
        mu3, _ = encode_posterior(model, batch["observation.state"], shaken, torch.zeros_like(batch["action_is_pad"]))
        assert not torch.allclose(mu3, mu, atol=1e-2), "Without the padding mask, the padded values should matter."
        print("Exercise 8B.2 passed")
        """,
            use_intro="Encode real demonstration chunks (with their natural end-of-episode padding) and check that padded values do not leak into the latent.",
            use="""
        SMALL_REAL = load_act_samples(CONFIGS, DATA, "bowl_plate", "validation", count=8, chunk_size=20, image_size=64)
        learner = copy.deepcopy(LEARNER).eval()
        with torch.no_grad():
            mu, logvar = encode_posterior(learner, SMALL_REAL["observation.state"], SMALL_REAL["action"], SMALL_REAL["action_is_pad"])
            shaken = torch.where(SMALL_REAL["action_is_pad"].unsqueeze(-1), torch.full_like(SMALL_REAL["action"], 25.0), SMALL_REAL["action"])
            mu_shaken, _ = encode_posterior(learner, SMALL_REAL["observation.state"], shaken, SMALL_REAL["action_is_pad"])
        print("padded steps per window:", SMALL_REAL["action_is_pad"].sum(1).tolist())
        print("largest change of mu when padded actions are replaced:", float((mu - mu_shaken).abs().max()))
        print("mu shape:", tuple(mu.shape))
        """,
        )
    )

    L.items.append(
        Exercise(
            key="8B.3",
            title="Choosing the latent: sample in training, zero at inference",
            minutes="10–15",
            purpose="The decoder needs a latent vector. In training it is a sample from the posterior. At inference there are no future actions, so use the prior mean: zero. Getting this branch wrong is a common ACT bug.",
            contract="""
        - `model.training` (bool), `model.settings.latent_dim` (`L`), `model.hooks.posterior(model, state, actions, action_is_pad) -> (mu, logvar)`.
        - `batch` has `observation.state (B, S)`; in training it also has `action (B, T, A)` and `action_is_pad (B, T)`.
        - **Training branch** (`model.training` and `"action"` in `batch`): get `(mu, logvar)` from `model.hooks.posterior`, then `z = mu + exp(logvar / 2) * noise`. `noise` defaults to standard normal noise of the shape of `mu`; if given, use it exactly. Use `reparameterize` (supplied, from the training unit) for the sampling step. Return `(z, mu, logvar)`.
        - **Otherwise** (evaluation mode, or no actions in the batch): return `(zeros (B, L), None, None)` with the dtype and device of `batch["observation.state"]`. No random numbers.
        """,
            method="Reparameterization trick (Kingma and Welling 2014) and the ACT paper's inference rule: at test time the latent is set to the mean of the prior, zero. Pinned code: the `else` branch that creates `latent_sample = torch.zeros(...)`.",
            pointers="One `if` with two branches. In the first branch call the hook, then `reparameterize(mu, logvar, noise=noise)`. In the second, only create zeros.",
            hint="""
        ```python
        state = batch["observation.state"]
        if model.training and "action" in batch:
            mu, logvar = model.hooks.posterior(model, state, batch["action"], batch["action_is_pad"])
            return reparameterize(mu, logvar, noise=noise), mu, logvar
        zeros = torch.zeros(state.shape[0], model.settings.latent_dim, dtype=state.dtype, device=state.device)
        return zeros, None, None
        ```
        """,
            names=["select_latent"],
            module="vla_lab.teaching.act_model",
            stub="""
        def select_latent(model, batch, noise=None):
            # Return (latent, mu, logvar): sampled in training, zeros (and None, None) otherwise.
            raise NotImplementedError("Complete Exercise 8B.3: select_latent")
        """,
            check="""
        model = copy.deepcopy(LEARNER)
        model.hooks = ACTHooks(posterior=ref.encode_posterior)     # the posterior hook is the package answer here
        batch = CHECK_BATCH
        model.train()
        noise = torch.randn(3, 6)
        z, mu, logvar = select_latent(model, batch, noise)
        ref_mu, ref_logvar = ref.encode_posterior(model, batch["observation.state"], batch["action"], batch["action_is_pad"])
        close(mu, ref_mu, "mu must come from the posterior hook"); close(logvar, ref_logvar, "logvar must come from the posterior hook")
        close(z, ref_mu + (ref_logvar / 2).exp() * noise, "z must be mu + exp(logvar / 2) * noise")
        z0, _, _ = select_latent(model, batch, torch.zeros(3, 6))
        close(z0, ref_mu, "zero noise gives z = mu")
        torch.manual_seed(5); a, _, _ = select_latent(model, batch)
        torch.manual_seed(5); b, _, _ = select_latent(model, batch)
        assert torch.equal(a, b) and not torch.allclose(a, ref_mu, atol=1e-3), "Without noise argument, use random noise from the global generator."
        model.eval()
        z, mu, logvar = select_latent(model, batch)
        assert mu is None and logvar is None and z.shape == (3, 6) and not z.any(), "Evaluation mode must give a zero latent and no posterior."
        model.train()
        z, mu, logvar = select_latent(model, {"observation.state": batch["observation.state"]})
        assert mu is None and not z.any(), "Without actions the latent is zero even in training mode."
        print("Exercise 8B.3 passed")
        """,
            use_intro="See both branches on real data: the sampled latent in training mode and the zero latent at inference.",
            use="""
        model = copy.deepcopy(LEARNER)
        model.hooks = ACTHooks(posterior=ref.encode_posterior, latent=select_latent)
        with torch.no_grad():
            model.train()
            z, mu, logvar = select_latent(model, SMALL_REAL)
            print("training latent mean |z|:", float(z.abs().mean()), "| |mu|:", float(mu.abs().mean()))
            model.eval()
            z, mu, logvar = select_latent(model, SMALL_REAL)
            print("inference latent is zero:", not bool(z.any()), "| mu is None:", mu is None)
        """,
        )
    )

    L.items.append(
        Exercise(
            key="8B.4",
            title="Encoder tokens: latent, state and camera cells",
            minutes="15–20",
            purpose="Assemble everything the encoder reads into one sequence, with a position for every token.",
            contract="""
        - `model` has `latent_proj` (`Linear(L, D)`), `state_proj` (`Linear(S, D)`), `camera_proj` (`Conv2d(C, D, 1)`), `pos_1d` (`nn.Embedding(2, D)`), `settings.dim_model` (`D`), and `hooks.camera_positions(height, width, dim) -> (1, dim, height, width)`.
        - Inputs: `latent (B, L)`, `state (B, S)`, `feature_maps`: a list with one `(B, C, h, w)` tensor per camera.
        - Return `tokens (B, N, D)` and `pos (1, N, D)` with `N = 2 + sum(h * w)`.
        - **Token order:** latent token (`latent_proj(latent)`), state token (`state_proj(state)`), then camera 1 cells row by row, then camera 2, … A feature map becomes tokens by `camera_proj` then flattening `(h, w)` to `h*w` (row-major) with the feature axis last.
        - **Positions, same order and device:** the two rows of `model.pos_1d.weight` for the latent and state tokens; then for each camera `model.hooks.camera_positions(h, w, D)` flattened the same way (call the hook once per camera). Convert each camera position tensor to the feature tensor's device and dtype before you append it.
        """,
            method="ACT encoder input: `[latent, state, image feature map pixels]` with learned positions for the first two and sinusoidal 2D positions for the pixels ([ACT paper](https://arxiv.org/abs/2304.13705), appendix; pinned `ACT.forward`).",
            pointers="Start two lists, `tokens` and `positions`. Each entry has the shape `(B, n, D)` or `(1, n, D)`. Use `flatten(2)` and `transpose(1, 2)` for `(B, D, h, w) → (B, h*w, D)`.",
            hint="""
        ```python
        tokens = [model.latent_proj(latent)[:, None], model.state_proj(state)[:, None]]    # (B, 1, D) each
        positions = [model.pos_1d.weight[None]]                                           # (1, 2, D)
        for features in feature_maps:
            height, width = features.shape[-2:]
            tokens.append(model.camera_proj(features).flatten(2).transpose(1, 2))          # (B, h*w, D)
            camera_pos = model.hooks.camera_positions(height, width, model.settings.dim_model)
            positions.append(camera_pos.to(device=features.device, dtype=features.dtype).flatten(2).transpose(1, 2))  # (1, h*w, D)
        return torch.cat(tokens, dim=1), torch.cat(positions, dim=1)
        ```
        """,
            names=["build_encoder_tokens"],
            module="vla_lab.teaching.act_model",
            stub="""
        def build_encoder_tokens(model, latent, state, feature_maps):
            # Return (tokens (B, N, D), pos (1, N, D)): latent, state, then each camera's cells in row-major order.
            raise NotImplementedError("Complete Exercise 8B.4: build_encoder_tokens")
        """,
            check="""
        model = copy.deepcopy(LEARNER).eval()
        batch = CHECK_BATCH
        maps = [model.backbone(batch[key])["feature_map"] for key in IMAGES]
        latent = torch.randn(3, 6)
        seen = []

        def counting_positions(height, width, dim):
            seen.append((height, width, dim))
            return camera_position_embedding_2d(height, width, dim)

        model.hooks = ACTHooks(camera_positions=counting_positions)
        tokens, pos = build_encoder_tokens(model, latent, batch["observation.state"], maps)
        cells = sum(m.shape[-2] * m.shape[-1] for m in maps)
        assert tokens.shape == (3, 2 + cells, 32) and pos.shape == (1, 2 + cells, 32), f"Expected tokens (3, {2 + cells}, 32) and pos (1, {2 + cells}, 32)."
        assert seen == [(maps[0].shape[-2], maps[0].shape[-1], 32)] * 2, f"Call the camera position hook once per camera with (h, w, D); saw {seen}."
        close(tokens[:, 0], model.latent_proj(latent), "token 0 must be the latent token")
        close(tokens[:, 1], model.state_proj(batch["observation.state"]), "token 1 must be the state token")
        close(pos[0, :2], model.pos_1d.weight, "positions 0 and 1 come from pos_1d")
        projected = model.camera_proj(maps[0])
        close(tokens[:, 2], projected[:, :, 0, 0], "first camera token is cell (row 0, column 0)")
        close(tokens[:, 2 + maps[0].shape[-1]], projected[:, :, 1, 0], "row-major order: the next row starts after one row of cells")
        oracle_tokens, oracle_pos = ref.build_encoder_tokens(model, latent, batch["observation.state"], maps)
        close(tokens, oracle_tokens, "tokens differ from the reference"); close(pos, oracle_pos, "positions differ from the reference")
        print("Exercise 8B.4 passed")
        """,
            use_intro="Build the encoder input for real camera frames and push it through the (supplied) Transformer encoder.",
            use="""
        model = copy.deepcopy(LEARNER).eval()
        model.hooks = ACTHooks(camera_positions=camera_position_embedding_2d)
        with torch.no_grad():
            maps = [model.backbone(SMALL_REAL[key])["feature_map"] for key in IMAGES]
            tokens, pos = build_encoder_tokens(model, torch.zeros(8, 6), SMALL_REAL["observation.state"], maps)
            memory = model.encoder(tokens, pos)
        print("token sequence:", tuple(tokens.shape), "= latent + state +", [int(m.shape[-2] * m.shape[-1]) for m in maps], "camera cells")
        print("encoder memory:", tuple(memory.shape))
        """,
        )
    )

    L.items.append(
        Exercise(
            key="8B.5",
            title="Action queries, decoder and head",
            minutes="10–15",
            purpose="Turn the encoder memory into a chunk of actions. The decoder input is zeros: only the learned query positions say which future step each query is.",
            contract="""
        - `model` has `settings.chunk_size` (`T`), `settings.dim_model` (`D`), `decoder` (callable `decoder(x, memory, query_pos, memory_pos) -> (B, T, D)`), `query_embed` (`nn.Embedding(T, D)`) and `action_head` (`Linear(D, A)`).
        - `encoder_out (B, N, D)`, `pos (1, N, D)` (the positions of the encoder tokens).
        - Decoder input `x`: zeros `(B, T, D)` with the dtype and device of `encoder_out`. Query position: `model.query_embed.weight[None]`, shape `(1, T, D)`. Memory position: `pos`.
        - Return `action_head(decoder_output)`, shape `(B, T, A)`.
        """,
            method="DETR-style learned object queries, used by ACT as *action queries* ([ACT paper](https://arxiv.org/abs/2304.13705)). Pinned code: the `decoder_in = torch.zeros(...)` lines of `ACT.forward`.",
            pointers="Make zeros, call the decoder with the three position-related arguments, apply the head.",
            hint="""
        ```python
        batch = encoder_out.shape[0]
        queries = torch.zeros(batch, model.settings.chunk_size, model.settings.dim_model,
                              dtype=encoder_out.dtype, device=encoder_out.device)
        out = model.decoder(queries, encoder_out, model.query_embed.weight[None], pos)
        return model.action_head(out)
        ```
        """,
            names=["decode_actions"],
            module="vla_lab.teaching.act_model",
            stub="""
        def decode_actions(model, encoder_out, pos):
            # Zero queries + learned query positions, cross-attend to the encoder output, then the action head.
            raise NotImplementedError("Complete Exercise 8B.5: decode_actions")
        """,
            check='''
        class RecordingDecoder(nn.Module):
            """Stands in for the decoder module and records its arguments."""
            def forward(self, x, memory, query_pos=None, memory_pos=None, memory_key_padding_mask=None):
                self.args = (x, memory, query_pos, memory_pos)
                return x + 1.0

        model = copy.deepcopy(LEARNER).eval()
        encoder_out, pos = torch.randn(3, 9, 32), torch.randn(1, 9, 32)
        actions = decode_actions(model, encoder_out, pos)
        assert actions.shape == (3, 20, 7), "Expected (B, chunk_size, action_dim)."
        close(actions, ref.decode_actions(model, encoder_out, pos), "actions differ from the reference decoding")
        probe = copy.deepcopy(model)
        probe.decoder = RecordingDecoder()
        out = decode_actions(probe, encoder_out, pos)
        x, memory, query_pos, memory_pos = probe.decoder.args
        assert x.shape == (3, 20, 32) and not x.any(), "The decoder input must be zeros (B, T, D)."
        assert torch.equal(memory, encoder_out) and torch.equal(memory_pos, pos), "Pass the encoder output and its positions."
        assert torch.equal(query_pos, probe.query_embed.weight[None]), "The query position is the learned query_embed table."
        close(out, probe.action_head(torch.ones(3, 20, 32)), "the head must be applied to the decoder output")
        print("Exercise 8B.5 passed")
        ''',
            use_intro="Run the five functions together on real data without training. The prediction is from reference weights in a small random LeRobot ACT, so its numbers mean nothing; the shapes and the zero-latent path are the point.",
            use="""
        model = copy.deepcopy(LEARNER).eval()
        model.hooks = ACTHooks(camera_positions=camera_position_embedding_2d, posterior=encode_posterior, latent=select_latent,
                               encoder_tokens=build_encoder_tokens, decode=decode_actions)
        with torch.no_grad():
            actions, (mu, logvar) = model(SMALL_REAL)
        print("predicted chunk:", tuple(actions.shape), "| posterior returned:", mu, logvar)
        """,
        )
    )
    return L


def lesson_08b_tail(L: Lesson) -> Lesson:
    L.items += [
        Md(r"""
        ## Assemble: your five functions inside the real model

        The next cells build an `ACTModel` whose five hooks are **your** functions. The model has the pinned structure (posterior encoder, encoder, decoder, head). We then copy the weights of the small real LeRobot ACT into it. Before that copy, the learner has random weights; after it, it holds *reference weights in the learner* and must give the same numbers as the reference. Only then is it safe to use.
        """),
        Code(
            "# Supplied: assemble the model from YOUR functions (random weights at first)\n"
            + tidy(r"""
            YOUR_HOOKS = ACTHooks(
                camera_positions=camera_position_embedding_2d, posterior=encode_posterior, latent=select_latent,
                encoder_tokens=build_encoder_tokens, decode=decode_actions,
            )
            settings = ACTSettings.from_lerobot(REFERENCE.config)
            backbone, channels = make_act_backbone(settings.vision_backbone)
            YOURS = ACTModel(settings, backbone, channels, YOUR_HOOKS)
            before = verify_act_parity(YOURS, REFERENCE, CHECK_BATCH)
            print("random weights, largest difference from reference:", {k: round(v, 3) for k, v in before.items()})
        """)
        ),
        Code(
            "# Supplied: copy the reference weights into your module (explicit converter, strict loading)\n"
            + tidy(r"""
            converted = convert_lerobot_act_state_dict(REFERENCE.state_dict())
            print("tensors after conversion:", len(converted), "| LeRobot tensors:", len(REFERENCE.state_dict()),
                  "(each packed attention tensor becomes query, key and value tensors)")
            print(load_lerobot_act(YOURS, REFERENCE))
        """)
        ),
        Code(
            "# Check: identical weights and inputs must give identical outputs\n"
            + tidy(r"""
            after = verify_act_parity(YOURS, REFERENCE, CHECK_BATCH)
            print({k: f"{v:.1e}" for k, v in after.items()})
            assert after["inference_actions"] < 1e-4, "Zero-latent inference path differs from the pinned ACT."
            assert after["training_actions"] < 1e-4 and after["mu"] < 1e-5 and after["logvar"] < 1e-5, "Posterior path differs from the pinned ACT."
            print("Parity with the pinned ACT: passed")
        """)
        ),
        Code(
            "# Check: your model plus the training unit's loss callback equals the pinned policy loss\n"
            + tidy(r"""
            from lerobot.policies.act import ACTPolicy

            pinned_policy = ACTPolicy(REFERENCE.config)
            pinned_policy.model.load_state_dict(REFERENCE.state_dict())
            pinned_policy.train(); YOURS.train()
            torch.manual_seed(7)
            pinned_loss, pinned_parts = pinned_policy(dict(CHECK_BATCH))        # the pinned loss path, used only for comparison
            torch.manual_seed(7)
            your_loss, your_parts = act_loss_fn(YOURS, CHECK_BATCH, kl_weight=REFERENCE.config.kl_weight)
            print("pinned policy loss:", float(pinned_loss), "| your model + act_loss_fn:", float(your_loss))
            assert abs(float(pinned_loss) - float(your_loss)) < 1e-4, "The two loss paths must agree for equal weights."
        """)
        ),
        Md("""
        ### Prepared checkpoint: real-size parity before any live use

        The same check at real size: the local **prepared** ACT checkpoint `act-bowl-plate-probe-200` (a 200-step LeRobot training run) is copied into your module and compared on real camera frames at 256×256. This reads local files only; no download starts. If the files are missing, the cell says how to install them. Parity must hold before you run your module in the simulator.
        """),
        Code(r"""
            RUN_CHECKPOINT_PARITY = True

            if RUN_CHECKPOINT_PARITY:
                from lerobot.policies.act import ACTPolicy

                checkpoint = DATA / "runs/act-bowl-plate-probe-200/checkpoint-000200"
                if not (checkpoint / "model.safetensors").is_file():
                    print("Prepared ACT checkpoint not found. Install the reference bundle with `just references-install` (outside this notebook).")
                else:
                    config = ACTConfig.from_pretrained(checkpoint, local_files_only=True)
                    config.device = "cpu"
                    prepared = ACTPolicy.from_pretrained(checkpoint, config=config, strict=True, local_files_only=True)
                    full_size = act_from_lerobot(prepared.model, hooks=YOUR_HOOKS)        # reference weights in the learner
                    real = load_act_samples(CONFIGS, DATA, "bowl_plate", "validation", count=4, chunk_size=config.chunk_size)
                    diffs = verify_act_parity(full_size, prepared.model, real)
                    print("largest differences at real size:", {k: f"{v:.1e}" for k, v in diffs.items()})
                    assert max(diffs.values()) < 5e-4, "Reference weights in your module must reproduce the pinned model."
                    saved = yaml.safe_load((WORKSPACE / "manifests/act_references.yaml").read_text())["tasks"]["bowl_plate"]
                    print("Prepared policy result (saved by the project, simulator development states; NOT produced here):",
                          saved["development_successes"], "of", saved["development_episodes"])
            else:
                print("Prepared-checkpoint parity is off. Set RUN_CHECKPOINT_PARITY=True to run it.")
        """),
        Md(r"""
        ## Experiment — which inputs does your ACT need? (30–60 min)

        **Question.** ACT reads state, two cameras and 2D camera positions. Which of these does a *small, briefly trained* ACT use, measured on held-out real demonstrations at inference (zero latent)?

        **Setup (real data).** 256 training windows and 64 validation windows, spread evenly over the frozen `bowl_plate` train and validation episodes; each has both camera images (box-filtered to 32×32 so that CPU training is fast), the 8-value state and 20 actions (standardized with training statistics; windows near the end of an episode have padded actions). The model is a small ACT (width 64, 4 heads, latent 8) with a ResNet-18 backbone and **your five functions**. The loss is the training unit's `act_loss_fn` with `kl_weight` below, run by `train_loop`.

        **Fixed for every trial.** The data above; weight seed `0`; batch seed `1`; **150 updates**, batch 8, AdamW with weight decay `1e-4`. These are *learner trained weights*: they start from random numbers, not from the prepared checkpoint.

        **Allowed changes (per trial).** `arm` (one of `full`, `no_cam_pos`, `one_camera`, `state_only`, `cameras_only` — each swaps one hook function), `lr` (`3e-4` or `1e-3`), `kl_weight` (`0.1` or `10`).

        **Measures.** `val_l1_zero`: masked L1 on validation windows with the zero latent (the inference path). `val_l1_posterior`: the same with the latent set to the posterior mean (an optimistic bound that uses the true future actions). `val_kl`: posterior KL. Also report a *do-nothing baseline*: the L1 of always predicting the training mean action.

        **Prediction first.** In the next cell write which arm you expect to be best and which worst, and whether `val_l1_posterior` will be far below `val_l1_zero`.

        **Keep failed trials.** Every trial is appended to `TRIALS`. A run that is no better than the baseline is a valid result. One hundred fifty updates are too few to get a good policy; this experiment compares *inputs*, not final quality.
        """),
        Code('PREDICTION = "TODO: replace this text with your prediction."\nprint(PREDICTION)'),
        Code("# Supplied: trial table helper\n" + tidy(EXPERIMENT_TABLE_HELPER)),
        Code(r"""
            # Supplied: data for the experiment (real windows, 32x32 images)
            SIZE, CHUNK = 32, 20
            TRAIN = load_act_samples(CONFIGS, DATA, "bowl_plate", "train", count=256, chunk_size=CHUNK, image_size=SIZE)
            VALID = load_act_samples(CONFIGS, DATA, "bowl_plate", "validation", count=64, chunk_size=CHUNK, image_size=SIZE)
            baseline_prediction = TRAIN["action"][~TRAIN["action_is_pad"]].mean(0)
            do_nothing = masked_action_l1(baseline_prediction.expand_as(VALID["action"]), VALID["action"], VALID["action_is_pad"]).item()
            print("train windows:", len(TRAIN["action"]), "validation windows:", len(VALID["action"]), "| image size:", SIZE)
            print("do-nothing baseline (always the mean training action), L1:", round(do_nothing, 4))
        """),
        Code(r"""
            # Supplied: harness. Every arm changes ONE of your hook functions; the rest stay yours.
            STEPS, BATCH, SEED_WEIGHTS, SEED_BATCH, WEIGHT_DECAY = 150, 8, 0, 1, 1e-4
            SMALL = ACTSettings(state_dim=8, action_dim=7, image_keys=IMAGES, chunk_size=CHUNK, dim_model=64, n_heads=4,
                                dim_feedforward=128, n_encoder_layers=2, n_vae_encoder_layers=2, n_decoder_layers=1,
                                latent_dim=8, dropout=0.0)


            def hooks_for(arm):
                if arm == "full":
                    return YOUR_HOOKS
                if arm == "no_cam_pos":      # camera cells get no position: the encoder cannot tell where a cell is
                    return ACTHooks(camera_positions=lambda h, w, d: torch.zeros(1, d, h, w), posterior=encode_posterior,
                                    latent=select_latent, encoder_tokens=build_encoder_tokens, decode=decode_actions)
                variants = {
                    "one_camera": lambda m, z, s, maps: build_encoder_tokens(m, z, s, maps[:1]),
                    "state_only": lambda m, z, s, maps: build_encoder_tokens(m, z, s, []),
                    "cameras_only": lambda m, z, s, maps: build_encoder_tokens(m, z, torch.zeros_like(s), maps),
                }
                return ACTHooks(camera_positions=camera_position_embedding_2d, posterior=encode_posterior, latent=select_latent,
                                encoder_tokens=variants[arm], decode=decode_actions)


            BASELINE = {"arm": "full", "lr": 1e-3, "kl_weight": 10.0}
            TRIALS = []


            def run_trial(config):
                started = time.perf_counter()
                torch.manual_seed(SEED_WEIGHTS)
                model = ACTModel.build(SMALL, hooks=hooks_for(config["arm"]))       # random weights: learner trained weights
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=WEIGHT_DECAY)
                draws = torch.Generator().manual_seed(SEED_BATCH)
                batches = []
                for _ in range(STEPS):
                    pick = torch.randint(0, len(TRAIN["action"]), (BATCH,), generator=draws)
                    batches.append({key: value[pick] for key, value in TRAIN.items()})
                loss_fn = partial(act_loss_fn, kl_weight=config["kl_weight"])
                history = train_loop(model, batches, optimizer, loss_fn, steps=STEPS)
                model.eval()
                with torch.no_grad():
                    zero, _ = model(VALID)
                    model.train()
                    posterior, (mu, logvar) = model(VALID, noise=torch.zeros(len(VALID["action"]), SMALL.latent_dim))
                    model.eval()
                return {
                    "val_l1_zero": masked_action_l1(zero, VALID["action"], VALID["action_is_pad"]).item(),
                    "val_l1_posterior": masked_action_l1(posterior, VALID["action"], VALID["action_is_pad"]).item(),
                    "val_kl": (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean().item(),
                    "last_train_loss": history[-1]["loss"],
                    "seconds": time.perf_counter() - started,
                }


            def record(config):
                config = {**BASELINE, **config}
                try:
                    TRIALS.append({"config": config, "status": "ok", **run_trial(config)})
                except Exception as error:  # a failed trial is evidence too
                    TRIALS.append({"config": config, "status": f"failed: {type(error).__name__}: {error}"})


            record(BASELINE)
            COLUMNS = ["val_l1_zero", "val_l1_posterior", "val_kl", "last_train_loss", "seconds"]
            show_trials(TRIALS, COLUMNS);
        """),
        Md(
            "**Your trials.** Add at least two trials, each changing one setting. Budget: 150 updates per trial."
        ),
        Pair(
            learner="""
            # Implement: choose your trials.
            MY_TRIALS = [
                # {"arm": "state_only"},
            ]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
            reference="""
            # Reference choice of trials: one change per trial.
            MY_TRIALS = [{"arm": "state_only"}, {"arm": "no_cam_pos"}, {"arm": "one_camera"}, {"kl_weight": 0.1}]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
        ),
        Code(r"""
            # Check: the protocol was followed.
            assert TRIALS[0]["config"] == BASELINE, "The first trial must be the baseline."
            assert len(TRIALS) >= 3, "Run the baseline and at least two trials of your own."
            assert all(set(t["config"]) == set(BASELINE) for t in TRIALS), "Only arm, lr and kl_weight may change."
            assert all(t["config"]["arm"] in {"full", "no_cam_pos", "one_camera", "state_only", "cameras_only"} for t in TRIALS)
            ok = [t for t in TRIALS if t["status"] == "ok"]
            assert all(torch.isfinite(torch.tensor([t["val_l1_zero"], t["val_l1_posterior"]])).all() for t in ok), "Measures must be finite."
            print(f"{len(TRIALS)} trials kept ({len(ok)} ok). Budget: {STEPS} updates, batch {BATCH}, seeds {SEED_WEIGHTS}/{SEED_BATCH}.")
        """),
        Code(r"""
            # Use your function: every trial against the do-nothing baseline
            ok = [t for t in TRIALS if t["status"] == "ok"]
            labels = [", ".join(f"{k}={v}" for k, v in t["config"].items() if v != BASELINE[k]) or "baseline" for t in ok]
            fig, axis = plt.subplots(figsize=(8, 3.5))
            axis.bar([i - 0.2 for i in range(len(ok))], [t["val_l1_zero"] for t in ok], width=0.4, label="zero latent (inference)")
            axis.bar([i + 0.2 for i in range(len(ok))], [t["val_l1_posterior"] for t in ok], width=0.4, label="posterior mean (uses true actions)")
            axis.axhline(do_nothing, color="k", linestyle="--", label="do-nothing baseline")
            axis.set_xticks(range(len(ok))); axis.set_xticklabels(labels, rotation=30, ha="right"); axis.set_ylabel("validation masked L1")
            axis.legend(); plt.tight_layout(); plt.show()
        """),
        Md("""
        **Explain (2–4 sentences, edit this cell).** Which arm was best and which worst? Is every arm better than the do-nothing baseline? Was `val_l1_posterior` much lower than `val_l1_zero`? What can you *not* conclude from 150 updates at 32×32 resolution?

        *Optional extension:* repeat the best two arms with another `SEED_BATCH`. Keep both sets of rows.
        """),
        Md("""
        ## Optional: fine-tune the reference weights in your learner (opt-in)

        This is an actual, bounded full-size training run. It starts from reference weights in the learner and runs 10 updates with your model, the training loss callback, and `train_loop`. It reports held-out L1 before and after training. It writes no checkpoint and does not measure simulator success. The code selects CUDA when available and otherwise uses CPU. Set `RUN_FULL_SIZE_FINETUNE = True` to run it.
        """),
        Code(r"""
            RUN_FULL_SIZE_FINETUNE = False
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            FINETUNE_STEPS, FINETUNE_BATCH, FINETUNE_LR = 10, 4, 1e-5   # the pinned recipe's learning rate

            if RUN_FULL_SIZE_FINETUNE:
                from lerobot.policies.act import ACTPolicy

                checkpoint = DATA / "runs/act-bowl-plate-probe-200/checkpoint-000200"
                if not (checkpoint / "model.safetensors").is_file():
                    raise FileNotFoundError(f"Prepared ACT checkpoint missing: {checkpoint}")
                config = ACTConfig.from_pretrained(checkpoint, local_files_only=True)
                config.device = DEVICE
                prepared = ACTPolicy.from_pretrained(checkpoint, config=config, strict=True, local_files_only=True)
                learner_full = act_from_lerobot(prepared.model, hooks=YOUR_HOOKS).to(DEVICE)  # reference weights in the learner
                train = load_act_samples(CONFIGS, DATA, "bowl_plate", "train", count=24, chunk_size=config.chunk_size)
                valid = load_act_samples(CONFIGS, DATA, "bowl_plate", "validation", count=16, chunk_size=config.chunk_size)
                train = {key: value.to(DEVICE) if isinstance(value, torch.Tensor) else value for key, value in train.items()}
                valid = {key: value.to(DEVICE) if isinstance(value, torch.Tensor) else value for key, value in valid.items()}

                def held_out(model):
                    model.eval()
                    with torch.no_grad():
                        return masked_action_l1(model(valid)[0], valid["action"], valid["action_is_pad"]).item()

                before_l1 = held_out(learner_full)
                optimizer = torch.optim.AdamW(learner_full.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
                draws = torch.Generator(device=DEVICE).manual_seed(1)
                batches = []
                for _ in range(FINETUNE_STEPS):
                    pick = torch.randint(0, len(train["action"]), (FINETUNE_BATCH,), generator=draws, device=DEVICE)
                    batches.append({key: value[pick] for key, value in train.items()})
                history = train_loop(learner_full, batches, optimizer, partial(act_loss_fn, kl_weight=config.kl_weight), steps=FINETUNE_STEPS)
                print("held-out L1 before (reference weights in the learner):", round(before_l1, 4))
                print("held-out L1 after  (learner trained weights):         ", round(held_out(learner_full), 4))
                print("training losses:", [round(h["loss"], 3) for h in history])
            else:
                print("Full-size fine-tuning is off. Set RUN_FULL_SIZE_FINETUNE=True to run", FINETUNE_STEPS, "updates on", DEVICE)
        """),
        Md("""
        ## What you built

        - Five functions that make up ACT, wired into the real model through hooks and verified to give the pinned model's outputs for identical weights, on small random weights and on the prepared checkpoint at real size.
        - A clear separation of *reference weights in the learner*, *learner trained weights* and *prepared policy results*.

        **Next:** Unit 9 starts diffusion policy: the forward noising process and the training objective.
        """),
    ]
    return L


DIFFUSION_FIXTURE = r'''
IMAGES = ("observation.images.image", "observation.images.image2")


def tiny_policy(seed=5):
    """A small *real* pinned DiffusionPolicy with random weights (64x64 images, horizon 16)."""
    torch.manual_seed(seed)
    config = DiffusionConfig(
        input_features={
            IMAGES[0]: PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            IMAGES[1]: PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        n_obs_steps=2, horizon=16, n_action_steps=8, drop_n_last_frames=7,
        down_dims=(16, 32, 64), spatial_softmax_num_keypoints=8, diffusion_step_embed_dim=32,
        pretrained_backbone_weights=None, crop_is_random=False, num_train_timesteps=100,
        num_inference_steps=20, do_mask_loss_for_padding=True, push_to_hub=False, device="cpu",
    )
    policy = DiffusionPolicy(config)
    for parameter in policy.parameters():
        if parameter.dim() == 1:
            nn.init.normal_(parameter, std=0.05)
    return policy.eval()


def algorithm_check_batch(batch=3, seed=1):
    """Random numbers in the right shapes. Clearly labelled: an algorithm check, not data."""
    g = torch.Generator().manual_seed(seed)
    pad = torch.zeros(batch, 16, dtype=torch.bool)
    pad[1, 12:] = True
    pad[2, 3:] = True
    return {
        "observation.state": torch.randn(batch, 2, 8, generator=g),
        IMAGES[0]: torch.rand(batch, 2, 3, 64, 64, generator=g),
        IMAGES[1]: torch.rand(batch, 2, 3, 64, 64, generator=g),
        "action": torch.rand(batch, 16, 7, generator=g) * 2 - 1,
        "action_is_pad": pad,
    }


POLICY = tiny_policy()
CHECK_BATCH = algorithm_check_batch()
'''

DIFFUSION_BACKGROUND = """
## Background: the forward process and what the network learns

Start from a clean action window `x0` (16 steps × 7 values, scaled to `[-1, 1]`). The **forward process** adds Gaussian noise in `T = 100` small steps. A closed form jumps straight to step `t`:

`x_t = sqrt(ᾱ_t) · x0 + sqrt(1 − ᾱ_t) · ε`, with `ε ~ N(0, I)` and `ᾱ_t = (1 − β_0)(1 − β_1)…(1 − β_t)`.

Small `t` is almost clean; large `t` is almost pure noise. The **network** `ε_θ(x_t, t, observation)` is trained to predict the noise `ε` that was added. The loss is the mean squared error over *real* action steps. Sampling later runs the process in reverse (Unit 11).

The pinned recipe, from `lerobot/policies/diffusion` at revision `e0d50211ef236143ae867228662b7dfaba554f02`: `T = 100`, the `squaredcos_cap_v2` (cosine) beta schedule, noise prediction, padded steps masked (`do_mask_loss_for_padding=True`), observation history 2, horizon 16, 8 executed actions. The current trainer uses Adam with a cosine learning-rate schedule and **no EMA** (exponential moving average of weights). Sources: [DDPM](https://arxiv.org/abs/2006.11239) sections 2 and 3, equations 4 and 12, Algorithm 1; [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/).

**Supplied here:** the beta schedule, the denoiser network (`ConditionalUnet1D`, which you write in Unit 10), the data reader and the training loop (`optimizer_update`, `train_loop` from the training unit). **You write** the five functions that turn a clean window into a training signal.
"""


def lesson_09() -> Lesson:
    L = Lesson("09-diffusion-training", "09-diffusion-training", "Diffusion training")
    L.items += [
        Md(r"""
        # Unit 9 — Diffusion policy training: noise, objective, update

        **Your result:** the functions that train a diffusion policy: cumulative alphas, forward noising, timestep selection, a masked noise loss, and a loss callback that the training loop uses to update a real denoiser. For the same random numbers your loss equals the pinned LeRobot loss.

        Five short exercises (10–20 minutes each) and a larger experiment (30–60 minutes). These times are estimates, not measurements. Full answers are in [the reference notebook](09-diffusion-training-reference.ipynb).
        """),
        setup_cell("""
            import math
            import time
            import torch
            from torch import nn
            import matplotlib.pyplot as plt

            import vla_lab.teaching.diffusion as ref  # package reference: used by checks only
            from vla_lab.teaching.diffusion import (
                ConditionalUnet1D, DDPMSchedule, DiffusionDenoiser, LeRobotDenoiser, cosine_betas,
                flat_state_condition, load_state_action_windows, stack_observation, take_batch,
            )
            from vla_lab.teaching.training import optimizer_update, train_loop  # from the training unit (package versions)
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
            from lerobot.configs import FeatureType, PolicyFeature
            from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy

            torch.manual_seed(0)
            print("PyTorch:", torch.__version__)
        """),
        Code("# Supplied: a helper that the checks use\n" + tidy(CLOSE_HELPER)),
        Md(DIFFUSION_BACKGROUND),
        Md("""
        **Real data used in this unit.** Real LIBERO `bowl_plate` demonstrations are read from local files (no download starts here). A *window* has the robot state at the last 2 frames and 16 actions that start one frame before the current one. The first window slot is an action *already taken*; this is the pinned alignment, and Unit 11 uses it to pick the executed actions. Windows near an episode edge repeat the nearest frame and set `action_is_pad=True`. The recording rate is 10 Hz; nothing is resampled to the simulator's 20 Hz controller. Values are scaled to `[-1, 1]` with training-set extrema. If the files are missing, the cell tells you the setup command.
        """),
        Code(r"""
            # Supplied: real windows (train and validation episodes are kept apart)
            TRAIN = load_state_action_windows(CONFIGS, DATA, "bowl_plate", "train")
            VALID = load_state_action_windows(CONFIGS, DATA, "bowl_plate", "validation")
            print("train windows:", len(TRAIN["action"]), "| validation windows:", len(VALID["action"]))
            print("state:", tuple(VALID["observation.state"].shape[1:]), "| action:", tuple(VALID["action"].shape[1:]))
            EXAMPLE = int((VALID["action_is_pad"].sum(1) == 4).nonzero()[0])   # a window whose last 4 steps are padding
            example = VALID["action"][EXAMPLE]
        """),
        Code(
            "# Supplied: a small real pinned DiffusionPolicy and a batch for algorithm checks\n"
            + tidy(DIFFUSION_FIXTURE)
        ),
    ]

    L.items.append(
        Exercise(
            key="9.1",
            title="Cumulative alphas",
            minutes="10",
            purpose="The noise level at step `t` depends on the product of all `(1 − β)` up to `t`. Compute it once for all steps.",
            contract="""
        - `cumulative_alphas(betas)`: `betas` is a `float32` tensor `(T,)` with values in `(0, 1)`.
        - Return `alpha_bar` `(T,)`, `float32`, with `alpha_bar[t] = prod_{s <= t} (1 - betas[s])` (includes step `t`). So `alpha_bar[0] = 1 - betas[0]`.
        - Do not loop in Python over steps; one tensor operation is enough.
        """,
            method="`α_t = 1 − β_t`, `ᾱ_t = ∏_{s=1}^{t} α_s` — [DDPM](https://arxiv.org/abs/2006.11239), section 2, just before equation 4.",
            pointers="Turn betas into alphas first. Then use a cumulative product along the only axis.",
            hint="`torch.cumprod(1.0 - betas, dim=0)`.",
            names=["cumulative_alphas"],
            module="vla_lab.teaching.diffusion",
            stub="""
        def cumulative_alphas(betas):
            # betas (T,) -> alpha_bar (T,): alpha_bar[t] = product of (1 - beta[s]) for s <= t.
            raise NotImplementedError("Complete Exercise 9.1: cumulative_alphas")
        """,
            check="""
        betas = cosine_betas(100)
        alpha_bar = cumulative_alphas(betas)
        assert alpha_bar.shape == (100,) and alpha_bar.dtype == torch.float32, "Expected a float32 tensor of shape (T,)."
        close(alpha_bar[0], 1 - betas[0], "alpha_bar[0] is 1 - beta[0]")
        close(alpha_bar[2], (1 - betas[0]) * (1 - betas[1]) * (1 - betas[2]), "alpha_bar[2] is the product of three alphas")
        assert (alpha_bar[1:] < alpha_bar[:-1]).all(), "alpha_bar must decrease with t."
        scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
        close(alpha_bar, scheduler.alphas_cumprod, "differs from diffusers' alphas_cumprod", atol=1e-7)
        close(cumulative_alphas(torch.tensor([0.5, 0.5, 0.5])), torch.tensor([0.5, 0.25, 0.125]), "constant betas", atol=1e-7)
        print("Exercise 9.1 passed")
        """,
            use_intro="Compare two noise schedules: the cosine schedule used by the pinned policy and a linear one. Where is the signal gone?",
            use="""
        linear_betas = torch.linspace(1e-4, 0.02, 100)
        for name, b in (("cosine (pinned)", cosine_betas(100)), ("linear 1e-4..0.02", linear_betas)):
            plt.plot(cumulative_alphas(b), label=name)
        plt.xlabel("timestep t"); plt.ylabel("alpha_bar[t] (signal kept)"); plt.legend(); plt.grid(alpha=0.3); plt.show()
        print("alpha_bar at the last step: cosine", float(cumulative_alphas(cosine_betas(100))[-1]), "| linear", float(cumulative_alphas(linear_betas)[-1]))
        """,
        )
    )

    L.items.append(
        Exercise(
            key="9.2",
            title="Forward noising",
            minutes="15",
            purpose="Jump from a clean action window to its noisy version at any step in one operation.",
            contract="""
        - `x0` and `noise`: `(B, H, A)`. `timesteps`: `(B,)` `long`, values in `[0, T)`. `alpha_bar`: `(T,)`.
        - Each batch row uses its *own* timestep.
        - Return `x_t` `(B, H, A)` with `x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * noise`.
        - Do not change `x0` or `noise` in place. Gradients must flow to `x0` and `noise`.
        """,
            method="`q(x_t | x_0) = N(√ᾱ_t x_0, (1 − ᾱ_t) I)` — [DDPM](https://arxiv.org/abs/2006.11239), equation 4. This is also what `DDPMScheduler.add_noise` computes.",
            pointers="Look up the per-row coefficients with the timestep tensor. They have shape `(B,)`; reshape them so they broadcast against `(B, H, A)`.",
            hint="""
        ```python
        abar = alpha_bar.to(device=x0.device, dtype=x0.dtype)[timesteps]       # (B,)
        abar = abar.reshape(-1, 1, 1)                                          # (B, 1, 1)
        return abar.sqrt() * x0 + (1.0 - abar).sqrt() * noise
        ```
        """,
            names=["forward_noise"],
            module="vla_lab.teaching.diffusion",
            stub="""
        def forward_noise(x0, noise, timesteps, alpha_bar):
            # x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * noise, with a separate t for each batch row.
            raise NotImplementedError("Complete Exercise 9.2: forward_noise")
        """,
            check="""
        scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
        alpha_bar = cumulative_alphas(cosine_betas(100))
        x0, noise = torch.randn(4, 16, 7, requires_grad=True), torch.randn(4, 16, 7)
        t = torch.tensor([0, 17, 50, 99])
        out = forward_noise(x0, noise, t, alpha_bar)
        assert out.shape == x0.shape, "Return the same shape as x0."
        close(out, scheduler.add_noise(x0.detach(), noise, t), "x_t differs from diffusers' add_noise")
        row = forward_noise(x0[1:2].detach(), noise[1:2], torch.tensor([17]), alpha_bar)
        close(out[1:2], row, "a row must depend only on its own timestep")
        low = forward_noise(x0.detach(), noise, torch.zeros(4, dtype=torch.long), alpha_bar)
        assert (low - x0.detach()).abs().max() < 0.25, "At t = 0 the sample is almost the clean window."
        out.sum().backward()
        close(x0.grad[0], alpha_bar[0].sqrt().expand(16, 7), "d x_t / d x0 is sqrt(alpha_bar[t])", atol=1e-6)
        print("Exercise 9.2 passed")
        """,
            use_intro="Noise a real action window at five steps with the same fixed noise, and look at one action channel.",
            use="""
        alpha_bar = cumulative_alphas(cosine_betas(100))
        noise = torch.randn(1, 16, 7, generator=torch.Generator().manual_seed(0))
        steps = [0, 25, 50, 75, 99]
        fig, axis = plt.subplots(figsize=(7, 3.2))
        for t in steps:
            noisy = forward_noise(example[None], noise, torch.tensor([t]), alpha_bar)[0]
            axis.plot(noisy[:, 0], label=f"t={t}")
        axis.plot(example[:, 0], "k--", linewidth=2, label="clean")
        axis.axvspan(12 - 0.5, 15.5, color="gray", alpha=0.15, label="padded steps")
        axis.set_xlabel("action step"); axis.set_ylabel("channel 0"); axis.legend(ncol=3, fontsize=8); plt.show()
        """,
        )
    )

    L.items.append(
        Exercise(
            key="9.3",
            title="Choose noise and timesteps",
            minutes="10–15",
            purpose="Each training example needs its own noise and its own random noise level. Draw both in a fixed order so a run can be repeated exactly.",
            contract="""
        - `actions (B, H, A)` float tensor, `num_train_timesteps` (int `T`), `generator` (a `torch.Generator` on the same device as `actions`, or `None` for the global generator).
        - Draw **first** `noise ~ N(0, I)` with the shape, dtype and device of `actions`; **then** `timesteps`, `B` integers drawn uniformly from `0 … T-1` (`T` itself is excluded), dtype `long`, on the same device.
        - Use the same `generator` for both draws. Return `(noise, timesteps)`.
        """,
            method="Algorithm 1 in [DDPM](https://arxiv.org/abs/2006.11239): `t ~ Uniform({1, …, T})`, `ε ~ N(0, I)`. Indices here start at 0. The pinned `compute_loss` draws noise first, then `torch.randint`.",
            pointers="Two calls: `torch.randn` with an explicit shape, `torch.randint` with `low`, `high` and a size. Pass the generator to both.",
            hint="""
        ```python
        noise = torch.randn(actions.shape, generator=generator, device=actions.device, dtype=actions.dtype)
        timesteps = torch.randint(0, num_train_timesteps, (actions.shape[0],), generator=generator, device=actions.device)
        return noise, timesteps
        ```
        """,
            names=["sample_noise_and_timesteps"],
            module="vla_lab.teaching.diffusion",
            stub="""
        def sample_noise_and_timesteps(actions, num_train_timesteps, generator=None):
            # Draw noise first, then one integer timestep in [0, T) per row, from the same generator.
            raise NotImplementedError("Complete Exercise 9.3: sample_noise_and_timesteps")
        """,
            check="""
        actions = torch.zeros(2000, 16, 7)
        noise, t = sample_noise_and_timesteps(actions, 100, torch.Generator().manual_seed(3))
        assert noise.shape == actions.shape and t.shape == (2000,) and t.dtype == torch.long, "Expected noise like actions and long timesteps (B,)."
        assert t.min() == 0 and t.max() == 99, "Timesteps must cover 0 … T-1 and never reach T."
        assert abs(float(noise.mean())) < 0.01 and abs(float(noise.std()) - 1) < 0.01, "Noise must be standard normal."
        again = sample_noise_and_timesteps(actions, 100, torch.Generator().manual_seed(3))
        assert torch.equal(noise, again[0]) and torch.equal(t, again[1]), "The same seeded generator must give the same draws."
        torch.manual_seed(9); expected_noise = torch.randn(actions.shape); expected_t = torch.randint(0, 100, (2000,))
        torch.manual_seed(9); drawn = sample_noise_and_timesteps(actions, 100)
        assert torch.equal(drawn[0], expected_noise) and torch.equal(drawn[1], expected_t), "Draw the noise first, then the timesteps (the pinned order)."
        print("Exercise 9.3 passed")
        """,
            use_intro="Draw the noise levels for a real training batch and check that all levels are used about equally.",
            use="""
        batch = take_batch(TRAIN, torch.arange(512))
        noise, timesteps = sample_noise_and_timesteps(batch["action"], 100, torch.Generator().manual_seed(0))
        plt.hist(timesteps.numpy(), bins=20); plt.xlabel("timestep"); plt.ylabel("count in a batch of 512"); plt.show()
        print("noise shape:", tuple(noise.shape), "| first timesteps:", timesteps[:8].tolist())
        """,
        )
    )

    L.items.append(
        Exercise(
            key="9.4",
            title="Masked noise loss",
            minutes="10–15",
            purpose="Padded action steps are copies, not decisions. They must not train the network. This is the loss that ignores them.",
            contract="""
        - `prediction`, `target`: `(B, H, A)`. `is_pad`: `(B, H)` bool, `True` = padded step.
        - Return one scalar tensor: `sum((prediction - target)**2 * valid) / max(valid_steps * A, 1)`, where `valid = ~is_pad` is broadcast over `A` and `valid_steps` counts the valid `(row, step)` pairs in the whole batch.
        - Changing `prediction` or `target` only at padded steps must change neither the loss nor the gradient. A batch that is entirely padding returns `0` (no `nan`).
        """,
            method="Mean squared error `‖ε − ε_θ‖²` ([DDPM](https://arxiv.org/abs/2006.11239), equation 12 / Algorithm 1) averaged over real elements only; the pinned `compute_loss` does `(loss * mask).sum() / (mask.sum() * action_dim).clamp_min(1)`.",
            pointers="Square the error. Multiply by a float mask. Divide by the number of valid *scalar* values, not the number of rows.",
            hint="""
        ```python
        valid = (~is_pad.to(torch.bool)).unsqueeze(-1).to(prediction.dtype)        # (B, H, 1)
        denominator = (valid.sum() * prediction.shape[-1]).clamp_min(1)
        return ((prediction - target).square() * valid).sum() / denominator
        ```
        """,
            names=["masked_epsilon_loss"],
            module="vla_lab.teaching.diffusion",
            stub="""
        def masked_epsilon_loss(prediction, target, is_pad):
            # Mean squared error over valid (row, step) pairs and all A action values.
            raise NotImplementedError("Complete Exercise 9.4: masked_epsilon_loss")
        """,
            check="""
        pred = torch.zeros(2, 3, 2, requires_grad=True)
        target = torch.ones(2, 3, 2)
        pad = torch.tensor([[False, False, True], [False, True, True]])
        loss = masked_epsilon_loss(pred, target, pad)
        assert loss.shape == () and torch.isfinite(loss), "Return one finite scalar."
        close(loss, torch.tensor(1.0), "three valid steps, each with error 1")
        loss.backward()
        assert torch.count_nonzero(pred.grad[pad]) == 0, "Padded steps must have zero gradient."
        close(pred.grad[~pad].sum(), torch.tensor(-2 * 6 / 6.0), "gradient of the valid elements", atol=1e-6)
        shaken = pred.detach().clone(); shaken[pad] = 99.0
        close(masked_epsilon_loss(shaken, target, pad), loss.detach(), "changing padded predictions must not change the loss")
        assert masked_epsilon_loss(pred.detach(), target, torch.ones_like(pad)) == 0, "An all-padding batch gives exactly 0."
        print("Exercise 9.4 passed")
        """,
            use_intro="On real windows that contain padding, build a prediction that is perfect on real steps and wrong only on padded steps. A correct masked loss is exactly 0; a plain mean is not.",
            use="""
        has_pad = VALID["action_is_pad"].any(1)
        batch = take_batch(VALID, has_pad.nonzero().flatten())
        noise = torch.randn(batch["action"].shape, generator=torch.Generator().manual_seed(1))
        prediction = noise.clone()
        prediction[batch["action_is_pad"]] = 0.0              # wrong only where the step is padded
        print("windows with padding:", len(noise), "| padded fraction of their steps:", round(float(batch["action_is_pad"].float().mean()), 3))
        print("masked loss:", float(masked_epsilon_loss(prediction, noise, batch["action_is_pad"])))
        print("plain mean :", float((prediction - noise).square().mean()))
        """,
        )
    )

    L.items.append(
        Exercise(
            key="9.5",
            title="The loss callback and one denoiser update",
            minutes="20",
            purpose="Combine your four functions into the objective that the training loop calls. The loop owns the optimizer step; you own the loss.",
            contract="""
        - `loss_fn(model, batch, *, noise=None, time=None) -> (scalar_loss, metrics)`. The loss tensor keeps its autograd graph; `metrics` is a dict of scalar tensors.
        - `model(noisy_actions, timesteps, batch)` predicts the noise and `model.schedule` holds `.alpha_bar (T,)` and `.num_train_timesteps`.
        - `batch["action"]` is `(B, H, A)`; `batch["action_is_pad"]` is `(B, H)`. The model reads the observations from `batch` itself.
        - If `noise` and/or `time` (the timesteps) are given, use them exactly. Draw whichever is missing with `sample_noise_and_timesteps`, **noise first**.
        - Steps: `noisy = forward_noise(actions, noise, time, alpha_bar)`; `prediction = model(noisy, time, batch)`; `loss = masked_epsilon_loss(prediction, noise, batch["action_is_pad"])`. Return `loss, {"epsilon_mse": loss}`.
        """,
            method="[DDPM](https://arxiv.org/abs/2006.11239) Algorithm 1: sample, noise, predict, regress on the noise. The training unit's `optimizer_update(model, batch, optimizer, loss_fn, noise=…, time=…)` calls `loss_fn`, runs `backward()` and steps the optimizer.",
            pointers="Draw only what is missing. Do not detach the loss. The padded mask is `batch['action_is_pad']`.",
            hint="""
        ```python
        actions = batch["action"]
        drawn_noise, drawn_time = sample_noise_and_timesteps(actions, model.schedule.num_train_timesteps)
        noise = drawn_noise if noise is None else noise
        time = drawn_time if time is None else time
        noisy = forward_noise(actions, noise, time, model.schedule.alpha_bar)
        loss = masked_epsilon_loss(model(noisy, time, batch), noise, batch["action_is_pad"])
        return loss, {"epsilon_mse": loss}
        ```
        Drawing both and using only what is missing keeps the random stream the same as the pinned code.
        """,
            names=["epsilon_loss_fn"],
            module="vla_lab.teaching.diffusion",
            stub="""
        def epsilon_loss_fn(model, batch, *, noise=None, time=None):
            # Return (loss, {"epsilon_mse": loss}). Use forward_noise and masked_epsilon_loss; keep the graph.
            raise NotImplementedError("Complete Exercise 9.5: epsilon_loss_fn")
        """,
            check="""
        denoiser = LeRobotDenoiser(POLICY)            # the pinned network, wrapped so that it predicts noise
        batch = CHECK_BATCH
        torch.manual_seed(21)
        expected = POLICY.diffusion.compute_loss(stack_observation(POLICY, batch))     # the pinned loss
        torch.manual_seed(21)
        loss, metrics = epsilon_loss_fn(denoiser, batch)
        assert loss.shape == () and "epsilon_mse" in metrics, "Return (scalar loss, metrics with epsilon_mse)."
        close(loss, expected, "your loss differs from the pinned compute_loss for the same random numbers", atol=1e-6)
        noise, time = torch.randn_like(batch["action"]), torch.tensor([3, 40, 90])
        a, _ = epsilon_loss_fn(denoiser, batch, noise=noise, time=time)
        b, _ = epsilon_loss_fn(denoiser, batch, noise=noise, time=time)
        close(a, b, "explicit noise and time must make the loss repeatable", atol=1e-7)
        by_hand = masked_epsilon_loss(denoiser(forward_noise(batch["action"], noise, time, denoiser.schedule.alpha_bar), time, batch),
                                      noise, batch["action_is_pad"])
        close(a, by_hand, "the loss with explicit noise and time", atol=1e-6)
        a.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in POLICY.diffusion.unet.parameters()), "The loss must keep its graph (U-Net gradients)."
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in POLICY.diffusion.rgb_encoder.parameters()), "Gradients must also reach the image encoders."
        POLICY.zero_grad()
        print("Exercise 9.5 passed")
        """,
            use_intro="Now use the loss to train a denoiser with the training unit's loop: a small U-Net conditioned on the real robot state, on real windows. Watch the loss, then ask which noise levels are hard.",
            use="""
        torch.manual_seed(0)
        schedule = DDPMSchedule.cosine(100)
        unet = ConditionalUnet1D(7, 2 * 8, (32, 64, 128), step_embed_dim=32)       # supplied denoiser; you write it in Unit 10
        model = DiffusionDenoiser(unet, schedule, flat_state_condition)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.95, 0.999))
        draws = torch.Generator().manual_seed(1)
        batches = [take_batch(TRAIN, torch.randint(0, len(TRAIN["action"]), (64,), generator=draws)) for _ in range(3)]
        history = train_loop(model, batches, optimizer, epsilon_loss_fn, steps=3)
        plt.plot([h["loss"] for h in history]); plt.xlabel("update"); plt.ylabel("masked epsilon MSE (three-update integration check)"); plt.show()
        model.eval()
        with torch.no_grad():
            fixed_noise = torch.randn(VALID["action"].shape, generator=torch.Generator().manual_seed(7))
            per_step = [float(epsilon_loss_fn(model, VALID, noise=fixed_noise, time=torch.full((len(fixed_noise),), t))[0]) for t in range(0, 100, 11)]
        plt.plot(range(0, 100, 11), per_step, marker="o"); plt.xlabel("timestep"); plt.ylabel("validation epsilon MSE"); plt.show()
        """,
        )
    )
    return L


def lesson_09_tail(L: Lesson) -> Lesson:
    L.items += [
        Md(r"""
        ## Experiment — which noise schedule is easier to learn? (30–60 min)

        **Question.** Under the same update budget, how do the cosine schedule and two linear schedules compare at *matched noise levels*, and does masking the padded steps matter?

        **Why "matched noise levels".** Timestep `t = 50` means different amounts of noise under different schedules, so comparing the loss at the same `t` is unfair. Compare at the same `ᾱ` (signal kept): for each schedule pick the step whose `ᾱ` is closest to `0.9` and to `0.5`. Also read `final_alpha_bar`: a schedule whose last step still keeps much signal ends before pure noise, which matters for sampling in Unit 11.

        **Setup (real data).** The windows above, a small U-Net conditioned on the flattened 2×8 state history (`ConditionalUnet1D`, supplied), your functions `cumulative_alphas`, `forward_noise`, `sample_noise_and_timesteps`, `masked_epsilon_loss` and `epsilon_loss_fn` inside the real update (`train_loop`).

        **Fixed for every trial.** Weight seed `0`; batch seed `1`; **200 updates**, batch 64, Adam(`lr=1e-3`, betas `(0.95, 0.999)`); `T = 100`; validation noise seed `7` and the same validation windows for every trial.

        **Allowed changes (per trial).** `schedule`: `cosine`, `linear` (β from `1e-4` to `0.02`) or `linear_strong` (β from `1e-4` to `0.1`). `mask`: `True` uses the real padding mask; `False` pretends no step is padded (the padded copies then train the network).

        **Measures** (validation windows, real steps only, fixed noise): `eps@0.9`, `eps@0.5` — noise-prediction MSE; `x0@0.9`, `x0@0.5` — MSE of the clean-window estimate `(x_t − √(1−ᾱ) ε̂) / √ᾱ`; `final_alpha_bar`.

        **Prediction first.** Edit the next cell: which schedule will have the lowest `eps@0.5`, and will `mask=False` change the measures much?

        **Keep failed trials.** Every trial is appended to `TRIALS`. No improvement is a valid result.
        """),
        Code('PREDICTION = "TODO: replace this text with your prediction."\nprint(PREDICTION)'),
        Code("# Supplied: trial table helper\n" + tidy(EXPERIMENT_TABLE_HELPER)),
        Code(r"""
            # Supplied: harness. Your functions run inside every update.
            STEPS, BATCH, LR, SEED_WEIGHTS, SEED_BATCH, SEED_NOISE = 200, 64, 1e-3, 0, 1, 7


            def make_schedule(name):
                if name == "cosine":
                    return DDPMSchedule.from_betas(cosine_betas(100), cumulative_fn=cumulative_alphas)
                end = {"linear": 0.02, "linear_strong": 0.1}[name]
                return DDPMSchedule.from_betas(torch.linspace(1e-4, end, 100), cumulative_fn=cumulative_alphas)


            def matched_step(schedule, level):
                return int((schedule.alpha_bar - level).abs().argmin())


            BASELINE = {"schedule": "cosine", "mask": True}
            TRIALS = []


            def run_trial(config):
                started = time.perf_counter()
                torch.manual_seed(SEED_WEIGHTS)
                schedule = make_schedule(config["schedule"])
                model = DiffusionDenoiser(ConditionalUnet1D(7, 16, (32, 64, 128), step_embed_dim=32), schedule, flat_state_condition)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.95, 0.999))
                draws = torch.Generator().manual_seed(SEED_BATCH)
                batches = []
                for _ in range(STEPS):
                    batch = take_batch(TRAIN, torch.randint(0, len(TRAIN["action"]), (BATCH,), generator=draws))
                    if not config["mask"]:
                        batch["action_is_pad"] = torch.zeros_like(batch["action_is_pad"])
                    batches.append(batch)
                train_loop(model, batches, optimizer, epsilon_loss_fn, steps=STEPS)
                model.eval()
                result = {}
                with torch.no_grad():
                    noise = torch.randn(VALID["action"].shape, generator=torch.Generator().manual_seed(SEED_NOISE))
                    real = ~VALID["action_is_pad"]
                    for level in (0.9, 0.5):
                        t = matched_step(schedule, level)
                        times = torch.full((len(noise),), t)
                        noisy = forward_noise(VALID["action"], noise, times, schedule.alpha_bar)
                        eps = model(noisy, times, VALID)
                        abar = schedule.alpha_bar[t]
                        x0 = (noisy - (1 - abar).sqrt() * eps) / abar.sqrt()
                        result[f"eps@{level}"] = (eps - noise).square().mean(-1)[real].mean().item()
                        result[f"x0@{level}"] = (x0 - VALID["action"]).square().mean(-1)[real].mean().item()
                result["final_alpha_bar"] = float(schedule.alpha_bar[-1])
                result["seconds"] = time.perf_counter() - started
                return result


            def record(config):
                config = {**BASELINE, **config}
                try:
                    TRIALS.append({"config": config, "status": "ok", **run_trial(config)})
                except Exception as error:  # a failed trial is evidence too
                    TRIALS.append({"config": config, "status": f"failed: {type(error).__name__}: {error}"})


            record(BASELINE)
            COLUMNS = ["eps@0.9", "eps@0.5", "x0@0.9", "x0@0.5", "final_alpha_bar", "seconds"]
            show_trials(TRIALS, COLUMNS);
        """),
        Md(
            "**Your trials.** Add at least two trials that each change one setting. Budget: 200 updates per trial."
        ),
        Pair(
            learner="""
            # Implement: choose your trials.
            MY_TRIALS = [
                # {"schedule": "linear"},
            ]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
            reference="""
            # Reference choice of trials: one change per trial.
            MY_TRIALS = [{"schedule": "linear"}, {"schedule": "linear_strong"}, {"mask": False}]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
        ),
        Code(r"""
            # Check: the protocol was followed.
            assert TRIALS[0]["config"] == BASELINE, "The first trial must be the baseline."
            assert len(TRIALS) >= 3, "Run the baseline and at least two trials of your own."
            assert all(set(t["config"]) == set(BASELINE) for t in TRIALS), "Only schedule and mask may change."
            assert all(t["config"]["schedule"] in {"cosine", "linear", "linear_strong"} for t in TRIALS)
            ok = [t for t in TRIALS if t["status"] == "ok"]
            assert all(torch.isfinite(torch.tensor([t[c] for c in COLUMNS])).all() for t in ok), "Measures must be finite."
            print(f"{len(TRIALS)} trials kept ({len(ok)} ok). Budget: {STEPS} updates, batch {BATCH}, seeds {SEED_WEIGHTS}/{SEED_BATCH}/{SEED_NOISE}.")
        """),
        Code(r"""
            # Use your function: plot every trial
            ok = [t for t in TRIALS if t["status"] == "ok"]
            labels = [", ".join(f"{k}={v}" for k, v in t["config"].items() if v != BASELINE[k]) or "baseline" for t in ok]
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
            for axis, column in zip(axes, ["eps@0.5", "x0@0.5", "final_alpha_bar"]):
                axis.bar(range(len(ok)), [t[column] for t in ok])
                axis.set_xticks(range(len(ok))); axis.set_xticklabels(labels, rotation=30, ha="right"); axis.set_title(column)
            plt.tight_layout(); plt.show()
        """),
        Md("""
        **Explain (2–4 sentences, edit this cell).** Which schedule had the lowest loss at matched noise, and is the comparison fair for the schedule that ends with `final_alpha_bar` far from 0? Did masking change the validation numbers, which count real steps only? What is one limit of using the state alone (no images) as the condition?

        *Optional extension:* run the cosine baseline with two more batch seeds and report the spread of `eps@0.5`; compare it with the gaps between schedules.
        """),
        Md("""
        ## Optional: your loss on the real prepared policy (opt-in)

        This loads the **prepared** diffusion checkpoint `diffusion-bowl-plate-reference-2000` on CPU, reads two real training windows with their camera frames, and runs your `epsilon_loss_fn` through the pinned network (image encoders + U-Net) for the same random numbers as the pinned loss. It then makes **two Adam updates** with your loss through the training loop, so you see that your callback updates the real weights. Two updates are an integration check, not training evidence; nothing is saved. The run uses local files and two training windows. Set `RUN_REAL_DIFFUSION_UPDATE = True` to run it.
        """),
        Code(r"""
            RUN_REAL_DIFFUSION_UPDATE = False

            if RUN_REAL_DIFFUSION_UPDATE:
                import copy
                from torch.utils.data import default_collate
                from vla_lab.teaching.act_model import open_libero_dataset
                from vla_lab.teaching.diffusion import load_prepared_diffusion
                from vla_lab.training.diffusion import FPS, diffusion_delta_timestamps

                checkpoint = DATA / "runs/diffusion-bowl-plate-reference-2000/checkpoint-002000"
                if not (checkpoint / "model.safetensors").is_file():
                    raise FileNotFoundError(f"Prepared diffusion checkpoint missing: {checkpoint}")
                DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
                policy, preprocessor, _ = load_prepared_diffusion(checkpoint, device=DEVICE)
                dataset = open_libero_dataset(CONFIGS, DATA, "bowl_plate", "train", diffusion_delta_timestamps(policy.config, FPS), episodes=[379, 422])
                batch = preprocessor(default_collate([dataset[i] for i in (3, 40)]))
                denoiser = LeRobotDenoiser(policy)
                policy.eval()
                torch.manual_seed(2); pinned_loss, _ = policy(dict(batch))
                torch.manual_seed(2); your_loss, _ = epsilon_loss_fn(denoiser, batch)
                print("pinned loss:", float(pinned_loss.detach()), "| your loss:", float(your_loss.detach()))
                model = LeRobotDenoiser(policy)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.95, 0.999))
                before = policy.diffusion.unet.final_conv[1].weight.detach().clone()
                history = train_loop(model, [batch], optimizer, epsilon_loss_fn, steps=2)
                print("two updates, losses:", [round(h["loss"], 4) for h in history])
                print("U-Net weights changed:", not torch.equal(before, policy.diffusion.unet.final_conv[1].weight))
            else:
                print("Real diffusion update is off. Set RUN_REAL_DIFFUSION_UPDATE=True to run the two-update integration check.")
        """),
        Md("""
        ## What you built

        - `cumulative_alphas`, `forward_noise`, `sample_noise_and_timesteps`, `masked_epsilon_loss` and `epsilon_loss_fn`: the full training objective, equal to the pinned one for equal random numbers.
        - A fair comparison across noise schedules (matched `ᾱ`) with every trial kept.

        **Next:** Unit 10A opens the denoiser: the timestep embedding and the FiLM-conditioned residual block.
        """),
    ]
    return L


def lesson_10a() -> Lesson:
    L = Lesson("10a-denoiser-blocks", "10a-denoiser-blocks", "Denoiser blocks")
    L.items += [
        Md(r"""
        # Unit 10A — Denoiser blocks: timestep embedding and FiLM residual block

        **Your result:** the two blocks that tell a denoiser *how noisy its input is* and *what the robot sees*. Your versions will match the pinned LeRobot diffusion-policy blocks, for outputs and gradients, including the weights of the prepared checkpoint.

        Two short exercises (about 15 minutes each) and a larger experiment (30–60 minutes). These times are estimates, not measurements. Full answers are in [the reference notebook](10a-denoiser-blocks-reference.ipynb).

        **Sources.** [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) (1D temporal U-Net with FiLM conditioning), FiLM: [Perez et al. 2018](https://arxiv.org/abs/1709.07871), and the timestep embedding from [DDPM](https://arxiv.org/abs/2006.11239) (appendix B, after Vaswani et al.). Pinned code: `DiffusionSinusoidalPosEmb`, `DiffusionConv1dBlock` and `DiffusionConditionalResidualBlock1d` in `lerobot/policies/diffusion/modeling_diffusion.py` (revision `e0d50211ef236143ae867228662b7dfaba554f02`).

        **Where this unit sits.** In Unit 9 the denoiser was supplied. Here you open it. Unit 10B assembles these blocks into the full U-Net. Your Unit 9 functions are available as package versions; this unit's functions are wired into the real network through hooks.
        """),
        setup_cell("""
            import math
            import time
            import torch
            from torch import nn
            import matplotlib.pyplot as plt

            import vla_lab.teaching.diffusion as ref  # package reference: used by checks only
            from vla_lab.teaching.diffusion import (  # Unit 9 functions, as package versions
                ConditionalResidualBlock1d, ConditionalUnet1D, Conv1dBlock, DDPMSchedule, DiffusionDenoiser,
                cosine_betas, epsilon_loss_fn, flat_state_condition, forward_noise,
                load_state_action_windows, take_batch,
            )
            from vla_lab.teaching.training import train_loop  # from the training unit (package version)
            from lerobot.policies.diffusion.modeling_diffusion import (
                DiffusionConditionalResidualBlock1d, DiffusionSinusoidalPosEmb,
            )

            torch.manual_seed(0)
            print("PyTorch:", torch.__version__)
        """),
        Code("# Supplied: a helper that the checks use\n" + tidy(CLOSE_HELPER)),
        Md("""
        ## Background: how a convolutional denoiser gets its conditions

        The denoiser is a 1D convolutional network over the *time axis of the action window* (16 steps). It receives two kinds of side information:

        1. **How noisy is the input?** The integer timestep `t` is turned into a smooth vector (`sinusoidal_timestep_embedding`) and passed through a small MLP.
        2. **What does the robot see?** The observation features (state, camera features) form one *global condition* vector.

        The two are concatenated into a `cond` vector. Every residual block turns `cond` into per-channel numbers that *modulate* the convolution features. This is **FiLM** (feature-wise linear modulation): `features = scale * features + bias`, with `scale` and `bias` computed from `cond`. In the pinned policy `use_film_scale_modulation=True`, so both scale and bias are used.
        """),
        Md("""
        **Real data used in this unit.** Real LIBERO `bowl_plate` windows (state history of 2 frames and 16 actions with padding mask), read from local files; no download starts here. The recording rate is 10 Hz and is not resampled to the 20 Hz controller. If files are missing, the cell tells you the setup command.
        """),
        Code(r"""
            # Supplied: real windows
            TRAIN = load_state_action_windows(CONFIGS, DATA, "bowl_plate", "train")
            VALID = load_state_action_windows(CONFIGS, DATA, "bowl_plate", "validation")
            print("train windows:", len(TRAIN["action"]), "| validation windows:", len(VALID["action"]))
        """),
    ]
    L.items.append(
        Exercise(
            key="10A.1",
            title="Sinusoidal timestep embedding",
            minutes="10–15",
            purpose="Give the network a smooth, bounded code for the noise level, instead of a raw integer.",
            contract="""
        - `sinusoidal_timestep_embedding(timesteps, dim)`: `timesteps (B,)` integer or float tensor; `dim` even and at least 4. Return `(B, dim)` `float32`.
        - `half = dim // 2`. Frequencies: `f_k = exp(-k * log(10000) / (half - 1))` for `k = 0 … half-1`.
        - Output row = `[sin(t * f_0) … sin(t * f_{half-1}), cos(t * f_0) … cos(t * f_{half-1})]`: **all sines first, then all cosines** (not interleaved).
        - Timestep 0 gives `sin = 0` and `cos = 1` everywhere.
        """,
            method="The Transformer position code applied to the diffusion timestep ([DDPM](https://arxiv.org/abs/2006.11239), appendix B; [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762), 3.5). The layout differs from Exercise 7.4 (that one interleaves sin and cos and counts frequencies differently).",
            pointers="Make the frequency vector first. Multiply the column of timesteps with the frequency row to get `(B, half)` angles. Concatenate the sines and the cosines on the last axis.",
            hint="""
        ```python
        half = dim // 2
        scale = math.log(10000) / (half - 1)
        frequencies = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
        angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)
        ```
        """,
            names=["sinusoidal_timestep_embedding"],
            module="vla_lab.teaching.diffusion",
            stub="""
        def sinusoidal_timestep_embedding(timesteps, dim):
            # (B,) -> (B, dim): [sin(t f_0) ... sin(t f_{h-1}), cos(t f_0) ... cos(t f_{h-1})], f_k = exp(-k log(10000)/(h-1)).
            raise NotImplementedError("Complete Exercise 10A.1: sinusoidal_timestep_embedding")
        """,
            check="""
        t = torch.tensor([0, 1, 7, 99])
        for dim in (16, 32, 128):
            mine = sinusoidal_timestep_embedding(t, dim)
            assert mine.shape == (4, dim) and mine.dtype == torch.float32, f"Expected float32 (4, {dim})."
            close(mine, DiffusionSinusoidalPosEmb(dim)(t), f"dim {dim} differs from the pinned embedding", atol=1e-6)
        zero = sinusoidal_timestep_embedding(torch.tensor([0]), 8)
        assert torch.equal(zero[0, :4], torch.zeros(4)) and torch.equal(zero[0, 4:], torch.ones(4)), "Timestep 0 is all sines 0 then all cosines 1."
        close(sinusoidal_timestep_embedding(torch.tensor([5]), 8)[0, 0], torch.tensor(math.sin(5.0)), "first frequency is 1", atol=1e-6)
        close(sinusoidal_timestep_embedding(torch.tensor([5]), 8)[0, 3], torch.tensor(math.sin(5 * 1e-4)), "last frequency is 1/10000", atol=1e-6)
        print("Exercise 10A.1 passed")
        """,
            use_intro="Embed all 100 training timesteps. Neighbouring steps should look similar and far steps different.",
            use="""
        embedding = sinusoidal_timestep_embedding(torch.arange(100), 128)
        similarity = embedding @ embedding.T
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        axes[0].imshow(embedding.T, aspect="auto", cmap="coolwarm"); axes[0].set_xlabel("timestep"); axes[0].set_ylabel("feature")
        axes[1].imshow(similarity, cmap="viridis"); axes[1].set_title("similarity between timesteps")
        plt.tight_layout(); plt.show()
        """,
        )
    )
    L.items.append(
        Exercise(
            key="10A.2",
            title="FiLM-conditioned residual block",
            minutes="15–20",
            purpose="Write the block that is repeated through the whole U-Net: two convolutions, with the condition modulating the features between them.",
            contract="""
        - `block` has `conv1`, `conv2` (`Conv1dBlock`: Conv1d → GroupNorm → Mish, keeps the length), `cond_encoder` (`Mish → Linear(cond_dim, K)`), `residual_conv` (`Conv1d(in, out, 1)` or `Identity`), `out_channels` (`C`) and the bool `use_film_scale_modulation`.
        - `x (B, C_in, T)` (channels first), `cond (B, cond_dim)`. Return `(B, C, T)`.
        - `out = conv1(x)`. `embed = cond_encoder(cond)` reshaped to `(B, K, 1)` so it broadcasts over `T`.
        - **With scale modulation** (`K = 2C`): `scale = embed[:, :C]`, `bias = embed[:, C:]`, `out = scale * out + bias`. **Without** (`K = C`): `out = out + embed`.
        - `out = conv2(out)`. Return `out + residual_conv(x)`.
        - The pinned code uses `scale * out + bias`, not `(1 + scale) * out + bias`.
        """,
            method="FiLM: `γ(c) ⊙ F + β(c)` — [Perez et al. 2018](https://arxiv.org/abs/1709.07871). Diffusion Policy uses it in each 1D residual block ([paper site](https://diffusion-policy.cs.columbia.edu/)). Pinned class: `DiffusionConditionalResidualBlock1d`.",
            pointers="Follow the order of the contract line by line. The only decision is how to cut `embed` into `scale` and `bias`. The residual connection adds the *input* (after `residual_conv`), not the output of `conv1`.",
            hint="""
        ```python
        out = block.conv1(x)
        embed = block.cond_encoder(cond).unsqueeze(-1)              # (B, K, 1)
        if block.use_film_scale_modulation:
            scale, bias = embed[:, : block.out_channels], embed[:, block.out_channels :]
            out = scale * out + bias
        else:
            out = out + embed
        out = block.conv2(out)
        return out + block.residual_conv(x)
        ```
        """,
            names=["film_residual_forward"],
            module="vla_lab.teaching.diffusion",
            stub="""
        def film_residual_forward(block, x, cond):
            # conv1 -> FiLM modulation from cond -> conv2 -> add residual_conv(x).
            raise NotImplementedError("Complete Exercise 10A.2: film_residual_forward")
        """,
            check="""
        for scale in (True, False):
            torch.manual_seed(2)
            oracle = DiffusionConditionalResidualBlock1d(8, 16, 10, kernel_size=5, n_groups=8, use_film_scale_modulation=scale)
            mine = ConditionalResidualBlock1d(8, 16, 10, 5, 8, scale, forward_fn=film_residual_forward)
            mine.load_state_dict(oracle.state_dict(), strict=True)       # same layer names: identical weights
            x, cond = torch.randn(3, 8, 16, requires_grad=True), torch.randn(3, 10, requires_grad=True)
            out = mine(x, cond)
            assert out.shape == (3, 16, 16), "Expected (B, out_channels, T)."
            close(out, oracle(x, cond), f"output (scale modulation={scale}) differs from the pinned block")
            out.square().sum().backward()
            x2, cond2 = (t.detach().clone().requires_grad_(True) for t in (x, cond))
            oracle(x2, cond2).square().sum().backward()
            close(x.grad, x2.grad, f"input gradient (scale={scale})", atol=1e-5)
            close(cond.grad, cond2.grad, f"condition gradient (scale={scale})", atol=1e-5)
            close(mine.cond_encoder[1].weight.grad, oracle.cond_encoder[1].weight.grad, "FiLM layer gradient", atol=1e-5)
            assert not torch.allclose(mine(x, cond + 1.0), out.detach(), atol=1e-4), "The condition must change the output."
        same = ConditionalResidualBlock1d(16, 16, 10, 5, 8, True, forward_fn=film_residual_forward)
        assert isinstance(same.residual_conv, nn.Identity) and same(torch.randn(2, 16, 16), torch.randn(2, 10)).shape == (2, 16, 16), "Also check equal channels."
        print("Exercise 10A.2 passed")
        """,
            use_intro="Condition one block on a real state history and a timestep. Change only the timestep and see how much of the output moves.",
            use="""
        torch.manual_seed(1)
        block = ConditionalResidualBlock1d(7, 16, 32 + 16, 5, 8, True, forward_fn=film_residual_forward).eval()
        window = VALID["action"][:4].transpose(1, 2)                             # (4, 7 channels, 16 steps): real action windows
        state = VALID["observation.state"][:4].flatten(1)                        # (4, 16): real robot states
        def condition(t):
            return torch.cat([sinusoidal_timestep_embedding(torch.full((4,), t), 32), state], dim=-1)
        with torch.no_grad():
            base = block(window, condition(10))
            for t in (11, 50, 90):
                print(f"change timestep 10 -> {t}: relative output change", round(float((block(window, condition(t)) - base).norm() / base.norm()), 3))
            other = block(window, torch.cat([sinusoidal_timestep_embedding(torch.full((4,), 10), 32), state.flip(0)], dim=-1))
            print("same timestep, states of other windows: relative change", round(float((other - base).norm() / base.norm()), 3))
        """,
        )
    )
    L.items += [
        Md("""
        ### Prepared weights in your block (real-size parity)

        The prepared diffusion checkpoint's first U-Net block is copied into a block that runs **your** `film_residual_forward`. The layer names match, so the copy is a plain strict `load_state_dict`. Both blocks receive the same real-sized random input. The check reads the local checkpoint only (no download). If the file is missing, the cell says so.
        """),
        Code(r"""
            RUN_PREPARED_BLOCK_PARITY = True

            if RUN_PREPARED_BLOCK_PARITY:
                from safetensors.torch import load_file

                checkpoint = DATA / "runs/diffusion-bowl-plate-reference-2000/checkpoint-002000"
                if not (checkpoint / "model.safetensors").is_file():
                    print("Prepared diffusion checkpoint not found; skipping. It is part of the reference bundle (outside this notebook).")
                else:
                    tensors = load_file(checkpoint / "model.safetensors")
                    prefix = "diffusion.unet.down_modules.0.0."
                    state = {key[len(prefix):]: value for key, value in tensors.items() if key.startswith(prefix)}
                    channels_in, channels_out = state["conv1.block.0.weight"].shape[1], state["conv1.block.0.weight"].shape[0]
                    cond_dim = state["cond_encoder.1.weight"].shape[1]
                    prepared = ConditionalResidualBlock1d(channels_in, channels_out, cond_dim, 5, 8, True, forward_fn=film_residual_forward).eval()
                    prepared.load_state_dict(state, strict=True)
                    oracle = DiffusionConditionalResidualBlock1d(channels_in, channels_out, cond_dim, kernel_size=5, n_groups=8, use_film_scale_modulation=True).eval()
                    oracle.load_state_dict(state, strict=True)
                    x, cond = torch.randn(4, channels_in, 16), torch.randn(4, cond_dim)
                    with torch.no_grad():
                        print(f"block {channels_in}->{channels_out} channels, cond {cond_dim}: largest output difference", float((prepared(x, cond) - oracle(x, cond)).abs().max()))
            else:
                print("Prepared-block parity is off.")
        """),
        Md(r"""
        ## Experiment — how should the denoiser use the observation? (30–60 min)

        **Question.** Does a small denoiser trained briefly on real windows (a) benefit from FiLM *scale* in addition to *bias*, and (b) use the robot state at all?

        **Setup (real data).** A 1D U-Net (`down_dims = (32, 64, 128)`) built from **your** `sinusoidal_timestep_embedding` and `film_residual_forward`, trained with the Unit 9 loss callback on real windows. The condition is the flattened 2×8 state history, or nothing.

        **Fixed for every trial.** Cosine schedule, `T = 100`; weight seed `0`; batch seed `1`; **200 updates**, batch 64, Adam(`lr=1e-3`, betas `(0.95, 0.999)`); the same validation windows and noise (seed `7`) for every trial.

        **Allowed changes (per trial).** `film_scale` (`True` = scale and bias, `False` = bias only) and `cond` (`"state"` or `"none"`).

        **Measures.** `eps_grid`: noise-prediction MSE on validation windows (real steps only), averaged over timesteps `10, 30, 50, 70, 90` with fixed noise. `eps_shuffled`: the same with the states of *other* windows (for `cond="state"` only). `condition_gain = eps_shuffled − eps_grid`: how much the model relies on its own observation. Also the parameter count.

        **Prediction first.** In the next cell: will `cond="state"` beat `cond="none"`? Will scale modulation help?

        **Keep failed trials.** Every trial is appended to `TRIALS`. No improvement is a valid result.
        """),
        Code('PREDICTION = "TODO: replace this text with your prediction."\nprint(PREDICTION)'),
        Code("# Supplied: trial table helper\n" + tidy(EXPERIMENT_TABLE_HELPER)),
        Code(r"""
            # Supplied: harness. Your embedding and your FiLM block are in the real U-Net.
            STEPS, BATCH, LR, SEED_WEIGHTS, SEED_BATCH, SEED_NOISE = 200, 64, 1e-3, 0, 1, 7
            SCHEDULE = DDPMSchedule.cosine(100)
            GRID = (10, 30, 50, 70, 90)
            BASELINE = {"film_scale": True, "cond": "state"}
            TRIALS = []


            def run_trial(config):
                started = time.perf_counter()
                torch.manual_seed(SEED_WEIGHTS)
                use_state = config["cond"] == "state"
                unet = ConditionalUnet1D(
                    7, 16 if use_state else 0, (32, 64, 128), step_embed_dim=32,
                    use_film_scale_modulation=config["film_scale"],
                    block_forward_fn=film_residual_forward, embedding_fn=sinusoidal_timestep_embedding,
                )
                model = DiffusionDenoiser(unet, SCHEDULE, flat_state_condition if use_state else (lambda batch: None))
                optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.95, 0.999))
                draws = torch.Generator().manual_seed(SEED_BATCH)
                batches = [take_batch(TRAIN, torch.randint(0, len(TRAIN["action"]), (BATCH,), generator=draws)) for _ in range(STEPS)]
                train_loop(model, batches, optimizer, epsilon_loss_fn, steps=STEPS)
                model.eval()
                with torch.no_grad():
                    noise = torch.randn(VALID["action"].shape, generator=torch.Generator().manual_seed(SEED_NOISE))
                    real = ~VALID["action_is_pad"]
                    other = {**VALID, "observation.state": VALID["observation.state"][torch.randperm(len(noise), generator=torch.Generator().manual_seed(3))]}
                    own, shuffled = [], []
                    for t in GRID:
                        times = torch.full((len(noise),), t)
                        noisy = forward_noise(VALID["action"], noise, times, SCHEDULE.alpha_bar)
                        own.append((model(noisy, times, VALID) - noise).square().mean(-1)[real].mean().item())
                        if use_state:
                            shuffled.append((model(noisy, times, other) - noise).square().mean(-1)[real].mean().item())
                result = {"eps_grid": sum(own) / len(own), "parameters": sum(p.numel() for p in model.parameters())}
                result["eps_shuffled"] = sum(shuffled) / len(shuffled) if use_state else float("nan")
                result["condition_gain"] = result["eps_shuffled"] - result["eps_grid"] if use_state else float("nan")
                result["seconds"] = time.perf_counter() - started
                return result


            def record(config):
                config = {**BASELINE, **config}
                try:
                    TRIALS.append({"config": config, "status": "ok", **run_trial(config)})
                except Exception as error:  # a failed trial is evidence too
                    TRIALS.append({"config": config, "status": f"failed: {type(error).__name__}: {error}"})


            record(BASELINE)
            COLUMNS = ["eps_grid", "eps_shuffled", "condition_gain", "parameters", "seconds"]
            show_trials(TRIALS, COLUMNS);
        """),
        Md(
            "**Your trials.** Add at least two trials that each change one setting. Budget: 200 updates per trial. `condition_gain` is `nan` when there is no condition."
        ),
        Pair(
            learner="""
            # Implement: choose your trials.
            MY_TRIALS = [
                # {"film_scale": False},
            ]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
            reference="""
            # Reference choice of trials: one change per trial.
            MY_TRIALS = [{"film_scale": False}, {"cond": "none"}]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
        ),
        Code(r"""
            # Check: the protocol was followed.
            assert TRIALS[0]["config"] == BASELINE, "The first trial must be the baseline."
            assert len(TRIALS) >= 3, "Run the baseline and at least two trials of your own."
            assert all(set(t["config"]) == set(BASELINE) for t in TRIALS), "Only film_scale and cond may change."
            assert all(t["config"]["cond"] in {"state", "none"} for t in TRIALS)
            ok = [t for t in TRIALS if t["status"] == "ok"]
            assert all(torch.isfinite(torch.tensor(t["eps_grid"])) for t in ok), "eps_grid must be finite."
            print(f"{len(TRIALS)} trials kept ({len(ok)} ok). Budget: {STEPS} updates, batch {BATCH}, seeds {SEED_WEIGHTS}/{SEED_BATCH}/{SEED_NOISE}.")
        """),
        Code(r"""
            # Use your function: plot every trial
            ok = [t for t in TRIALS if t["status"] == "ok"]
            labels = [", ".join(f"{k}={v}" for k, v in t["config"].items() if v != BASELINE[k]) or "baseline" for t in ok]
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            for axis, column in zip(axes, ["eps_grid", "condition_gain"]):
                axis.bar(range(len(ok)), [t[column] for t in ok])
                axis.set_xticks(range(len(ok))); axis.set_xticklabels(labels, rotation=30, ha="right"); axis.set_title(column)
            plt.tight_layout(); plt.show()
        """),
        Md("""
        **Explain (2–4 sentences, edit this cell).** Which trial had the lowest `eps_grid`? Was a denoiser with no observation allowed to be better on this measure — and what does that say about judging a policy by noise-prediction loss alone? Does `condition_gain` show that the state is used? What would you add as a condition (Unit 10B and 11 use camera features too)?

        *Optional extension:* repeat the baseline with another `SEED_BATCH` to see the spread, and keep all rows.
        """),
        Md("""
        ## What you built

        - `sinusoidal_timestep_embedding` and `film_residual_forward`, equal to the pinned blocks in output and gradient, also with the weights of the prepared checkpoint.
        - Real-data evidence about conditioning, with every trial kept.

        **Next:** Unit 10B stacks these blocks into the full conditional U-Net: temporal down/up-sampling, skip connections, and a check against the pinned network.
        """),
    ]
    return L


UNET_FIXTURE = r'''
IMAGES = ("observation.images.image", "observation.images.image2")


def pinned_unet(down_dims=(16, 32, 64), cond_dim=6, seed=4):
    """A *real* pinned DiffusionConditionalUnet1d with random weights (7 actions, horizon 16)."""
    torch.manual_seed(seed)
    config = DiffusionConfig(
        input_features={
            IMAGES[0]: PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            IMAGES[1]: PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        n_obs_steps=2, horizon=16, n_action_steps=8, drop_n_last_frames=7, down_dims=tuple(down_dims),
        diffusion_step_embed_dim=32, pretrained_backbone_weights=None, push_to_hub=False, device="cpu",
    )
    unet = DiffusionConditionalUnet1d(config, global_cond_dim=cond_dim)
    for parameter in unet.parameters():          # distinct biases and scales make wrong wiring visible
        if parameter.dim() == 1:
            nn.init.normal_(parameter, std=0.1)
    return unet.eval()


PINNED = pinned_unet()
X, T_STEPS, COND = torch.randn(3, 16, 7), torch.tensor([0, 10, 99]), torch.randn(3, 6)    # algorithm-check inputs (random numbers)
print("pinned U-Net parameters:", sum(p.numel() for p in PINNED.parameters()))
'''


def lesson_10b() -> Lesson:
    L = Lesson("10b-unet", "10b-unet", "The conditional U-Net")
    L.items += [
        Md(r"""
        # Unit 10B — The conditional 1D U-Net

        **Your result:** the full denoiser network used by Diffusion Policy, assembled from the Unit 10A blocks. Fed the weights of the pinned LeRobot network it gives the same outputs, for the small test network and for the prepared checkpoint. You then change its architecture under a fixed budget and compare.

        Three short exercises (about 10–20 minutes each) and a larger experiment (30–60 minutes). These times are estimates, not measurements. Full answers are in [the reference notebook](10b-unet-reference.ipynb).

        **Sources.** [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) (temporal U-Net for actions), the U-Net idea in [DDPM](https://arxiv.org/abs/2006.11239) (appendix B), and the pinned `DiffusionConditionalUnet1d` in `lerobot/policies/diffusion/modeling_diffusion.py` (around line 654; revision `e0d50211ef236143ae867228662b7dfaba554f02`). The pinned model has **no EMA** in the current trainer; the weights you load are the raw trained weights.

        **Where this unit sits.** Units 9 and 10A are available as package versions. Your functions here (`unet_channel_plan`, `unet_forward`, the weight converter) run inside the real network through its hooks.
        """),
        setup_cell("""
            import time
            import torch
            from torch import nn
            import matplotlib.pyplot as plt

            import vla_lab.teaching.diffusion as ref  # package reference: used by checks only
            from vla_lab.teaching.diffusion import (  # earlier units, as package versions
                ConditionalUnet1D, DDPMSchedule, DiffusionDenoiser, epsilon_loss_fn, film_residual_forward,
                flat_state_condition, forward_noise, load_prepared_diffusion, load_state_action_windows,
                LeRobotDenoiser, sinusoidal_timestep_embedding, take_batch, unet_from_lerobot, verify_unet_parity,
            )
            from vla_lab.teaching.act_model import open_libero_dataset
            from vla_lab.teaching.training import train_loop
            from lerobot.configs import FeatureType, PolicyFeature
            from lerobot.policies.diffusion import DiffusionConfig
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d

            torch.manual_seed(0)
            print("PyTorch:", torch.__version__)
        """),
        Code("# Supplied: a helper that the checks use\n" + tidy(CLOSE_HELPER)),
        Md("""
        ## Background: a U-Net over time

        The input is an action window `(B, 16, 7)`. The U-Net treats the 16 steps as a 1D signal with 7 channels.

        ```
        encoder:   blocks (7→d0)(d0→d0)  ─skip→  ↓2   blocks (d0→d1)(d1→d1) ─skip→  ↓2   blocks (d1→d2)(d2→d2) ─skip→
        middle:    two blocks at width d2
        decoder:   [x ‖ skip] (2·d2→d1)(d1→d1) ↑2    [x ‖ skip] (2·d1→d0)(d0→d0) ↑2     head: conv → conv(→7)
        ```

        - Going *down*, the sequence is halved by a stride-2 convolution (all stages except the last) and the channel width grows.
        - Each stage keeps a **skip feature**. Going *up*, the newest skip is concatenated on the channel axis, so the decoder blocks take twice the channels.
        - Every block is conditioned on `cond = [timestep MLP output, global observation condition]`.
        - The decoder has one stage fewer than the encoder (`n − 1` for `n` encoder stages). It upsamples after *every* decoder stage, including the last, with a transposed convolution. The sequence length falls by `2^(n-1)` in the encoder and rises by `2^(n-1)` in the decoder, so it returns to 16.

        **Supplied:** `Conv1dBlock`, the residual block with your Unit 10A function, the module class `ConditionalUnet1D` (it builds itself from your plan function and runs your forward function), and the data reader. **You write** the channel plan, the forward pass with skip connections, and the weight converter.
        """),
        Code(
            "# Supplied: a small real pinned U-Net and algorithm-check inputs\n"
            + tidy(UNET_FIXTURE)
        ),
    ]
    L.items.append(
        Exercise(
            key="10B.1",
            title="The channel plan",
            minutes="15",
            purpose="Decide which block takes how many channels, where the sequence is downsampled, and what the decoder receives after the skip connection is concatenated. A wrong plan builds a network that cannot load the real weights.",
            contract="""
        - `unet_channel_plan(action_dim, down_dims)` returns a dict with the keys `"down"`, `"mid"`, `"up"`.
        - Let `pairs = [(action_dim, down_dims[0]), (down_dims[0], down_dims[1]), …, (down_dims[-2], down_dims[-1])]`.
        - `"down"`: one entry per pair, `(in_channels, out_channels, downsample)`. Each stage has two residual blocks (`in → out`, `out → out`). `downsample` is `True` for every stage **except the last**.
        - `"mid"`: `[(c, c), (c, c)]` with `c = down_dims[-1]`.
        - `"up"`: one entry per pair **in reverse order, leaving out the first pair** `(action_dim, down_dims[0])`. For a reversed pair `(width_out, width_in)` (that is `(d_{k-1}, d_k)` read as `(out, in)`) the entry is `(2 * width_in, width_out, True)`: the first block takes the decoder state plus the concatenated skip, hence `2 * width_in`. The pinned network upsamples after every `"up"` stage, so `upsample` is `True` for all of them.
        """,
            method="Read the constructor of the pinned `DiffusionConditionalUnet1d`: the lists `in_out`, `down_modules`, `mid_modules`, `up_modules`. The check builds the real network and compares every channel number and every stride.",
            pointers="First build `pairs`. Then make `down` from `pairs` with a flag for the last one. Make `up` from `reversed(pairs[1:])`. Print your plan for `(8, 16, 32)`-style inputs and compare with the picture above.",
            hint="""
        ```python
        pairs = [(action_dim, down_dims[0]), *zip(down_dims[:-1], down_dims[1:])]
        down = [(i, o, index < len(pairs) - 1) for index, (i, o) in enumerate(pairs)]
        mid = [(down_dims[-1], down_dims[-1])] * 2
        up = [(width_in * 2, width_out, True) for width_out, width_in in reversed(pairs[1:])]
        return {"down": down, "mid": mid, "up": up}
        ```
        For `down_dims = (16, 32, 64)` and `action_dim = 7`: `pairs = [(7, 16), (16, 32), (32, 64)]`; `up` is `[(128, 32, True), (64, 16, True)]`.
        """,
            names=["unet_channel_plan"],
            module="vla_lab.teaching.diffusion",
            stub="""
        def unet_channel_plan(action_dim, down_dims):
            # Return {"down": [(in, out, downsample)...], "mid": [(c, c), (c, c)], "up": [(in, out, upsample)...]}.
            raise NotImplementedError("Complete Exercise 10B.1: unet_channel_plan")
        """,
            check="""
        for dims in ((16, 32), (16, 32, 64), (8, 16, 24, 32)):
            pinned = pinned_unet(dims)
            plan = unet_channel_plan(7, dims)
            assert set(plan) == {"down", "mid", "up"}, "Return the keys down, mid and up."
            assert len(plan["down"]) == len(pinned.down_modules) and len(plan["up"]) == len(pinned.up_modules) and len(plan["mid"]) == len(pinned.mid_modules), f"Wrong number of stages for {dims}."
            for (i, o, down), (res1, res2, downsample) in zip(plan["down"], pinned.down_modules, strict=True):
                assert (res1.conv1.block[0].in_channels, res1.conv1.block[0].out_channels) == (i, o), f"down stage channels for {dims}: expected {(res1.conv1.block[0].in_channels, res1.conv1.block[0].out_channels)}, got {(i, o)}"
                assert res2.conv2.block[0].out_channels == o
                assert isinstance(downsample, nn.Conv1d) == bool(down), f"downsample flag for {dims}"
            for (c_in, c_out), block in zip(plan["mid"], pinned.mid_modules, strict=True):
                assert (block.conv1.block[0].in_channels, block.conv1.block[0].out_channels) == (c_in, c_out)
            for (i, o, up), (res1, res2, upsample) in zip(plan["up"], pinned.up_modules, strict=True):
                assert (res1.conv1.block[0].in_channels, res1.conv1.block[0].out_channels) == (i, o), f"up stage channels for {dims}: expected {(res1.conv1.block[0].in_channels, res1.conv1.block[0].out_channels)}, got {(i, o)}"
                assert isinstance(upsample, nn.ConvTranspose1d) == bool(up), f"upsample flag for {dims}"
        print("Exercise 10B.1 passed")
        """,
            use_intro="Build a U-Net from your plan and look at the sequence length and channel width along the way.",
            use="""
        unet = ConditionalUnet1D(7, 6, (16, 32, 64), step_embed_dim=32, plan_fn=unet_channel_plan)
        print("plan:", unet_channel_plan(7, (16, 32, 64)))
        print("parameters:", sum(p.numel() for p in unet.parameters()), "| pinned network with the same sizes:", sum(p.numel() for p in PINNED.parameters()))
        x = torch.randn(1, 16, 7)
        length = 16
        for res1, res2, downsample in unet.down_blocks:
            length = length // 2 if isinstance(downsample, nn.Conv1d) else length
            print("after encoder stage: sequence length", length)
        """,
        )
    )
    L.items.append(
        Exercise(
            key="10B.2",
            title="The forward pass with skip connections",
            minutes="15–20",
            purpose="Run the encoder, remember each stage's output, run the middle, then the decoder with the skips concatenated in reverse order.",
            contract="""
        - `unet` has `time_mlp` (timestep → vector), `down_blocks`, `mid_blocks`, `up_blocks`, `head`. `down_blocks` and `up_blocks` hold triples `(res1, res2, resample)`; `res1(x, cond)` and `res2(x, cond)` are residual blocks and `resample(x)` is a downsampling or upsampling module (or `Identity`).
        - `x (B, H, A)` (**time before channels**), `timestep (B,)` long, `global_cond (B, G)` or `None`. Return `(B, H, A)`.
        - Move to channels first: `(B, A, H)`. `cond = unet.time_mlp(timestep)`; if `global_cond` is not `None`, concatenate it on the last axis: `cond = [time vector, global_cond]`.
        - Encoder: for each triple apply `res1`, then `res2`, **save that output as a skip feature**, then apply `resample`.
        - Middle: apply each block of `mid_blocks` to `(x, cond)`.
        - Decoder: for each triple, first concatenate `x` with the **most recently saved skip** on the channel axis (`dim=1`), then `res1`, `res2`, then `resample`. Skips are used in reverse order of saving.
        - End with `head(x)` and move back to `(B, H, A)`.
        """,
            method="U-Net with skip connections ([Ronneberger et al. 2015](https://arxiv.org/abs/1505.04597), used for denoising in [DDPM](https://arxiv.org/abs/2006.11239) appendix B). Pinned code: `DiffusionConditionalUnet1d.forward`.",
            pointers="Use a Python list as a stack: `append` after the second residual block of each encoder stage, `pop` in the decoder. Use `torch.cat((x, skip), dim=1)` for channels-first tensors.",
            hint="""
        ```python
        x = x.transpose(1, 2)                                   # (B, A, H)
        cond = unet.time_mlp(timestep)
        if global_cond is not None:
            cond = torch.cat([cond, global_cond], dim=-1)
        skips = []
        for res1, res2, downsample in unet.down_blocks:
            x = res2(res1(x, cond), cond)
            skips.append(x)
            x = downsample(x)
        for block in unet.mid_blocks:
            x = block(x, cond)
        for res1, res2, upsample in unet.up_blocks:
            x = torch.cat((x, skips.pop()), dim=1)
            x = res2(res1(x, cond), cond)
            x = upsample(x)
        return unet.head(x).transpose(1, 2)
        ```
        """,
            names=["unet_forward"],
            module="vla_lab.teaching.diffusion",
            stub="""
        def unet_forward(unet, x, timestep, global_cond=None):
            # Encoder with skip features, middle blocks, decoder with concatenated skips, head.
            raise NotImplementedError("Complete Exercise 10B.2: unet_forward")
        """,
            check="""
        # The learner network has your forward function and the package plan; weights come from the pinned network.
        learner = unet_from_lerobot(PINNED, forward_fn=unet_forward, plan_fn=ref.unet_channel_plan)
        out = learner(X, T_STEPS, COND)
        assert out.shape == X.shape, "Return (B, H, A)."
        close(out, PINNED(X, T_STEPS, COND), "output differs from the pinned U-Net")
        x1 = X.clone().requires_grad_(True); learner(x1, T_STEPS, COND).square().sum().backward()
        x2 = X.clone().requires_grad_(True); PINNED(x2, T_STEPS, COND).square().sum().backward()
        close(x1.grad, x2.grad, "input gradient", atol=1e-4)
        close(learner.head[1].weight.grad, PINNED.final_conv[1].weight.grad, "head gradient", atol=1e-4)
        close(learner.down_blocks[0][0].conv1.block[0].weight.grad, PINNED.down_modules[0][0].conv1.block[0].weight.grad, "first-block gradient (flows through the skip)", atol=1e-4)
        close(learner(X, T_STEPS, COND), out, "repeatable")
        no_cond = unet_from_lerobot(pinned_unet((16, 32), cond_dim=0), forward_fn=unet_forward, plan_fn=ref.unet_channel_plan)
        close(no_cond(X, T_STEPS, None), pinned_unet((16, 32), cond_dim=0)(X, T_STEPS, None), "network without a global condition", atol=1e-5)
        print("Exercise 10B.2 passed")
        """,
            use_intro="Run your plan and your forward pass together in one network, with weights copied from the pinned one, and compare outputs.",
            use="""
        learner = unet_from_lerobot(PINNED, forward_fn=unet_forward, plan_fn=unet_channel_plan)
        print("largest output difference from the pinned U-Net:", verify_unet_parity(learner, PINNED, X, T_STEPS, COND))
        """,
        )
    )
    L.items.append(
        Exercise(
            key="10B.3",
            title="The weight converter",
            minutes="10",
            purpose="Move trained weights from the pinned network into your modules *explicitly*, so that a changed network cannot load by accident.",
            contract="""
        - `convert_lerobot_unet_state_dict(source)`: `source` maps tensor names to tensors (a LeRobot `state_dict`). Return a **new dict** for `ConditionalUnet1D`.
        - Only the first name part changes: `diffusion_step_encoder → time_mlp`, `down_modules → down_blocks`, `mid_modules → mid_blocks`, `up_modules → up_blocks`, `final_conv → head`. Everything after the first dot is unchanged (for example `down_modules.0.0.conv1.block.0.weight` becomes `down_blocks.0.0.conv1.block.0.weight`).
        - Every source tensor appears exactly once in the result (copy them).
        - A name whose first part is not in the list must raise `KeyError`. Do not skip it silently.
        """,
            method="An explicit rename table with a loud failure is the safe way to move weights between two implementations; `load_state_dict(..., strict=True)` then catches missing or extra tensors on the other side.",
            pointers="Split each name at the first dot with `partition('.')`. Look the first part up in a dict. Raise when it is missing.",
            hint="""
        ```python
        renames = {"diffusion_step_encoder": "time_mlp", "down_modules": "down_blocks", "mid_modules": "mid_blocks",
                   "up_modules": "up_blocks", "final_conv": "head"}
        converted = {}
        for name, value in source.items():
            top, _, rest = name.partition(".")
            if top not in renames:
                raise KeyError(f"no conversion rule for LeRobot U-Net tensor {name!r}")
            converted[f"{renames[top]}.{rest}"] = value.detach().clone()
        return converted
        ```
        """,
            names=["convert_lerobot_unet_state_dict"],
            module="vla_lab.teaching.diffusion",
            stub="""
        def convert_lerobot_unet_state_dict(source):
            # Rename the first part of every key; raise KeyError for an unknown first part.
            raise NotImplementedError("Complete Exercise 10B.3: convert_lerobot_unet_state_dict")
        """,
            check="""
        converted = convert_lerobot_unet_state_dict(PINNED.state_dict())
        assert len(converted) == len(PINNED.state_dict()), "Every source tensor must appear exactly once."
        fresh = ConditionalUnet1D(7, 6, (16, 32, 64), step_embed_dim=32)
        assert set(converted) == set(fresh.state_dict()), f"Names differ from the learner module: {sorted(set(converted) ^ set(fresh.state_dict()))[:4]}"
        for name, tensor in converted.items():
            assert tensor.shape == fresh.state_dict()[name].shape, f"shape of {name}"
        fresh.load_state_dict(converted, strict=True)
        close(fresh.eval()(X, T_STEPS, COND), PINNED(X, T_STEPS, COND), "converted weights must reproduce the pinned output")
        try:
            convert_lerobot_unet_state_dict({"mystery.weight": torch.zeros(1)})
        except KeyError:
            pass
        else:
            raise AssertionError("An unknown tensor name must raise KeyError.")
        source_key = next(iter(PINNED.state_dict()))
        assert converted[convert_lerobot_unet_state_dict({source_key: torch.zeros(1)}).popitem()[0]] is not PINNED.state_dict()[source_key], "Copy the tensors."
        print("Exercise 10B.3 passed")
        """,
            use_intro="Load the *prepared* checkpoint's U-Net into your modules through your converter, your plan and your forward function. Check numeric parity on real camera frames before using the network anywhere.",
            use="""
        RUN_PREPARED_UNET_PARITY = True

        if RUN_PREPARED_UNET_PARITY:
            import copy
            from torch.utils.data import default_collate
            from vla_lab.teaching.act_model import open_libero_dataset
            from vla_lab.teaching.diffusion import load_prepared_diffusion, stack_observation
            from vla_lab.training.diffusion import FPS, diffusion_delta_timestamps

            checkpoint = DATA / "runs/diffusion-bowl-plate-reference-2000/checkpoint-002000"
            if not (checkpoint / "model.safetensors").is_file():
                print("Prepared diffusion checkpoint not found; skipping this check.")
            else:
                policy, preprocessor, _ = load_prepared_diffusion(checkpoint, device="cpu")
                prepared_unet = policy.diffusion.unet
                yours = unet_from_lerobot(prepared_unet, converter=convert_lerobot_unet_state_dict, plan_fn=unet_channel_plan,
                                          forward_fn=unet_forward, block_forward_fn=film_residual_forward, embedding_fn=sinusoidal_timestep_embedding)
                dataset = open_libero_dataset(CONFIGS, DATA, "bowl_plate", "validation", diffusion_delta_timestamps(policy.config, FPS))
                batch = preprocessor(default_collate([dataset[i] for i in (5, 60)]))
                with torch.no_grad():
                    cond = policy.diffusion._prepare_global_conditioning(stack_observation(policy, batch))   # real frames -> condition
                    noisy = torch.randn(2, 16, 7, generator=torch.Generator().manual_seed(0))
                    difference = verify_unet_parity(yours, prepared_unet, noisy, torch.tensor([7, 77]), cond)
                print("real-size U-Net, real camera condition: largest output difference", difference)
                assert difference < 1e-4, "Reference weights in your U-Net must reproduce the pinned network before any experiment."
        else:
            print("Prepared-U-Net parity is off.")
        """,
        )
    )
    L.items += [
        Md("""
        ## Opt-in: update your U-Net inside a real policy

        This opt-in check loads the prepared diffusion policy and copies its U-Net weights into your U-Net. It checks real-condition parity, then runs two updates with the Unit 9 `epsilon_loss_fn` through the Unit 9 training loop. The image encoder stays frozen. Only local files are used. The update budget is two; this is an integration check, not a training result. Set `RUN_REAL_UNET_UPDATE = True` to run it.
        """),
        Code(r"""
            RUN_REAL_UNET_UPDATE = False
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            REAL_UPDATES = 2

            if RUN_REAL_UNET_UPDATE:
                from torch.utils.data import default_collate
                from vla_lab.training.diffusion import FPS, diffusion_delta_timestamps

                checkpoint = DATA / "runs/diffusion-bowl-plate-reference-2000/checkpoint-002000"
                if not (checkpoint / "model.safetensors").is_file():
                    raise FileNotFoundError(f"Prepared diffusion checkpoint missing: {checkpoint}")
                policy, preprocessor, _ = load_prepared_diffusion(checkpoint, device=DEVICE)
                policy.eval()
                for parameter in policy.parameters():
                    parameter.requires_grad_(False)
                pinned_unet = policy.diffusion.unet
                learner_unet = unet_from_lerobot(
                    pinned_unet, converter=convert_lerobot_unet_state_dict, plan_fn=unet_channel_plan,
                    forward_fn=unet_forward, block_forward_fn=film_residual_forward,
                    embedding_fn=sinusoidal_timestep_embedding,
                ).to(DEVICE)
                dataset = open_libero_dataset(CONFIGS, DATA, "bowl_plate", "train", diffusion_delta_timestamps(policy.config, FPS), episodes=[379, 422])
                batch = preprocessor(default_collate([dataset[i] for i in (3, 40)]))
                class FrozenPolicyDenoiser(LeRobotDenoiser):
                    def train(self, mode=True):
                        super().train(mode)
                        self.policy.eval()
                        self.unet.train(mode)
                        return self

                denoiser = FrozenPolicyDenoiser(policy, unet=learner_unet)
                noisy = torch.randn(2, policy.config.horizon, policy.config.action_feature.shape[0], device=DEVICE,
                                    generator=torch.Generator(device=DEVICE).manual_seed(3))
                timestep = torch.tensor([7, 77], dtype=torch.long, device=DEVICE)
                with torch.no_grad():
                    condition = denoiser.global_condition(batch)
                    difference = verify_unet_parity(learner_unet, pinned_unet, noisy, timestep, condition)
                print("real-condition parity difference:", difference)
                assert difference < 1e-4, "Your U-Net must match the pinned network before training."
                optimizer = torch.optim.AdamW(learner_unet.parameters(), lr=1e-5)
                before = learner_unet.head[1].weight.detach().clone()
                torch.manual_seed(4)
                history = train_loop(denoiser, [batch], optimizer, epsilon_loss_fn, steps=REAL_UPDATES)
                changed = not torch.equal(before, learner_unet.head[1].weight)
                print("updates:", [round(row["loss"], 5) for row in history], "| learner U-Net changed:", changed)
                assert changed, "The real training path must update your U-Net weights."
                assert all(parameter.grad is None for parameter in policy.parameters()), "The pinned policy weights must stay frozen."
            else:
                print("Real U-Net updates are off. Set RUN_REAL_UNET_UPDATE=True to run the two-update check.")
        """),
    ]
    L.items += [
        Md(r"""
        ## Experiment — change the architecture under a fixed budget (30–60 min)

        **Question.** With the same number of updates, how do depth and kernel size of the U-Net change the held-out noise-prediction error and the cost?

        **Gate first.** The harness refuses to run any trial unless your U-Net (your plan, forward pass and converter) reproduces the pinned network to `1e-4` on the algorithm-check inputs. An experiment on a network you have not checked is not an experiment.

        **Setup (real data).** Real windows as in Unit 9 (state-conditioned, no images), cosine schedule, your Unit 10 functions in the real network, the Unit 9 loss callback in `train_loop`.

        **Fixed for every trial.** Weight seed `0`; batch seed `1`; **200 updates**, batch 64, Adam(`lr=1e-3`, betas `(0.95, 0.999)`); the same validation windows and noise (seed `7`).

        **Allowed changes (per trial).** `dims`: `shallow` = `(32, 64)`, `base` = `(32, 64, 128)`, `deep` = `(16, 32, 64, 128)`. `kernel`: `3` or `5`. (The horizon 16 is divisible by `2^4`, so four stages are valid.)

        **Measures.** `eps_grid`: noise-prediction MSE on validation windows (real steps), averaged over timesteps `10, 30, 50, 70, 90`, fixed noise. `parameters`. `ms_per_update`: wall-clock milliseconds per training update on this CPU (noisy; compare only within one session).

        **Prediction first.** In the next cell: which architecture will have the lowest `eps_grid`, and will it be the biggest one?

        **Keep failed trials.** Every trial is appended to `TRIALS`; an architecture that is too deep for the horizon is a failed trial worth keeping.
        """),
        Code('PREDICTION = "TODO: replace this text with your prediction."\nprint(PREDICTION)'),
        Code("# Supplied: trial table helper\n" + tidy(EXPERIMENT_TABLE_HELPER)),
        Code(r"""
            # Supplied: the parity gate and the real windows
            gate = unet_from_lerobot(PINNED, converter=convert_lerobot_unet_state_dict, plan_fn=unet_channel_plan,
                                     forward_fn=unet_forward, block_forward_fn=film_residual_forward, embedding_fn=sinusoidal_timestep_embedding)
            gate_difference = verify_unet_parity(gate, PINNED, X, T_STEPS, COND)
            assert gate_difference < 1e-4, f"Your U-Net differs from the pinned network by {gate_difference:.1e}; fix the exercises before the experiment."
            print("parity gate passed:", gate_difference)

            TRAIN = load_state_action_windows(CONFIGS, DATA, "bowl_plate", "train")
            VALID = load_state_action_windows(CONFIGS, DATA, "bowl_plate", "validation")
        """),
        Code(r"""
            # Supplied: harness. Your plan, forward pass and Unit 10A blocks build and run the network.
            STEPS, BATCH, LR, SEED_WEIGHTS, SEED_BATCH, SEED_NOISE = 200, 64, 1e-3, 0, 1, 7
            SCHEDULE, GRID = DDPMSchedule.cosine(100), (10, 30, 50, 70, 90)
            DIMS = {"shallow": (32, 64), "base": (32, 64, 128), "deep": (16, 32, 64, 128)}
            BASELINE = {"dims": "base", "kernel": 5}
            TRIALS = []


            def run_trial(config):
                torch.manual_seed(SEED_WEIGHTS)
                unet = ConditionalUnet1D(
                    7, 16, DIMS[config["dims"]], kernel_size=config["kernel"], step_embed_dim=32,
                    plan_fn=unet_channel_plan, forward_fn=unet_forward,
                    block_forward_fn=film_residual_forward, embedding_fn=sinusoidal_timestep_embedding,
                )
                model = DiffusionDenoiser(unet, SCHEDULE, flat_state_condition)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.95, 0.999))
                draws = torch.Generator().manual_seed(SEED_BATCH)
                batches = [take_batch(TRAIN, torch.randint(0, len(TRAIN["action"]), (BATCH,), generator=draws)) for _ in range(STEPS)]
                started = time.perf_counter()
                train_loop(model, batches, optimizer, epsilon_loss_fn, steps=STEPS)
                seconds = time.perf_counter() - started
                model.eval()
                with torch.no_grad():
                    noise = torch.randn(VALID["action"].shape, generator=torch.Generator().manual_seed(SEED_NOISE))
                    real = ~VALID["action_is_pad"]
                    own = []
                    for t in GRID:
                        times = torch.full((len(noise),), t)
                        noisy = forward_noise(VALID["action"], noise, times, SCHEDULE.alpha_bar)
                        own.append((model(noisy, times, VALID) - noise).square().mean(-1)[real].mean().item())
                return {"eps_grid": sum(own) / len(own), "parameters": sum(p.numel() for p in model.parameters()),
                        "ms_per_update": 1000 * seconds / STEPS}


            def record(config):
                config = {**BASELINE, **config}
                try:
                    TRIALS.append({"config": config, "status": "ok", **run_trial(config)})
                except Exception as error:  # a failed trial is evidence too
                    TRIALS.append({"config": config, "status": f"failed: {type(error).__name__}: {error}"})


            record(BASELINE)
            COLUMNS = ["eps_grid", "parameters", "ms_per_update"]
            show_trials(TRIALS, COLUMNS);
        """),
        Md(
            "**Your trials.** Add at least two trials that each change one setting. Budget: 200 updates per trial."
        ),
        Pair(
            learner="""
            # Implement: choose your trials.
            MY_TRIALS = [
                # {"dims": "shallow"},
            ]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
            reference="""
            # Reference choice of trials: one change per trial.
            MY_TRIALS = [{"dims": "shallow"}, {"dims": "deep"}, {"kernel": 3}]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
        ),
        Code(r"""
            # Check: the protocol was followed.
            assert TRIALS[0]["config"] == BASELINE, "The first trial must be the baseline."
            assert len(TRIALS) >= 3, "Run the baseline and at least two trials of your own."
            assert all(set(t["config"]) == set(BASELINE) for t in TRIALS), "Only dims and kernel may change."
            assert all(t["config"]["dims"] in DIMS and t["config"]["kernel"] in (3, 5) for t in TRIALS)
            ok = [t for t in TRIALS if t["status"] == "ok"]
            assert all(torch.isfinite(torch.tensor(t["eps_grid"])) for t in ok), "eps_grid must be finite."
            print(f"{len(TRIALS)} trials kept ({len(ok)} ok). Budget: {STEPS} updates, batch {BATCH}, seeds {SEED_WEIGHTS}/{SEED_BATCH}/{SEED_NOISE}.")
        """),
        Code(r"""
            # Use your function: quality against cost, every trial shown
            ok = [t for t in TRIALS if t["status"] == "ok"]
            labels = [", ".join(f"{k}={v}" for k, v in t["config"].items() if v != BASELINE[k]) or "baseline" for t in ok]
            fig, axis = plt.subplots(figsize=(5.5, 4))
            axis.scatter([t["parameters"] for t in ok], [t["eps_grid"] for t in ok])
            for t, label in zip(ok, labels):
                axis.annotate(label, (t["parameters"], t["eps_grid"]), fontsize=8)
            axis.set_xlabel("parameters"); axis.set_ylabel("validation eps_grid"); axis.grid(alpha=0.3); plt.show()
        """),
        Md("""
        **Explain (2–4 sentences, edit this cell).** Which architecture was best at this budget, and is the largest network the best one? Compare quality with parameters and time. What do the failed trials, if any, tell you about the 16-step horizon? Why can you not conclude that a deeper network is better for the *robot*?

        *Optional extension:* train the baseline longer (change `STEPS` for every trial and keep the old rows) to see whether the ranking changes.
        """),
        Md("""
        ## What you built

        - `unet_channel_plan`, `unet_forward` and `convert_lerobot_unet_state_dict`: a conditional U-Net that loads the pinned weights and reproduces the pinned outputs on random inputs and on real camera conditions.
        - A fixed-budget architecture comparison gated by numeric parity, with every trial kept.

        **Next:** Unit 11 runs the network backwards: DDPM sampling, skipped timesteps, and the actions that are actually executed.
        """),
    ]
    return L


def lesson_11() -> Lesson:
    L = Lesson("11-diffusion-sampling", "11-diffusion-sampling", "Diffusion sampling")
    L.items += [
        Md(r"""
        # Unit 11 — Diffusion sampling: reverse steps, skipped timesteps, executed actions

        **Your result:** the code that turns noise into robot actions. Your reverse step equals the `diffusers` DDPM step at every timestep (including the final step with no noise and the jumps of a shortened schedule). Your sampler then produces the actions of a *real* policy object that you can run in the simulator.

        Five short exercises (10–20 minutes each) and a larger experiment (30–60 minutes). These times are estimates, not measurements. Full answers are in [the reference notebook](11-diffusion-sampling-reference.ipynb).

        **Sources.** [DDPM](https://arxiv.org/abs/2006.11239): equation 7 (posterior mean and variance), Algorithm 2 (sampling), equation 15 (the clean-sample estimate). [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) for which actions are executed. Pinned code (revision `e0d50211ef236143ae867228662b7dfaba554f02`): `DiffusionModel.conditional_sample`, `DiffusionModel.generate_actions` and `DiffusionPolicy.select_action`; the numerical step lives in `diffusers.DDPMScheduler.step`, which this unit uses **only as an oracle in checks**, never in your answers.

        **Where this unit sits.** Units 9–10 are available as package versions (the forward process, the trained denoiser classes). The five functions of this unit are wired into a real policy wrapper through hooks; the wrapper never calls `scheduler.step`.
        """),
        setup_cell("""
            import math
            import time
            import torch
            from torch import nn
            import matplotlib.pyplot as plt

            import vla_lab.teaching.diffusion as ref  # package reference: used by checks only
            from vla_lab.teaching.diffusion import (  # earlier units, as package versions
                CallbackDiffusionPolicy, ConditionalUnet1D, DDPMSchedule, DiffusionDenoiser, LeRobotDenoiser,
                cosine_betas, cumulative_alphas, epsilon_loss_fn, flat_state_condition, forward_noise,
                load_prepared_diffusion, load_state_action_windows, make_step_noises, take_batch,
            )
            from vla_lab.teaching.training import masked_action_l1, train_loop
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

            torch.manual_seed(0)
            print("PyTorch:", torch.__version__)
        """),
        Code("# Supplied: a helper that the checks use\n" + tidy(CLOSE_HELPER)),
        Md("""
        ## Background: running the process backwards

        Training learned `ε_θ(x_t, t)`. Sampling starts from pure noise `x_T ~ N(0, I)` and repeats one **reverse step** until `x_0`:

        1. Estimate the clean window: `x̂_0 = (x_t − √(1−ᾱ_t) · ε_θ) / √ᾱ_t`, then **clip** it to `[-1, 1]` (the pinned recipe has `clip_sample=True`; actions are scaled to `[-1, 1]`).
        2. Take the posterior mean `μ = c₀ · x̂_0 + c_t · x_t`, with coefficients from the *adjacent* or *skipped* step.
        3. Add Gaussian noise with the posterior variance, **except at the final step**.

        **Shortened schedules.** The pinned policy trains with `T = 100` steps but samples with `n = 20` steps: `t = 95, 90, …, 5, 0`. A jump from `t` to `prev < t − 1` has no single `β`. Use the ratio `α_t = ᾱ_t / ᾱ_prev` (the product of the skipped alphas) and `β_t = 1 − α_t`.

        **Which actions are executed.** The network predicts a window of 16 actions that starts *one frame before* the current observation (`n_obs_steps = 2`). The policy executes indices `n_obs_steps − 1 … n_obs_steps − 1 + n_action_steps − 1` = `1 … 8`, then samples again.
        """),
    ]
    L.items.append(
        Exercise(
            key="11.1",
            title="Posterior coefficients for adjacent and skipped steps",
            minutes="20",
            purpose="Derive the weights of the reverse step. This is where a shortened schedule differs from the textbook formula: the one-step `β_t` no longer applies.",
            contract="""
        - `posterior_coefficients(abar_t, abar_prev)` takes the cumulative alphas at the current step and at the step we jump to (python floats or 0-d tensors). `abar_prev = 1` means "the clean sample" (used for the final step).
        - Compute `alpha_t = abar_t / abar_prev` and `beta_t = 1 - alpha_t`.
        - Return three 0-d `float32` tensors `(coef_x0, coef_xt, variance)`:
          `coef_x0 = sqrt(abar_prev) * beta_t / (1 - abar_t)`;
          `coef_xt = sqrt(alpha_t) * (1 - abar_prev) / (1 - abar_t)`;
          `variance = (1 - abar_prev) / (1 - abar_t) * beta_t`, clamped to at least `1e-20`.
        - For `abar_prev = 1` the result is `coef_x0 = 1`, `coef_xt = 0` (the mean is the clean estimate).
        """,
            method="Posterior `q(x_prev | x_t, x_0) = N(μ̃, β̃ I)` with `μ̃ = (√ᾱ_prev β_t / (1−ᾱ_t)) x_0 + (√α_t (1−ᾱ_prev) / (1−ᾱ_t)) x_t` — [DDPM](https://arxiv.org/abs/2006.11239), equation 7. Skipped steps: apply the same equation to the coarser chain whose one-step alpha is `ᾱ_t / ᾱ_prev`. Why this works: the marginals `q(x_t | x_0)` are the same Gaussians whatever steps we skip, so the chain over the chosen steps has per-jump alpha `ᾱ_t / ᾱ_prev`.",
            pointers="Write the ratio and `beta_t` first. Then the three formulas in the order of the contract. Use `torch.as_tensor(..., dtype=torch.float32)` so floats and tensors both work. Derive for yourself why `alpha_t * abar_prev = abar_t`.",
            hint="""
        ```python
        abar_t = torch.as_tensor(abar_t, dtype=torch.float32)
        abar_prev = torch.as_tensor(abar_prev, dtype=torch.float32)
        alpha_t = abar_t / abar_prev
        beta_t = 1.0 - alpha_t
        coef_x0 = abar_prev.sqrt() * beta_t / (1.0 - abar_t)
        coef_xt = alpha_t.sqrt() * (1.0 - abar_prev) / (1.0 - abar_t)
        variance = ((1.0 - abar_prev) / (1.0 - abar_t) * beta_t).clamp_min(1e-20)
        return coef_x0, coef_xt, variance
        ```
        """,
            names=["posterior_coefficients"],
            module="vla_lab.teaching.diffusion",
            stub="""
        def posterior_coefficients(abar_t, abar_prev):
            # Return (coef_x0, coef_xt, variance) for a jump from step t to an earlier step with cumulative alpha abar_prev.
            raise NotImplementedError("Complete Exercise 11.1: posterior_coefficients")
        """,
            check="""
        scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
        abar, betas = scheduler.alphas_cumprod, scheduler.betas
        for t, prev in ((50, 49), (50, 45), (99, 94), (5, 0)):
            c0, ct, var = posterior_coefficients(abar[t], abar[prev])
            assert c0.shape == () and ct.shape == () and var.shape == (), "Return three scalar tensors."
            product = torch.prod(1 - betas[prev + 1 : t + 1])               # the skipped alphas multiplied
            close(abar[t] / abar[prev], product, f"alpha_t for the jump {t} -> {prev}", atol=1e-6)
            beta = 1 - product
            close(c0, abar[prev].sqrt() * beta / (1 - abar[t]), f"coef_x0 for {t} -> {prev}", atol=1e-6)
            close(ct, product.sqrt() * (1 - abar[prev]) / (1 - abar[t]), f"coef_xt for {t} -> {prev}", atol=1e-6)
            close(var, (1 - abar[prev]) / (1 - abar[t]) * beta, f"variance for {t} -> {prev}", atol=1e-7)
        c0, ct, var = posterior_coefficients(abar[0], 1.0)
        close(c0, torch.tensor(1.0), "final step: coef_x0 is 1", atol=1e-6)
        close(ct, torch.tensor(0.0), "final step: coef_xt is 0", atol=1e-6)
        assert 0 < float(var) <= 1e-19, "Clamp the variance to at least 1e-20 (it is 0 at the final step)."
        c0b, _, _ = posterior_coefficients(float(abar[50]), float(abar[40]))
        close(c0b, posterior_coefficients(abar[50], abar[40])[0], "python floats must work too", atol=1e-7)
        print("Exercise 11.1 passed")
        """,
            use_intro="Plot how the weights change along a 20-step schedule and a 100-step schedule.",
            use="""
        abar = cumulative_alphas(cosine_betas(100))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        for steps, style in ((100, "-"), (20, "o-")):
            ts = ref.inference_timesteps(100, steps)
            rows = [posterior_coefficients(abar[t], abar[ts[i + 1]] if i + 1 < len(ts) else 1.0) for i, t in enumerate(ts)]
            for axis, index, name in zip(axes, range(3), ("coef_x0 (weight on the clean estimate)", "coef_xt (weight on x_t)", "std of the added noise")):
                values = [float(r[index]) for r in rows]
                axis.plot(ts, [v ** 0.5 if index == 2 else v for v in values], style, label=f"{steps} steps"); axis.set_title(name, fontsize=9); axis.set_xlabel("t")
        axes[0].legend(); plt.tight_layout(); plt.show()
        """,
        )
    )
    L.items.append(
        Exercise(
            key="11.2",
            title="One reverse step (with the final step)",
            minutes="20",
            purpose="Write the single update that the sampler repeats. The final step is special: there is no noise and the result is the clipped clean estimate.",
            contract="""
        - `ddpm_reverse_step(x_t, epsilon, t, prev_t, alpha_bar, noise=None, clip_range=1.0)`.
        - `x_t`, `epsilon`, optional `noise`: `(B, H, A)`. `t`, `prev_t`: python ints; `prev_t < 0` means "the final step" (after it there is no earlier step). `alpha_bar`: `(T,)` cumulative alphas.
        - `x0_hat = (x_t - sqrt(1 - abar_t) * epsilon) / sqrt(abar_t)`, then clamp to `[-clip_range, clip_range]` if `clip_range` is not `None`.
        - `abar_prev = alpha_bar[prev_t]`, or exactly `1` when `prev_t < 0`. Get coefficients from your Exercise 11.1 function. `mean = coef_x0 * x0_hat + coef_xt * x_t`.
        - If `prev_t >= 0`: return `mean + sqrt(variance) * noise`; raise `ValueError` if `noise` is `None`. If `prev_t < 0`: return `mean` and **never** use `noise` (it may be `None` or any value).
        - Use only the passed tensors. **Do not call a scheduler.**
        """,
            method="DDPM Algorithm 2, line 4: `x_{t-1} = μ̃(x_t, x̂_0) + σ_t z`, with `z = 0` at the last step; `x̂_0` from equation 15 and clipping as in the reference implementation. This is what `DDPMScheduler.step` computes with `prediction_type='epsilon'`.",
            pointers="Estimate `x0_hat`, clip, fetch `abar_prev` (watch the final-step convention), call `posterior_coefficients`, combine. Draw no random numbers inside this function: the caller passes the noise.",
            hint="""
        ```python
        abar_t = alpha_bar[t].to(x_t.dtype)
        abar_prev = alpha_bar[prev_t].to(x_t.dtype) if prev_t >= 0 else torch.ones((), dtype=x_t.dtype)
        x0_hat = (x_t - (1.0 - abar_t).sqrt() * epsilon) / abar_t.sqrt()
        if clip_range is not None:
            x0_hat = x0_hat.clamp(-clip_range, clip_range)
        coef_x0, coef_xt, variance = posterior_coefficients(abar_t, abar_prev)
        mean = coef_x0.to(x_t.dtype) * x0_hat + coef_xt.to(x_t.dtype) * x_t
        if prev_t < 0:
            return mean
        if noise is None:
            raise ValueError("noise is required for every step except the final one")
        return mean + variance.sqrt().to(x_t.dtype) * noise
        ```
        """,
            names=["ddpm_reverse_step"],
            module="vla_lab.teaching.diffusion",
            stub="""
        def ddpm_reverse_step(x_t, epsilon, t, prev_t, alpha_bar, noise=None, clip_range=1.0):
            # x0_hat (clipped) -> posterior mean -> add sqrt(variance) * noise unless this is the final step.
            raise NotImplementedError("Complete Exercise 11.2: ddpm_reverse_step")
        """,
            check="""
        for steps in (5, 20, 100):
            scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, clip_sample_range=1.0)
            scheduler.set_timesteps(steps)
            times = scheduler.timesteps.tolist()
            abar = scheduler.alphas_cumprod
            g = torch.Generator().manual_seed(0)
            for index, t in enumerate(times):
                prev = times[index + 1] if index + 1 < len(times) else -1
                x = torch.randn(2, 16, 7, generator=g) * 1.5
                eps = torch.randn(2, 16, 7, generator=g)
                noise = torch.randn(2, 16, 7, generator=torch.Generator().manual_seed(100 + index))
                expected = scheduler.step(eps, t, x, generator=torch.Generator().manual_seed(100 + index)).prev_sample   # oracle only
                mine = ddpm_reverse_step(x, eps, t, prev, abar, noise if prev >= 0 else None, 1.0)
                close(mine, expected, f"{steps}-step schedule, t={t} -> prev={prev} differs from diffusers", atol=1e-5)
                if prev < 0:
                    x0 = ((x - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()).clamp(-1, 1)
                    close(mine, x0, "the final step returns the clipped clean estimate", atol=1e-5)
                    close(ddpm_reverse_step(x, eps, t, prev, abar, torch.full_like(x, 99.0), 1.0), mine, "the final step must ignore any noise", atol=0)
        abar = DDPMSchedule.cosine(100).alpha_bar
        try:
            ddpm_reverse_step(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), 10, 5, abar, None)
        except ValueError:
            pass
        else:
            raise AssertionError("A non-final step without noise must raise ValueError.")
        unclipped = ddpm_reverse_step(torch.full((1, 2, 2), 3.0), torch.zeros(1, 2, 2), 0, -1, abar, None, clip_range=None)
        clipped = ddpm_reverse_step(torch.full((1, 2, 2), 3.0), torch.zeros(1, 2, 2), 0, -1, abar, None, clip_range=1.0)
        assert float(unclipped.max()) > 1.5 and float(clipped.max()) <= 1.0 + 1e-6, "clip_range=None must not clamp; 1.0 must."
        print("Exercise 11.2 passed")
        """,
            use_intro="Check the noise your step adds: with a fixed `x_t` and a fixed `ε`, many different noises should spread the result by exactly `sqrt(variance)`.",
            use="""
        abar = cumulative_alphas(cosine_betas(100))
        x_t = torch.zeros(4000, 1, 1)
        eps = torch.zeros(4000, 1, 1)
        noise = torch.randn(4000, 1, 1, generator=torch.Generator().manual_seed(0))
        samples = ddpm_reverse_step(x_t, eps, 50, 45, abar, noise)
        expected_std = float(posterior_coefficients(abar[50], abar[45])[2].sqrt())
        print("empirical spread:", float(samples.std()), "| predicted sqrt(variance):", expected_std)
        """,
        )
    )
    L.items.append(
        Exercise(
            key="11.3",
            title="Timesteps of a shortened schedule",
            minutes="10",
            purpose="Choose which of the 100 training steps the sampler visits. Fewer steps are faster; the choice must keep the last step at 0.",
            contract="""
        - `inference_timesteps(num_train_timesteps, num_inference_steps)` returns a Python `list` of ints in **descending** order.
        - With `ratio = num_train_timesteps // num_inference_steps` the list is `[(n-1)*ratio, …, 2*ratio, ratio, 0]` (length `n`). For `T=100, n=20` that is `[95, 90, …, 5, 0]`.
        - The last entry is always `0`. Raise `ValueError` if `n < 1` or `n > T`.
        """,
            method="Evenly spaced ('leading') timesteps as in `diffusers` `DDPMScheduler.set_timesteps`; the pinned policy samples 20 of its 100 steps.",
            pointers="Make the ascending multiples of the ratio, then reverse them. Validate the inputs first.",
            hint="""
        ```python
        if not 1 <= num_inference_steps <= num_train_timesteps:
            raise ValueError("num_inference_steps must be between 1 and the training step count")
        ratio = num_train_timesteps // num_inference_steps
        return [index * ratio for index in reversed(range(num_inference_steps))]
        ```
        """,
            names=["inference_timesteps"],
            module="vla_lab.teaching.diffusion",
            stub="""
        def inference_timesteps(num_train_timesteps, num_inference_steps):
            # Descending evenly spaced training timesteps ending at 0.
            raise NotImplementedError("Complete Exercise 11.3: inference_timesteps")
        """,
            check="""
        for steps in (1, 5, 10, 20, 50, 100):
            scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
            scheduler.set_timesteps(steps)
            mine = inference_timesteps(100, steps)
            assert isinstance(mine, list) and mine == scheduler.timesteps.tolist(), f"{steps} steps: expected {scheduler.timesteps.tolist()[:4]}…, got {mine[:4]}…"
            assert mine[-1] == 0 and mine == sorted(mine, reverse=True), "Descending, ending at 0."
        assert inference_timesteps(100, 20)[:3] == [95, 90, 85]
        for bad in (0, 101):
            try:
                inference_timesteps(100, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{bad} steps must raise ValueError.")
        print("Exercise 11.3 passed")
        """,
            use_intro="How much signal does each step budget leave at its first step? The first timestep decides how far the sampler starts from pure noise.",
            use="""
        abar = cumulative_alphas(cosine_betas(100))
        for steps in (2, 5, 10, 20, 100):
            ts = inference_timesteps(100, steps)
            print(f"{steps:>3} steps: first t = {ts[0]:>2}, alpha_bar there = {float(abar[ts[0]]):.4f}, visited: {ts[:4]}…")
        """,
        )
    )
    L.items.append(
        Exercise(
            key="11.4",
            title="The reverse loop with explicit noise",
            minutes="20",
            purpose="Run the process from noise to actions. All randomness is passed in, so the same inputs always give the same actions.",
            contract="""
        - `sample_actions(denoise, initial_noise, schedule, num_inference_steps, step_noises, clip_range=1.0)`.
        - `denoise(x_t, t_batch) -> epsilon` where `t_batch` is a **`long` tensor `(B,)` filled with the current timestep** (same device as `x_t`). `initial_noise (B, H, A)`. `schedule` has `.alpha_bar` and `.num_train_timesteps`.
        - `step_noises` is a sequence of `(B, H, A)` tensors: `step_noises[i]` is the noise for step `i`. There must be exactly `num_inference_steps - 1` of them (the final step uses none); otherwise raise `ValueError`.
        - Use `inference_timesteps` (Exercise 11.3). Start from `x = initial_noise`. For each timestep: call `denoise(x, t_batch)`, then `x = ddpm_reverse_step(...)` (Exercise 11.2). **Carry the new `x` into the next step.** The previous timestep of the last entry is `-1`.
        - Return the final `x`. Do not call a scheduler and do not draw random numbers.
        """,
            method="DDPM Algorithm 2 with the shortened timesteps of Exercise 11.3. The pinned `conditional_sample` is the same loop with `noise_scheduler.step` inside.",
            pointers="Compute the list of timesteps once. Loop with `enumerate` so that you know the next timestep. Build `t_batch` with `torch.full((batch,), t, dtype=torch.long, device=x.device)`.",
            hint="""
        ```python
        timesteps = inference_timesteps(schedule.num_train_timesteps, num_inference_steps)
        if len(step_noises) != len(timesteps) - 1:
            raise ValueError("need one noise tensor for every step except the last")
        x = initial_noise
        for index, t in enumerate(timesteps):
            prev_t = timesteps[index + 1] if index + 1 < len(timesteps) else -1
            t_batch = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
            epsilon = denoise(x, t_batch)
            noise = step_noises[index] if prev_t >= 0 else None
            x = ddpm_reverse_step(x, epsilon, t, prev_t, schedule.alpha_bar, noise, clip_range)
        return x
        ```
        """,
            names=["sample_actions"],
            module="vla_lab.teaching.diffusion",
            stub="""
        def sample_actions(denoise, initial_noise, schedule, num_inference_steps, step_noises, clip_range=1.0):
            # For each timestep: predict epsilon with denoise, take a reverse step, carry x to the next step.
            raise NotImplementedError("Complete Exercise 11.4: sample_actions")
        """,
            check="""
        schedule = DDPMSchedule.cosine(100)
        torch.manual_seed(8)
        net = nn.Linear(7, 7)
        toy = lambda x, t: net(x) * (t.float().reshape(-1, 1, 1) / 100.0)

        # (1) Same noise as the diffusers loop gives the same actions.
        for steps in (3, 20, 100):
            scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, clip_sample_range=1.0)
            scheduler.set_timesteps(steps)
            g = torch.Generator().manual_seed(1)
            initial = torch.randn(2, 16, 7, generator=g)
            expected = initial
            for t in scheduler.timesteps:                                       # oracle loop (uses scheduler.step)
                expected = scheduler.step(toy(expected, torch.full((2,), int(t))), int(t), expected, generator=g).prev_sample
            g2 = torch.Generator().manual_seed(1)
            initial2 = torch.randn(2, 16, 7, generator=g2)
            noises = make_step_noises(steps, initial2.shape, g2)
            close(sample_actions(toy, initial2, schedule, steps, noises, 1.0), expected, f"{steps} steps differ from the diffusers loop", atol=1e-4)

        # (2) Behaviour: a perfect denoiser recovers the clean window exactly, whatever the noise.
        clean = torch.rand(2, 16, 7) * 1.6 - 0.8
        def oracle(x, t):
            ab = schedule.alpha_bar[t].reshape(-1, 1, 1)
            return (x - ab.sqrt() * clean) / (1 - ab).sqrt()
        seen = []
        def spy(x, t):
            seen.append((x.clone(), t.clone()))
            return oracle(x, t)
        start = torch.randn(2, 16, 7, generator=torch.Generator().manual_seed(2))
        out = sample_actions(spy, start, schedule, 10, make_step_noises(10, start.shape, torch.Generator().manual_seed(3)))
        close(out, clean, "a perfect denoiser must recover the clean window", atol=1e-4)
        assert [int(t[0]) for _, t in seen] == [90, 80, 70, 60, 50, 40, 30, 20, 10, 0], "Visit inference_timesteps(100, 10) in order."
        assert all(t.dtype == torch.long and t.shape == (2,) for _, t in seen), "t_batch must be a long tensor of shape (B,)."
        assert torch.equal(seen[0][0], start) and not torch.equal(seen[1][0], start), "Carry each new sample into the next step."
        # (3) Repeatable and strict about the noise count.
        noises = make_step_noises(10, start.shape, torch.Generator().manual_seed(3))
        assert torch.equal(sample_actions(toy, start, schedule, 10, noises), sample_actions(toy, start, schedule, 10, noises))
        try:
            sample_actions(toy, start, schedule, 10, noises[:-1])
        except ValueError:
            pass
        else:
            raise AssertionError("A wrong number of step noises must raise ValueError.")
        print("Exercise 11.4 passed")
        """,
            use_intro="Sample from a *perfect* denoiser built from a real action window (it knows the answer), for several step budgets. The sampler must return the real window each time; with a learned denoiser (the experiment) it only approximates it.",
            use="""
        VALID_REAL = load_state_action_windows(CONFIGS, DATA, "bowl_plate", "validation")
        truth = VALID_REAL["action"][100:104]
        schedule = DDPMSchedule.cosine(100)
        def perfect(x, t):
            ab = schedule.alpha_bar[t].reshape(-1, 1, 1)
            return (x - ab.sqrt() * truth) / (1 - ab).sqrt()
        for steps in (1, 5, 20, 100):
            start = torch.randn(truth.shape, generator=torch.Generator().manual_seed(0))
            out = sample_actions(perfect, start, schedule, steps, make_step_noises(steps, start.shape, torch.Generator().manual_seed(1)))
            print(f"{steps:>3} steps: largest error to the real window = {float((out - truth).abs().max()):.1e}")
        """,
        )
    )
    L.items.append(
        Exercise(
            key="11.5",
            title="Which actions are executed",
            minutes="10",
            purpose="The network predicts 16 actions, but the robot executes only 8 of them, starting from the one that belongs to the *current* observation. Off by one here means acting on the past.",
            contract="""
        - `executed_action_slice(chunk, n_obs_steps, n_action_steps)`: `chunk (B, H, A)` is the sampled window; it starts `n_obs_steps - 1` steps before the current observation.
        - Return `chunk[:, start:end]` with `start = n_obs_steps - 1` and `end = start + n_action_steps`.
        - For `n_obs_steps=2`, `n_action_steps=8` the returned indices are `1, 2, …, 8` (index 8 included, 9 excluded). The result has `n_action_steps` rows: `(B, n_action_steps, A)`.
        - Raise `ValueError` if `n_obs_steps < 1`, `n_action_steps < 1` or `end > H`.
        """,
            method="[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) (observation horizon, prediction horizon, action horizon) and the table in the docstring of the pinned `DiffusionPolicy.select_action`; the pinned `generate_actions` uses `start = n_obs_steps - 1`, `end = start + n_action_steps`.",
            pointers="Two numbers, one slice. Test it with a chunk whose values are their own indices.",
            hint="""
        ```python
        start = n_obs_steps - 1
        end = start + n_action_steps
        if n_obs_steps < 1 or n_action_steps < 1 or end > chunk.shape[1]:
            raise ValueError("the executed slice must fit inside the sampled window")
        return chunk[:, start:end]
        ```
        """,
            names=["executed_action_slice"],
            module="vla_lab.teaching.diffusion",
            stub="""
        def executed_action_slice(chunk, n_obs_steps, n_action_steps):
            # Return chunk[:, n_obs_steps - 1 : n_obs_steps - 1 + n_action_steps].
            raise NotImplementedError("Complete Exercise 11.5: executed_action_slice")
        """,
            check="""
        chunk = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1).expand(2, 16, 3).clone()
        taken = executed_action_slice(chunk, 2, 8)
        assert taken.shape == (2, 8, 3), "Expected (B, n_action_steps, A)."
        assert taken[0, :, 0].tolist() == [float(i) for i in range(1, 9)], f"For 2/16/8 the indices must be 1..8; got {taken[0, :, 0].tolist()}."
        assert executed_action_slice(chunk, 1, 4)[0, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0], "With one observation the window starts at the current action."
        assert executed_action_slice(chunk, 2, 15)[0, -1, 0] == 15.0, "The last allowed slice ends at index H-1."
        for bad in ((2, 16), (0, 4), (2, 0)):
            try:
                executed_action_slice(chunk, *bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{bad} must raise ValueError.")
        print("Exercise 11.5 passed")
        """,
            use_intro="The timeline of one chunk, in real units. The dataset is recorded at 10 Hz; the simulator controller runs at 20 Hz. The reference policy sends each action for one controller step and does **not** resample; that behaviour is retained here and is a known mismatch.",
            use="""
        horizon, n_obs, n_act = 16, 2, 8
        chunk = torch.arange(horizon).reshape(1, horizon, 1).float()
        picked = executed_action_slice(chunk, n_obs, n_act)[0, :, 0].int().tolist()
        print("window index        :", list(range(horizon)))
        print("observed (history)  :", list(range(n_obs)), "<- index", n_obs - 1, "is the current frame")
        print("executed            :", picked)
        print(f"recorded demonstration time of the window: {horizon / 10:.1f} s at 10 Hz; executed actions: {n_act / 10:.1f} s of demonstration time")
        print(f"as sent to the 20 Hz controller: {n_act / 20:.2f} s of simulated time per chunk (not resampled)")
        """,
        )
    )
    return L


def lesson_11_tail(L: Lesson) -> Lesson:
    L.items += [
        Md(r"""
        ## Experiment — how many sampling steps are enough? (30–60 min)

        **Question.** The pinned policy trains with 100 steps and samples with 20. How does the number of sampling steps change the accuracy of the executed actions and the time per chunk, with the noise held fixed?

        **Setup (real data, quick CPU version).** A small state-conditioned denoiser is trained once, here, with the Unit 9 loss (300 updates on real windows; this is *learner trained weights*, not the prepared policy). Then **your** `sample_actions`, `ddpm_reverse_step` and `executed_action_slice` turn fixed noise into action chunks for 256 validation windows. The opt-in sections below run the *same functions* inside the real prepared policy.

        **Fixed for every trial.** The trained denoiser; the 256 validation windows; initial noise from seed `11` and the step noises from the same generator; `clip_range` as listed. The 100-step chunk with the same initial noise is the reference for `l1_to_100`.

        **Allowed changes (per trial).** `steps` (any of `1, 2, 5, 10, 20, 50, 100`) and `clip` (`True`: clip the clean estimate to `[-1, 1]`; `False`: no clipping).

        **Measures.** `l1_to_demo`: masked L1 between the executed actions (indices `1…8`) and the demonstrated actions at those steps. `l1_to_100`: L1 between the executed actions and those of the 100-step sample. `ms_batch1`: milliseconds for one chunk at batch size 1 (the simulator case; noisy, compare within one session). Padded demonstration steps are excluded.

        **Prediction first.** In the next cell: at what step budget will `l1_to_demo` stop improving, and what happens with one step?

        **Keep failed trials.** Every trial is appended to `TRIALS`. A budget that is much worse is still a result.
        """),
        Code('PREDICTION = "TODO: replace this text with your prediction."\nprint(PREDICTION)'),
        Code("# Supplied: trial table helper\n" + tidy(EXPERIMENT_TABLE_HELPER)),
        Code(r"""
            # Supplied: real windows and a briefly trained small denoiser (learner trained weights)
            TRAIN = load_state_action_windows(CONFIGS, DATA, "bowl_plate", "train")
            VALID = load_state_action_windows(CONFIGS, DATA, "bowl_plate", "validation")
            SCHEDULE = DDPMSchedule.cosine(100)

            torch.manual_seed(0)
            SMALL = DiffusionDenoiser(ConditionalUnet1D(7, 16, (32, 64, 128), step_embed_dim=32), SCHEDULE, flat_state_condition)
            draws = torch.Generator().manual_seed(1)
            training_batches = [take_batch(TRAIN, torch.randint(0, len(TRAIN["action"]), (64,), generator=draws)) for _ in range(300)]
            started = time.perf_counter()
            train_loop(SMALL, training_batches, torch.optim.Adam(SMALL.parameters(), lr=1e-3, betas=(0.95, 0.999)), epsilon_loss_fn, steps=300)
            SMALL.eval()
            print(f"trained 300 updates in {time.perf_counter() - started:.0f} s")

            pick = torch.linspace(0, len(VALID["action"]) - 1, 256).round().long()
            EVAL = take_batch(VALID, pick)
            CONDITION = SMALL.condition_fn(EVAL)
            denoise = lambda x, t: SMALL.unet(x, t, global_cond=CONDITION)
            DEMO, DEMO_PAD = EVAL["action"][:, 1:9], EVAL["action_is_pad"][:, 1:9]    # the executed indices of the demonstration window
        """),
        Code(r"""
            # Supplied: harness. Your sampler, reverse step and slice are in the real path.
            SEED_NOISE, N_OBS, N_ACT = 11, 2, 8
            SHAPE = EVAL["action"].shape
            BASELINE = {"steps": 20, "clip": True}
            TRIALS = []


            def sample_chunk(steps, clip, batch_size=None):
                rows = SHAPE[0] if batch_size is None else batch_size
                g = torch.Generator().manual_seed(SEED_NOISE)
                initial = torch.randn(SHAPE, generator=g)[:rows]
                noises = [n[:rows] for n in make_step_noises(steps, SHAPE, g)]
                cond = CONDITION[:rows]
                started = time.perf_counter()
                with torch.no_grad():
                    window = sample_actions(lambda x, t: SMALL.unet(x, t, global_cond=cond), initial, SCHEDULE, steps, noises, 1.0 if clip else None)
                return window, time.perf_counter() - started


            REFERENCE_WINDOW, _ = sample_chunk(100, True)
            REFERENCE_EXECUTED = executed_action_slice(REFERENCE_WINDOW, N_OBS, N_ACT)


            def run_trial(config):
                window, _ = sample_chunk(config["steps"], config["clip"])
                executed = executed_action_slice(window, N_OBS, N_ACT)
                timings = [sample_chunk(config["steps"], config["clip"], batch_size=1)[1] for _ in range(5)]
                return {
                    "l1_to_demo": masked_action_l1(executed, DEMO, DEMO_PAD).item(),
                    "l1_to_100": (executed - REFERENCE_EXECUTED).abs().mean().item(),
                    "ms_batch1": 1000 * sorted(timings)[2],
                }


            def record(config):
                config = {**BASELINE, **config}
                try:
                    TRIALS.append({"config": config, "status": "ok", **run_trial(config)})
                except Exception as error:  # a failed trial is evidence too
                    TRIALS.append({"config": config, "status": f"failed: {type(error).__name__}: {error}"})


            record(BASELINE)
            COLUMNS = ["l1_to_demo", "l1_to_100", "ms_batch1"]
            show_trials(TRIALS, COLUMNS);
        """),
        Md(
            "**Your trials.** Add at least three trials. Changing `steps` is the main question; try `clip=False` once. Budget: 300 updates per trial."
        ),
        Pair(
            learner="""
            # Implement: choose your trials.
            MY_TRIALS = [
                # {"steps": 5},
            ]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
            reference="""
            # Reference choice of trials: one change per trial.
            MY_TRIALS = [{"steps": 1}, {"steps": 2}, {"steps": 5}, {"steps": 10}, {"steps": 50}, {"steps": 100}, {"clip": False}]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
        ),
        Code(r"""
            # Check: the protocol was followed.
            assert TRIALS[0]["config"] == BASELINE, "The first trial must be the baseline."
            assert len(TRIALS) >= 4, "Run the baseline and at least three trials of your own."
            assert all(set(t["config"]) == set(BASELINE) for t in TRIALS), "Only steps and clip may change."
            assert all(t["config"]["steps"] in (1, 2, 5, 10, 20, 50, 100) for t in TRIALS)
            ok = [t for t in TRIALS if t["status"] == "ok"]
            assert all(torch.isfinite(torch.tensor([t["l1_to_demo"], t["l1_to_100"], t["ms_batch1"]])).all() for t in ok), "Measures must be finite."
            print(f"{len(TRIALS)} trials kept ({len(ok)} ok). Fixed noise seed {SEED_NOISE}; {SHAPE[0]} validation windows; denoiser trained once for 300 updates.")
        """),
        Code(r"""
            # Use your function: accuracy against time for every trial
            ok = [t for t in TRIALS if t["status"] == "ok"]
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
            for t in ok:
                label = f"{t['config']['steps']}" + ("" if t["config"]["clip"] else " (no clip)")
                axes[0].scatter(t["ms_batch1"], t["l1_to_demo"]); axes[0].annotate(label, (t["ms_batch1"], t["l1_to_demo"]), fontsize=8)
                axes[1].scatter(t["ms_batch1"], t["l1_to_100"]); axes[1].annotate(label, (t["ms_batch1"], t["l1_to_100"]), fontsize=8)
            axes[0].set_xlabel("ms per chunk (batch 1)"); axes[0].set_ylabel("l1_to_demo")
            axes[1].set_xlabel("ms per chunk (batch 1)"); axes[1].set_ylabel("l1_to_100")
            plt.tight_layout(); plt.show()
        """),
        Md("""
        **Explain (2–4 sentences, edit this cell).** Which budget would you choose for a robot, and why? How does the distance to the 100-step sample differ from the distance to the demonstration, and which one measures task success? What does `clip=False` change? Remember that this denoiser sees only the state (no images) and was trained for 300 updates.

        *Optional extension:* repeat the baseline and two other budgets with another `SEED_NOISE`; report the spread. Keep all rows.
        """),
        Md("""
        ## Opt-in A: your sampler inside the real prepared policy (offline, CPU)

        `CallbackDiffusionPolicy` runs the **prepared** diffusion policy's image encoders and U-Net, but its sampling loop is **your** `sample_actions` (with your `ddpm_reverse_step`) and its action choice is **your** `executed_action_slice`; it never calls `scheduler.step`. This section samples chunks for 12 real validation frames with the same fixed noise for every budget, and compares them with the demonstrated actions and with the 100-step sample. It reads local files only. The workload is 12 validation windows and up to 100 sampling steps per window. Set `RUN_REAL_POLICY_OFFLINE = True` to run it.
        """),
        Code(r"""
            RUN_REAL_POLICY_OFFLINE = False
            REAL_SEED_NOISE = 11
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            STEP_BUDGETS = (5, 10, 20, 50, 100)

            if RUN_REAL_POLICY_OFFLINE:
                import json
                import pandas as pd
                from torch.utils.data import default_collate
                from vla_lab.teaching.act_model import open_libero_dataset
                from vla_lab.teaching.diffusion import stack_observation
                from vla_lab.training.diffusion import FPS, diffusion_delta_timestamps

                checkpoint = DATA / "runs/diffusion-bowl-plate-reference-2000/checkpoint-002000"
                if not (checkpoint / "model.safetensors").is_file():
                    raise FileNotFoundError(f"Prepared diffusion checkpoint missing: {checkpoint}")
                policy, preprocessor, _ = load_prepared_diffusion(checkpoint, device=DEVICE)
                config = policy.config
                dataset = open_libero_dataset(CONFIGS, DATA, "bowl_plate", "validation", diffusion_delta_timestamps(config, FPS))
                frames = torch.linspace(0, len(dataset) - 1, 12).round().long().tolist()
                batch = preprocessor(default_collate([dataset[i] for i in frames]))
                observation = stack_observation(policy, batch)
                wrapper = CallbackDiffusionPolicy(policy, sample_actions, executed_action_slice, seed=REAL_SEED_NOISE)
                real_demo = batch["action"][:, config.n_obs_steps - 1 : config.n_obs_steps - 1 + config.n_action_steps]
                real_pad = batch["action_is_pad"][:, config.n_obs_steps - 1 : config.n_obs_steps - 1 + config.n_action_steps]
                shape = (len(frames), config.horizon, 7)
                windows, rows = {}, []
                for steps in STEP_BUDGETS:
                    g = torch.Generator(device=DEVICE).manual_seed(REAL_SEED_NOISE)
                    initial = torch.randn(shape, generator=g, device=DEVICE)
                    noises = make_step_noises(steps, shape, g, device=DEVICE)
                    started = time.perf_counter()
                    windows[steps] = wrapper.predict_window(observation, initial, noises, num_inference_steps=steps)
                    rows.append({"steps": steps, "seconds_for_12_chunks": time.perf_counter() - started})
                reference = executed_action_slice(windows[max(STEP_BUDGETS)], config.n_obs_steps, config.n_action_steps)
                for row in rows:
                    executed = executed_action_slice(windows[row["steps"]], config.n_obs_steps, config.n_action_steps)
                    row["l1_to_demo"] = masked_action_l1(executed, real_demo, real_pad).item()
                    row["l1_to_max_steps"] = (executed - reference).abs().mean().item()
                print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
                saved = DATA.parent / "artifacts/reference/diffusion/sampler_ablation.json"
                if saved.is_file():
                    ablation = json.loads(saved.read_text())["sampling_steps"]
                    print("Prepared policy results (saved by the project: simulator, development states 0-4, RTX 3070; NOT produced here):")
                    for key, value in ablation.items():
                        print(f"  {key:>3} steps: {value['successes']}/{value['attempts']} successes, mean sampler time {value['weighted_mean_sampler_ms']:.0f} ms")
            else:
                print("Offline real-policy comparison is off. Set RUN_REAL_POLICY_OFFLINE=True to run the 12-window comparison.")
        """),
        Md("""
        ## Opt-in B: closed-loop rollouts on development states 10–14 (simulator)

        This runs the prepared policy with **your sampler** in the LIBERO simulator through the project's `run_policy_episode`, on the *methods* development states `10–14` only (states `0–9` are for exploration, `15–19` for the intervention unit, and the final states `20–49` are never used in these units). Reset seed `20260930 + state` and the policy noise seed are fixed; every attempt, including failures and errors, is kept in `ATTEMPTS`. Each episode has at most 300 simulator steps (the controller runs at 20 Hz; the demonstrations were recorded at 10 Hz and the policy is **not** resampled). The episode budget is 300 simulator steps per attempt. The code selects CUDA when available and otherwise uses CPU; it loads one model only. Set `RUN_DEV_STATE_ROLLOUTS = True` to run it.
        """),
        Code(r"""
            RUN_DEV_STATE_ROLLOUTS = False
            DEV_STATES = (10, 11, 12, 13, 14)     # methods states; narrow this list to run fewer episodes
            ROLLOUT_BUDGETS = (5, 20)             # sampler steps to compare
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            MAX_STEPS = 300
            RESET_BASE_SEED = 20260930
            ATTEMPTS = []

            if RUN_DEV_STATE_ROLLOUTS:
                import pandas as pd
                from vla_lab.sim.episode import open_libero_task, run_policy_episode

                assert all(10 <= s <= 14 for s in DEV_STATES), "Use development states 10-14 only in this unit."
                checkpoint = DATA / "runs/diffusion-bowl-plate-reference-2000/checkpoint-002000"
                if not (checkpoint / "model.safetensors").is_file():
                    raise FileNotFoundError(f"Prepared diffusion checkpoint missing: {checkpoint}")
                policy, preprocessor, postprocessor = load_prepared_diffusion(checkpoint, device=DEVICE)
                for steps in ROLLOUT_BUDGETS:
                    for state_id in DEV_STATES:
                        wrapper = CallbackDiffusionPolicy(policy, sample_actions, executed_action_slice,
                                                          num_inference_steps=steps, seed=RESET_BASE_SEED + state_id)
                        environment = None
                        try:
                            environment = open_libero_task("bowl_plate", state_id, CONFIGS, DATA)
                            result = run_policy_episode(environment, wrapper, preprocessor, postprocessor, "bowl_plate", state_id, CONFIGS,
                                                        partition="development", max_steps=MAX_STEPS, reset_seed=RESET_BASE_SEED + state_id)
                            ATTEMPTS.append({"steps": steps, "state": state_id, "success": result.success, "env_steps": result.steps,
                                             "ended": result.termination_reason, "chunks": wrapper.chunks_made,
                                             "mean_sampler_ms": 1000 * sum(wrapper.sample_seconds) / max(len(wrapper.sample_seconds), 1)})
                        except Exception as error:  # keep failed attempts
                            ATTEMPTS.append({"steps": steps, "state": state_id, "success": False, "ended": f"error: {type(error).__name__}: {error}"})
                        finally:
                            if environment is not None:
                                try:
                                    environment.close()
                                except Exception as error:
                                    ATTEMPTS.append({"steps": steps, "state": state_id, "success": False,
                                                     "ended": f"close error: {type(error).__name__}: {error}"})
                table = pd.DataFrame(ATTEMPTS)
                print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
                print(table.groupby("steps")["success"].agg(["sum", "count"]).rename(columns={"sum": "successes", "count": "attempts"}))
            else:
                print("Closed-loop rollouts are off. Set RUN_DEV_STATE_ROLLOUTS=True to run", len(DEV_STATES) * len(ROLLOUT_BUDGETS), "episodes.")
        """),
        Md("""
        ## What you built

        - `posterior_coefficients`, `ddpm_reverse_step`, `inference_timesteps`, `sample_actions` and `executed_action_slice`: DDPM sampling that equals `diffusers` at every timestep (final and skipped steps included) without calling it, and a slice that picks the right actions.
        - A real policy object whose actions come from your sampler, with fixed noise, step budgets, timing and every trial or attempt kept.

        **Next:** Unit 12A opens a vision-language-action policy (SmolVLA): inputs and trainable parameters. Unit 12B replaces diffusion by flow matching.
        """),
    ]
    return L


TINY_FIXTURE = r'''
VLM_DIR = DATA / "models/smolvlm2-500m-video-instruct"     # local tokenizer files and config; the 500M weights are NOT read here
IMAGES = ("observation.images.image", "observation.images.image2")
PROMPT = "put the bowl on the plate"


def make_tiny_smolvla(seed=1):
    """A small *real* pinned SmolVLAPolicy: the same classes and code path as the 450M model,
    with a tiny VLM (width 64, 2 layers, 64x64 images). Random weights; it checks mechanics only."""
    tiny = Path(tempfile.mkdtemp(prefix="tiny_smolvlm_"))
    for path in VLM_DIR.iterdir():
        if path.is_file() and path.name != "model.safetensors":
            shutil.copy(path, tiny / path.name)
    vlm = json.loads((VLM_DIR / "config.json").read_text())
    vlm["text_config"].update(hidden_size=64, intermediate_size=128, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=16, num_hidden_layers=2)
    vlm["vision_config"].update(hidden_size=32, num_attention_heads=2, num_hidden_layers=1, intermediate_size=64,
                                image_size=64, patch_size=16, max_image_size={"longest_edge": 64}, size={"longest_edge": 64})
    vlm["scale_factor"] = 2
    (tiny / "config.json").write_text(json.dumps(vlm))
    config = SmolVLAConfig(
        device="cpu", push_to_hub=False, vlm_model_name=str(tiny), load_vlm_weights=False,
        input_features={
            IMAGES[0]: PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            IMAGES[1]: PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        chunk_size=10, n_action_steps=5, num_vlm_layers=2, resize_imgs_with_padding=(64, 64),
        max_state_dim=16, max_action_dim=16, prefix_length=0, num_steps=4, expert_width_multiplier=0.5,
        tokenizer_max_length=24,
    )
    torch.manual_seed(seed)
    return SmolVLAPolicy(config)


TOKENIZER = AutoTokenizer.from_pretrained(str(VLM_DIR))
POLICY = make_tiny_smolvla()
print("tiny SmolVLA parameters:", sum(p.numel() for p in POLICY.parameters()))
'''


def lesson_12a() -> Lesson:
    L = Lesson("12a-vla-inputs", "12a-vla-inputs", "VLA inputs and trainable parameters")
    L.items += [
        Md(r"""
        # Unit 12A — A vision-language-action policy: its inputs and its trainable parameters

        **Your result:** the input path and the parameter selection of SmolVLA. Your functions turn camera frames, a robot state and a text instruction into the tensors the model reads, and freeze the vision-language model (VLM) so that only the action expert and its projections are trained. They match the pinned LeRobot SmolVLA implementation exactly.

        Four short exercises (10–20 minutes each) and a larger experiment (30–60 minutes). These times are estimates, not measurements. Full answers are in [the reference notebook](12a-vla-inputs-reference.ipynb).

        **Sources.** [SmolVLA method description](https://huggingface.co/blog/smolvla#method) (a frozen SmolVLM2 backbone, a small action expert trained with flow matching, a visual and language prefix plus the robot state), and the pinned code in `lerobot/policies/smolvla` (`modeling_smolvla.py`, `smolvlm_with_expert.py`, `processor_smolvla.py`; revision `e0d50211ef236143ae867228662b7dfaba554f02`).

        **Do not mix up two things.** The SmolVLA *action expert* is a small language-model-style stack that reads the VLM's cached keys and values. It is **not** the ACT Transformer you built in Unit 8; do not assume they compute the same thing. Also keep apart the *published* policy (`smolvla-libero`, trained by others), the *prepared learner* (`smolvla-c2-5000-development`, trained in this project), and your own runs.
        """),
        setup_cell("""
            import json
            import shutil
            import tempfile
            import time
            from pathlib import Path

            import torch
            import torch.nn.functional as F
            from torch import nn
            import matplotlib.pyplot as plt
            from transformers import AutoTokenizer

            import vla_lab.teaching.smolvla as ref  # package reference: used by checks only
            from vla_lab.teaching.smolvla import SmolVLAAdapter, flow_loss_fn  # Unit 12B function, package version (supplied here)
            from vla_lab.teaching.act_model import load_act_samples
            from vla_lab.teaching.training import optimizer_update
            from lerobot.configs import FeatureType, PolicyFeature
            from lerobot.policies.common.vla_utils import resize_with_pad
            from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy
            from lerobot.processor import NewLineTaskProcessorStep, TokenizerProcessorStep, load_pretrained_policy_processors

            torch.manual_seed(0)
            print("PyTorch:", torch.__version__)
        """),
        Code("# Supplied: a helper that the checks use\n" + tidy(CLOSE_HELPER)),
        Md("""
        ## Background: what the model reads

        SmolVLA reads a **prefix** and writes a chunk of actions:

        - **Images.** Each camera frame (`float`, `[0, 1]`) is resized *keeping its aspect ratio* and padded on the top and left to the model size (512×512 for the published policy), then mapped to `[-1, 1]` as the vision encoder (SigLIP) expects.
        - **Text.** The instruction ends with a newline, is tokenized, and is padded on the right to the longest prompt in the batch. An **attention mask** says which tokens are real; padded tokens must not be read.
        - **State.** The (already normalized) robot state is zero-padded to `max_state_dim = 32`.
        - **Trainable parameters.** The VLM (SigLIP + language model) is frozen. The *action expert* layers, the state projection, the action input/output projections and the action-time MLP are trained.

        **Supplied here:** a small *real* SmolVLA (the pinned classes with a tiny VLM and random weights; no download, no large model load), the data reader, and the Unit 12B flow loss (package version). **You write** the four input and parameter functions.
        """),
        Md("""
        **Real data used in this unit.** Real LIBERO `bowl_plate` frames, robot states and action chunks, read from local files (no download starts here). The recording rate is 10 Hz; nothing is resampled to the simulator's 20 Hz controller. If files are missing, the cell tells you the setup command.
        """),
        Code("# Supplied: a small real SmolVLA and its tokenizer\n" + tidy(TINY_FIXTURE)),
    ]
    L.items.append(
        Exercise(
            key="12A.1",
            title="Camera images: resize, pad, normalize",
            minutes="15",
            purpose="Fit any camera frame into the square the vision encoder expects without distorting it, and put the pixel values in the range it was trained on.",
            contract="""
        - `resize_pad_normalize_image(image, height, width)`: `image (B, 3, h, w)` `float32` in `[0, 1]`. Return `(B, 3, height, width)` in `[-1, 1]`.
        - If `(h, w)` already equals `(height, width)`, only rescale values. Otherwise: `ratio = max(w / width, h / height)`; resize bilinearly (`align_corners=False`) to `(int(h / ratio), int(w / ratio))`; **pad zeros on the top and left** until the size is `(height, width)`; then map `v → 2 * v - 1`.
        - Padding is applied before the value mapping, so padded pixels end up at `-1`.
        - The aspect ratio must be preserved: a wide image fills the width and the missing rows are on top.
        """,
            method="SmolVLA preprocessing (`SmolVLAPolicy.prepare_images`, `resize_with_pad` with top/left padding and `pad_value=0`, then `img * 2 - 1`), described in the [SmolVLA method section](https://huggingface.co/blog/smolvla#method). SigLIP expects values in `[-1, 1]`.",
            pointers="Compute the ratio and the resized size first. `F.interpolate` does the resize; `F.pad(x, (left, right, top, bottom), value=0)` pads the last two axes (the order is left, right, top, bottom). Do the value mapping last.",
            hint="""
        ```python
        current_height, current_width = image.shape[-2:]
        if (current_height, current_width) != (height, width):
            ratio = max(current_width / width, current_height / height)
            resized = (int(current_height / ratio), int(current_width / ratio))
            image = F.interpolate(image, size=resized, mode="bilinear", align_corners=False)
            image = F.pad(image, (width - resized[1], 0, height - resized[0], 0), value=0.0)
        return image * 2.0 - 1.0
        ```
        """,
            names=["resize_pad_normalize_image"],
            module="vla_lab.teaching.smolvla",
            stub="""
        def resize_pad_normalize_image(image, height, width):
            # (B,3,h,w) in [0,1] -> (B,3,height,width) in [-1,1]; keep the aspect ratio; pad top and left with zeros.
            raise NotImplementedError("Complete Exercise 12A.1: resize_pad_normalize_image")
        """,
            check="""
        for h, w in ((64, 64), (48, 64), (64, 32), (128, 96), (256, 256)):
            image = torch.rand(2, 3, h, w)
            mine = resize_pad_normalize_image(image, 64, 64)
            expected, _ = POLICY.prepare_images({IMAGES[0]: image, IMAGES[1]: image})
            assert mine.shape == (2, 3, 64, 64), f"Expected (2, 3, 64, 64) for a {h}x{w} input."
            close(mine, expected[0], f"{h}x{w} input differs from the pinned preparation", atol=1e-6)
        wide = resize_pad_normalize_image(torch.ones(1, 3, 32, 64), 64, 64)
        assert (wide[:, :, :32] == -1).all() and (wide[:, :, 32:] == 1).all(), "A wide image fills the width; the padding rows are on the top and equal -1."
        tall = resize_pad_normalize_image(torch.ones(1, 3, 64, 32), 64, 64)
        assert (tall[..., :32] == -1).all() and (tall[..., 32:] == 1).all(), "A tall image fills the height; the padding columns are on the left."
        same = resize_pad_normalize_image(torch.zeros(1, 3, 64, 64), 64, 64)
        assert (same == -1).all(), "No resize when the size already fits: values only map [0, 1] -> [-1, 1]."
        print("Exercise 12A.1 passed")
        """,
            use_intro="Prepare a real 256×256 camera frame for the tiny model (64×64) and for the published model (512×512), and look at both.",
            use="""
        frames = load_act_samples(CONFIGS, DATA, "bowl_plate", "validation", count=2, chunk_size=2)     # real frames, 256x256
        frame = frames[IMAGES[0]][:1]
        small, large = resize_pad_normalize_image(frame, 64, 64), resize_pad_normalize_image(frame, 512, 512)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
        for axis, image, title in zip(axes, (frame, small, large), ("raw 256x256", "tiny model 64x64", "published model 512x512")):
            axis.imshow(((image[0] + 1) / 2 if image.min() < 0 else image[0]).permute(1, 2, 0).clamp(0, 1)); axis.set_title(title); axis.axis("off")
        plt.show()
        print("value range after preparation:", float(small.min()), "to", float(small.max()))
        """,
        )
    )
    L.items.append(
        Exercise(
            key="12A.2",
            title="Prompts: newline, tokens and attention mask",
            minutes="15",
            purpose="Turn the instruction into token ids plus a mask that tells the model which positions are real words and which are padding.",
            contract="""
        - `tokenize_prompts(tokenizer, prompts, max_length)`: `prompts` is one string or a list of strings. `tokenizer` is a Hugging Face tokenizer.
        - Append `"\\n"` to every prompt that does not already end with a newline.
        - Call the tokenizer with `max_length=max_length`, `truncation=True`, `padding="longest"`, `padding_side="right"`, `return_tensors="pt"`. So the length `L` is the longest prompt in the batch, at most `max_length`.
        - Return `(tokens, mask)`: `tokens (B, L)` `long`, `mask (B, L)` `bool`, `True` for real tokens and `False` for padding.
        """,
            method="SmolVLA's processor pipeline: `NewLineTaskProcessorStep` then `TokenizerProcessorStep(padding='longest', padding_side='right', max_length=48)` (`processor_smolvla.py`). The mask is later used to hide padded tokens from attention.",
            pointers="Normalize the input to a list first. Add the newline. One tokenizer call does the rest; convert the mask to `bool` at the end.",
            hint="""
        ```python
        texts = [prompts] if isinstance(prompts, str) else list(prompts)
        texts = [text if text.endswith("\\n") else f"{text}\\n" for text in texts]
        encoded = tokenizer(texts, max_length=max_length, truncation=True, padding="longest",
                            padding_side="right", return_tensors="pt")
        return encoded["input_ids"].long(), encoded["attention_mask"].to(torch.bool)
        ```
        """,
            names=["tokenize_prompts"],
            module="vla_lab.teaching.smolvla",
            stub="""
        def tokenize_prompts(tokenizer, prompts, max_length):
            # Newline, tokenize with right padding to the longest prompt, return (tokens long (B,L), mask bool (B,L)).
            raise NotImplementedError("Complete Exercise 12A.2: tokenize_prompts")
        """,
            check="""
        tokens, mask = tokenize_prompts(TOKENIZER, ["put the bowl on the plate", "open it\\n"], 48)
        assert tokens.dtype == torch.long and mask.dtype == torch.bool and tokens.shape == mask.shape and tokens.shape[0] == 2
        assert mask[0].all() and mask[1].sum() < mask[0].sum(), "The shorter prompt must be right padded."
        assert not mask[1][int(mask[1].sum()):].any() and mask[1][:int(mask[1].sum())].all(), "Real tokens first, then padding."
        single, single_mask = tokenize_prompts(TOKENIZER, "put the bowl on the plate", 48)
        assert torch.equal(single[0], tokens[0]) and single_mask.all(), "A single string is a batch of one; the newline is added."
        step = TokenizerProcessorStep(tokenizer=TOKENIZER, max_length=48, padding="longest", padding_side="right")
        pinned = step._tokenize_text([NewLineTaskProcessorStep().complementary_data({"task": "put the bowl on the plate"})["task"]])
        assert torch.equal(single, pinned["input_ids"]), "Tokens must equal the pinned processor's tokens."
        assert tokenize_prompts(TOKENIZER, "word " * 100, 12)[0].shape == (1, 12), "Truncate to max_length."
        newline_twice, _ = tokenize_prompts(TOKENIZER, "open it\\n", 48)
        once, _ = tokenize_prompts(TOKENIZER, "open it", 48)
        assert torch.equal(newline_twice, once), "Do not add a second newline."
        print("Exercise 12A.2 passed")
        """,
            use_intro="Tokenize the canonical instruction, a paraphrase and a long conflicting instruction together, and see how padding and the mask line up.",
            use="""
        prompts = ["put the bowl on the plate", "place the bowl onto the plate please", "turn on the stove and then put the bowl on the plate"]
        tokens, mask = tokenize_prompts(TOKENIZER, prompts, 48)
        print("token matrix:", tuple(tokens.shape), "| real tokens per prompt:", mask.sum(1).tolist())
        for row in range(3):
            print(f"prompt {row}:", [TOKENIZER.decode([int(t)]) if m else "<pad>" for t, m in zip(tokens[row], mask[row])][:14])
        """,
        )
    )
    L.items.append(
        Exercise(
            key="12A.3",
            title="Assemble the model inputs",
            minutes="15–20",
            purpose="Put the three kinds of input together in the form the model's prefix reads: image tensors with masks, token ids with masks, and a padded state.",
            contract="""
        - `pad_to_width(values, width)`: append zeros on the last axis until it has `width` entries; return the input unchanged if it already has `width`; raise `ValueError` if the last axis is wider than `width`.
        - `prepare_vla_inputs(observation, prompts, tokenizer, config)`: `observation` has `"observation.state" (B, S)` (already normalized) and, for every key of `config.image_features` (in that order), an image `(B, 3, h, w)` in `[0, 1]`. `config.resize_imgs_with_padding = (width, height)` (note the order), `config.max_state_dim`, `config.tokenizer_max_length`.
        - Return a dict: `images` (list of `(B, 3, height, width)` from Exercise 12A.1, camera order as in `config.image_features`), `img_masks` (list of `(B,)` **bool** tensors, all `True`: every camera is present), `lang_tokens`, `lang_masks` (from Exercise 12A.2 with `config.tokenizer_max_length`), `state` (`(B, config.max_state_dim)`, zero padded).
        - If `prompts` is one string but the batch has `B > 1` rows, give every row the same tokens and mask (expand the single row).
        """,
            method="`SmolVLAPolicy.prepare_images`, `prepare_state` and the tokenizer steps, called with the same shapes by `VLAFlowMatching.embed_prefix` ([SmolVLA method](https://huggingface.co/blog/smolvla#method)).",
            pointers="Call your two earlier functions. Keep the camera order of `config.image_features`. Create the masks on the same device as the images.",
            hint="""
        ```python
        def pad_to_width(values, width):
            missing = width - values.shape[-1]
            if missing < 0:
                raise ValueError(f"last axis has {values.shape[-1]} entries, more than width {width}")
            return F.pad(values, (0, missing))

        def prepare_vla_inputs(observation, prompts, tokenizer, config):
            width, height = config.resize_imgs_with_padding
            images = [resize_pad_normalize_image(observation[key], height, width) for key in config.image_features]
            batch = images[0].shape[0]
            tokens, token_mask = tokenize_prompts(tokenizer, prompts, config.tokenizer_max_length)
            if tokens.shape[0] == 1 and batch > 1:
                tokens, token_mask = tokens.expand(batch, -1), token_mask.expand(batch, -1)
            return {"images": images, "img_masks": [torch.ones(batch, dtype=torch.bool, device=images[0].device) for _ in images],
                    "lang_tokens": tokens.to(images[0].device), "lang_masks": token_mask.to(images[0].device),
                    "state": pad_to_width(observation["observation.state"], config.max_state_dim)}
        ```
        """,
            names=["pad_to_width", "prepare_vla_inputs"],
            module="vla_lab.teaching.smolvla",
            stub="""
        def pad_to_width(values, width):
            # Zero-pad the last axis to `width`; ValueError if it is already wider.
            raise NotImplementedError("Complete Exercise 12A.3: pad_to_width")


        def prepare_vla_inputs(observation, prompts, tokenizer, config):
            # Return {"images", "img_masks", "lang_tokens", "lang_masks", "state"} from the observation and prompts.
            raise NotImplementedError("Complete Exercise 12A.3: prepare_vla_inputs")
        """,
            check="""
        out = pad_to_width(torch.ones(2, 3), 5)
        assert out.shape == (2, 5) and out[:, :3].eq(1).all() and out[:, 3:].eq(0).all(), "Pad with zeros on the right."
        assert torch.equal(pad_to_width(torch.ones(2, 5), 5), torch.ones(2, 5))
        try:
            pad_to_width(torch.ones(2, 6), 5)
        except ValueError:
            pass
        else:
            raise AssertionError("A wider input must raise ValueError.")

        g = torch.Generator().manual_seed(0)
        observation = {IMAGES[0]: torch.rand(2, 3, 48, 64, generator=g), IMAGES[1]: torch.rand(2, 3, 64, 64, generator=g),
                       "observation.state": torch.randn(2, 8, generator=g)}
        prompts = ["put the bowl on the plate", "open it"]
        mine = prepare_vla_inputs(observation, prompts, TOKENIZER, POLICY.config)
        tokens, mask = ref.tokenize_prompts(TOKENIZER, prompts, POLICY.config.tokenizer_max_length)
        pinned_images, _ = POLICY.prepare_images(observation)
        assert len(mine["images"]) == 2 and len(mine["img_masks"]) == 2, "One image and one mask per camera."
        for a, b in zip(mine["images"], pinned_images, strict=True):
            close(a, b, "images differ from the pinned preparation", atol=1e-6)
        assert all(m.dtype == torch.bool and m.shape == (2,) and m.all() for m in mine["img_masks"]), "img_masks: bool (B,) all True."
        assert torch.equal(mine["lang_tokens"], tokens) and torch.equal(mine["lang_masks"], mask), "Tokens and masks from tokenize_prompts."
        assert mine["state"].shape == (2, 16) and torch.equal(mine["state"][:, :8], observation["observation.state"]) and not mine["state"][:, 8:].any(), "State padded to max_state_dim with zeros."
        one = prepare_vla_inputs(observation, "put the bowl on the plate", TOKENIZER, POLICY.config)
        assert one["lang_tokens"].shape[0] == 2 and torch.equal(one["lang_tokens"][0], one["lang_tokens"][1]), "A single prompt is shared by the whole batch."
        print("Exercise 12A.3 passed")
        """,
            use_intro="Feed your inputs to the model and test the mask: the model's output for a prompt must not change when that prompt is padded to a longer length because another prompt in the batch is longer.",
            use="""
        policy = POLICY.eval()
        adapter = SmolVLAAdapter(policy)
        samples = load_act_samples(CONFIGS, DATA, "bowl_plate", "validation", count=2, chunk_size=10, image_size=64)
        observation = {key: samples[key] for key in (*IMAGES, "observation.state")}
        short, long_ = "put the bowl on the plate", "put the bowl on the plate and then turn on the stove next to it"
        alone = prepare_vla_inputs(observation, [short, short], TOKENIZER, policy.config)
        padded = prepare_vla_inputs(observation, [short, long_], TOKENIZER, policy.config)
        print("prompt length alone:", alone["lang_tokens"].shape[1], "| padded to:", padded["lang_tokens"].shape[1])
        noisy = torch.randn(2, 10, 16, generator=torch.Generator().manual_seed(0))
        time = torch.tensor([0.5, 0.5])
        with torch.no_grad():
            v_alone = adapter.velocity(alone, noisy, time)
            v_padded = adapter.velocity(padded, noisy, time)
        print("output change for row 0 when its prompt is padded:", float((v_alone[0] - v_padded[0]).abs().max()))
        print("output change for row 1 (a different prompt):", float((v_alone[1] - v_padded[1]).abs().max()))
        """,
        )
    )
    L.items.append(
        Exercise(
            key="12A.4",
            title="Trainable parameters and the frozen VLM",
            minutes="15–20",
            purpose="Adapting a VLA should change the action expert, not the large backbone. Freeze it, give the optimizer exactly the right parameters, and keep the frozen backbone in evaluation mode when the rest trains.",
            contract="""
        - `select_trainable_parameters(policy, frozen_prefix="model.vlm_with_expert.vlm.")`: set `requires_grad = False` on every parameter whose name (from `policy.named_parameters()`) starts with `frozen_prefix`. **Do not change any other flag**: a parameter that was already frozen on purpose (for example the state projection) stays frozen. Return the list of parameters with `requires_grad=True`, each once, in `named_parameters()` order.
        - `train_mode_with_frozen_vlm(policy, vlm_name="model.vlm_with_expert.vlm")`: put `policy` in training mode, then call `.eval()` on the submodule named `vlm_name` (and so on all its children). Return `policy`.
        - After one optimizer update with these parameters, the VLM weights must be bit-for-bit unchanged and the expert weights must have changed.
        """,
            method="[SmolVLA](https://huggingface.co/blog/smolvla#method): the VLM is frozen and only the action expert is trained. Pinned: `SmolVLMWithExpertModel.set_requires_grad`/`train` (`train_expert_only=True`). A frozen module in training mode would still use dropout and update normalization statistics — a silent change.",
            pointers="Loop over `named_parameters()` once for the freeze, then collect. For the mode: `model.train()` switches *everything* on; fix the backbone afterwards with `get_submodule(name).eval()`.",
            hint="""
        ```python
        def select_trainable_parameters(policy, frozen_prefix="model.vlm_with_expert.vlm."):
            for name, parameter in policy.named_parameters():
                if name.startswith(frozen_prefix):
                    parameter.requires_grad_(False)
            return [parameter for _, parameter in policy.named_parameters() if parameter.requires_grad]

        def train_mode_with_frozen_vlm(policy, vlm_name="model.vlm_with_expert.vlm"):
            policy.train()
            policy.get_submodule(vlm_name).eval()
            return policy
        ```
        """,
            names=["select_trainable_parameters", "train_mode_with_frozen_vlm"],
            module="vla_lab.teaching.smolvla",
            stub="""
        def select_trainable_parameters(policy, frozen_prefix="model.vlm_with_expert.vlm."):
            # Freeze every parameter under frozen_prefix; leave all other flags alone; return the trainable parameters.
            raise NotImplementedError("Complete Exercise 12A.4: select_trainable_parameters")


        def train_mode_with_frozen_vlm(policy, vlm_name="model.vlm_with_expert.vlm"):
            # policy.train(), then keep the VLM submodule in eval mode. Return policy.
            raise NotImplementedError("Complete Exercise 12A.4: train_mode_with_frozen_vlm")
        """,
            check="""
        model = make_tiny_smolvla(seed=3)
        for parameter in model.parameters():                      # something unfroze everything
            parameter.requires_grad_(True)
        trainable = select_trainable_parameters(model)
        owner = {id(p): name for name, p in model.named_parameters()}
        assert len({id(p) for p in trainable}) == len(trainable), "Return each parameter once."
        assert not any(owner[id(p)].startswith("model.vlm_with_expert.vlm.") for p in trainable), "No VLM parameter may be returned."
        assert all(not p.requires_grad for n, p in model.named_parameters() if n.startswith("model.vlm_with_expert.vlm.")), "Every VLM parameter must be frozen."
        names = [owner[id(p)] for p in trainable]
        assert any(n.startswith("model.vlm_with_expert.lm_expert.") for n in names) and any(n.startswith("model.action_out_proj") for n in names), "Expert and projections stay trainable."
        model.model.state_proj.requires_grad_(False)               # a deliberate freeze must survive
        assert not any(owner[id(p)].startswith("model.state_proj") for p in select_trainable_parameters(model)), "Do not unfreeze parameters outside the prefix."

        model.model.vlm_with_expert.vlm.train()                    # something switched the backbone on
        assert train_mode_with_frozen_vlm(model) is model
        assert model.training and model.model.action_out_proj.training and model.model.vlm_with_expert.lm_expert.training, "The rest of the model must be in training mode."
        assert not any(m.training for m in model.model.vlm_with_expert.vlm.modules()), "Every VLM submodule must be in eval mode."

        # One real update: the VLM stays bit-for-bit identical, the expert changes.
        model = make_tiny_smolvla(seed=3)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        params = select_trainable_parameters(model)
        optimizer = torch.optim.AdamW(params, lr=1e-2)
        before_vlm = {n: p.detach().clone() for n, p in model.model.vlm_with_expert.vlm.named_parameters()}
        before_head = model.model.action_out_proj.weight.detach().clone()
        adapter = SmolVLAAdapter(model, train_mode_fn=train_mode_with_frozen_vlm)
        g = torch.Generator().manual_seed(0)
        batch = {IMAGES[0]: torch.rand(2, 3, 64, 64, generator=g), IMAGES[1]: torch.rand(2, 3, 64, 64, generator=g),
                 "observation.state": torch.randn(2, 8, generator=g),
                 "observation.language.tokens": torch.randint(5, 100, (2, 6)), "observation.language.attention_mask": torch.ones(2, 6, dtype=torch.bool),
                 "action": torch.randn(2, 10, 7, generator=g), "action_is_pad": torch.zeros(2, 10, dtype=torch.bool)}
        optimizer_update(adapter, batch, optimizer, flow_loss_fn, noise=torch.randn(2, 10, 16, generator=g), time=torch.tensor([0.3, 0.7]))
        assert all(torch.equal(before_vlm[n], p) for n, p in model.model.vlm_with_expert.vlm.named_parameters()), "The frozen VLM must not change."
        assert not torch.equal(before_head, model.model.action_out_proj.weight), "The expert side must change."
        print("Exercise 12A.4 passed")
        """,
            use_intro="Count where the parameters of the small model are, from your selection. The same grouping is used for the 450M prepared learner.",
            use="""
        model = make_tiny_smolvla(seed=3)
        total = sum(p.numel() for p in model.parameters())
        trainable = select_trainable_parameters(model)
        groups = {"VLM (frozen)": "model.vlm_with_expert.vlm.", "action expert": "model.vlm_with_expert.lm_expert.", "state projection": "model.state_proj.",
                  "action in/out projections": ("model.action_in_proj.", "model.action_out_proj."), "action-time MLP": ("model.action_time_mlp_in.", "model.action_time_mlp_out.")}
        for label, prefix in groups.items():
            count = sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix))
            trainable_count = sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad)
            print(f"{label:<26} {count:>9,} parameters, {trainable_count:>9,} trainable")
        print(f"total {total:,} | selected for the optimizer {sum(p.numel() for p in trainable):,}")
        """,
        )
    )
    return L


def lesson_12a_tail(L: Lesson) -> Lesson:
    L.items += [
        Md("""
        ### Your inputs against the real published configuration

        The published policy (`smolvla-libero`) uses 512×512 images, a 48-token prompt limit and 32-wide state and action vectors. This check builds the *real* preprocessing pipeline of that policy from its saved files (configuration, tokenizer, normalization; no 450M weights are loaded) and compares your `prepare_vla_inputs` with it on real camera frames. It reads local files only. If they are missing, the cell says so.
        """),
        Code(r"""
            RUN_REAL_CONFIG_CHECK = True

            if RUN_REAL_CONFIG_CHECK:
                model_dir = DATA / "models/smolvla-libero"
                if not (model_dir / "policy_preprocessor.json").is_file():
                    print("Published SmolVLA files not found; skipping. Install them with `just data-models` (this downloads; run it outside this notebook).")
                else:
                    real_config = SmolVLAConfig.from_pretrained(str(model_dir))
                    pipeline, _ = load_pretrained_policy_processors(
                        str(model_dir), preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": str(VLM_DIR)}, "device_processor": {"device": "cpu"}})
                    real_frames = load_act_samples(CONFIGS, DATA, "bowl_plate", "validation", count=2, chunk_size=2)
                    raw = {IMAGES[0]: real_frames[IMAGES[0]][0], IMAGES[1]: real_frames[IMAGES[1]][0],
                           "observation.state": real_frames["observation.state"][0], "task": PROMPT}
                    pinned_batch = pipeline(raw)                                   # the real pipeline: tokens, masks, normalized state
                    mine = prepare_vla_inputs({k: v for k, v in pinned_batch.items() if k.startswith("observation.")}, PROMPT, TOKENIZER, real_config)
                    print("image size:", tuple(mine["images"][0].shape), "| state width:", mine["state"].shape[-1], "| prompt tokens:", mine["lang_tokens"].shape[1])
                    for a, key in zip(mine["images"], real_config.image_features, strict=True):
                        expected = resize_with_pad(pinned_batch[key], 512, 512, pad_value=0) * 2 - 1
                        assert (a - expected).abs().max() < 1e-6, f"{key} differs from the pinned resize/pad/scale"
                    assert torch.equal(mine["lang_tokens"], pinned_batch["observation.language.tokens"]), "tokens differ from the real pipeline"
                    assert torch.equal(mine["lang_masks"], pinned_batch["observation.language.attention_mask"]), "mask differs from the real pipeline"
                    assert mine["state"].shape == (1, 32) and torch.equal(mine["state"][:, :8], pinned_batch["observation.state"])
                    print("Your inputs equal the real published preprocessing on real frames.")
            else:
                print("Real-configuration check is off.")
        """),
        Md(r"""
        ## Experiment — which parameters should be trained? (30–60 min)

        **Question.** With the same updates, how do three adaptation settings differ in trainable parameters, held-out flow loss, and whether the frozen backbone really stays fixed?

        **Setup (real data, small real model).** 128 training and 64 validation windows, evenly spread over the frozen `bowl_plate` train and validation episodes: both camera frames (box-filtered to 64×64), the standardized state, 10 standardized actions, padding where the chunk passes the episode end, and the instruction `put the bowl on the plate`. The model is the small real SmolVLA from the supplied cell (random weights: this is a mechanics experiment, not a robot policy). **Your functions** `resize_pad_normalize_image`, `tokenize_prompts`, `pad_to_width`, `prepare_vla_inputs`, `select_trainable_parameters` and `train_mode_with_frozen_vlm` are all in the real update: the adapter's `input_fn` builds the inputs with them, and `train_mode_fn` keeps the VLM frozen. The loss is the Unit 12B flow loss (package version) through the training unit's `optimizer_update`.

        **Fixed for every trial.** Weight seed `1`; batch seed `1`; **300 updates**, batch 16, AdamW `lr=1e-3`; the noise and the times of every update come from the seeded generator (times uniform in `(0.001, 0.999)`); the same validation noise (seed `7`) at times `0.2, 0.5, 0.8` for every trial.

        **Allowed changes (per trial).** `trainable`: `expert_only` (your selection: VLM frozen, expert and projections trained), `expert_and_vlm_text` (also unfreeze the VLM language model), `projections_only` (freeze the expert layers as well).

        **Measures.** `trainable_params`; `val_flow_loss`; `vlm_drift`: the largest absolute change of any VLM weight (must be exactly `0` when the VLM is frozen); `seconds`.

        **Prediction first.** In the next cell: which setting reaches the lowest validation loss, and which keeps `vlm_drift` at 0?

        **Keep failed trials.** Every trial is appended to `TRIALS`. A setting that is no better than the cheaper one is a valid result.
        """),
        Code('PREDICTION = "TODO: replace this text with your prediction."\nprint(PREDICTION)'),
        Code("# Supplied: trial table helper\n" + tidy(EXPERIMENT_TABLE_HELPER)),
        Code(r"""
            # Supplied: real windows for the small model (64x64 frames, 10-step chunks)
            TRAIN = load_act_samples(CONFIGS, DATA, "bowl_plate", "train", count=128, chunk_size=10, image_size=64)
            VALID = load_act_samples(CONFIGS, DATA, "bowl_plate", "validation", count=64, chunk_size=10, image_size=64)
            print("train windows:", len(TRAIN["action"]), "| validation windows:", len(VALID["action"]), "| instruction:", repr(PROMPT))
        """),
        Code(r"""
            # Supplied: harness. Your input and parameter functions are in the real update.
            STEPS, BATCH, LR, SEED_WEIGHTS, SEED_BATCH, SEED_NOISE = 300, 16, 1e-3, 1, 1, 7
            BASELINE = {"trainable": "expert_only"}
            TRIALS = []


            def run_trial(config):
                started = time.perf_counter()
                policy = make_tiny_smolvla(seed=SEED_WEIGHTS)
                for parameter in policy.parameters():
                    parameter.requires_grad_(True)
                selected = select_trainable_parameters(policy)                      # YOUR function freezes the VLM
                vlm = policy.model.vlm_with_expert.vlm
                if config["trainable"] == "expert_and_vlm_text":
                    for parameter in vlm.model.text_model.parameters():
                        parameter.requires_grad_(True)
                if config["trainable"] == "projections_only":
                    for parameter in policy.model.vlm_with_expert.lm_expert.parameters():
                        parameter.requires_grad_(False)
                params = [p for p in policy.parameters() if p.requires_grad]
                before = {name: p.detach().clone() for name, p in vlm.named_parameters()}

                def input_fn(adapter, raw):                                         # YOUR input functions build every batch
                    observation = {key: value for key, value in raw.items() if key.startswith("observation.")}
                    inputs = prepare_vla_inputs(observation, PROMPT, TOKENIZER, policy.config)
                    if "action" in raw:
                        inputs["actions"] = pad_to_width(raw["action"], policy.config.max_action_dim)
                    return inputs

                adapter = SmolVLAAdapter(policy, train_mode_fn=train_mode_with_frozen_vlm, input_fn=input_fn)
                optimizer = torch.optim.AdamW(params, lr=LR)
                draws = torch.Generator().manual_seed(SEED_BATCH)
                for _ in range(STEPS):
                    pick = torch.randint(0, len(TRAIN["action"]), (BATCH,), generator=draws)
                    batch = {key: value[pick] for key, value in TRAIN.items()}
                    noise = torch.randn(BATCH, 10, policy.config.max_action_dim, generator=draws)
                    time_ = torch.rand(BATCH, generator=draws) * 0.998 + 0.001
                    optimizer_update(adapter, batch, optimizer, flow_loss_fn, noise=noise, time=time_)
                adapter.eval()
                with torch.no_grad():
                    losses = []
                    for t in (0.2, 0.5, 0.8):
                        noise = torch.randn(len(VALID["action"]), 10, policy.config.max_action_dim, generator=torch.Generator().manual_seed(SEED_NOISE))
                        loss, _ = flow_loss_fn(adapter, VALID, noise=noise, time=torch.full((len(noise),), t))
                        losses.append(loss.item())
                return {
                    "trainable_params": sum(p.numel() for p in params),
                    "val_flow_loss": sum(losses) / len(losses),
                    "vlm_drift": max((p - before[name]).abs().max().item() for name, p in vlm.named_parameters()),
                    "seconds": time.perf_counter() - started,
                }


            def record(config):
                config = {**BASELINE, **config}
                try:
                    TRIALS.append({"config": config, "status": "ok", **run_trial(config)})
                except Exception as error:  # a failed trial is evidence too
                    TRIALS.append({"config": config, "status": f"failed: {type(error).__name__}: {error}"})


            record(BASELINE)
            COLUMNS = ["trainable_params", "val_flow_loss", "vlm_drift", "seconds"]
            show_trials(TRIALS, COLUMNS);
        """),
        Md("**Your trials.** Add at least two trials. Budget: 300 updates per trial."),
        Pair(
            learner="""
            # Implement: choose your trials.
            MY_TRIALS = [
                # {"trainable": "projections_only"},
            ]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
            reference="""
            # Reference choice of trials.
            MY_TRIALS = [{"trainable": "expert_and_vlm_text"}, {"trainable": "projections_only"}]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
        ),
        Code(r"""
            # Check: the protocol was followed.
            assert TRIALS[0]["config"] == BASELINE, "The first trial must be the baseline."
            assert len(TRIALS) >= 3, "Run the baseline and at least two trials of your own."
            assert all(t["config"]["trainable"] in {"expert_only", "expert_and_vlm_text", "projections_only"} for t in TRIALS)
            ok = [t for t in TRIALS if t["status"] == "ok"]
            assert all(torch.isfinite(torch.tensor(t["val_flow_loss"])) for t in ok), "val_flow_loss must be finite."
            frozen = [t for t in ok if t["config"]["trainable"] != "expert_and_vlm_text"]
            assert all(t["vlm_drift"] == 0.0 for t in frozen), "A frozen VLM must not drift: check select_trainable_parameters and train_mode_with_frozen_vlm."
            print(f"{len(TRIALS)} trials kept ({len(ok)} ok). Budget: {STEPS} updates, batch {BATCH}, seeds {SEED_WEIGHTS}/{SEED_BATCH}/{SEED_NOISE}.")
        """),
        Code(r"""
            # Use your function: parameters against held-out loss, every trial shown
            ok = [t for t in TRIALS if t["status"] == "ok"]
            labels = [t["config"]["trainable"] for t in ok]
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            axes[0].bar(range(len(ok)), [t["val_flow_loss"] for t in ok]); axes[0].set_title("val_flow_loss")
            axes[1].bar(range(len(ok)), [t["trainable_params"] for t in ok]); axes[1].set_title("trainable parameters"); axes[1].set_yscale("log")
            for axis in axes:
                axis.set_xticks(range(len(ok))); axis.set_xticklabels(labels, rotation=20, ha="right")
            plt.tight_layout(); plt.show()
        """),
        Md("""
        **Explain (2–4 sentences, edit this cell).** Which setting gave the best held-out loss for its cost? Which setting broke the "frozen VLM" rule, and what did `vlm_drift` show? This VLM has random weights and 128 training windows: what can you *not* conclude about the real 450M model?

        *Optional extension:* rerun the baseline with another `SEED_WEIGHTS` and keep both rows.
        """),
        Md("""
        ## Opt-in: the prepared learner at real size (CPU or GPU, one model)

        This loads the **prepared learner** (`smolvla-c2-5000-development`, 450M parameters, trained in this project) and its processors from local files (hash-checked, no download), then:

        1. runs **your** `select_trainable_parameters` on it and compares the group counts with the project's own parameter audit;
        2. builds one real batch (two camera frames, state, instruction) with the real pipeline, and compares your `prepare_vla_inputs` with the pinned path at the 512×512 size;
        3. computes the velocity from both input paths at fixed noise and time (`SmolVLAAdapter.velocity`) and compares them.

        It makes no update and writes nothing. The code selects CUDA when available and otherwise uses CPU. It loads one model. Set `RUN_REAL_SMOLVLA = True` to run it.
        """),
        Code(r"""
            RUN_REAL_SMOLVLA = False
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

            if RUN_REAL_SMOLVLA:
                from torch.utils.data import default_collate
                from vla_lab.policy.smolvla_learner import audit_learner_parameter_groups, load_learner_bundle
                from vla_lab.teaching.act_model import open_libero_dataset

                checkpoint = DATA / "runs/smolvla-c2-5000-development/checkpoint-005000"
                if not (checkpoint / "model.safetensors").is_file() or not VLM_DIR.joinpath("model.safetensors").is_file():
                    raise FileNotFoundError("Prepared learner or VLM weights missing (see manifests/smolvla_reference.yaml).")
                learner, _, _, manifest = load_learner_bundle(checkpoint, VLM_DIR, device=DEVICE)      # weights and hashes are checked
                preprocessor, _ = load_pretrained_policy_processors(
                    str(checkpoint), preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": str(VLM_DIR)}, "device_processor": {"device": DEVICE}})
                for parameter in learner.parameters():
                    parameter.requires_grad_(True)
                trainable = select_trainable_parameters(learner)
                groups = audit_learner_parameter_groups(learner)                       # raises if a flag breaks the project's rule
                counted = sum(p.numel() for p in trainable)
                total = sum(p.numel() for p in learner.parameters())
                audited = sum(group["trainable_parameters"] for group in groups.values())
                print(f"your selection: {counted:,} trainable of {total:,} parameters | project audit: {audited:,} | frozen VLM: {total - counted:,}")
                assert counted == audited, "your selection differs from the project's parameter audit"
                vlm_params = sum(p.numel() for n, p in learner.named_parameters() if n.startswith("model.vlm_with_expert.vlm."))
                assert total - counted >= vlm_params, "every VLM parameter must be frozen"
                dataset = open_libero_dataset(CONFIGS, DATA, "bowl_plate", "validation", {"action": [i / 10 for i in range(learner.config.chunk_size)]})
                batch = preprocessor(default_collate([dataset[5]]))
                adapter = SmolVLAAdapter(learner.eval())
                pinned_inputs = adapter.pinned_inputs(batch)
                observation = {k: v for k, v in batch.items() if k.startswith("observation.") and "language" not in k}
                mine = prepare_vla_inputs(observation, PROMPT, AutoTokenizer.from_pretrained(str(VLM_DIR)), learner.config)
                same = all(torch.equal(a, b) for a, b in zip(mine["images"], pinned_inputs["images"], strict=True)) and torch.equal(mine["lang_tokens"], pinned_inputs["lang_tokens"]) \
                    and torch.equal(mine["lang_masks"], pinned_inputs["lang_masks"]) and torch.equal(mine["state"], pinned_inputs["state"])
                print("your inputs equal the pinned inputs at real size:", same)
                noise = torch.randn(1, learner.config.chunk_size, learner.config.max_action_dim, generator=torch.Generator().manual_seed(0)).to(DEVICE)
                time_ = torch.tensor([0.5], device=DEVICE)
                with torch.no_grad():
                    difference = (adapter.velocity(mine, noise, time_) - adapter.velocity(pinned_inputs, noise, time_)).abs().max().item()
                print("largest velocity difference between the two input paths:", difference)
            else:
                print("Real SmolVLA check is off. Set RUN_REAL_SMOLVLA=True to run it (loads one 450M model).")
        """),
        Md("""
        ## What you built

        - `resize_pad_normalize_image`, `tokenize_prompts`, `pad_to_width`, `prepare_vla_inputs`, `select_trainable_parameters` and `train_mode_with_frozen_vlm`: the input path and parameter selection of SmolVLA, equal to the pinned implementation, with a mask that keeps padded prompt tokens from changing the output.
        - A fixed-budget comparison of adaptation settings with every trial kept, and a check that a frozen backbone does not move.

        **Next:** Unit 12B writes the flow-matching objective and the Euler sampler, and puts them into the same real model.
        """),
    ]
    return L


def lesson_12b() -> Lesson:
    L = Lesson("12b-flow", "12b-flow", "Flow matching for actions")
    L.items += [
        Md(r"""
        # Unit 12B — Flow matching: objective, Euler sampler and the policy hook

        **Your result:** the training objective and the sampler of SmolVLA's action expert, written by you and running inside the real model. Your loss equals the pinned LeRobot loss for the same noise and times; your Euler loop equals the pinned integrator; and a real policy wrapper uses your sampler to produce the actions it hands to the simulator.

        Four short exercises (10–20 minutes each) and a larger experiment (30–60 minutes). These times are estimates, not measurements. Full answers are in [the reference notebook](12b-flow-reference.ipynb).

        **Sources.** [SmolVLA method description](https://huggingface.co/blog/smolvla#method) (flow-matching action expert; the VLM is frozen); the pinned code in `lerobot/policies/smolvla/modeling_smolvla.py` (`VLAFlowMatching.forward`, `sample_actions`, `denoise_step`) and `lerobot/policies/common/flow_matching.py` (`euler_integrate`), revision `e0d50211ef236143ae867228662b7dfaba554f02`. Flow matching: [Lipman et al. 2023](https://arxiv.org/abs/2210.02747).

        **Do not mix up two things.** The SmolVLA *action expert* is a small language-model-style stack that attends to the VLM's cached keys and values. It is **not** the ACT Transformer of Unit 8: do not assume they are the same computation. And flow matching here is **not** DDPM: there is no noise schedule `ᾱ`, no clipping and no random step noise. The sampler is deterministic given the starting noise.

        **Where this unit sits.** The six Unit 12A functions are available as package versions. The *raw velocity adapter* (`SmolVLAAdapter.velocity`, the pinned model's forward pass up to the velocity) is supplied and tested against the pinned `VLAFlowMatching.forward`; you write the objective around it.
        """),
        setup_cell("""
            import json
            import shutil
            import tempfile
            import time
            from pathlib import Path

            import torch
            from torch import nn
            import matplotlib.pyplot as plt
            import yaml
            from transformers import AutoTokenizer

            import vla_lab.teaching.smolvla as ref  # package reference: used by checks only
            from vla_lab.teaching.smolvla import (  # Unit 12A functions, as package versions
                CallbackSmolVLAPolicy, SmolVLAAdapter, prepare_vla_inputs, select_trainable_parameters,
                train_mode_with_frozen_vlm, pad_to_width, load_prepared_smolvla,
            )
            from vla_lab.teaching.act_model import load_act_samples, open_libero_dataset
            from vla_lab.teaching.training import masked_action_l1, optimizer_update, train_loop
            from lerobot.configs import FeatureType, PolicyFeature
            from lerobot.policies.common.flow_matching import euler_integrate
            from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy

            torch.manual_seed(0)
            print("PyTorch:", torch.__version__)
        """),
        Code("# Supplied: a helper that the checks use\n" + tidy(CLOSE_HELPER)),
        Md("""
        ## Background: the flow and its sign convention

        Flow matching learns a **velocity field** that moves noise to actions along straight lines. SmolVLA uses time `t ∈ [0, 1]` with this convention (it is the pinned one; the opposite convention is common in other code, so check signs):

        - `t = 0` is the **clean action chunk**; `t = 1` is **pure noise**.
        - Training pair: `x_t = t · noise + (1 − t) · actions` and target velocity `u_t = noise − actions`. (Because `d x_t / d t = noise − actions`.)
        - The expert predicts `v(x_t, t, observation)`; the loss is the mean squared error `‖v − u_t‖²` over real action steps and real action columns.
        - **Sampling** integrates from `t = 1` (noise) to `t = 0` with forward Euler: `dt = −1 / n`, `x ← x + dt · v(x, t)`, `t = 1, 1 − 1/n, …`.
        - Actions are zero-padded to `max_action_dim = 32` columns inside the model. Only the first `action_dim = 7` columns and the non-padded time steps count in the loss.
        - Training times come from a prior that favors noisy times: `t = 0.001 + 0.999 · Beta(1.5, 1)`.
        """),
        Md("""
        **Real data used in this unit.** Real LIBERO `bowl_plate` frames, states and action chunks, read from local files (no download starts here). The recording rate is 10 Hz; nothing is resampled to the simulator's 20 Hz controller. If files are missing, the cell tells you the setup command.
        """),
        Code(
            "# Supplied: a small real SmolVLA (the pinned classes, a tiny VLM, random weights) and its tokenizer\n"
            + tidy(TINY_FIXTURE)
        ),
        Code(r"""
            # Supplied: a small batch for algorithm checks (random numbers) in the shapes the model reads
            def algorithm_check_batch(batch=2, chunk=10, seed=0):
                g = torch.Generator().manual_seed(seed)
                tokens, mask = ref.tokenize_prompts(TOKENIZER, ["put the bowl on the plate", "open it"], 24)
                pad = torch.zeros(batch, chunk, dtype=torch.bool)
                pad[1, 7:] = True
                return {
                    IMAGES[0]: torch.rand(batch, 3, 64, 64, generator=g), IMAGES[1]: torch.rand(batch, 3, 64, 64, generator=g),
                    "observation.state": torch.randn(batch, 8, generator=g),
                    "observation.language.tokens": tokens, "observation.language.attention_mask": mask,
                    "action": torch.randn(batch, chunk, 7, generator=g), "action_is_pad": pad,
                }

            CHECK_BATCH = algorithm_check_batch()
        """),
    ]
    L.items.append(
        Exercise(
            key="12B.1",
            title="The flow training pair",
            minutes="10",
            purpose="Make the noisy input and the velocity target for flow matching. Getting the direction or the sign wrong trains a model that moves actions the wrong way.",
            contract="""
        - `flow_training_pair(actions, noise, time)`: `actions`, `noise`: `(B, T, D)`; `time`: `(B,)` `float32` in `[0, 1]`.
        - Return `(x_t, u_t)`, both `(B, T, D)`:
          `x_t = t * noise + (1 - t) * actions` and `u_t = noise - actions`, with each row using its own `t`.
        - `t = 0` gives the clean actions; `t = 1` gives the noise.
        """,
            method="Linear interpolation path of [flow matching](https://arxiv.org/abs/2210.02747). Pinned: `x_t = time_expanded * noise + (1 - time_expanded) * actions` and `u_t = noise - actions` in `VLAFlowMatching.forward` (the opposite of the usual diffusion-literature direction).",
            pointers="Reshape `time` to `(B, 1, 1)` so that it broadcasts. The target does not depend on time.",
            hint="""
        ```python
        t = time.reshape(-1, 1, 1)
        return t * noise + (1.0 - t) * actions, noise - actions
        ```
        """,
            names=["flow_training_pair"],
            module="vla_lab.teaching.smolvla",
            stub="""
        def flow_training_pair(actions, noise, time):
            # x_t = t * noise + (1 - t) * actions; u_t = noise - actions. t = 0 is clean, t = 1 is noise.
            raise NotImplementedError("Complete Exercise 12B.1: flow_training_pair")
        """,
            check="""
        actions, noise = torch.randn(3, 6, 4), torch.randn(3, 6, 4)
        x0, u = flow_training_pair(actions, noise, torch.zeros(3))
        assert x0.shape == actions.shape and u.shape == actions.shape, "Both outputs are (B, T, D)."
        assert torch.equal(x0, actions), "At t = 0 the sample is the clean actions."
        close(u, noise - actions, "the target is noise - actions (not actions - noise)")
        x1, _ = flow_training_pair(actions, noise, torch.ones(3))
        close(x1, noise, "At t = 1 the sample is the noise")
        x, _ = flow_training_pair(actions, noise, torch.tensor([0.25, 0.5, 0.75]))
        close(x[0], 0.25 * noise[0] + 0.75 * actions[0], "row 0 uses its own t = 0.25")
        close(x[2], 0.75 * noise[2] + 0.25 * actions[2], "row 2 uses its own t = 0.75")
        # the velocity is the derivative of the path: moving along -u from t=1 reaches the actions
        t = torch.tensor([0.4, 0.4, 0.4])
        xt, ut = flow_training_pair(actions, noise, t)
        close(xt - 0.4 * ut, actions, "x_t - t * u_t is the clean action: the path is straight", atol=1e-6)
        print("Exercise 12B.1 passed")
        """,
            use_intro="Draw the straight path from real actions to noise for one action channel.",
            use="""
        real = load_act_samples(CONFIGS, DATA, "bowl_plate", "validation", count=4, chunk_size=10, image_size=64)
        actions = real["action"][:1, :, :7]
        noise = torch.randn(actions.shape, generator=torch.Generator().manual_seed(0))
        fig, axis = plt.subplots(figsize=(6.5, 3.2))
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            x_t, u_t = flow_training_pair(actions, noise, torch.tensor([t]))
            axis.plot(x_t[0, :, 0], label=f"t={t}")
        axis.set_xlabel("action step"); axis.set_ylabel("channel 0"); axis.legend(ncol=3, fontsize=8); plt.show()
        """,
        )
    )
    L.items.append(
        Exercise(
            key="12B.2",
            title="Masked velocity loss",
            minutes="10–15",
            purpose="The model works with 32 action columns and a fixed chunk length, but only 7 columns and the non-padded steps are real. The loss must ignore the rest.",
            contract="""
        - `masked_velocity_loss(prediction, target, is_pad, action_dim)`: `prediction`, `target`: `(B, T, D_max)`; `is_pad`: `(B, T)` bool, `True` = padded step; `action_dim <= D_max`.
        - Squared error `(prediction - target)**2` on the **first `action_dim` columns only**, zero on padded steps.
        - Return `sum(error * valid) / max(valid_steps * action_dim, 1)` where `valid_steps` is the number of non-padded `(row, step)` pairs in the whole batch.
        - Changing `prediction` or `target` in padded steps or in columns `>= action_dim` must change neither the loss nor the gradient. An all-padding batch returns `0`.
        """,
            method="Pinned `SmolVLAPolicy.forward`: `losses = losses[:, :, :action_dim]`, multiply by `~action_is_pad`, then `losses.sum() / (num_valid * losses.shape[-1]).clamp_min(1)`.",
            pointers="Slice the columns first. Then use a float mask of shape `(B, T, 1)`. The denominator counts valid *scalar* values.",
            hint="""
        ```python
        squared = (prediction[..., :action_dim] - target[..., :action_dim]).square()
        valid = (~is_pad.to(torch.bool)).unsqueeze(-1).to(squared.dtype)
        return (squared * valid).sum() / (valid.sum() * action_dim).clamp_min(1)
        ```
        """,
            names=["masked_velocity_loss"],
            module="vla_lab.teaching.smolvla",
            stub="""
        def masked_velocity_loss(prediction, target, is_pad, action_dim):
            # Mean squared error over real steps and the first action_dim columns.
            raise NotImplementedError("Complete Exercise 12B.2: masked_velocity_loss")
        """,
            check="""
        pred = torch.zeros(2, 3, 5, requires_grad=True)
        target = torch.ones(2, 3, 5)
        pad = torch.tensor([[False, False, True], [False, True, True]])
        loss = masked_velocity_loss(pred, target, pad, 3)
        assert loss.shape == () and torch.isfinite(loss), "Return one finite scalar."
        close(loss, torch.tensor(1.0), "three real steps, three real columns, error 1")
        loss.backward()
        assert torch.count_nonzero(pred.grad[pad]) == 0, "Padded steps need zero gradient."
        assert torch.count_nonzero(pred.grad[..., 3:]) == 0, "Padded action columns need zero gradient."
        other = pred.detach().clone()
        other[..., 3:] = 50.0
        other[pad] = 50.0
        close(masked_velocity_loss(other, target, pad, 3), loss.detach(), "changing padded steps or columns must not change the loss")
        assert masked_velocity_loss(pred.detach(), target, torch.ones_like(pad), 3) == 0, "An all-padding batch gives exactly 0."
        wide = masked_velocity_loss(torch.zeros(1, 2, 7), torch.ones(1, 2, 7), torch.zeros(1, 2, dtype=torch.bool), 7)
        close(wide, torch.tensor(1.0), "all columns real")
        print("Exercise 12B.2 passed")
        """,
            use_intro="On a real batch, build a prediction that is exact on real steps and columns and wrong only where they are padded: the loss must be exactly zero.",
            use="""
        real = load_act_samples(CONFIGS, DATA, "bowl_plate", "validation", count=16, chunk_size=10, image_size=64)
        target = pad_to_width(real["action"], 32) * 0.0 + torch.randn(16, 10, 32, generator=torch.Generator().manual_seed(1))
        prediction = target.clone()
        prediction[real["action_is_pad"]] = 0.0
        prediction[..., 7:] = 9.0
        print("padded steps in this batch:", int(real["action_is_pad"].sum()), "of", real["action_is_pad"].numel())
        print("loss:", float(masked_velocity_loss(prediction, target, real["action_is_pad"], 7)), "(0 means only real steps and columns count)")
        """,
        )
    )
    L.items.append(
        Exercise(
            key="12B.3",
            title="The loss callback with explicit noise and time",
            minutes="20",
            purpose="Connect your pair and your loss to the real model. The training loop calls this function; passing the noise and the time explicitly makes every update repeatable and lets you compare with the pinned loss.",
            contract="""
        - `flow_loss_fn(model, batch, *, noise=None, time=None) -> (scalar_loss, metrics)`. The loss keeps its autograd graph; `metrics` is a dict of scalar tensors.
        - `model` is a `SmolVLAAdapter`. `inputs = model.prepare_inputs(batch)` gives a dict with `inputs["actions"]` already zero-padded to `(B, T, D_max)`.
        - `noise`: `(B, T, D_max)` standard normal noise. If `None`, draw `torch.randn_like(actions)`. If given with another shape, raise `ValueError`.
        - `time`: `(B,)` float in `(0, 1)`. If `None`, use `model.sample_time(B, device)` (the pinned Beta prior).
        - `x_t, u_t = flow_training_pair(actions, noise, time)`; `v = model.velocity(inputs, x_t, time)` (shape `(B, T, D_max)`); `loss = masked_velocity_loss(v, u_t, batch["action_is_pad"], model.action_dim)`.
        - Return `loss, {"flow_mse": loss}`.
        """,
            method="Pinned `SmolVLAPolicy.forward` / `VLAFlowMatching.forward`: noise and time may be passed in; otherwise noise is standard normal and time follows the Beta prior; the loss is masked as in Exercise 12B.2. The velocity adapter (`SmolVLAAdapter.velocity`) re-implements the pinned forward pass up to the velocity so that this loss sits on top of the *real* expert.",
            pointers="Get the inputs, then fill in the missing noise and time, then the three calls. Do not detach anything.",
            hint="""
        ```python
        inputs = model.prepare_inputs(batch)
        actions = inputs["actions"]
        if noise is None:
            noise = torch.randn_like(actions)
        if noise.shape != actions.shape:
            raise ValueError(f"noise must have the padded action shape {tuple(actions.shape)}")
        if time is None:
            time = model.sample_time(actions.shape[0], actions.device)
        x_t, target = flow_training_pair(actions, noise, time)
        velocity = model.velocity(inputs, x_t, time)
        loss = masked_velocity_loss(velocity, target, batch["action_is_pad"], model.action_dim)
        return loss, {"flow_mse": loss}
        ```
        """,
            names=["flow_loss_fn"],
            module="vla_lab.teaching.smolvla",
            stub="""
        def flow_loss_fn(model, batch, *, noise=None, time=None):
            # Use model.prepare_inputs, flow_training_pair, model.velocity and masked_velocity_loss. Keep the graph.
            raise NotImplementedError("Complete Exercise 12B.3: flow_loss_fn")
        """,
            check="""
        policy = make_tiny_smolvla(seed=1)
        adapter = SmolVLAAdapter(policy)
        noise, time = torch.randn(2, 10, 16), torch.tensor([0.3, 0.6])
        expected, _ = policy(dict(CHECK_BATCH), noise=noise, time=time)           # the pinned loss
        loss, metrics = flow_loss_fn(adapter, CHECK_BATCH, noise=noise, time=time)
        assert loss.shape == () and "flow_mse" in metrics, "Return (scalar loss, metrics with flow_mse)."
        close(loss, expected, "your loss differs from the pinned SmolVLA loss for the same noise and time", atol=1e-6)
        again, _ = flow_loss_fn(adapter, CHECK_BATCH, noise=noise, time=time)
        close(again, loss.detach(), "explicit noise and time must make the loss repeatable", atol=1e-7)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.model.vlm_with_expert.lm_expert.parameters()), "The loss must reach the action expert (keep the graph)."
        assert all(p.grad is None for p in policy.model.vlm_with_expert.vlm.parameters()), "The frozen VLM must get no gradient."
        try:
            flow_loss_fn(adapter, CHECK_BATCH, noise=torch.randn(2, 10, 7), time=time)
        except ValueError:
            pass
        else:
            raise AssertionError("Noise with the wrong shape must raise ValueError.")
        drawn, _ = flow_loss_fn(adapter, CHECK_BATCH)                             # no noise, no time: drawn inside
        assert torch.isfinite(drawn), "Draw noise and time when they are not given."
        all_pad = dict(CHECK_BATCH, action_is_pad=torch.ones(2, 10, dtype=torch.bool))
        assert flow_loss_fn(adapter, all_pad, noise=noise, time=time)[0] == 0, "An all-padding batch has zero loss."
        print("Exercise 12B.3 passed")
        """,
            use_intro="Update the expert with the training unit's `optimizer_update`, which calls your callback. The frozen VLM must stay exactly where it is, and the loss on this batch should fall.",
            use="""
        policy = make_tiny_smolvla(seed=1)
        for parameter in policy.parameters():
            parameter.requires_grad_(True)
        params = select_trainable_parameters(policy)                               # Unit 12A (package version)
        adapter = SmolVLAAdapter(policy, train_mode_fn=train_mode_with_frozen_vlm)
        optimizer = torch.optim.AdamW(params, lr=1e-3)
        vlm_before = {n: p.detach().clone() for n, p in policy.model.vlm_with_expert.vlm.named_parameters()}
        noise, time = torch.randn(2, 10, 16, generator=torch.Generator().manual_seed(0)), torch.tensor([0.4, 0.7])
        history = [optimizer_update(adapter, CHECK_BATCH, optimizer, flow_loss_fn, noise=noise, time=time)["loss"] for _ in range(20)]
        print("loss, first and last update:", round(history[0], 4), round(history[-1], 4))
        print("VLM unchanged:", all(torch.equal(vlm_before[n], p) for n, p in policy.model.vlm_with_expert.vlm.named_parameters()))
        """,
        )
    )
    L.items.append(
        Exercise(
            key="12B.4",
            title="The Euler sampler from noise to actions",
            minutes="15",
            purpose="Turn the learned velocity into actions by integrating from `t = 1` to `t = 0`. Time runs *backwards*; the step is negative.",
            contract="""
        - `euler_sample(velocity_fn, noise, num_steps)`: `noise (B, T, D)`; `velocity_fn(x_t, t) -> v` where `t` is a **`float32` tensor `(B,)`** filled with the current time and `v` has the shape of `x_t`.
        - `dt = -1.0 / num_steps`. Start at `x = noise`. For `step = 0 … num_steps - 1`: `t = 1.0 + step * dt`; `x = x + dt * velocity_fn(x, t)`. The velocity is therefore evaluated at `t = 1, 1 - 1/n, …, 1/n` (never at 0).
        - Return `x` `(B, T, D)`. No randomness inside.
        """,
            method="Forward Euler for the ODE `dx/dt = v` from `t = 1` to `t = 0`; pinned `euler_integrate` in `flow_matching.py` (`dt = -1/num_steps`, `time = 1.0 + step * dt`). If `v = noise - actions` exactly, one Euler step already lands on the actions (straight path).",
            pointers="A for-loop with one update line. Build the time tensor with `torch.full((batch,), t, dtype=torch.float32, device=x.device)`.",
            hint="""
        ```python
        dt = -1.0 / num_steps
        x = noise
        for step in range(num_steps):
            t = torch.full((x.shape[0],), 1.0 + step * dt, dtype=torch.float32, device=x.device)
            x = x + dt * velocity_fn(x, t)
        return x
        ```
        """,
            names=["euler_sample"],
            module="vla_lab.teaching.smolvla",
            stub="""
        def euler_sample(velocity_fn, noise, num_steps):
            # Integrate from t = 1 to t = 0 with dt = -1/num_steps: x = x + dt * velocity_fn(x, t).
            raise NotImplementedError("Complete Exercise 12B.4: euler_sample")
        """,
            check="""
        noise, field = torch.randn(2, 6, 4), torch.randn(2, 6, 4)
        seen = []

        def velocity(x, t):
            seen.append(t.clone())
            return field + 0.1 * x

        mine = euler_sample(velocity, noise, 5)
        assert [round(float(t[0]), 6) for t in seen] == [1.0, 0.8, 0.6, 0.4, 0.2], "Times must run 1, 0.8, ..., 0.2 (never 0)."
        assert all(t.dtype == torch.float32 and t.shape == (2,) for t in seen), "t must be a float32 tensor of shape (B,)."
        close(mine, euler_integrate(lambda x, t: field + 0.1 * x, noise, 5), "differs from the pinned euler_integrate", atol=1e-6)
        actions = torch.randn(2, 6, 4)
        for steps in (1, 3, 10):
            close(euler_sample(lambda x, t: noise - actions, noise, steps), actions, f"the straight-line field is exact in {steps} step(s)", atol=1e-5)
        assert torch.equal(euler_sample(velocity, noise, 4), euler_sample(velocity, noise, 4)), "No randomness inside the sampler."
        print("Exercise 12B.4 passed")
        """,
            use_intro="Sample actions from the small real model with your sampler through the supplied policy hook, and compare with the pinned `sample_actions` for the same noise. The weights are random, so only the *agreement* matters.",
            use="""
        policy = make_tiny_smolvla(seed=1).eval()
        adapter = SmolVLAAdapter(policy)
        inputs = adapter.prepare_inputs(CHECK_BATCH)
        noise = torch.randn(2, 10, 16, generator=torch.Generator().manual_seed(0))
        mine = adapter.sample_chunk(inputs, euler_sample, noise, 4)                  # your sampler, the real prefix cache and expert
        with torch.no_grad():
            pinned = policy.model.sample_actions(inputs["images"], inputs["img_masks"], inputs["lang_tokens"], inputs["lang_masks"],
                                                 inputs["state"], noise=noise)[:, :, :7]
        print("chunk shape:", tuple(mine.shape), "| largest difference from the pinned sampler:", float((mine - pinned).abs().max()))
        wrapper = CallbackSmolVLAPolicy(policy, euler_sample, num_steps=4, seed=1)    # a policy object for the simulator loop
        observation = {k: v for k, v in CHECK_BATCH.items() if k not in ("action", "action_is_pad")}
        actions = [wrapper.select_action(observation) for _ in range(6)]
        print("actions per call:", tuple(actions[0].shape), "| chunks sampled for 6 calls:", wrapper.chunks_made, "(n_action_steps = 5)")
        """,
        )
    )
    return L


def lesson_12b_tail(L: Lesson) -> Lesson:
    L.items += [
        Md(r"""
        ## Experiment — time prior and Euler steps (30–60 min)

        **Question.** SmolVLA trains with a Beta time prior that favors noisy times and samples with 10 Euler steps. With a small real SmolVLA trained by your loss, how do the training time prior and the number of Euler steps change the executed actions and the time per chunk?

        **Setup (real data, small real model).** 128 training and 64 validation windows from the frozen `bowl_plate` episodes (64×64 frames, standardized state, 10 standardized actions, padding at episode ends, the instruction `put the bowl on the plate`). The expert and projections are trained for 300 updates with **your** `flow_loss_fn` (through `optimizer_update` with explicit noise and time) after your Unit 12A parameter selection; the sampler is **your** `euler_sample` through the adapter's prefix cache. The VLM has random weights: this is a mechanics experiment, not a robot policy.

        **Fixed for every trial.** Weight seed `1`; batch/noise seed `1`; **300 updates**, batch 16, AdamW `lr=1e-3`; validation initial noise seed `11` (the same noise for every step count); first 5 actions of the chunk are the "executed" ones (`n_action_steps = 5`).

        **Allowed changes (per trial).** `time_prior`: `beta` (pinned: `0.001 + 0.999 · Beta(1.5, 1)`) or `uniform` (`0.001 + 0.998 · U(0, 1)`) — this retrains the model. `steps`: Euler steps at inference, any of `1, 2, 5, 10, 20, 50`.

        **Measures.** `l1_to_demo`: masked L1 between the executed actions and the demonstrated first 5 actions. `l1_to_50`: L1 between the executed actions and those of the 50-step sample of the same model and noise. `ms_batch1`: milliseconds to sample one chunk at batch 1 (noisy; compare within one session).

        **Prediction first.** In the next cell: which `time_prior` will be better, and where will extra Euler steps stop helping?

        **Keep failed trials.** Every trial is appended to `TRIALS`. No improvement is a valid result.
        """),
        Code('PREDICTION = "TODO: replace this text with your prediction."\nprint(PREDICTION)'),
        Code("# Supplied: trial table helper\n" + tidy(EXPERIMENT_TABLE_HELPER)),
        Code(r"""
            # Supplied: real windows for the small model
            TRAIN = load_act_samples(CONFIGS, DATA, "bowl_plate", "train", count=128, chunk_size=10, image_size=64)
            VALID = load_act_samples(CONFIGS, DATA, "bowl_plate", "validation", count=64, chunk_size=10, image_size=64)
            print("train windows:", len(TRAIN["action"]), "| validation windows:", len(VALID["action"]))
        """),
        Code(r'''
            # Supplied: harness. Your loss callback and your sampler are in the real path.
            STEPS, BATCH, LR, SEED_WEIGHTS, SEED_BATCH, SEED_NOISE, N_ACT = 300, 16, 1e-3, 1, 1, 11, 5
            BASELINE = {"time_prior": "beta", "steps": 10}
            TRIALS, TRAINED, REFERENCE_CHUNKS = [], {}, {}


            def input_fn(adapter, raw):
                observation = {key: value for key, value in raw.items() if key.startswith("observation.")}
                inputs = prepare_vla_inputs(observation, PROMPT, TOKENIZER, adapter.policy.config)
                if "action" in raw:
                    inputs["actions"] = pad_to_width(raw["action"], adapter.policy.config.max_action_dim)
                return inputs


            def train_model(time_prior):
                """Train once per time prior (cached): your flow loss, explicit noise and time, seeded."""
                if time_prior in TRAINED:
                    return TRAINED[time_prior]
                policy = make_tiny_smolvla(seed=SEED_WEIGHTS)
                for parameter in policy.parameters():
                    parameter.requires_grad_(True)
                params = select_trainable_parameters(policy)
                adapter = SmolVLAAdapter(policy, train_mode_fn=train_mode_with_frozen_vlm, input_fn=input_fn)
                optimizer = torch.optim.AdamW(params, lr=LR)
                torch.manual_seed(SEED_BATCH)
                for _ in range(STEPS):
                    pick = torch.randint(0, len(TRAIN["action"]), (BATCH,))
                    batch = {key: value[pick] for key, value in TRAIN.items()}
                    noise = torch.randn(BATCH, 10, policy.config.max_action_dim)
                    if time_prior == "beta":
                        time_ = 0.001 + 0.999 * torch.distributions.Beta(1.5, 1.0).sample((BATCH,))
                    else:
                        time_ = 0.001 + 0.998 * torch.rand(BATCH)
                    optimizer_update(adapter, batch, optimizer, flow_loss_fn, noise=noise, time=time_)
                TRAINED[time_prior] = adapter.eval()
                return TRAINED[time_prior]


            def sample(adapter, batch, steps):
                inputs = adapter.prepare_inputs(batch)
                noise = torch.randn(len(batch["action"]), 10, adapter.policy.config.max_action_dim, generator=torch.Generator().manual_seed(SEED_NOISE))
                started = time.perf_counter()
                chunk = adapter.sample_chunk(inputs, euler_sample, noise, steps)             # YOUR sampler
                return chunk, time.perf_counter() - started


            def run_trial(config):
                adapter = train_model(config["time_prior"])
                with torch.no_grad():
                    if config["time_prior"] not in REFERENCE_CHUNKS:
                        REFERENCE_CHUNKS[config["time_prior"]] = sample(adapter, VALID, 50)[0][:, :N_ACT]
                    chunk, _ = sample(adapter, VALID, config["steps"])
                    executed = chunk[:, :N_ACT]
                    one = {key: value[:1] for key, value in VALID.items()}
                    timings = sorted(sample(adapter, one, config["steps"])[1] for _ in range(5))
                return {
                    "l1_to_demo": masked_action_l1(executed, VALID["action"][:, :N_ACT], VALID["action_is_pad"][:, :N_ACT]).item(),
                    "l1_to_50": (executed - REFERENCE_CHUNKS[config["time_prior"]]).abs().mean().item(),
                    "ms_batch1": 1000 * timings[2],
                }


            def record(config):
                config = {**BASELINE, **config}
                try:
                    TRIALS.append({"config": config, "status": "ok", **run_trial(config)})
                except Exception as error:  # a failed trial is evidence too
                    TRIALS.append({"config": config, "status": f"failed: {type(error).__name__}: {error}"})


            record(BASELINE)
            COLUMNS = ["l1_to_demo", "l1_to_50", "ms_batch1"]
            show_trials(TRIALS, COLUMNS);
        '''),
        Md(
            "**Your trials.** Add at least three trials. Each setting uses a fixed 300-update budget."
        ),
        Pair(
            learner="""
            # Implement: choose your trials.
            MY_TRIALS = [
                # {"steps": 2},
            ]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
            reference="""
            # Reference choice of trials: one change per trial.
            MY_TRIALS = [{"steps": 1}, {"steps": 2}, {"steps": 5}, {"steps": 20}, {"steps": 50}, {"time_prior": "uniform"}]
            for trial in MY_TRIALS:
                record(trial)
            show_trials(TRIALS, COLUMNS);
            """,
        ),
        Code(r"""
            # Check: the protocol was followed.
            assert TRIALS[0]["config"] == BASELINE, "The first trial must be the baseline."
            assert len(TRIALS) >= 4, "Run the baseline and at least three trials of your own."
            assert all(set(t["config"]) == set(BASELINE) for t in TRIALS), "Only time_prior and steps may change."
            assert all(t["config"]["time_prior"] in {"beta", "uniform"} and t["config"]["steps"] in (1, 2, 5, 10, 20, 50) for t in TRIALS)
            ok = [t for t in TRIALS if t["status"] == "ok"]
            assert all(torch.isfinite(torch.tensor([t["l1_to_demo"], t["l1_to_50"], t["ms_batch1"]])).all() for t in ok), "Measures must be finite."
            print(f"{len(TRIALS)} trials kept ({len(ok)} ok). Budget: {STEPS} updates, batch {BATCH}, seeds {SEED_WEIGHTS}/{SEED_BATCH}/{SEED_NOISE}.")
        """),
        Code(r"""
            # Use your function: accuracy against time for every trial
            ok = [t for t in TRIALS if t["status"] == "ok"]
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
            for t in ok:
                label = f"{t['config']['time_prior'][:1]}{t['config']['steps']}"
                axes[0].scatter(t["ms_batch1"], t["l1_to_demo"]); axes[0].annotate(label, (t["ms_batch1"], t["l1_to_demo"]), fontsize=8)
                axes[1].scatter(t["ms_batch1"], t["l1_to_50"]); axes[1].annotate(label, (t["ms_batch1"], t["l1_to_50"]), fontsize=8)
            axes[0].set_xlabel("ms per chunk (batch 1)"); axes[0].set_ylabel("l1_to_demo")
            axes[1].set_xlabel("ms per chunk (batch 1)"); axes[1].set_ylabel("l1_to_50")
            plt.tight_layout(); plt.show()
        """),
        Md("""
        **Explain (2–4 sentences, edit this cell).** Which time prior and step count would you pick, and what is the cost? Flow matching has no noise schedule and no random step noise: what does that change about repeating a sample? What can you *not* conclude from a tiny model with random VLM weights?

        *Optional extension:* repeat the baseline and one other trial with another `SEED_NOISE`; keep all rows.
        """),
        Md("""
        ## Opt-in A: your flow loss on the prepared learner at real size

        This loads the **prepared learner** (450M parameters, `smolvla-c2-5000-development`) from local files, builds one real batch with the real pipeline (two 256×256 frames, state, the instruction, 50 actions), and compares **your** `flow_loss_fn` with the pinned `policy(batch, noise, time)` for the same noise and time. It then makes `REAL_UPDATES` AdamW update(s) with your loss on the expert only (the VLM must not move) and prints the loss before and after. One or two updates are an integration check, not training evidence; nothing is saved. The code selects CUDA when available and otherwise uses CPU. It loads one model. Set `RUN_REAL_FLOW_LOSS = True` to run it.
        """),
        Code(r"""
            RUN_REAL_FLOW_LOSS = False
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            REAL_UPDATES = 1

            if RUN_REAL_FLOW_LOSS:
                from torch.utils.data import default_collate

                learner, preprocessor, _ = load_prepared_smolvla("learner", DATA, device=DEVICE)
                dataset = open_libero_dataset(CONFIGS, DATA, "bowl_plate", "train", {"action": [i / 10 for i in range(learner.config.chunk_size)]}, episodes=[379])
                batch = preprocessor(default_collate([dataset[7]]))
                adapter = SmolVLAAdapter(learner)
                noise = torch.randn(1, learner.config.chunk_size, learner.config.max_action_dim, generator=torch.Generator().manual_seed(0)).to(DEVICE)
                time_ = torch.tensor([0.6], device=DEVICE)
                with torch.no_grad():
                    pinned, _ = learner(dict(batch), noise=noise, time=time_)
                    yours, _ = flow_loss_fn(adapter, dict(batch), noise=noise, time=time_)
                print("pinned loss:", float(pinned), "| your loss:", float(yours))
                assert abs(float(pinned) - float(yours)) < 1e-4, "Your loss must equal the pinned loss for the same noise and time."
                for parameter in learner.parameters():
                    parameter.requires_grad_(True)
                params = select_trainable_parameters(learner)
                adapter = SmolVLAAdapter(learner, train_mode_fn=train_mode_with_frozen_vlm)
                optimizer = torch.optim.AdamW(params, lr=1e-5)
                vlm_probe = next(p for n, p in learner.named_parameters() if n.startswith("model.vlm_with_expert.vlm."))
                head_probe = learner.model.action_out_proj.weight
                before = (vlm_probe.detach().clone(), head_probe.detach().clone())
                history = [optimizer_update(adapter, dict(batch), optimizer, flow_loss_fn, noise=noise, time=time_)["loss"] for _ in range(REAL_UPDATES)]
                print("loss before the updates:", float(pinned), "| losses during the update(s):", [round(h, 5) for h in history])
                print("VLM weight unchanged:", torch.equal(before[0], vlm_probe), "| action head changed:", not torch.equal(before[1], head_probe))
            else:
                print("Real flow-loss check is off. Set RUN_REAL_FLOW_LOSS=True to run it (loads one 450M model).")
        """),
        Md("""
        ## Opt-in B: your sampler in closed loop on development states 10–14

        This runs a prepared policy with **your** `euler_sample` (through `CallbackSmolVLAPolicy`, which uses the pinned vision-language prefix and action expert but not the pinned sampler) in the LIBERO simulator on the *methods* development states `10–14` only (states `0–9` are for exploration, `15–19` for the intervention unit; the final states `20–49` are never used here). The reset seed is `20260930 + state`, the policy noise seed is the same, and every attempt, including failures and errors, is kept in `ATTEMPTS`. The instruction is the task's own text. The controller runs at 20 Hz and the demonstrations were recorded at 10 Hz; the policy is **not** resampled. This uses one 450M model and a bounded 300-step episode. Set `RUN_DEV_STATE_ROLLOUTS = True` to run it. Saved results of the same policies, produced earlier by the project, are printed for orientation and are **not** produced by this notebook.
        """),
        Code(r"""
            RUN_DEV_STATE_ROLLOUTS = False
            ROLLOUT_POLICY = "learner"             # "learner" (trained in this project) or "published" (smolvla-libero)
            DEV_STATES = (10, 11, 12, 13, 14)      # narrow this list to run fewer episodes
            ROLLOUT_STEPS = (10, 3)                # Euler steps to compare (10 is the pinned default)
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            MAX_STEPS = 300
            RESET_BASE_SEED = 20260930
            ATTEMPTS = []

            if RUN_DEV_STATE_ROLLOUTS:
                import pandas as pd
                from vla_lab.sim.episode import open_libero_task, run_policy_episode

                assert all(10 <= s <= 14 for s in DEV_STATES), "Use development states 10-14 only in this unit."
                task_text = yaml.safe_load((CONFIGS / "data" / "episodes.yaml").read_text())["tasks"]["bowl_plate"]["task_name"]
                policy, preprocessor, postprocessor = load_prepared_smolvla(ROLLOUT_POLICY, DATA, device=DEVICE, n_action_steps=10)
                for steps in ROLLOUT_STEPS:
                    for state_id in DEV_STATES:
                        wrapper = CallbackSmolVLAPolicy(policy, euler_sample, num_steps=steps, seed=RESET_BASE_SEED + state_id)
                        environment = None
                        try:
                            environment = open_libero_task("bowl_plate", state_id, CONFIGS, DATA)
                            result = run_policy_episode(environment, wrapper, preprocessor, postprocessor, "bowl_plate", state_id, CONFIGS,
                                                        partition="development", max_steps=MAX_STEPS, task_prompt=task_text, reset_seed=RESET_BASE_SEED + state_id)
                            ATTEMPTS.append({"steps": steps, "state": state_id, "success": result.success, "env_steps": result.steps, "ended": result.termination_reason,
                                             "chunks": wrapper.chunks_made, "mean_sampler_ms": 1000 * sum(wrapper.sample_seconds) / max(len(wrapper.sample_seconds), 1)})
                        except Exception as error:  # keep failed attempts
                            ATTEMPTS.append({"steps": steps, "state": state_id, "success": False, "ended": f"error: {type(error).__name__}: {error}"})
                        finally:
                            if environment is not None:
                                try:
                                    environment.close()
                                except Exception as error:
                                    ATTEMPTS.append({"steps": steps, "state": state_id, "success": False,
                                                     "ended": f"close error: {type(error).__name__}: {error}"})
                table = pd.DataFrame(ATTEMPTS)
                print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
                print(table.groupby("steps")["success"].agg(["sum", "count"]).rename(columns={"sum": "successes", "count": "attempts"}))
                saved = yaml.safe_load((WORKSPACE / "manifests/smolvla_reference.yaml").read_text())["published_reference"]["selected_result"]
                print("Saved published-policy result (development states 0-19, 10 executed actions; NOT produced here):", saved["successes"], "of", saved["episodes"])
            else:
                print("Closed-loop rollouts are off. Set RUN_DEV_STATE_ROLLOUTS=True to run", len(DEV_STATES) * len(ROLLOUT_STEPS), "episodes (one 450M model).")
        """),
        Md("""
        ## Opt-in C: adapt the prepared learner and test its sampler

        This compares two learning rates from the same prepared learner weights. Each setting uses the same two training examples, batch size 1, five updates, noise, and time values. The Unit 12B `flow_loss_fn` updates the selected action parameters through `optimizer_update`; the VLM stays frozen. The code measures held-out loss before and after training. It then runs your Euler sampler for at most 40 simulator steps on development state 10. Every training and simulator failure stays in `REAL_ADAPTATION_RESULTS`. The code loads one model at a time and writes no weights. It uses local files only. Set `RUN_REAL_ADAPTATION = True` to run these two bounded trials.
        """),
        Code(r"""
            RUN_REAL_ADAPTATION = False
            ADAPTATION_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            ADAPTATION_LRS = (5e-6, 1e-5)
            ADAPTATION_STEPS = 5
            ADAPTATION_SEED = 19
            REAL_ADAPTATION_RESULTS = []

            if RUN_REAL_ADAPTATION:
                import hashlib
                from torch.utils.data import default_collate
                from vla_lab.teaching.act_model import open_libero_dataset
                from vla_lab.sim.episode import open_libero_task, run_policy_episode

                train_indices, validation_indices = (3, 40), (7, 12)
                timestamps = {"action": [i / 10 for i in range(50)]}
                setup_error = None
                try:
                    train_data = open_libero_dataset(CONFIGS, DATA, "bowl_plate", "train", timestamps, episodes=[379])
                    validation_data = open_libero_dataset(CONFIGS, DATA, "bowl_plate", "validation", timestamps)
                except Exception as error:
                    train_data = validation_data = None
                    setup_error = f"dataset setup failed: {type(error).__name__}: {error}"
                checkpoint_file = DATA / "runs/smolvla-c2-5000-development/checkpoint-005000/model.safetensors"
                checkpoint_hash = None
                try:
                    if not checkpoint_file.is_file():
                        raise FileNotFoundError(f"Prepared SmolVLA checkpoint missing: {checkpoint_file}")
                    file_digest = hashlib.sha256()
                    with checkpoint_file.open("rb") as checkpoint_stream:
                        for block in iter(lambda: checkpoint_stream.read(1024 * 1024), b""):
                            file_digest.update(block)
                    checkpoint_hash = file_digest.hexdigest()
                except Exception as error:
                    setup_error = setup_error or f"checkpoint setup failed: {type(error).__name__}: {error}"

                def module_hash(module):
                    digest = hashlib.sha256()
                    for name, parameter in module.named_parameters():
                        digest.update(name.encode())
                        value = parameter.detach().to(device="cpu", copy=True).contiguous()
                        digest.update(value.view(torch.uint8).numpy().tobytes())
                    return digest.hexdigest()

                previous_initial_head = None
                for trial_index, learning_rate in enumerate(ADAPTATION_LRS):
                    result = {"lr": learning_rate, "updates": ADAPTATION_STEPS, "checkpoint_sha256": checkpoint_hash,
                              "status": "failed", "rollout": {"state": 10, "max_steps": 40, "status": "not run"}}
                    policy = adapter = environment = None
                    parameters = optimizer = preprocessor = postprocessor = training_batches = validation_batch = None
                    train_noises = train_times = validation_noise = validation_time = None
                    noise_generator = head_before = initial_head = frozen_vlm = wrapper = batch = None
                    before_loss = after_loss = vlm_hash_before = vlm_hash_after = episode = task_text = rollout = None
                    try:
                        if setup_error:
                            raise RuntimeError(setup_error)
                        policy, preprocessor, postprocessor = load_prepared_smolvla("learner", DATA, device=ADAPTATION_DEVICE, n_action_steps=10)
                        torch.manual_seed(ADAPTATION_SEED)
                        initial_head = policy.model.action_out_proj.weight.detach().clone()
                        if previous_initial_head is None:
                            previous_initial_head = initial_head.detach().cpu().clone()
                        else:
                            assert torch.equal(initial_head.cpu(), previous_initial_head), "Every learning-rate trial must start from the same learner weights."
                        for parameter in policy.parameters():
                            parameter.requires_grad_(True)
                        parameters = select_trainable_parameters(policy)
                        adapter = SmolVLAAdapter(policy, train_mode_fn=train_mode_with_frozen_vlm)
                        training_batches = [preprocessor(default_collate([train_data[index]])) for index in train_indices]
                        validation_batch = preprocessor(default_collate([validation_data[index] for index in validation_indices]))
                        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
                        noise_generator = torch.Generator(device=ADAPTATION_DEVICE).manual_seed(ADAPTATION_SEED)
                        train_noises = [torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device=ADAPTATION_DEVICE,
                                                    generator=noise_generator) for _ in range(ADAPTATION_STEPS)]
                        train_times = [torch.tensor([value], device=ADAPTATION_DEVICE)
                                       for value in (0.25, 0.5, 0.75, 0.4, 0.6)]
                        validation_noise = torch.randn(len(validation_indices), policy.config.chunk_size, policy.config.max_action_dim,
                                                       device=ADAPTATION_DEVICE, generator=noise_generator)
                        validation_time = torch.full((len(validation_indices),), 0.5, device=ADAPTATION_DEVICE)
                        adapter.eval()
                        frozen_vlm = policy.model.vlm_with_expert.vlm
                        vlm_hash_before = module_hash(frozen_vlm)
                        with torch.no_grad():
                            before_loss, _ = flow_loss_fn(adapter, validation_batch, noise=validation_noise, time=validation_time)
                        head_before = policy.model.action_out_proj.weight.detach().clone()
                        for step in range(ADAPTATION_STEPS):
                            batch = training_batches[step % len(training_batches)]
                            optimizer_update(adapter, batch, optimizer, flow_loss_fn, noise=train_noises[step], time=train_times[step])
                        adapter.eval()
                        with torch.no_grad():
                            after_loss, _ = flow_loss_fn(adapter, validation_batch, noise=validation_noise, time=validation_time)
                        vlm_frozen = all(not parameter.requires_grad and parameter.grad is None for parameter in frozen_vlm.parameters())
                        vlm_hash_after = module_hash(frozen_vlm)
                        expert_changed = not torch.equal(head_before, policy.model.action_out_proj.weight)
                        result.update({"status": "trained", "val_loss_before": float(before_loss), "val_loss_after": float(after_loss),
                                       "expert_changed": expert_changed, "vlm_frozen": vlm_frozen,
                                       "vlm_hash_unchanged": vlm_hash_before == vlm_hash_after})
                        assert expert_changed, "The learner update did not change the action head."
                        assert vlm_frozen, "A VLM parameter is trainable or has a gradient."
                        assert vlm_hash_before == vlm_hash_after, "The frozen VLM weights changed."

                        rollout = {"lr": learning_rate, "state": 10, "max_steps": 40, "status": "failed"}
                        try:
                            task_text = yaml.safe_load((CONFIGS / "data" / "episodes.yaml").read_text())["tasks"]["bowl_plate"]["task_name"]
                            environment = open_libero_task("bowl_plate", 10, CONFIGS, DATA)
                            wrapper = CallbackSmolVLAPolicy(policy, euler_sample, num_steps=10, seed=20260930 + 10)
                            episode = run_policy_episode(environment, wrapper, preprocessor, postprocessor, "bowl_plate", 10, CONFIGS,
                                                         partition="development", max_steps=40, task_prompt=task_text, reset_seed=20260930 + 10)
                            rollout.update({"status": "ok", "success": episode.success, "steps": episode.steps,
                                            "ended": episode.termination_reason, "chunks": wrapper.chunks_made})
                        except Exception as error:
                            rollout["error"] = f"{type(error).__name__}: {error}"
                        finally:
                            if environment is not None:
                                try:
                                    environment.close()
                                except Exception as error:
                                    rollout["close_error"] = f"{type(error).__name__}: {error}"
                        result["rollout"] = rollout
                    except Exception as error:
                        result["status"] = f"failed: {type(error).__name__}: {error}"
                    finally:
                        REAL_ADAPTATION_RESULTS.append(result)
                        del parameters, optimizer, preprocessor, postprocessor, training_batches, validation_batch
                        del train_noises, train_times, validation_noise, validation_time
                        del noise_generator, head_before, initial_head, frozen_vlm, wrapper, batch
                        del before_loss, after_loss, vlm_hash_before, vlm_hash_after, episode, task_text, rollout, environment
                        if policy is not None:
                            del policy
                        if adapter is not None:
                            del adapter
                        if ADAPTATION_DEVICE == "cuda":
                            torch.cuda.empty_cache()
                print(json.dumps(REAL_ADAPTATION_RESULTS, indent=2))
            else:
                print("Real adaptation is off. Set RUN_REAL_ADAPTATION=True to run two five-update trials and state-10 rollouts.")
        """),
        Md("""
        ## What you built

        - `flow_training_pair`, `masked_velocity_loss`, `flow_loss_fn` and `euler_sample`: the flow-matching objective and sampler of SmolVLA, equal to the pinned loss and integrator, running through the real prefix cache and action expert, with the VLM frozen.
        - A policy object whose actions come from your sampler, with fixed noise, step budgets, timing and every trial or attempt kept.

        **Next:** the comparison and runtime units reuse these functions: package versions of everything in Units 7–12 are available from `vla_lab.teaching`.
        """),
    ]
    return L


# ==== REGISTRY ====
LESSON_BUILDERS = [
    lambda: lesson_07_tail(lesson_07()),
    lambda: lesson_08a(),
    lambda: lesson_08b_tail(lesson_08b()),
    lambda: lesson_09_tail(lesson_09()),
    lambda: lesson_10a(),
    lambda: lesson_10b(),
    lambda: lesson_11_tail(lesson_11()),
    lambda: lesson_12a_tail(lesson_12a()),
    lambda: lesson_12b_tail(lesson_12b()),
]


def lessons():
    return [builder() for builder in LESSON_BUILDERS]


if __name__ == "__main__":
    raise SystemExit(main())
