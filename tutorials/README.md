# The 22-lesson course

This course starts with a working policy playground and introduces all three models. You then implement small functions for policy execution, training, data, and model methods. Each lesson has two notebooks in the same folder:

- The learner notebook contains the full exercise prompts, checks, and places to write your answers.
- The reference notebook contains complete answers for review.

Open the learner notebook first. Save a copy before you work so your answers remain available. You do not need to run notebooks in one shared kernel. Lessons that reuse an earlier concept import its tested package function.

## Read for context before you implement

Start with [From demonstrations to control](concepts/control-and-learning.md). Then open the relevant [concept guide](concepts/README.md) beside your notebook: [ACT](concepts/act.md), [Diffusion Policy](concepts/diffusion-policy.md), [SmolVLA](concepts/smolvla.md), or [data and evaluation](concepts/data-and-evaluation.md). These guides explain the problem, the complete model path, and where each lesson fits. They include system diagrams and short examples.

Each notebook starts with the method's purpose, the component you will work on, and a question to explain before coding. Read that introduction before the tensor contract. Use the guide for the wider view, and return to the notebook for the exact exercise. Mermaid diagrams stay in the Markdown guides; notebook Markdown uses SVG diagrams and tables.

## What you need

The course assumes Python and NumPy. It explains required PyTorch operations in the notebook where they first appear. Follow the repository [setup guide](../docs/initial-setup.md) and [environment guide](../docs/environment.md). Some live policy cells need local model and simulator assets. Check them with `just doctor-live` and follow its setup guidance if they are absent.

Each lesson has short checks and clearly marked GPU or simulator experiments with bounded step counts. The notebooks use prepared local assets. Running an experiment cell starts its stated operation. Short function exercises are about 15 minutes; larger experiments take longer. These are learning targets, not measured run times. The [course validation record](../artifacts/reference/course/l40s_validation.json) reports course validation runs, including machine, workload, and measurement scope. It documents code execution and resource use; it is not a policy benchmark or a learner-result claim.

## How to work through an exercise

Read the contract above the function. It states inputs, shapes, units, masks, and outputs. Implement the function in the learner notebook, run its check, then use that same function in the lesson's experiment. The check must test your function; it must not call the answer from the reference notebook.

Hints are plain text first. More revealing hints can be collapsed. Use the same-lesson reference notebook after you try the exercise. Later lessons may import a concept after you have implemented and checked it.

A code check shows that one function passed its check. A model update shows that the optimizer changed model parameters. A simulator result shows how a policy performed in the stated conditions. These are separate results. Keep the task, data partition, seeds, checkpoint and data provenance, and timing with any experiment result.

## Check course notebooks

From the repository root, run the static checks first:

```bash
.venv/bin/python tools/check_course.py tutorials --report .local-data/course-runs/checks.json
```

Then check one notebook at a time with a quick profile in a fresh local Python kernel:

```bash
.venv/bin/python tools/check_course.py tutorials/06-act-attention/06-act-attention-reference.ipynb --execute --profile quick --report .local-data/course-runs/checks.json
```

The report records a result for each notebook. Learner notebooks pass this check when they stop at the first unfinished exercise with `NotImplementedError`. Reference notebooks and the playground must complete their default quick path. The quick profile skips all experiment cells. The full profile executes the marked experiment cells. The quick profile hides CUDA; both profiles disable model and data downloads. Select a named experiment with `--profile full --experiment NAME` to run it and its prerequisites. Each notebook cell runs directly, with no execution flags.

The [course validation record](../artifacts/reference/course/l40s_validation.json) lists bounded runs that passed on the stated machine. Its wall times include kernel startup. Execution times include CPU notebook setup. The record shows code execution and resource use; it does not measure policy quality or successful learning.

## Data splits

- States 0–9: exploration.
- States 10–14: method checks.
- States 15–19: data-intervention checks.
- States 20–49: final evaluation after Lesson 21 and recipe freeze.

Do not use final results to select methods or tune settings. Keep failed attempts in every denominator. Compute normalization statistics from training episodes only.

## Lessons

See the [course index](index.ipynb) for the exact learner and reference links. Lessons 1–9 cover ACT: playground, execution, data, training, CVAE, attention, decoder, assembly, and adaptation. Lessons 10–15 cover diffusion: execution, training, blocks, U-Net, sampling, and adaptation. Lessons 16–20 cover SmolVLA: execution, training, conditioning, flow, and adaptation. Lessons 21–22 cover data quality and model comparison.
