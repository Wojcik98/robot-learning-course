# The 14-unit course

This course starts with a working policy playground. You then implement small functions for policy execution, training, data, and model methods. Each unit has two notebooks in the same folder:

- The learner notebook contains the full exercise prompts, checks, and places to write your answers.
- The reference notebook contains complete answers for review.

Open the learner notebook first. Save a copy before you work so your answers remain available. You do not need to run notebooks in one shared kernel. Units that reuse an earlier concept import its tested package function.

## What you need

The course assumes Python and NumPy. It explains required PyTorch operations in the notebook where they first appear. Follow the repository [setup guide](../docs/initial-setup.md) and [environment guide](../docs/environment.md). Some live policy cells need local model and simulator assets. Check them with `just doctor-live` and follow its setup guidance if they are absent.

CPU checks run by default. A real GPU experiment is clearly marked and has a bounded step count. The notebooks do not download assets or start long training automatically. Short function exercises are about 15 minutes; larger experiments take longer. These are learning targets, not measured run times. The [bounded local-resource record](../artifacts/reference/course/local_resources.json) reports course validation runs, including machine, workload, and measurement scope. It is not a policy benchmark or a learner-result claim.

## How to work through an exercise

Read the contract above the function. It states inputs, shapes, units, masks, and outputs. Implement the function in the learner notebook, run its check, then use that same function in the unit's experiment. The check must test your function; it must not call the answer from the reference notebook.

Hints are plain text first. More revealing hints can be collapsed. Use the same-unit reference notebook after you try the exercise. Later units may import a concept after you have implemented and checked it.

A code check shows that one function passed its check. A model update shows that the optimizer changed model parameters. A simulator result shows how a policy performed in the stated conditions. These are separate results. Keep the task, data partition, seeds, checkpoint and data provenance, and timing with any experiment result.

## Check course notebooks

From the repository root, run the static checks first:

```bash
.venv/bin/python tools/check_course.py tutorials --report .local-data/course-redesign/notebook-validation.json
```

Then run each notebook in a fresh local Python kernel:

```bash
.venv/bin/python tools/check_course.py tutorials --execute --report .local-data/course-redesign/notebook-validation.json
```

The report records a result for each notebook. Learner notebooks pass this check when they stop at the first unfinished exercise with `NotImplementedError`. Reference notebooks and the playground must complete their default quick path. The execution check hides CUDA and disables model and data downloads. Real GPU and robot experiments stay behind explicit flags and run separately.

The [local resource record](../artifacts/reference/course/local_resources.json) lists bounded runs that passed on the stated machine. Its wall times include kernel startup. Execution times include CPU notebook setup. The record shows code execution and resource use; it does not measure policy quality or successful learning.

## Data splits

- States 0–9: exploration.
- States 10–14: method checks.
- States 15–19: data-intervention checks.
- States 20–49: final evaluation after Unit 13 and recipe freeze.

Do not use final results to select methods or tune settings. Keep failed attempts in every denominator. Compute normalization statistics from training episodes only.

## Units

See the [course index](index.ipynb) for the exact learner and reference links. The sequence is: playground, execution, ACT training, adapters, data, CVAE, attention, ACT decoder, ACT assembly, diffusion training, denoiser blocks, U-Net, sampling, VLA inputs, flow matching, data improvement, and comparison.

The earlier lab text guides remain available. Use them for command details about runtime, RTC, and deployment.
