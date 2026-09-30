# Lab 0: Run ACT in a simulator

In this lab, you load a supplied ACT checkpoint and run a robot task in LIBERO. You do not need to train a model first.

## Before you start

Run these commands from the repository root. Use the locked environment in [the environment guide](../../docs/environment.md). Make sure that the local data and ACT checkpoint files are present. Run `robot-lab doctor` to check them.

```bash
uv sync --locked
.venv/bin/robot-lab doctor
.venv/bin/jupyter lab tutorials/00-first-run/00-first-run.ipynb
```

Select the `Python 3 (ipykernel)` kernel that uses this repository's `.venv`. Start Jupyter from the repository root. The notebook writes runs to `.local-data/runs/`; Git ignores that directory.

The equivalent terminal command is:

```bash
.venv/bin/robot-lab demo --task bowl_plate --state-id 0 --output .local-data/runs/my-first-act-run
```

Each `--output` path must be new or empty. Use `bowl_stove` with its matching checkpoint for the second task. The task name selects the simulator task and its matching checkpoint. ACT does not read the task name as an instruction.

## What to check

The notebook shows a video and a run record. The video is a live policy rollout in LIBERO. The `success` value comes from the simulator. A run can fail even when the checkpoint has succeeded on other development states. One run does not measure the policy's general success rate.

The notebook uses development states 0 to 19. Keep final states 20 to 49 for the final evaluation. Do not use them to select a checkpoint or change this lab.

If the GPU, EGL renderer, video encoder, assets, or checkpoint is missing, run `robot-lab doctor` and use the reported failed check. Do not use a CPU fallback as a GPU result.
