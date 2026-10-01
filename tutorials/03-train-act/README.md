# Lab 3: Train ACT and compare settings

This lab trains a task policy from public demonstrations, restores a checkpoint, and checks three declared variants on the same 20 development states. The [notebook](03-train-act.ipynb) starts a short detached training probe and then reads a prepared local reference suite. Its plots use saved training metrics and live simulator results.

Run from the repository root after Labs 0–2 and the training data download:

```bash
.venv/bin/robot-lab data fetch --scope training
.venv/bin/jupyter lab tutorials/03-train-act/03-train-act.ipynb
```

The full reference command is:

```bash
.venv/bin/python -m vla_lab.experiments.act \
  --config configs/train/act_task.yaml \
  --output .local-data/runs/my-act-suite
```

Run a new output directory each time. The suite trains 10-episode and 39-episode ACT policies with a 20-action chunk, then trains a 39-episode policy with a 10-action chunk. Each run gets the statistics of its selected training episodes. The 10 validation episodes stay fixed. All three policies run on the same development states; final states 20–49 stay closed.

The reference uses the local GPU, AV1 decoder, and EGL renderer. It does not use Brev credit. Allow several GiB of disk for the pinned data, checkpoints, and rollout videos. Plan 3–5 learner hours for the notebook, reading, and exercise. The measured local suite took about 6 minutes 21 seconds. Its variants succeeded on 11/20, 17/20, and 12/20 development states in the order above, with zero infrastructure errors. These are one-seed development results; they do not establish a stable ranking. See the [reference record](../../artifacts/reference/act/lab3_suite.json).

For a detached local run, use the job controls after the worker lifecycle passes on your host:

```bash
.venv/bin/robot-lab jobs submit \
  --config configs/train/act_task.yaml \
  --purpose act-lab3-reference --seed 20260930 -- \
  .venv/bin/python -m vla_lab.experiments.act \
  --config configs/train/act_task.yaml \
  --output '{run_dir}/experiment'
```

The job command prints a run ID. Use `robot-lab jobs status RUN_ID`, `logs RUN_ID`, and `artifacts RUN_ID` after a notebook or terminal restart. Use `cancel RUN_ID` to stop a local worker. The [job controls](../../docs/jobs.md) give the tested behavior and current host limit.
