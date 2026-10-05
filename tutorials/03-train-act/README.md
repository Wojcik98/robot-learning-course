# Lab 3: Train ACT and compare settings

This guide describes the earlier lab. Its notebook was removed. Use the [course index](../index.ipynb) for current exercises.

This lab trains a task policy from public demonstrations, restores a checkpoint, and checks three declared variants on the same 20 development states. The notebook starts a short detached training probe and then reads a prepared local reference suite. Its plots use saved training metrics and live simulator results.

Run from the repository root after Labs 0–2 and the training data download:

```bash
just data-training
```

The full reference command is:

```bash
just act-suite \
  --config configs/train/act_task.yaml \
  --output .local-data/runs/my-act-suite
```

Run a new output directory each time. The suite trains 10-episode and 39-episode ACT policies with a 20-action chunk, then trains a 39-episode policy with a 10-action chunk. Each run gets the statistics of its selected training episodes. The 10 validation episodes stay fixed. All three policies run on the same development states; final states 20–49 stay closed.

The reference uses the local GPU, AV1 decoder, and EGL renderer. It does not use Brev credit. Allow several GiB of disk for the pinned data, checkpoints, and rollout videos. Plan 3–5 learner hours for the notebook, reading, and exercise. Training the three variants took about 49 seconds. The seeded evaluations took about 287 seconds of recorded episode time. With explicit simulator reset seeds, the variants succeeded on 10/20, 18/20, and 11/20 development states in the order above, with zero infrastructure errors. These are one-seed development results; they do not establish a stable ranking. See the [reference record](../../artifacts/reference/act/lab3_suite.json).

For a detached local run, use the job controls after the worker lifecycle passes on your host:

```bash
just act-suite-job
```

The job command prints a run ID. Use `just jobs status RUN_ID`, `just jobs logs RUN_ID`, and `just jobs artifacts RUN_ID` after a notebook or terminal restart. Use `just jobs cancel RUN_ID` to stop a local worker. The [job controls](../../docs/jobs.md) give the tested behavior and current host limit.
