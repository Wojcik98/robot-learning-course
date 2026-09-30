# End-to-end robot learning

This project is building a notebook course on ACT, SmolVLA, and Diffusion Policy in simulation. Lab 0 is ready to run on the prepared local machine. Later labs are in progress.

## Setup

- [Course and implementation plan](docs/plans/vla-tutorial-plan.md)
- [Initial setup checklist and GPU access](docs/initial-setup.md)
- [Selected local environment](docs/environment.md)
- [Measured feasibility and open gates](docs/feasibility.md)

## First live lab

Run these commands from this repository's root after the [environment setup](docs/environment.md):

```bash
uv sync --locked
.venv/bin/robot-lab data fetch --scope demo
.venv/bin/robot-lab references install --archive .local-data/bundles/act-references.zip
.venv/bin/robot-lab doctor --live
.venv/bin/jupyter lab tutorials/index.ipynb
```

Open [Lab 0](tutorials/00-first-run/00-first-run.ipynb) from the index. It loads a trained ACT checkpoint, runs a LIBERO-Goal task, saves a live video, and resets the task for a second run. The [Lab 0 guide](tutorials/00-first-run/README.md) gives a terminal path and an exercise.

The ACT reference checkpoints and simulator data are large files in ignored `.local-data/`. A prepared archive is at `.local-data/bundles/act-references.zip` on this machine. For another machine, copy that archive and run:

```bash
.venv/bin/robot-lab references install --archive /path/to/act-references.zip
```

The installer checks each file against [the pinned reference manifest](manifests/act_references.yaml). The archive does not contain the simulator data. The distribution location for a new learner machine is still open.

Before a training lab, run `.venv/bin/robot-lab data fetch --scope training` to add the selected data shards and camera videos. The command uses the exact Hub revisions in [the core manifest](manifests/core.yaml).

Use [local job controls](docs/jobs.md) for longer training runs. They keep a run ID, heartbeat, log, cancel command, and bounded artifact copy after a notebook closes.

[Brev cost and stop controls](docs/cloud.md) are under test. The local labs do not use cloud credit.

The [course data flow](docs/architecture.md) now has validated packets, a basic action queue, exact-state evaluation, and paired reports. [Lab 1](tutorials/01-act/01-act.ipynb) traces ACT tensors, loss terms, and action chunks. Lab 2 is in progress.
