# End-to-end robot learning

This project is a notebook course on ACT, SmolVLA, and Diffusion Policy in simulation. Labs 0–11 are available on the prepared local machine. The revised keyboard controls passed a user retest. The paid cloud path remains open.

## Setup

- [Course and implementation plan](docs/plans/vla-tutorial-plan.md)
- [Initial setup checklist and GPU access](docs/initial-setup.md)
- [Selected local environment](docs/environment.md)
- [Measured feasibility and open gates](docs/feasibility.md)
- [Operations and rollback](docs/operations.md)
- [Checked evidence and limits](docs/evidence.md)
- [Add a lesson](docs/extending.md)

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

The [course data flow](docs/architecture.md) has validated packets, a basic action queue, exact-state evaluation, and paired reports. [Lab 1](tutorials/01-act/01-act.ipynb) traces ACT tensors, loss terms, and action chunks. [Lab 2](tutorials/02-data-evaluation/02-data-evaluation.ipynb) audits a complete demonstration and runs two fixed development states.

[Lab 3](tutorials/03-train-act/03-train-act.ipynb) trains ACT, resumes a checkpoint, and compares three saved local variants on the same 20 development states. The [reference record](artifacts/reference/act/lab3_suite.json) has every count and checkpoint hash. A long detached job was interrupted on this managed command host, so the full reference used a monitored local command. The [job guide](docs/jobs.md) records the remaining lifecycle check.

[Labs 4 and 5](tutorials/04-smolvla/04-smolvla.ipynb) run a published SmolVLA checkpoint and adapt a separate learner. [Lab 6](tutorials/06-diffusion/06-diffusion.ipynb) trains and evaluates Diffusion Policy. [Lab 7](tutorials/07-comparison/07-comparison.ipynb) reads checked final results for bowl-plate and stove-on tasks. [Lab 8](tutorials/08-data-quality/08-data-quality.ipynb) checks action timing and a bounded recovery continuation.

[Lab 9](tutorials/09-runtime/09-runtime.ipynb) tests delayed action chunks on virtual controller ticks. [Lab 10](tutorials/10-rtc/10-rtc.ipynb) checks RTC masks and guidance, then reads the [frozen final runtime report](artifacts/reference/runtime/final_modes.json). [Lab 11](tutorials/11-deployment/11-deployment.ipynb) checks provenance, a reversible rollback, and a career assessment memo. The [course index](tutorials/index.ipynb) gives the lesson order and time estimates.
