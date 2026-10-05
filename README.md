# Robot learning course

Start with the [14-unit implementation course](tutorials/index.ipynb). It begins with a working policy playground, then asks you to implement small functions and use them in the same notebook. Each lesson has a learner notebook and a complete reference notebook in the same folder. Your saved notebook copy keeps your answers.

The course assumes Python and NumPy. Each notebook explains the PyTorch operations that it needs. CPU checks run by default. Longer training and GPU experiments are explicit steps. The notebooks do not download assets or start long jobs automatically. Bring the local assets required by an experiment before you run it; use the setup and asset notes below.

## Setup

- [Initial setup checklist and GPU access](docs/initial-setup.md)
- [Selected local environment](docs/environment.md)
- [Runtime requirements and limits](docs/feasibility.md)
- [Operations and rollback](docs/operations.md)
- [Checked evidence and limits](docs/evidence.md)
- [How to add or change a lesson](docs/extending.md)
- [Keyboard recording commands](docs/recording.md)

Run `just` at the repository root to list available recipes. Install `just` and `uv` with the [initial setup checklist](docs/initial-setup.md). `just jupyter` starts one Jupyter server and opens the course index. To open a notebook directly, run `just jupyter tutorials/01-playground/01-playground.ipynb`. The server listens on `127.0.0.1`. If it is already running, open another notebook in its file browser.

Run `just doctor` for local checks. Run `just doctor-live` for live-rendering checks. Use `just test` and `just lint` to check the code. Run `.venv/bin/python tools/check_course.py tutorials` for static notebook checks. These checks do not run the course's real GPU experiments.

## Assets, data, and compute

The first live playground needs the local policy and simulator assets described in [initial setup](docs/initial-setup.md). For current asset state, run `just doctor-live`; follow the printed setup instructions when an asset is absent. The course does not fetch assets automatically.

Training experiments use local data and can take longer than the function checks. Start them only from their marked cells or commands. Use [local job controls](docs/jobs.md) for longer training runs; they provide a run ID, heartbeat, log, cancel command, and bounded artifact copy after a notebook closes. Cloud tasks are separate and use the controls in [the cloud guide](docs/cloud.md).

Keep each result tied to its data partition, task, seed, checkpoint provenance, and timing settings. Exploration results use states 0–9. States 10–14 are for method checks. States 15–19 are for data-intervention checks. Final states 20–49 are for frozen recipes after Unit 13. Do not use final results to select a method or tune a setting. A short training run or successful check does not prove that a trained policy succeeds in the simulator.
