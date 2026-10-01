# Course evidence and limits

This page points to the checked records for the prepared local course. Run commands from the repository root. Large checkpoints, recordings, videos, and executed notebooks stay in ignored `.local-data/`.

The [Release 3 notebook record](../artifacts/reference/release/course_notebooks.json) checks the source and executed hashes for Labs 0–11. Twelve fresh kernels ran 84 code cells with zero error outputs. It also records the course index links and short local CLI checks. The run reused saved long jobs; it did not retrain every model.

A [clean local checkout check](../artifacts/reference/release/clean_checkout.json) cloned commit `66e9ac0` into a new directory on the same workstation. A normal locked `uv sync` installed the environment, the ACT archive restored all 14 checked files, the pinned public demo assets downloaded, and `doctor --live` passed 20 checks. A new ACT demo succeeded in 74 steps. An offline first install could not find the pinned TorchCodec wheel in the cache; a normal online install passed. This check did not use a new machine.

## What the course checks

| Area | Checked record | Result and limit |
|---|---|---|
| Data and controller | [feasibility](feasibility.md), [pinned source manifest](../manifests/core.yaml) | Real AV1 frames decode. LIBERO uses 50 ms control ticks. Dataset time labels do not prove physical elapsed time. |
| ACT reference | [ACT manifest](../manifests/act_references.yaml), [release record](../artifacts/reference/release/act_bowl_plate_200.json) | Bowl-plate ACT succeeded on 19/20 seeded development states and 30/30 frozen final states. One training seed and one simulator task do not prove transfer. |
| Learner SmolVLA | [SmolVLA manifest](../manifests/smolvla_reference.yaml) | The learner C2 checkpoint succeeded on 14/20 development states and 18/30 bowl-plate final states. The published checkpoint B has unknown training-data provenance. |
| Diffusion Policy | [Diffusion manifest](../manifests/diffusion_reference.yaml) | The selected 20-step sampler succeeded on 15/20 development states and 20/30 bowl-plate final states. |
| Three-method comparison | [frozen recipe](../configs/experiments/comparison.yaml), [paired final report](../artifacts/reference/comparison/final/three_method_report.json) | All 90 bowl-plate attempts completed. Different pretraining, processors, and training steps limit causal claims. |
| Recovery data | [recovery report](../artifacts/reference/data_quality/recovery_reference.json), [human keyboard retest](../artifacts/reference/data_quality/keyboard_human_retest.json) | Five policy-generated episodes gave no net success gain on 15 held-out development states. The user confirmed the revised keyboard UI works; the saved retest has arm motion. No human-recovery training result exists. |
| Runtime and RTC | [frozen recipe](../configs/experiments/latency_final.yaml), [final report](../artifacts/reference/runtime/final_modes.json) | Five modes used one SmolVLA model on the same ten final states. Request times exceeded the set 100 ms virtual delay. This is a static simulator study, not a real-time robot result. |
| Local job recovery | [interruption check](../artifacts/reference/jobs/act_interrupted_restore.json) | A new process resumed ACT after `SIGTERM` and completed one live simulator step. The restored weights were close, but not bitwise equal. A new machine and a long detached worker remain untested. |
| Release and rollback | [Lab 11 check](../artifacts/reference/deployment/lab11_notebook.json), [operations guide](operations.md) | The lab wrote records and moved a release pointer in a temporary directory. Its exercise ends with a six-question career memo. It did not deploy to a robot or a service. |

## Commands that match the notebooks

The [course index](../tutorials/index.ipynb) links Labs 0–11. Each lab guide gives a CLI command or a Python module command for its live or training path. These commands were checked on the prepared local Ubuntu host with the locked environment. In particular:

```bash
.venv/bin/robot-lab doctor --live
.venv/bin/robot-lab demo --task bowl_plate --state-id 0 --output .local-data/runs/check-demo
.venv/bin/robot-lab eval --checkpoint .local-data/runs/act-bowl-plate-probe-200/checkpoint-000200 --state-id 0 --output .local-data/runs/check-eval
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests
just lint
```

Use [the setup guide](initial-setup.md) for Ubuntu, `uv`, `just`, data fetch, EGL, and GPU access. Use [the operations guide](operations.md) for saved jobs, checkpoint restore, and rollback. The [lesson guide](extending.md) shows how Lab 11 reuses an existing policy, evaluator, config, fixture, notebook, solution, and index entry.

## Resource and cost boundary

The course used the local RTX 3070. [The measured resource table](../artifacts/reference/comparison/training_resources.json) gives optimizer-loop times for the three bowl-plate learners. It excludes setup, downloads, and evaluation. The [environment record](environment.md) gives selected download and disk measurements. No paid Brev job ran, so measured cloud spend is US$0. The user reported a US$29.94 prepaid balance with auto recharge off. A live balance, a live quote, provider stop, independent stop, budget limit, and monitoring still need checks before a paid allocation. See [the cloud gate](cloud.md).

## Open limits

- The large local assets need a distribution location for learners who cannot access this workstation. The ACT archive passed a clean local checkout, but no public download location or new-machine check exists.
- Source demonstration overlap with simulator initial states is unresolved.
- No physical robot, new GPU type, or new host has run the course.
- Human keyboard use was checked, but the five recovery demonstrations and the training continuation came from an ACT policy, not a person.
- The final runtime matrix has ten states and one checkpoint. Do not use it as a stable rank of runtime modes.
