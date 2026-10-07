# Course evidence and limits

This page points to the checked records for the prepared local course. Run commands from the repository root. Large checkpoints, recordings, videos, and executed notebooks stay in ignored `.local-data/`.

## What the course checks

The frozen reference scores below are records from the original reference experiments. They are not results from the current learner notebooks or proof that a new adaptation has the same score.

| Area | Checked record | Result and limit |
|---|---|---|
| Current course checks | [L40S validation record](../artifacts/reference/course/l40s_validation.json) | 599 tests passed, one was skipped, and all 45 quick notebook checks passed or stopped at the expected learner exercise. Required training, reload, and simulator paths completed on the L40S. |
| Current adapted-policy comparison | [L40S comparison](../artifacts/reference/course/l40s_comparison.json) | After 1,000 adaptation updates each, ACT succeeded on 25/30 final states, SmolVLA on 24/30, and Diffusion on 23/30. All 90 attempts had zero policy and infrastructure errors. One task, one adaptation seed, and different prior training budgets limit this result. |
| Earlier course resource checks | [local resource record](../artifacts/reference/course/local_resources.json) | Historical bounded checks report code execution and memory use. They do not establish policy quality. |
| Data and controller | [feasibility](feasibility.md), [pinned source manifest](../manifests/core.yaml) | Real AV1 frames decode. LIBERO uses 50 ms control ticks. Dataset time labels do not prove physical elapsed time. |
| Explicit control timing | [three-policy timing check](../artifacts/reference/course/control_timing.json) | ACT, Diffusion, and published SmolVLA each issued 20 actions and advanced the measured simulator clock by 1.0 s. Videos use 20 FPS and include the reset frame. This short check does not measure task success. |
| ACT reference | [ACT manifest](../manifests/act_references.yaml), [release record](../artifacts/reference/release/act_bowl_plate_200.json) | Bowl-plate ACT succeeded on 19/20 seeded development states and 30/30 frozen final states. One training seed and one simulator task do not prove transfer. |
| Learner SmolVLA | [SmolVLA manifest](../manifests/smolvla_reference.yaml) | The learner C2 checkpoint succeeded on 14/20 development states and 18/30 bowl-plate final states. The published checkpoint B has unknown training-data provenance. |
| Diffusion Policy | [Diffusion manifest](../manifests/diffusion_reference.yaml) | The selected 20-step sampler succeeded on 15/20 development states and 20/30 bowl-plate final states. |
| Three-method comparison | [frozen recipe](../configs/experiments/comparison.yaml), [paired final report](../artifacts/reference/comparison/final/three_method_report.json) | All 90 bowl-plate attempts completed. Different pretraining, processors, and training steps limit causal claims. |
| Recovery data | [recovery report](../artifacts/reference/data_quality/recovery_reference.json), [keyboard recording sample](../artifacts/reference/data_quality/keyboard_human_retest.json) | Five policy-generated episodes gave no net success gain on 15 held-out development states. The saved keyboard sample contains nonzero arm actions. No human-recovery training result exists. |
| Runtime and RTC | [frozen recipe](../configs/experiments/latency_final.yaml), [final report](../artifacts/reference/runtime/final_modes.json) | Five modes used one SmolVLA model on the same ten final states. Request times exceeded the set 100 ms virtual delay. This is a static simulator study, not a real-time robot result. |
| Local job recovery | [interruption check](../artifacts/reference/jobs/act_interrupted_restore.json) | A new process resumed ACT after `SIGTERM` and completed one live simulator step. The restored weights were close, but not bitwise equal. A new machine and a long detached worker remain untested. |

## Commands that match the notebooks

The [course index](../tutorials/index.ipynb) links the current learner and reference notebooks. Run a cell to run its stated operation. Automated checks select named experiments from cell metadata. The quick profile reports which experiments it skips; the full profile runs the selected experiment and its prerequisites without changing the notebook. Use the locked environment. For example:

```bash
.venv/bin/robot-lab doctor --live
.venv/bin/robot-lab demo --task bowl_plate --state-id 0 --output .local-data/runs/check-demo
.venv/bin/robot-lab eval --checkpoint .local-data/runs/act-bowl-plate-probe-200/checkpoint-000200 --state-id 0 --output .local-data/runs/check-eval
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests
just lint
.venv/bin/python tools/check_course.py --execute --profile quick
.venv/bin/python tools/check_course.py --execute --profile full --experiment act-data-local tutorials/03-act-data/03-act-data-reference.ipynb
```

Use [the setup guide](initial-setup.md) for Ubuntu, `uv`, `just`, data fetch, EGL, and GPU access. Use [the operations guide](operations.md) for saved jobs, checkpoint restore, and rollback. Use the [lesson guide](extending.md) to add or change a lesson.

## Resource and cost boundary

The course used the local RTX 3070 and a Brev L40S with 46068 MiB of GPU memory. The [current validation record](../artifacts/reference/course/l40s_validation.json) contains training, reload, simulator, timing, and resource checks. Each required training experiment completed within two hours. The final comparison used 1,000 adaptation updates per model: about 58 s for ACT, 89 s for Diffusion, and 275 s for SmolVLA. These times exclude preparation and evaluation.

The cloud results and source snapshot were exported to `.local-data/course-work/course-validation.tar.zst` and extracted under `.local-data/course-work/cloud-export`. The archive SHA-256 and all three final model hashes were verified. Brev then confirmed that the instance was absent. The quoted compute and storage rate was US$1.875113 per hour; the elapsed-time cost estimate is US$3.62, below the US$5 session limit. The actual invoice was not obtained. The validation record contains the archive hash, provider response, timestamps, and cost calculation. See [the cloud controls](cloud.md).

## Open limits

- The large local assets need a distribution location for learners who cannot access this workstation. No public download location exists. The cloud check used an explicitly prepared asset transfer.
- Source demonstration overlap with simulator initial states is unresolved.
- No physical robot has run the course. The remote L40S checks do not establish physical robot safety or transfer.
- Human keyboard use was checked, but the five recovery demonstrations and the training continuation came from an ACT policy, not a person.
- The final runtime matrix has ten states and one checkpoint. Do not use it as a stable rank of runtime modes.
