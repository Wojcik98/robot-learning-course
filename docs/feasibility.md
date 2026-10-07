# Runtime requirements and limits

The reference environment uses Ubuntu 24.04, Python 3.12, and an RTX 3070 with 8 GiB of GPU memory. Use [the setup guide](initial-setup.md) to create the locked environment. Use [the experiment records](evidence.md) to inspect saved results and their limits.

The required training experiments target a 48 GB GPU and a maximum of two hours per experiment. The 8 GB local GPU can run small interface checks and policy rollouts. It is not the reference for the full training memory budget. Use the local GPU first, then use a bounded cloud run when memory is insufficient.

## Data and simulator

The pinned dataset has two 256 × 256 RGB cameras, an eight-value state, and a seven-value action. The selected tasks are bowl on plate, bowl on stove, and stove on. Each task has 50 simulator initial states. The [episode split](../configs/data/episodes.yaml) separates training and validation demonstrations. The [state split](../configs/eval/splits.yaml) reserves states 0–9 for exploration, 10–14 for method checks, 15–19 for data-intervention checks, and 20–49 for final evaluation.

LIBERO executes relative control at 20 Hz. Dataset timestamp labels record 10 Hz storage frequency; this is not proof of physical recording frequency. The training pipeline does not resample source actions. Keyboard recordings retain every 50 ms controller tick. Source filtering and overlap between demonstrations and simulator initial states are unknown. Keep storage timing, simulator execution time, and wallclock inference time distinct. Final-state results do not prove transfer to new object placements.

## Policy requirements

| Policy | Inputs and action settings | Resource record |
|---|---|---|
| ACT | Two cameras, state, 20 predicted actions in the supplied checkpoint | [ACT manifest](../manifests/act_references.yaml) |
| Learner SmolVLA | Two cameras, state, instruction, frozen pretrained VLM, trained action expert | [SmolVLA manifest](../manifests/smolvla_reference.yaml) |
| Diffusion Policy | Two observation frames, 16 predicted actions, eight executed actions | [Diffusion manifest](../manifests/diffusion_reference.yaml) |
| RTC | SmolVLA sampling with gradients for guidance | [Guidance memory record](../artifacts/reference/rtc/p0_guidance_probe.json) |

The [training resource record](../artifacts/reference/comparison/training_resources.json) gives optimizer-loop times on the reference GPU. It excludes setup, downloads, and evaluation. Memory use depends on batch size, model, renderer, and other GPU programs. Check `nvidia-smi` and run a short probe with your intended settings before a long job.

## Current limits

- Large checkpoints and full recordings are in ignored `.local-data/`. The ACT transfer archive has no public download location. SmolVLA learner, Diffusion Policy, and recovery assets also require transfer from a prepared machine.
- A long detached worker can stop when its command host exits. Test worker survival from your intended terminal or Jupyter host before unattended use. See [job controls](jobs.md).
- Cloud lifecycle checks use simulated providers. Real allocation, transfer, and stop controls need a supervised check before unattended use. See [cloud controls](cloud.md).
- Runtime studies use virtual reply delays. Measured requests exceed the selected 100 ms delay on the reference GPU. The studies do not measure real-time robot control.
- The course has no physical robot backend. Results from one GPU and fixed simulator tasks do not establish performance on another host or robot.
