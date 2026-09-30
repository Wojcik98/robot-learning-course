# Feasibility record

**Date:** 2026-09-30
**Gate:** P0 in progress. No course release or trained ACT reference is ready.

## Verified

- The locked project environment imports PyTorch 2.11.0+cu128, TorchCodec 0.11.0+cpu, LeRobot, MuJoCo 3.8.1, and LIBERO. A CUDA matrix operation returned finite values on the RTX 3070 through the approved host route.
- A pinned `lerobot/libero` AV1 file from each camera decoded. Both first frames have shape `3 × 256 × 256` and nonconstant pixel values. The main frame mean was 68.08 on the 0–255 scale; the wrist frame mean was 67.78.
- The data metadata declares 1,693 episodes, 273,465 frames, 40 task labels, 10 Hz timestamps, two cameras, an eight-value state, and a seven-value action. A local scan checked all 377 action/state shards against episode boundaries. It found 1,693 complete episodes.
- The selected task demonstration counts are 49 for bowl on plate, 48 for bowl on stove, and 50 for stove on. The explicit training and validation episode IDs are in [episodes.yaml](../configs/data/episodes.yaml). Training-only state and action statistics are in `configs/data/stats/`.
- The pinned LIBERO-Goal registry resolves bowl on plate to task ID 8, bowl on stove to 1, and stove on to 7. Each has 50 stored initial states. Their hashes and explicit development/final IDs are in [splits.yaml](../configs/eval/splits.yaml).
- A live EGL reset and one relative-control step passed for bowl on plate at initial state 0. Both cameras returned nonconstant `256 × 256 × 3` images. The seven-value action was accepted. The task did not report success after the no-op step.
- An ACT shape probe used one real bowl-on-plate demonstration, both cameras, a 20-action prediction chunk, and a reduced transformer size. The training loss was finite (`81.3164`); backward gradients and one selected seven-value action were finite. PyTorch reported 166 MiB peak allocated CUDA memory for this policy-only probe. The simulator did not run in that process, so this is not the combined memory gate.

## Open gates

- Check camera order and state/action transforms through the actual LeRobot policy processor. The image names and shapes match the metadata, but this check alone does not prove the complete transform.
- Run ACT save/reload and a policy rollout. Run SmolVLA and Diffusion Policy inference, finite-gradient, save/reload, and resource probes with one environment.
- Train the two ACT task references on training episodes. Select only on development states. The first lab needs a checkpoint that meets the plan's demo gate.
- Measure cold and warm policy latency, peak VRAM with rendering active, and the full reference training cost.
- Check the cloud account and spending controls before any paid job. No paid cloud job has started.

No final simulator state has been used for model selection. No policy success rate is available yet. The one simulator step is a rendering and control-path check only.
