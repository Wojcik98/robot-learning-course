# Feasibility record

**Date:** 2026-09-30
**Gate:** P0 in progress. Both ACT references meet the development-state demo gate. No course release is ready.

## Verified

- The locked project environment imports PyTorch 2.11.0+cu128, TorchCodec 0.11.0+cpu, LeRobot, MuJoCo 3.8.1, and LIBERO. A CUDA matrix operation returned finite values on the RTX 3070 through the approved host route.
- A pinned `lerobot/libero` AV1 file from each camera decoded. Both first frames have shape `3 × 256 × 256` and nonconstant pixel values. The main frame mean was 68.08 on the 0–255 scale; the wrist frame mean was 67.78.
- The data metadata declares 1,693 episodes, 273,465 frames, 40 task labels, 10 Hz timestamps, two cameras, an eight-value state, and a seven-value action. A local scan checked all 377 action/state shards against episode boundaries. It found 1,693 complete episodes.
- The selected task demonstration counts are 49 for bowl on plate, 48 for bowl on stove, and 50 for stove on. The explicit training and validation episode IDs are in [episodes.yaml](../configs/data/episodes.yaml). Training-only state and action statistics are in `configs/data/stats/`.
- The pinned LIBERO-Goal registry resolves bowl on plate to task ID 8, bowl on stove to 1, and stove on to 7. Each has 50 stored initial states. Their hashes and explicit development/final IDs are in [splits.yaml](../configs/eval/splits.yaml).
- A live EGL reset and one relative-control step passed for bowl on plate at initial state 0. Both cameras returned nonconstant `256 × 256 × 3` images. The seven-value action was accepted. The task did not report success after the no-op step.
- An ACT shape probe used one real bowl-on-plate demonstration, both cameras, a 20-action prediction chunk, and a reduced transformer size. The training loss was finite (`81.3164`); backward gradients and one selected seven-value action were finite. PyTorch reported 166 MiB peak allocated CUDA memory for this policy-only probe. The simulator did not run in that process, so this is not the combined memory gate.
- The local ACT recipe uses the real LeRobot processor for two camera images, an eight-value state, and a seven-value action. State and action use the frozen training statistics. Images remain on their decoded 0–1 scale. A saved ACT bundle reloads its weights and both processors in a new process.
- A live policy action on bowl-on-plate development state 0 passed with the simulator and model in one process. The first untrained action did not complete the task. Peak PyTorch allocation was 82 MiB for that one-step run. The simulator used the pinned local assets, rather than LIBERO's default asset cache.
- The 200-step ACT bowl-on-plate run used 39 training episodes and 3,661 frames. Batch size was eight. It took about 24 seconds on the RTX 3070. The last training loss was 1.2733. Its saved checkpoint succeeded on **19 of 20** development states. State 8 failed. A no-op baseline failed on the two development states checked.
- The 200-step ACT bowl-on-stove run used the frozen training episodes with the same recipe. It took about 25 seconds. The last training loss was 1.3356. Its saved checkpoint succeeded on **20 of 20** development states.
- Both 200-step runs saved complete ACT bundles, optimizer state, step metrics, and per-state development results. One successful state-0 video per task was encoded and checked with `ffprobe` at 3.75 seconds. Small results and SHA-256 file hashes are in [the ACT reference manifest](../manifests/act_references.yaml). The large files remain in ignored `.local-data/`.

## Open gates

- Check camera order and state/action meaning against source demonstrations and simulator frames. The processor path is now tested, but visual and motion alignment still need an independent check.
- Run SmolVLA and Diffusion Policy inference, finite-gradient, save/reload, and resource probes with one environment.
- Measure cold and warm policy latency and peak total VRAM with rendering active. PyTorch allocation alone does not include every renderer allocation.
- Give learners a reliable way to retrieve the ACT bundles on a new machine. The current reference paths are local to this workstation.
- Check the cloud account and spending controls before any paid job. No paid cloud job has started.

No final simulator state has been used for model selection. The reported ACT rates are development results for this fixed setup. They are not benchmark estimates. The source demonstration overlap with simulator initial states is not resolved. The bowl-on-stove run clipped 1,317 action values across 1,446 steps; bowl-on-plate clipped 541 values across 1,735 steps. The course must show this action limit and study its effect.
