# Lab 6 exercise guide

The fixed recipe uses two observation frames, a 16-step action horizon, and eight executed actions. It uses 10 frames per second. The observation offsets are `[-0.1, 0.0]` seconds. The action offsets start one frame before the current frame. The executed actions are indices 1–8 in the predicted window.

The training split contains 39 episodes and 3,388 eligible windows. The loader drops the last seven frames of each episode as current frames. The final source row of an episode has padded actions. The policy marks those values, and the training loss masks them. This prevents repeated end frames from acting like real future actions.

The images use the pinned ResNet-18 ImageNet weights and ImageNet channel mean and standard deviation. The state and action normalizers use extrema from the frozen training episodes. The statistics file and checkpoint manifest carry the training split hash. Development data does not contribute to those values.

The trained reference uses 2,000 steps, batch size 8, and seed `20260930`. Its recorded training loop took 195.3 seconds. Its last training loss is about 0.2203. This loss measures noise prediction on training windows. It does not measure success in the simulator.

The full seeded development result is 15/20, with 0 policy errors and 0 infrastructure errors. It uses states 0–19 and base seed `20260930`. Each state uses reset seed `20260930 + state_id`. Final states 20–49 are not used.

Use `manifests/diffusion_reference.yaml` and its referenced ablation records for the exact paired 5-step and 20-step outcomes and per-state sampler latency. Do not infer missing values from the full 20-step result. The 5-step sampler uses fewer denoising iterations. It should reduce action-generation work, but can change the predicted action sequence and task behavior. Five states and one training seed are not enough to show a stable success rate.

The live rollout uses development state 2 and reset seed `20260932`. The saved reference succeeded on this state. Its new result can vary with runtime and hardware. Record its model hash, success, termination reason, elapsed time, and sampler-call latency. Inspect the saved state-0 failure. Keep task failures, policy errors, and infrastructure errors separate.

The checkpoint loader verifies the model, config, statistics, and processor hashes before it loads the model. A matching hash verifies file identity. It does not prove that the model is good at the task.

| Development state | 5-step result | 20-step result |
| --- | --- | --- |
| 0 | failure | failure |
| 1 | success | failure |
| 2 | success | success |
| 3 | failure | success |
| 4 | failure | failure |

The weighted mean sampler latency was 29.052 ms with 5 steps and 99.263 ms with 20 steps. The 20-step value was 3.417 times the 5-step value. The timing measured CUDA-synchronized `generate_actions` calls on the local RTX 3070. The mean includes the first cold call. It excludes simulator and processor time. Both settings succeeded on 2/5 states, but each setting succeeded on a different pair except state 2. Keep 20 steps as the selected setting because its full seeded development run succeeded on 15/20. The five-state ablation cannot rank behavior or give a stable success rate.
