# Lab 7 exercise guide

The frozen config SHA-256 is `b927d1af8602af3a098091b88dd21fc55f09d89d70046ba98cf168dbd166d207`. Each policy has 30 ordered final attempts for state IDs 20–49. Each reset seed is `20260930 + state_id`. The saved comparison has zero policy errors and zero infrastructure errors.

| Recipe | Successes | 95% Wilson interval |
| --- | ---: | ---: |
| ACT | 30/30 | 0.886–1.000 |
| Learner SmolVLA C2 | 18/30 | 0.423–0.754 |
| Diffusion Policy | 20/30 | 0.488–0.808 |

ACT succeeded on 10 states where Diffusion Policy failed. Diffusion Policy succeeded on no state where ACT failed. The paired success difference is 10/30. The exact McNemar value is 0.001953 for these fixed attempts. It does not show that ACT architecture caused the difference.

All three recipes use 39 robot training episodes and 3,661 source frames. ACT used 200 optimizer steps and 1,600 sampled windows. Diffusion Policy used 2,000 steps and 16,000 sampled windows, with a pinned ImageNet ResNet-18 visual encoder. Learner SmolVLA C2 used 5,000 steps and 5,000 sampled windows, with a pinned SmolVLM2-500M-Video-Instruct VLM source. A repeated window counts again. The recorded local optimizer-loop times were 16.655 s, 195.329 s, and 828.829 s, respectively. Full VLM pretraining exposure is unknown. The models also use different processors, history, and action settings. These differences prevent a controlled architecture claim.

For the fixed bowl-plate goal, select the prepared ACT recipe if task success and local training time are the main constraints. It succeeded on 30/30 saved final states and used the least recorded robot training time. This choice does not predict a new goal, visual shift, new scene, or physical robot. One training seed does not estimate training-seed variation. Source demonstrations and simulator initial states are separate partitions, but the course has no proof of a new-placement test.

ACT and Diffusion Policy do not take a text instruction in this course. To act on another goal, an external system must choose a task-specific checkpoint. That selection adds a component and its own possible errors. The published SmolVLA reference did react to prompt changes on one bowl-plate development state. This one state does not establish general language understanding. The published reference is separate from learner SmolVLA C2, and its training data are unknown.

[RT-2 Section 3.2](https://arxiv.org/html/2307.15818) maps discretized robot action dimensions into output tokens for a vision-language model. This course's ACT predicts continuous action chunks; learner SmolVLA uses a flow-based action expert; Diffusion Policy iteratively denoises continuous action sequences. A possible prediction is that an action-token model can use text changes without an external checkpoint selector. This course has no matched RT-2 checkpoint or controlled test for that prediction. Do not infer its task success from these three runs.

The stove-on third-task gate is not met by the first ACT reference: 5/20 development successes. Its final states remain closed. Do not add it to the frozen bowl-plate report.
