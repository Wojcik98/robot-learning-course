# Lab 7 exercise: Choose a recipe for one task

1. Verify the SHA-256 of `configs/experiments/comparison.yaml` against the report. Check that all three raw result files contain final state IDs 20–49 in order, with the declared reset seeds and no policy or infrastructure errors.
2. Report each success count and its 95% Wilson interval. Show the paired result for ACT against Diffusion Policy, including the states where only one policy succeeded.
3. Compare robot training episodes, optimizer steps, sampled windows, training loop time, and the source of pretrained weights. Explain why this is a practical recipe comparison and not an architecture study.
4. Choose one recipe for this fixed bowl-plate task. Name the task goal, the main constraint you used, and the exact result that supports your choice. State two conditions where the result does not apply.
5. Explain the cost of using ACT or Diffusion Policy for a new text goal: another system must choose a task-specific checkpoint. Use the one-state published SmolVLA prompt study only as a prompt-sensitivity example.
6. Read [RT-2 Section 3.2](https://arxiv.org/html/2307.15818). State how its action representation differs from this course's ACT, SmolVLA, and Diffusion Policy recipes. Give one testable prediction or one limit of this exercise.

Do not tune a checkpoint with the saved final states. Use development states for a new task or a new setting.
