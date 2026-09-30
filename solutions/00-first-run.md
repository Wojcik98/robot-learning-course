# Lab 0 exercise guide

Use the `run.json` files from two development runs. Compare `state_id`, `steps`, and `success`. Open the two videos and look at the initial robot and object positions. The exact step count and result can differ by start state. Do not interpret two runs as a general success rate.

The checkpoint hash must be the same in both run records if you changed only the start state. The `task_alias` must match the checkpoint task. ACT uses images and robot state for action prediction; `task_alias` selects the environment and checkpoint outside the policy.
