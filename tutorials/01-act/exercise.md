# Exercise: Change the action execution length

The checkpoint predicts 20 actions at each policy call, but its saved setting executes 10 actions before the next call.

1. For a 74-step rollout, calculate the number of policy calls for execution lengths 10 and 5. Use `ceil(steps / length)`.
2. In the notebook's synthetic queue cell, change `ensemble_decay` from `0.0` to `1.0`. Write down how the action at tick 11 changes and which chunk gets more weight.
3. Explain one possible benefit and one cost of a shorter execution length. Keep the trained chunk size at 20. Do not edit the saved checkpoint to make a new trained model.

Check your reasoning with the [solution guide](../../solutions/01-act.md) after you finish.
