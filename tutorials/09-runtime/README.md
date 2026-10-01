# Lab 9: Run delayed action chunks on virtual ticks

This lab teaches how a delayed policy reply reaches the robot controller. It uses a deterministic scheduler with 50 ms virtual ticks. It uses synthetic `ActionChunk` objects that you can read by eye.

Start from the repository root:

```bash
.venv/bin/jupyter lab tutorials/09-runtime/09-runtime.ipynb
```

The notebook runs on CPU. It needs no GPU, no LIBERO assets, and no model checkpoint. Cloud cost is US$0.

## What you learn

- A delay `d` is a whole number of ticks. Delays of 0, 1, 2, and 4 ticks are 0, 50, 100, and 200 ms.
- A chunk made at observation tick `t` is available at tick `t + d`. At controller tick `k`, the scheduler uses chunk index `k - t`. A delayed chunk does not restart at index 0.
- With jitter, replies can arrive in a different order. The scheduler rejects duplicate, reordered, and exhausted replies. A reply from an old episode or reset cannot enter.
- When no chunk covers a tick, the action is zero arm motion and the last gripper value.
- A reset clears pending replies, accepted chunks, the gripper hold value, and the tick counter.
- Wall clock latency is a separate measurement. Virtual ticks do not read a wall clock.

## Code that the lab uses

- `src/vla_lab/runtime/virtual_scheduler.py`: `VirtualScheduler`, `TickRecord`, `latency_to_ticks`.
- `src/vla_lab/runtime/action_queue.py`: `ActionQueue`.
- `src/vla_lab/runtime/timing.py`: `measure_calls`.
- Tests in `tests/test_virtual_scheduler.py`, `tests/test_action_queue.py`, and `tests/test_runtime_timing.py`.

## Limits

This lab is a draft. It does not run SmolVLA. It does not run RTC. It does not run LIBERO. It does not measure task success or GPU cost. The scheduler tests show queue rules only.

The scheduler records camera and policy availability. Only policy outage changes delivery. A camera fault is recorded and does not change the action.

The wall clock cell prints numbers that depend on your host. The notebook does not assert them. Do not compare them across hosts.

Read the [exercise](exercise.md) after the notebook. Read the [solution guide](../../solutions/09-runtime.md) after you save your answer.
