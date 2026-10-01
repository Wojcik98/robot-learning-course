# Lab 9 exercise guide

One tick is 50 ms. A chunk has `t` (observation tick), `d` (delay), and `H` (action count). It is available at tick `t + d`. It has an action for ticks `t` to `t + H - 1`. At tick `k`, the scheduler uses index `k - t`.

1. For `t = 3`, `d = 2`, `H = 6`: the first tick is 5 and the last tick is 8. At tick 6 the index is 3. The chunk cannot be created with `d >= H`.
2. The first used index equals `d`: 0, 1, 2, 4. The delays are 0, 50, 100, and 200 ms. Underflow before the first used tick is `d` ticks for a single chunk. The chunk ends after tick 7, so ticks 8 and 9 also underflow.
3. With `H = 8` and `s = 4`, the rule `d <= s <= H - d` holds for `d` of 0, 1, 2, and 4. It fails for `d = 5`. For `s = 1` and `d = 4`, `d <= s` is false. A delay that is longer than the stride means that new requests start before older replies arrive. A stride longer than `H - d` leaves a gap. The check in the notebook is local teaching code, not the RTC validator.
4. The availability ticks are 3, 2, 6, and 7. The reply from `t = 0` (tick 3) arrives after the reply from `t = 2` (tick 2). The scheduler accepts the newer reply first and rejects the older reply as `reordered` at tick 3.
5. The second copy has the same request ID, so the reason is `duplicate`. The reply with a new ID and the same observation tick is `reordered`. The scheduler tests the observation tick with "not newer than", so an equal tick is rejected.
6. Use `t = 0`, `d = 2`, `H = 3`. Hold the policy unavailable at ticks 2 and 3. The reply waits. At tick 4 it is delivered, has no action left, and is rejected as `exhausted`. Tick 4 is also an underflow tick.
7. At the first underflow tick the arm values are zero and the gripper value is 1. The scheduler repeats the last gripper command, so an underflow does not open or close the gripper. It does not prove that the real controller stays still under zero input. That needs a simulator check.
8. After the reset, the pending reply and all accepted chunks are gone. The tick counter starts again at 0. A chunk with the old episode ID or reset counter makes `submit` raise `ValueError`. The runner must also reset policy state and history. The scheduler does not own them.
9. The cold and warm values depend on your host and run. Use the printed values from your run. `latency_to_ticks` rounds up to the next tick. A latency of 51 ms gives 2 ticks, so a measured delay is never rounded down. Wall clock time on a CPU matrix product says nothing about GPU time. A virtual delay is a setting that you choose. It is not a measurement.
10. Example answer: "This lab used synthetic chunks and a deterministic scheduler. It did not run SmolVLA, RTC guidance, or LIBERO, so it shows no task success, no GPU cost, and no RTC benefit."
