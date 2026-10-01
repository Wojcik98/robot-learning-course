# Lab 9 exercise: Predict and check delayed actions

Use the notebook helper `make_chunk` and `VirtualScheduler`. Write your prediction before you run each step.

1. A chunk has `t = 3`, `d = 2`, and `H = 6`. List the first and last tick where the controller can use it. State the chunk index at tick 6.
2. Run delays 0, 1, 2, and 4 for one chunk with `H = 8`. Give the first used index and the number of underflow ticks before it. Convert each delay to milliseconds.
3. Use `H = 8` and `s = 4`. Find the setting `d` in {0, 1, 2, 4, 5} where `d <= s <= H - d` fails. Explain why that rule matters. Show that `s = 1, d = 4` fails the local check.
4. Make four requests at `t = 0, 2, 4, 6` with delays `[3, 0, 2, 1]`. Predict the availability tick of each reply. Predict which reply the scheduler rejects, and the reason. Then run it and compare.
5. Send one reply twice. Send a second reply with a new ID and the same observation tick. State the rejection reason for each.
6. Make an exhausted reply. Use a policy outage. State the tick where the scheduler rejects it.
7. Let a chunk end while the gripper value is 1. Show the action at the first underflow tick. Explain why the scheduler holds the gripper.
8. Reset to a new reset counter while a reply is pending. Show that the old reply cannot reach the controller. State what else the runner must reset outside the scheduler.
9. Time one CPU call with `measure_calls`. Report cold time, warm p50, warm p95, and deadline misses. Convert warm p95 to a delay with `latency_to_ticks`. State why this conversion is conservative. State why you cannot call it a GPU speed result.
10. Write two sentences that state what this lab does not prove about SmolVLA, RTC, and LIBERO.
