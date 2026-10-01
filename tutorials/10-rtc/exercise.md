# Lab 10 exercise: Predict and check the RTC mask

Write your answer before you run each step. Use the notebook functions `eq5` and `prefix_weights`.

1. **Prediction.** For `H = 50`, `s = 10`, `d = 2`, state how many weights are above 0. State whether the weight at index 2 is above or below 0.9. Check it.
2. For `H = 6`, `L = 4`: write the soft mask for `d = 0` and for `d = 4`. Which one equals the hard mask?
3. Try `d = 5` with `L = 4`. State the result and the reason. Then state why `s = 1, d = 4` is also invalid for RTC.
4. In the toy guidance check, set `t = 0.5`. State the correction. Explain why with the factor `1 - a t`.
5. A guidance weight needs clipping near `t = 1`. Give the clip value that this course uses. State what could go wrong without it.
6. The development report shows 4/5 for `synchronous` and `rtc_hard`, and 2/5 for `rtc_soft`. Write two sentences. State what this shows and what it does not show about RTC and success.
7. The virtual delay is 100 ms. The measured p50 request time is about 177–319 ms. Convert both to ticks. State which delay the report used.
8. Read the frozen final report. Compare `naive_async` with `rtc_soft` on the same ten states. Give the successes, the discordant counts, the exact McNemar value, and the mean adjacent-action L2 values. State one limit of this comparison.
