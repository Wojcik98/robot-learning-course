# Lab 10 solution guide

1. `L = 40`. Forty weights are above 0. The weight at index 2 is 0.935, so it is above 0.9. The first two weights are 1.
2. For `d = 0`: `[0.571, 0.287, 0.114, 0.026, 0, 0]`. All guided weights are soft. For `d = 4 = L`: `[1, 1, 1, 1, 0, 0]`. The soft region is empty, so this equals the hard mask.
3. `d = 5` is larger than `L = 4`, so `prefix_weights` raises `ValueError`. The delayed actions would extend past the guided overlap. For `s = 1, d = 4`, the rule `d <= s` fails: new requests start before the older reply arrives. RTC requires `d <= s <= H - d`.
4. The correction is 0. With `a = 2` and `t = 0.5`, the factor `1 - a t` is 0. The estimate does not depend on the input, so the vector-Jacobian product is 0. A guidance term without the model derivative would not be 0. That is the defect that the checked guidance code fixes.
5. The course uses `β = 5`. The weight is `(t² + (1-t)²) / (t (1-t))`. It becomes very large when `t` is near 0 or 1. Without the clip, the correction can dominate the velocity and give non-finite or unstable values.
6. The report shows 2/5 to 4/5 on 5 development states with one checkpoint. It does not show a stable ranking, and it does not show that RTC improves or hurts success. A lower discontinuity for `rtc_soft` does not show a better task result.
7. Virtual: 100 ms is 2 ticks. Measured: 177 ms is 4 ticks and 319 ms is 7 ticks (round up). The report used the virtual 2 ticks. It did not run the measured delay.
8. `naive_async` succeeded on 9/10 and `rtc_soft` on 4/10. There are five states where only `naive_async` succeeded and none where only `rtc_soft` succeeded. The exact paired McNemar value is 0.0625. Their mean adjacent-action L2 values are 0.334 and 0.274 in bounded controller-action units. Ten states and one checkpoint do not establish a stable ranking; the virtual 100 ms delay was shorter than measured request time.
