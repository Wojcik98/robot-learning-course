# Lab 11 exercise guide

The Lab 11 rule is: reject on a hash or state-ID mismatch, reject on any error, promote only on more successes, and otherwise keep the current release.

1. Both variants have 20 rows. The state IDs are 0 to 19. Each reset seed is `20260930 + state_id`. `demos39_chunk20` has 18 successes. `demos39_chunk10` has 11 successes. These counts equal `development_successes` in `lab3_suite.json`.
2. Both variants have `split_sha256` `ef237195fc6ace8a78ff69fcf472362a89adb64a4ff079b5fb84724157bb82f4`. They used the same training and validation episodes. The match does not show that the training runs, seeds, or model files are the same. The model hashes differ (`5c5ae4155dd1…` and `ff7ae2283036…`). The chunk size differs: 20 for `demos39_chunk20` and 10 for `demos39_chunk10`. Both use 10 action steps.
3. The decision is `keep`. The reason string is `candidate 11 vs current 18`. The state IDs and reset seeds match, no row has an error, and 11 is not more than 18. The Lab 3 rows have no `error` key, so `decide` uses `r.get("error")` and finds no error.
4. Only the current release succeeded on states `[0, 1, 3, 8, 13, 15, 16, 18]`. That is 8 states. Only the candidate succeeded on state `[11]`. That is 1 state. The exact McNemar value is `0.0390625`. It equals the pair record in `lab3_suite.json` (`a_only` 8, `b_only` 1).
5. Example records:

   ```python
   current = {"release_id": "demos39_chunk20", "model_sha256": "5c5ae4155dd1…full hash…", "parent_release": None,
              "evaluation": {"partition": "development", "state_ids": list(range(20)), "successes": 18}}
   candidate = {"release_id": "demos39_chunk10", "model_sha256": "ff7ae2283036…full hash…", "parent_release": "demos39_chunk20",
                "evaluation": {"partition": "development", "state_ids": list(range(20)), "successes": 11}}
   ```

   Use the full hashes from `lab3_suite.json`. Write `current.json` with `{"release_id": "demos39_chunk20", "model_sha256": <current hash>}`.
6. After the what-if promotion, `current.json` names `demos39_chunk10`. After the rollback, it names `demos39_chunk20` and the current model hash again. Hash the two record files before and after. The hashes are equal. A rollback writes a pointer and a history line. It writes no record.
7. Example answer: "A matching model hash proves that two files have the same bytes. It does not show that the policy works. This result uses one training seed and 20 development states, and source demonstrations may overlap the simulator initial states, so it does not show a general advantage."

Check your answer with these assertions:

```python
assert (sum(r["success"] for r in current_rows), sum(r["success"] for r in candidate_rows)) == (18, 11)
assert decide(current_rows, candidate_rows) == ("keep", "candidate 11 vs current 18")
assert (current_only, candidate_only) == ([0, 1, 3, 8, 13, 15, 16, 18], [11])
assert exact_mcnemar(len(current_only), len(candidate_only)) == 0.0390625
```

## Example career assessment memo

This is one example. Your preferences can differ. Keep the measured result separate from your preference.

1. I liked tracing an image and robot state through a policy to a live action. I also liked finding the reset-seed fault because a small integration error changed the measured result.
2. I would accept repeated data checks, failed-run review, and checkpoint hash checks. I would automate their mechanical parts and inspect exceptions by hand.
3. For this fixed bowl-plate task, the prepared ACT recipe reached 30/30 final states; learner SmolVLA reached 18/30. I would choose ACT for that declared task and setup. A VLA may help when many goals must share one policy, but this course did not test that benefit on new goals.
4. I would collect demonstrations from the real cameras and controller, with new object placements and recovery cases. I would check camera calibration, action units, stop behavior, operator safety, and failure rates before physical use.
5. Frame transforms, control timing, device logs, and reproducible tests used robotics integration skills. The course also needed data and model provenance checks.
6. I would reconsider the method if a VLA matched or exceeded the narrow policy on a frozen set of new goals with a measured action deadline, a comparable data budget, and no extra safety failures. I would reconsider the career choice if the daily data and incident work were less acceptable than the experiments.
