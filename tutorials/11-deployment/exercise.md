# Lab 11 exercise: Decide on a second candidate

Use the notebook helpers `sha256`, `read_json`, `read_rows`, `decide`, and `exact_mcnemar`. Use only tracked files. Write your prediction before you run each step.

The current release is `demos39_chunk20`. The candidate is `demos39_chunk10`. Both are ACT variants from the Lab 3 record `artifacts/reference/act/lab3_suite.json`. Each has a result folder named by `seeded_result_dir`.

1. Load `lab3_suite.json`. Find the two variants. For each one, read `episodes.jsonl` from its `seeded_result_dir`. Show that the state IDs are 0 to 19 in order. Show that each `reset_seed` equals `20260930 + state_id`. Show that the success count equals `development_successes`.
2. Show that both variants have the same `split_sha256`. State what this tells you about their training data. State what it does not tell you.
3. Predict the result of `decide(current_rows, candidate_rows)`. Then run it. Write the decision and the reason string.
4. List the states where only the current release succeeded. List the states where only the candidate succeeded. Compute the exact McNemar value. Show that it equals the `exact_mcnemar_p` for this pair in `lab3_suite.json`.
5. In a temporary directory, write one release record for each variant. Use the fields `release_id`, `model_sha256`, `parent_release`, and `evaluation`. Set the candidate `parent_release` to the current release ID. Write `current.json` for the current release.
6. Promote the candidate in the temporary directory. This is a what-if step. Roll back using `parent_release`. Show that `current.json` names the current release and its `model_sha256` again. Show that the SHA-256 value of each record file did not change. Remove the temporary directory.
7. Write two sentences. The first sentence states what a matching model hash proves. The second states one reason why this result does not show that `demos39_chunk20` is a better policy in general.

## Career assessment memo

Write a short memo for yourself. Answer all six questions. Use one course result where it helps. Personal preferences have no model answer.

1. Which part of the work was interesting to you?
2. Which repeated tasks would you accept in a job?
3. Was a VLA better than a narrow imitation policy for the fixed bowl-plate task in this course? What might change for many goals?
4. What data and checks would you need before a real robot deployment?
5. Which integration tasks used skills that you already have?
6. What measured result would change your decision about this work or method?

Do not open final states. Do not run a simulator. Do not start a cloud job.
