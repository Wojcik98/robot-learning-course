# Lab 4 exercise guide

Each run must record `evaluation_task: put the bowl on the plate`. The `model_prompt` field must contain the exact text sent to the model. Compare the three `episodes.jsonl` rows with the same state ID and seed.

The paraphrase asks for the same goal in other words. The conflicting prompt asks for another object relation. If the conflicting run fails the bowl-plate check, that result alone does not prove that the model followed the text. Inspect the object motion in a video or run more fixed states before you make a behavior claim. Three attempts on one state do not measure a stable success rate.

The stored model uses one action per policy call. The selected course check uses ten actions per call. A different execution setting changes when the model sees a new observation, even if its weights stay fixed.

The saved state-0 reference used ten actions and reset seed 20260930 in all three runs. The canonical prompt succeeded in 74 steps, and the paraphrase succeeded in 85 steps. The conflicting prompt failed after 300 steps. The simulator still checked `put the bowl on the plate` in each run. The three results show a difference on this state. They do not show which object the robot tried to move in the failed run.
