# Lab 1 exercise guide

For 74 controller steps, an execution length of 10 needs `ceil(74/10) = 8` policy calls. A length of 5 needs `ceil(74/5) = 15` calls. The model still predicts a 20-action chunk. A shorter execution length uses newer observations more often, but it adds policy calls and compute time. It does not guarantee better success.

At tick 11, the older chunk predicts `0.2` and the newer chunk predicts `0.6`. With decay `0.0`, both have equal weight and the queue returns `0.4`. With positive decay, the newer chunk gets more weight, so the result moves toward `0.6`. This queue exercise does not change the current ACT checkpoint's inference setting; its saved temporal ensembling coefficient is null.
