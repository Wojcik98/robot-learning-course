# Brev use and cost controls

The course uses local compute by default. Use Brev only when local GPU memory or run time is insufficient. Real allocation, SSH, file transfer, and provider stop are not verified for this course. Use supervised sessions until these paths and an independent stop path are checked.

## Account and job checks

Log in to the Brev CLI and select the intended organization. Before each allocation, check the live funded balance, currency, storage charges, automatic recharge, and any postpaid charge path. Get a fresh quote for a GPU with enough memory and a provider stop control.

The [budget code](../src/vla_lab/cloud/budget.py) uses these default limits:

| Limit | Value |
|---|---|
| Project cap | US$30, bounded by available funded credit |
| Reserve | US$5 |
| GPU hourly rate | US$3/hour |
| Job cost | US$5 |
| Job duration | Two hours |
| Quote and billing check age | Five minutes |

The cost bound includes startup, export, stop time, and quoted disk charges. The code rejects automatic recharge, postpaid overage, another running instance, insufficient memory, stale checks, and a job that cannot fit the deadline. A local cost calculation does not enforce the provider bill. Set your own funded budget before you use paid compute.

## Monitor, export, and stop

The [Brev adapter](../src/vla_lab/cloud/brev.py) gets quotes, reads instance state, and stops a named instance. The [monitor](../src/vla_lab/cloud/monitor.py) checks saved work counts, heartbeat age, lack of progress, projected overrun, cost, and the deadline. The [local watchdog](../src/vla_lab/cloud/watchdog.py) can stop a named instance at a saved deadline in a separate process. It depends on the local host and network; it is not a provider deadline.

Before unattended use, check allocation, SSH, file transfer, bounded export, provider stop, and an independent stop path in a short supervised session. Keep a fixed deadline and the exact instance name. Stop the instance if progress stops or the cost or time limit is near. Export required files before deletion.

Killing Python or shutting down the guest OS does not confirm that billing has stopped. Check the provider state after `brev stop INSTANCE_NAME`. Stopped instances can still incur storage charges. See [the operations guide](operations.md#71-cloud-failure-supervised-use-only) for failure steps.

Keep tokens out of notebooks, tracked files, run logs, and remote shell commands. Use the service credential store. Do not use `brev stop --all` or delete unrelated resources.
