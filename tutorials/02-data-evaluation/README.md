# Lab 2: Data and evaluation

This guide describes the earlier lab. Its notebook was removed. Use the [course index](../index.ipynb) for current exercises.

This lab checks a complete LIBERO demonstration, the fixed episode split, and the statistics used by the ACT processor. It ends with two live ACT runs on fixed development states. The dataset records 10 Hz labels. The simulator controller uses 20 Hz ticks. Do not use dataset timestamps as proof of simulator time.

From the repository root, run:

```bash
just data-training
```

The audit and format cells use a CPU. The live evaluation cell needs the local GPU, AV1 decoder, and EGL runtime. The pinned training files and two camera views use several GiB of disk. This local reference run has no cloud cost. Plan for 3–4 learner hours, including the reading and exercise.

The notebook saves two new development results under `.local-data/runs`. Its 20-state result is a saved ACT reference. The two-state run checks the evaluator path; it is too small for a useful success estimate. Keep final simulator states 20–49 closed until final evaluation.
