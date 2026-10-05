# Type Annotation Plan

**Goal:** Add concrete types to source code and notebook code. Use the same interfaces in small exercises and real policy paths.

**Scope:** `src/vla_lab`, `tools`, and tracked notebooks in `tutorials`.

**Constraints:** Preserve current user edits, notebook outputs, metadata, Markdown, and cell IDs. Preserve the user's implemented reset exercise. Allow interface changes needed for concrete types. Do not commit or push.

- [x] Complete function signatures in source and tools. Remove explicit `Any`. Use typed records for course data and results, and JSON value types at serialization boundaries.
- [x] Add shared policy, environment, processor, model block, and callback protocols. Check the installed ACT policy against the action-chunk interface.
- [x] Use matching function types in learner and reference notebooks. Update notebook builders to copy reference signatures into exercise stubs.
- [x] Add `just typecheck`, Mypy as a development dependency, and the package type marker. Check source and notebook annotation coverage, type imports, matching signatures, and five rejected policy calls.
- [x] Preserve task text in the general simulator and SmolVLA paths. Prepare tensor-only batches in the ACT exercise adapter.
- [x] Run checks and review the final interfaces.

**Checks:** 543 CPU tests passed. `just typecheck` passed for the teaching modules, simulator and recording interfaces, static policy checks, and both full unit 2 notebooks. It also checked annotations and matching signatures in all 35 tracked notebooks. Five invalid calls were rejected. Ruff lint and format checks passed. All 35 notebooks ran in fresh CPU kernels: 19 completed and 16 stopped at expected learner stubs. Live simulation was disabled only in the in-memory test copy. All 26 builder outputs had complete signatures, valid type imports, and no explicit `Any`. The model notebook pairs had matching signatures. Saved outputs, metadata, cell IDs, Markdown, and the user's reset exercise body were preserved. A separate code review found no further issue in the reviewed interfaces.

**Check limits:** CPU checks do not assess GPU policy quality or live simulator behavior. Mypy checks the selected course and runtime modules; annotation coverage checks all source and tool files.
