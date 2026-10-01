# Initial setup: one user checklist

**Status:** Labs 0–11 passed on the prepared local host. No paid cloud job has started. The user reported a US$29.94 Brev balance, auto recharge off, and no other active charge path. The Brev CLI login and organization list work. A read-only L4 quote was checked on 2026-10-01. The live billing screen, supervised allocation, and provider stop still need checks. A Hugging Face token file is present on this host; do not copy or print it.

Use the defaults below unless the user supplies an override. Do not ask again for decisions already made.

## 1. Defaults and existing authorization

| Item | Default or confirmed scope |
|---|---|
| Storage | `<workspace>/.local-data`, ignored by Git; 166 GB was free on 2026-09-30. Recheck before downloads. Allow another disk through configuration. |
| Shell | Bash on Ubuntu 24.04; test completion in a fresh shell. |
| Compute | Local RTX 3070 first; NVIDIA Brev when local memory or runtime is unsuitable. |
| Region | Available EU region where practical, within the price limit. |
| Implementation credit | User will provide about US$30 prepaid. Project cap is the lower of US$30 and the confirmed funded project balance. |
| Job limits | One VM, at most US$3/hour, US$5/job, and 2 hours/job; stricter remaining-budget and deadline limits apply. |
| Lifecycle | Start, monitor, checkpoint, export, and stop project jobs within that allowance. Stop hung or overlong jobs. No automatic top-ups. |
| Future course budget | €150/month is a planning limit, separate from the implementation spending authorization. |
| Git | Keep the plan uncommitted during review. After approval, use Conventional Commits directly on `main`. |
| Worker use | Sol leads; Luna workers; Claude Code help; Astra advice. |
| Usage pause | Above 85% of the Codex 5-hour window, pause workers/tasks, stop paid jobs, and set one resume timer for reset. This is already requested. |

Use the complete [budget and monitoring rules](plans/vla-tutorial-plan.md#7-cloud-operation-and-budget). Reserve US$5 of the implementation balance for storage, transfer, billing uncertainty, and shutdown. Quotes and measured duration must fit the remaining allowance. Do not assume that a local cost counter enforces the provider bill.

## 2. User actions to complete together

Sol checks which actions are actually needed, then presents one list:

1. **Hugging Face:** Log in if required for selected assets; accept any required model terms. Sol supplies exact repository IDs before this step.
2. **NVIDIA Brev:** Create/login to the account, select the project, and add the intended prepaid credit. Disable auto recharge. Verify any postpaid route, shared credit balance, currency, tax, billing granularity, and storage charges with Sol.
3. **Host packages:** Complete the single prepared administrator step if native dependencies are missing. Do not install a different driver to address sandbox-only GPU failure.
4. **Storage:** Supply an alternate path only if the default is unsuitable. Grant the app access to that directory when needed.
5. **External tools:** Complete Claude Code login if needed. Establish the available Astra advice route. If another chat is required, identify it and authorize messages there once.

Keep passwords and tokens out of chat, notebook cells, tracked files, and command logs. Use the service's credential store or secret facility. Local paths and private account settings belong in ignored `config.local.toml`.

### Hugging Face access

The installed CLI provides `hf auth login` and `hf auth whoami`, as described in the [CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli). Use a dedicated read token when access needs one. Public core assets worked without login in the P0 download. A token file is now present on this host; its access to each future asset is not yet verified.

Test metadata and a small download for each asset family. Keep the authentication location stable. Set `HF_HUB_CACHE` and `HF_DATASETS_CACHE` for large files without changing `HF_HOME` after login. On a VM, use a scoped credential through an encrypted channel or secret facility. Do not copy an entire local credential directory.

### Brev access and billing

Use the [Brev console](https://brev.nvidia.com) and the supported [CLI login workflow](https://docs.nvidia.com/brev/latest/cli/getting-started). The user installed Brev CLI v0.6.335 and completed browser login. A host CLI check returned no instances in the active organization. A separate API token is needed only if the chosen automation route requires one.

The [billing guide](https://docs.nvidia.com/brev/guides/console-reference#billing--usage) describes shared credit and auto recharge. The user checked the Billing page and reported no active charge path beyond the US$29.94 credit, with auto recharge off. Check the live quote and stop controls before allocation. If that boundary cannot be established, continue local work and report the cloud blocker.

Within the authorized allowance, verify one short supervised allocation: SSH, file transfer, command execution, export, and provider stop. Record cost and provider state. [Stopped instances can retain storage charges](https://docs.nvidia.com/brev/concepts/gpu-instances). Export required files before deletion or credit exhaustion. Never delete unrelated resources.

Before unattended remote jobs, P1 must verify the detached progress monitor, bounded export, actual provider stop, and an independent stop path. Guest OS shutdown or killing Python is not evidence that billing has stopped. Until these checks pass, cloud runs stay supervised with a fixed deadline.

## 3. Host dependencies and completion

`uv 0.9.28` and `just 1.36.0` were present in the workspace check. Sol rechecks them and prepares the exact package list. Start with Python 3.12 and a project-local uv environment.

Inspect FFmpeg shared libraries with AV1 support and EGL/OpenGL libraries. A working `ffmpeg` executable alone does not prove that the Python decoder can load its libraries. P0 must decode an actual dataset file and render an actual simulator observation. Prefer the compact video dataset; do not download the large image-based copy as an automatic decoder workaround.

Install completion for existing tools in S0. P1 adds and tests completion for the project CLI. Show any shell-startup change before applying it. Test recipe names with `just`; test detailed option and value completion with the activated `robot-lab` command.

## GPU access and tests

On 2026-09-30, read-only host execution reported:

```text
NVIDIA GeForce RTX 3070, 8192 MiB, 580.178.04
```

A host Python process also loaded `libcuda.so.1` and passed `cuInit(0)`, device enumeration, and device-name lookup. The restricted command environment could not access the GPU. No app mode or driver change is needed for the host route that passed. Codex can request host execution; the app's permission system still controls each request.

PyTorch was absent from the inspected default Python environment. Driver access does not establish project runtime compatibility. P0 must use the selected project environment to:

1. Allocate a CUDA tensor, run a matrix operation, synchronize, and check finite output.
2. Render a real simulator frame with the selected backend and inspect its content.
3. Decode actual dataset video frames.
4. Load ACT, predict, and complete a live rollout.
5. Repeat model checks for SmolVLA and Diffusion Policy. Measure memory with simulation/rendering active, and probe training and RTC guidance separately.

Save commands, exit codes, environment identity, and artifacts in the feasibility report. Do not count CPU fallback as a GPU pass. If host execution is rejected, preserve CPU progress and report the exact action and rejection. A virtual environment or container does not remove the session's execution boundary.

## 4. Setup acceptance record

Sol fills a compact redacted table here during S0; do not create another setup report.

| Check | Required evidence | Status |
|---|---|---|
| Storage | Writable selected path, free space, download reserve | Passed local path; `doctor --live` reported 128.7 GiB free on 2026-10-01 |
| Driver route | Host GPU query and CUDA driver API | Passed host check and live local GPU notebooks |
| Runtime | Exact package list, lock, tensor/render/decode checks | Passed pinned local runtime, GPU, AV1 decode, and EGL reset |
| Hub access | Exact asset IDs and successful small downloads | Public pinned assets downloaded; token access for future assets not checked |
| Cloud billing | Allocated credit, current quote, recharge/overage settings, job limits | User confirmed credit and charge path; live L4 quote checked; current billing screen pending |
| Remote lifecycle | SSH, export, provider stop, cost and storage record | Pending supervised S0 check |
| Unattended jobs | Monitor, deadline, independent stop, failure tests | Pending P1 |
| Workers | Claude login/usage access, Astra route | Claude Code CLI was used for bounded course tasks; an Astra route is not part of the learner setup |
| Completion | Bash tests for installed tools, then project CLI | `just --list` and the offline Bash completion tests passed; shell startup installation was not checked |

Collect missing user actions in one message. Local work can continue when a cloud or external-review route is unavailable, but dependent work must remain marked as blocked. Later questions are needed only for changed scope, failed access, costs outside the allowance, or enforced tool approvals. Do not repeat the standing spending or lifecycle questions.
