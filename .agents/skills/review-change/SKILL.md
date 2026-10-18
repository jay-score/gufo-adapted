---
name: review-change
description: Review a Gufo change, branch, commit or pull request against this repository's correctness, testing, Windows/Linux, API and evidence rules. Use when asked for a code review, PR review, diff review or merge-readiness check.
metadata:
  origin: gufo
---

# Review a Gufo change

This is a **read-only** review. Do not edit, stage, push or resolve review
threads unless the user also asks for fixes.

Read [AGENTS.md](../../../AGENTS.md) and the change-selection table in
[testing](../../../docs/TESTING.md) first. Use
[`tools/ci/review-context.py`](../../../tools/ci/review-context.py) to classify
the diff and select checks; then read the changed code and verify the claims.

## Intake

Establish before reviewing:

- base and head refs; target branch (`windows-port` unless the user says
  otherwise)
- stated intent: PR description, issue, commit messages, or the user's request
- whether the user wants a verdict, a fix list, or both
- what is actually available: GPU, model artifacts, time budget, `nix`, `gh`,
  Windows toolchain
- whether the review covers a committed range or also working-tree edits

If the intent is missing, infer it from the commits and say that the inference
is unconfirmed. If the diff is too large to read, review it by area and state
what was not read. Do not pretend a large diff received a full line-by-line
review.

## Build the context packet

Run from the repository root:

```sh
python3 tools/ci/review-context.py \
  --base origin/windows-port \
  --head HEAD \
  --intent "<PR description or one-paragraph intent>" \
  --run-checks \
  --output artifacts/agent/review/context.md
```

On Windows use `python`. Add `--format json` when another tool will consume the
packet. The tool is advisory: it classifies files, recommends checks and flags
hazards, but it does not build Gufo, load models or prove numerical parity.

If `gh` is installed, authenticated and the user authorized GitHub access,
`gh pr view <number>` / `gh pr diff <number>` can supply the intent and the
exact PR range. Otherwise use `git log`, `git diff --stat` and the commit
messages.

Read the changed files. The packet is an index, not a substitute.

## Review dimensions

### 1. Intent and scope

- Does the diff do what the stated intent says, and only that?
- Are unrelated formatting, renames or refactors mixed in?
- Are the commits focused, with Conventional Commit subjects?
- Does the change preserve unrelated work and local configuration?

### 2. Correctness and numerical contract

- Identify the arithmetic contract: quantization, accumulation order, FMA
  contraction, BF16 rounding, softmax reduction order, tolerance or golden
  values.
- For unchanged arithmetic, expect exact full-logit equality. For intentional
  changes, expect declared numerical limits and an independent reference.
- Reject "two Gufo paths agree" as proof of upstream correctness.
- Check awkward tails: 1/8/9/32/33 rows, partial tiles, ragged batches, empty
  inputs, saturated and unsaturated values.
- A green build is not quality evidence. A missing model or device skip is not
  a pass.

### 3. State, cache and snapshots

- Snapshot payload version, cache identity, prefix reuse, rollback, seeded
  replay, EOS commit and cancellation.
- Disk-cache restart, edited/rewound history, reasoning boundaries and image
  identity where applicable.
- If state files changed without `kSnapshotPayloadVersion` or an equivalent
  bump, require a justification.

### 4. API and client behavior

- Chat Completions and Responses, buffered and streaming.
- Reasoning boundaries, tool-call identity, arguments, partial calls and
  replay.
- History replay is stateless; preserve tool-call identity and image order.
- `/v1/models` and `/props` must reflect actual loaded capabilities, including
  image support.
- Client-facing errors, cancellation and queue/admission behavior.

### 5. Concurrency, lifetime and ownership

- Stream/scratch ownership, disjoint buffers, descriptor lifetime, device
  synchronization and rollback pointers.
- Per-request KV, recurrence, acceptance and RNG state independence.
- Cancellation before touching state; cleanup on failure.
- Allocation sizes and peak memory, especially on unified memory and Windows.

### 6. Platform

- Windows is the primary fork target; Linux compatibility is still required for
  shared code.
- POSIX shims in `compat/win32`, path handling, 8 MiB stack, socket semantics,
  `flock`, `renameat`, `posix_spawnp`.
- Unified-memory placement: a successful `hipHostRegister` is not proof that
  the next stream creation or GPU read works.
- DLL staging, ROCm kernel directories, GUI settings and executable selection.

### 7. Build, dependencies and licenses

- CMake presets, HIP arch (`gfx1151`), compiler flags and target wiring.
- Pinned ROCm/TheRock/clang/vcpkg versions; no unvalidated substitutions.
- `THIRD_PARTY_NOTICES.md`, vendored licenses and
  `check-dependencies.py`.
- Flakes only see tracked files; new files must be staged before a Nix build.

### 8. Tests and evidence

- Match the change to the table in [testing](../../../docs/TESTING.md).
- Distinguish what was executed from what was only inspected.
- Require exact commands, revisions, binary hashes, model/sidecar identities,
  corpus hashes and artifact paths for numerical claims.
- For API changes, start with `openai_chat_test` and `http_server_test`; include
  Qwen template tests when prompt rendering changes.
- For kernel/loader/state changes, require matched-token logits and perplexity
  against a recorded baseline, plus focused state/sampling/rollback checks.
- For performance, require matched workload identity and retained evidence;
  one unprofiled run is not a result.

### 9. Docs and consistency

- Model packages need the four documents, the model-index row and the
  `tools/ci/check-docs.py` model-list entry.
- Server flags need `PrintServeHelp` and parsing updated together; one
  canonical long option per behavior, no aliases.
- Successful optimizations become the default; do not keep experimental
  switches or duplicate tests.
- Update statements the new numbers contradict; do not leave stale TODO/N/A
  claims.

### 10. Performance claims

- Compare identical model artifacts, prompts, output counts, context depths,
  batching, sampling, cache history and concurrency.
- Separate request latency from aggregate throughput.
- Require a control and a mechanism-level explanation; a microbenchmark gain
  must survive the full inference path.
- Reject dead routes and environment switches that preserve experiments.

## Severity

- **Blocker:** wrong results, crash, data loss, API break, unqualified
  arithmetic, broken Windows or Linux behavior, license violation.
- **Major:** missing required check or evidence, wrong snapshot/cache version,
  capability-reporting drift, missing model docs, unsafe concurrency or
  lifetime, unvalidated toolchain change.
- **Minor:** missing edge-case test, incomplete docs, maintainability issue,
  unclear naming, avoidable allocation.
- **Nit:** style or wording that does not affect behavior.

## Output

Use this shape unless the user asks for something shorter:

```markdown
## Verdict
<merge-ready | ready with fixes | not ready | needs more evidence>

## Findings

### Blocker
- `path:line` — <problem, impact, required fix>

### Major
- ...

### Minor
- ...

### Nit
- ...

## Evidence

| Claim | Evidence | Status |
| --- | --- | --- |
| <claim from the change> | <command, artifact or file> | verified / inspected / unverified |

## Not verified
- <what was not run and why: no GPU, no model, no baseline, time budget>

## Open questions
- <only questions that block a verdict>
```

Status values:

- **verified**: a command or artifact in this review produced the result.
- **inspected**: the code/docs were read, but the behavior was not executed.
- **unverified**: the claim is asserted but not checked.

Never claim a test was run when it was not. Never turn a skipped model or
device into a pass.

## Rules

- Do not modify the worktree, stage files, commit, push or open a PR.
- Use `gh` only after checking `gh auth status` and only when GitHub access is
  authorized.
- Do not run full model sweeps, video generation or expensive benchmark
  campaigns during a review unless the user asks; review the evidence instead.
- Do not accept tolerance loosening, golden replacement or deleted tests as a
  way to make a change pass.
- Do not accept a benchmark number measured with a different workload identity.
- Do not mix unrelated code cleanup into a fix.
- Report what you could not verify explicitly; that is part of the review.

## Stop conditions

Stop and ask when:

- the base/head range is ambiguous or the history is shallow/unrelated
- the intent is unknown and the diff could be interpreted in materially
  different ways
- the diff is too large to review credibly in the available time
- required evidence depends on a model or GPU that is not available
- a blocker is found that makes further line-by-line review low value
