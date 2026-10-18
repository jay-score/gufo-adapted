---
name: qualify-inference-change
description: Qualify a Gufo inference change with the affected model's fast suite, matched-token logits/perplexity and state checks against a recorded baseline. Use when a change touches kernels, loading, tokenization, prefill/decode, sampling, caching or speculation.
metadata:
  origin: gufo
---

# Qualify an inference change

Read [AGENTS.md](../../../AGENTS.md), the change-selection table in
[testing](../../../docs/TESTING.md), and the affected model's
`docs/models/<model>/QUALITY.md` first. The declared model gate lives in
`docs/models/<model>/artifacts/qualify.json`; the runner is
[`tools/models/qualify.py`](../../../tools/models/qualify.py), implemented under
`tools/gufo/qualify/`. The Flash-Next declaration is the reference example.

The workflow is: **plan -> record baseline -> record candidate -> compare ->
verdict**. `plan` is advisory; `compare` is the gate. Do not report a
qualification as passed when a required item is missing, skipped or unverified.

## Intake

Establish before running:

- the intent, base ref and head ref (the working tree alone is not a baseline)
- which models and scopes `plan` selects, and whether any changed code path is
  uncovered
- the exact model files and sidecars (roles are declared in `qualify.json`)
- the baseline source: an isolated worktree or retained build at the planned
  base, with the same toolchain and model files
- whether arithmetic is expected to be unchanged (exact logits) or
  intentionally changed (declared model-owned limits)
- available GPU, model artifacts, time budget, and whether the run is on
  Windows native or inside `nix develop` on Linux

If the baseline binary or a required artifact is missing, stop. A missing
model, a skipped CTest test or an item with no required command for this
platform is not a pass.

## Stage 1: plan

```sh
python tools/models/qualify.py plan \
  --base origin/windows-port --head HEAD \
  --format markdown
```

For automation, write the JSON and keep it with the run:

```sh
python tools/models/qualify.py plan \
  --base origin/windows-port --head HEAD \
  --format json --output artifacts/qualify/<run>/plan.json
```

`plan` delegates diff collection to
[`review-context.py`](../../../tools/ci/review-context.py). It loads every
`qualify.json`, matches repository-relative path prefixes to scopes, unions the
declared checks and captures, and lists changed `src/` code that no declaration
covers.

Confirm from the plan:

- the affected model(s), scopes, checks, captures and baseline captures
- the declared profiles (`fast`, `full`, `reference`, `logits` where present)
- every uncovered path: either add the shared path to the affected model
  declarations or state explicitly why the workflow does not cover it
- every declared file and tool role you must supply

Use `--model` only for a deliberate manual selection; it does not replace a
matching trigger. Use `--scope` to force a scope the diff does not expose (for
example when validating a known change with a synthetic diff).

## Stage 2: record the baseline

Build the baseline from the planned base in an isolated worktree. Use the same
compiler, ROCm, dependency pins, model files and sidecars as the candidate.
Build any capture tool the plan names (for example the Flash-Next session test
or the Qwen27B target test) in that worktree.

```sh
python tools/models/qualify.py record \
  --model <model> --side baseline \
  --plan artifacts/qualify/<run>/plan.json \
  --gufo <baseline-worktree>/build/release/gufo \
  --file gguf=<target-gguf> \
  --file mtp=<sidecar> \
  --tool session_test=<baseline-worktree>/build/gpu-test/.../session_test \
  --repo <baseline-worktree> \
  --out artifacts/qualify/<run>/baseline
```

- Pass the first shard of a sharded GGUF; siblings are discovered and hashed.
- Pass all artifacts the selected captures use. The record hashes every file,
  the corpus, the config, the binary and any tools, and runs
  `gufo --version` plus `gufo diagnose --fingerprint --json`. If an older
  baseline binary cannot produce a fingerprint, pass the same
  `--fingerprint <json>` from this host to both records.
- On Windows, retain actual TheRock/clang, dependency and runtime identities
  separately. The built-in fingerprint can retain default compiler/driver
  labels; equal IDs alone do not establish matching toolchains.
- `{repo}` in a declaration means `--repo`, the checkout that built the binary;
  `--root` is the checkout that owns `qualify.json` and the corpus.
- Commands run in `--repo` by default; a relative declared `cwd` is relative
  to that checkout, so builds and CTest use the selected side's source tree.
- `{test:name}` locates a `gpu-test` executable in the build tree on either
  platform under `--repo`; it never falls back to the driver's build tree.
  `{tool:name}` is an explicitly supplied path, used when the
  baseline and candidate need different test builds.
- Baseline records captures only. Checks run on the candidate. A manual
  baseline capture can use `--profile logits`; the plan is preferred because it
  selects exactly the captures the triggered scopes require.
- A model with no two-build capture (currently DeepSeek) still needs the
  baseline identity record for the artifact/fingerprint check; `record` writes
  an identity-only record when the plan declares no baseline captures.
- A baseline may be reused only when its recorded config hash, file hashes,
  corpus hashes and fingerprint match the candidate run.

## Stage 3: record the candidate

Build the candidate from the planned head with the same toolchain and model
files, then run every check and capture the plan requires:

```sh
python tools/models/qualify.py record \
  --model <model> --side candidate \
  --plan artifacts/qualify/<run>/plan.json \
  --gufo build/release/gufo \
  --file gguf=<target-gguf> \
  --file mtp=<sidecar> \
  --tool session_test=build/gpu-test/.../session_test \
  --repo . \
  --out artifacts/qualify/<run>/candidate
```

`--profile fast` runs only `build` + `fast`; `--profile full` runs the declared
full check set. Explicit `--check`/`--capture`/`--profile` selection controls
what this invocation runs; the plan's required lists are still stored and
enforced by `compare`. A profile is a convenience for iteration, not a
substitute for the plan.

Use `--env NAME=VALUE` for declared environment placeholders (for example
`QUALIFY_IMAGE_A`, `QUALIFY_IMAGE_B` or `QUALIFY_IMAGE_DIR`) and child-process
environment overrides. Command-specific environment entries take precedence.
Relative `--out` and `--output` paths are resolved against `--root`. Each capture
invocation writes into a fresh directory so reruns cannot reuse earlier output.

On Linux, run the whole command inside `nix develop -c` so CMake, CTest and the
optional reference tools see the pinned environment. On Windows the declared
`build` check invokes `tools/windows/build.ps1`, which imports the MSVC
environment and stages runtime DLLs. Do not mix toolchains between baseline and
candidate.

## Stage 4: compare

```sh
python tools/models/qualify.py compare \
  --model <model> \
  --baseline artifacts/qualify/<run>/baseline \
  --candidate artifacts/qualify/<run>/candidate \
  --plan artifacts/qualify/<run>/plan.json \
  --out artifacts/qualify/<run>
```

The result is `qualification.json` plus a readable `qualification.md`. Exit 0
means `PASS` (or `CONTROL`); exit 1 means `FAIL` or `UNVERIFIED`.

`compare` refuses to produce a verdict when:

- the config, model files, corpus or toolchain fingerprint differ between sides
- the baseline or candidate revision is not the one the plan selected
- baseline and candidate are the same binary (unless `--control` is explicit)
- a required check or capture is missing, failed, skipped or not applicable
  (individual CTest results are retained as JUnit XML and checked for skips)
- a declared metric is absent or violates its model-owned limit
- a logit row differs under an exact comparator

`--control` runs the same binary on both sides to validate the harness. It is
never evidence that a change is qualified. Treat `UNVERIFIED` as not done:
fix the missing evidence or state exactly what remains unverified.

## Gates

- Unchanged arithmetic: require exact full-logit equality. The Flash-Next
  `logit-sha256` comparator checks every recorded raw row's SHA-256, not only
  perplexity. Qwen27B captures compare complete FP32 files with
  `file-sha256`.
- Intentional arithmetic change: require the model-owned limits from
  `QUALITY.md` and an independent reference where one exists. Declare a numeric
  `limits` entry only when a real model-owned threshold exists; exact
  comparators need none. Never loosen a tolerance, replace a golden or delete a
  gate to admit a change.
- State, sampling, rejection/rollback, EOS and cache behavior are separate
  checks. A logit match does not cover parser/API behavior, sampling draws or
  service replay.
- Missing model/device and CTest skip are not passes. Broadening a suite is
  better than silently accepting a gap.
- Commands for other platforms are recorded as `not-applicable` and skipped.
  Each item needs at least one required command for the current platform;
  commands marked `optional` are platform-only extras and cannot meet that
  requirement. Applicable commands must pass, including every CTest test.
- Preserve the exact corpus, tokenizer/template, schedules, seed, temperature
  and batching settings. A changed identity changes the comparison.
- `qualify.json` is part of the identity. If the declaration changes, re-record
  both sides; do not compare records made under different declarations.

## Evidence and finish

- Keep full records, logs and raw captures in the ignored top-level
  `artifacts/` directory. Commit only compact results that update a retained
  claim in `docs/models/<model>/QUALITY.md`, and say what was measured and what
  remains unverified.
- Do not commit multi-gigabyte logits, model files, profiler traces or private
  paths.
- Report: status (`PASS` / `FAIL` / `UNVERIFIED` / `CONTROL`), model and scopes,
  baseline/candidate revisions and binary hashes, model/sidecar/corpus hashes,
  commands, comparator results, limits, and every item not run.
- A benchmark number is not quality evidence. Run
  [benchmark-model](../benchmark-model/SKILL.md) only after the quality gate
  passes.

## Stop conditions

Stop and ask when:

- the baseline cannot be built at the planned base with the same toolchain
- model files or sidecars differ between sides
- the fingerprint differs without a stated, irrelevant reason
- a required check is skipped, missing or not applicable
- a mismatch is not attributed to the change
- a limit or tolerance change would be required
- an affected model has no `qualify.json`, or the plan leaves a changed
  inference path uncovered

## What this workflow does not do

- build a baseline automatically or choose the comparison revision
- decide whether an intentional arithmetic change is acceptable
- establish original-model or GGUF-conversion accuracy
- cover API/parser/reasoning/tool-call behavior (use the API-contract workflow)
- publish benchmarks or open pull requests
