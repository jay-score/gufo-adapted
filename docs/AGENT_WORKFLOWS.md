# Agent development workflows

Status: active roadmap. Last reassessed: 2026-10-18.

This document plans and tracks workflows that help AI agents maintain and
improve this Windows fork of Gufo. Canonical procedures stay in the linked
documents. Implementation, fixture tests and real-task validation are recorded
separately; the existence of a skill or a successful control run is not enough
to call a workflow complete.

## Purpose

The goal is a repeatable maintenance loop: understand the request and affected
fork behavior, reproduce or record a baseline, make a focused change, run the
required checks, review the result and retain evidence for the next agent.
Success means agents can finish representative tasks with fewer unsupported
claims, missed regressions and unnecessary interventions.

The repository already contains procedures in
[adding models](ADDING_MODELS.md), [testing](TESTING.md),
[benchmarking](BENCHMARKS.md), [performance](PERFORMANCE.md) and the existing
skills under [`.agents/skills/`](../.agents/skills/). The missing layer is
consistent orchestration: how to select and compose those procedures, what
evidence permits progress, when information is missing, and how to resume.
The [fork behavior inventory](UPSTREAM_SYNC.md#fork-behavior-to-preserve)
records obligations that incoming changes must preserve.

The benchmark system provides a useful pattern: declarative `bench.json`, one
driver, retained artifacts and rendered tables. Reuse that pattern where a
new executable gate is needed; reuse existing tools and tests elsewhere.

## Principles

1. **Skill = intake -> stages -> gates -> evidence -> stop.** Keep skills short
   and link to canonical docs instead of duplicating them.
2. **Enforce objective gates with tools.** Existing tests, presets and scripts
   may be sufficient. Add a runner only when it closes an enforcement or
   reproducibility gap; engineering judgment remains the agent's responsibility.
3. **Completion requires evidence.** Make objective acceptance executable and
   identify unverified work. Never accept "looks right" for numerical, state
   or platform work.
4. **Resumable state for multi-day work.** Model bring-up and performance
   campaigns need a small ledger: stage, decisions, artifacts, open questions,
   next step.
5. **Respect authorization and cost.** Continue authorized, reversible work.
   Establish scope before additional downloads, costly campaigns, toolchain
   changes or publication. Do not ask again for authorization already given;
   a stage boundary is an evidence gate, not an automatic permission request.
6. **Platform-aware.** Windows is the primary fork target; Linux CI is still
   required for shared code. Do not assume Nix, `rocprofv3`, `gh` or a GPU.
7. **No subagent requirement.** The workflow must work in a single session;
   local setups may not support multi-session agents.
8. **Test both tools and agent behavior.** Fixture tests check deterministic
   enforcement. Separate agent scenarios check whether the skill selects the
   right workflow and refuses unsupported conclusions. Neither replaces the
   other or real-task validation.
9. **Compose workflows.** Bug fixes, upstream integrations and optimizations
   should reuse review, qualification and benchmark procedures rather than
   maintain parallel versions of their gates.

## Progress and evidence

Use these maturity levels with an explicit scope:

- **Proposed:** requirements are recorded; implementation remains.
- **Implemented:** the skill and required tooling or existing check wiring exist.
- **Fixture-tested:** deterministic tests cover selection, success and refusal
  paths without requiring large artifacts or a GPU.
- **Exercised:** a bounded real diff or artifact has been used; identify which
  parts ran and whether the exercise was only a control.
- **Validated:** the workflow has completed a representative real task with
  retained commands, identities, results and an explicit coverage statement.

"Existing" in the catalog means a workflow predates this roadmap; its maturity
has not been reassessed here. It is not a validation claim. A workflow is done
only for its validated scope. Validation on one model or platform does not
establish support for every model or platform.

Keep workflow maturity separate from an individual run's verdict. A missing
required artifact, skipped required check, uncovered path or absent required comparator
cannot produce a pass. `CONTROL` validates the harness, not a changed build.
Use the runner's failure/unverified status and explain the gap.

Each validation record needs the task and scope, exact commands, source/build
and input identities, results, artifact locations and limitations. Multi-day
work also needs decisions, open questions and the next action. Keep large/raw
outputs in ignored `artifacts/`; retain compact evidence with the maintained
claim. Follow [testing](TESTING.md) and [artifact retention](BENCHMARKS.md).

## Workflow catalog

| Workflow | Status | Skill | Tooling | Acceptance |
| --- | --- | --- | --- | --- |
| Review a change | Exercised: context tool | [`review-change`](../.agents/skills/review-change/SKILL.md) | [`tools/ci/review-context.py`](../tools/ci/review-context.py) | Classifies a diff, selects required checks, flags hazards, runs cheap checks, produces a severity-ranked report with an explicit unverified list |
| Qualify an inference change | Exercised: GPU control only | [`qualify-inference-change`](../.agents/skills/qualify-inference-change/SKILL.md) | [`tools/models/qualify.py`](../tools/models/qualify.py), per-model `qualify.json` | Runs affected model checks and comparisons against a recorded baseline; a changed-build exercise remains |
| Qualify a quantization variant | Proposed: next implementation | proposed | proposed GGUF inspection and variant runner | Establishes runtime routes and bounded quality against a declared reference before benchmarks |
| Triage a bug | Proposed | proposed | existing tests, debugger and Git | Reproduces, minimizes, bisects when useful, fixes and adds a regression test; handles device fault/OOM/hang branches |
| CI failure triage | Proposed | proposed | existing presets and CI logs | Separates infrastructure from code, reproduces the smallest failing check locally |
| Change the API contract | Proposed | proposed | existing server tests | Covers Chat Completions/Responses, tool calls, reasoning, streaming, cache/replay and capability reporting |
| Windows port change | Proposed | proposed | existing Windows scripts | Checks POSIX shims, memory placement, DLL staging, GUI settings and Linux preservation |
| Investigate a regression | Proposed | proposed | existing benchmark artifacts; compare helper if needed | Reproduces with matched settings, bisects, profiles and attributes the mechanism |
| State format change | Proposed | proposed | existing state tests | Snapshot version, migration or explicit invalidation, replay and rollback compatibility |
| Dependency or toolchain upgrade | Proposed | proposed | existing checks | Validated pins, kernel resource budgets, Linux CI and Windows runtime checks |
| Add a model | Proposed | proposed | completeness checker first; scaffold later | Staged bring-up with an independent-reference parity gate before generation, API or performance work |
| Add a quantization type | Proposed | proposed | existing dequant/GEMM tests | Byte-layout parity, scalar dequant/dot correctness, then native kernels with declared arithmetic limits |
| Upstream sync | Existing | [`gufo-upstream-sync`](../.agents/skills/gufo-upstream-sync/SKILL.md) | [`tools/upstream-sync.py`](../tools/upstream-sync.py) | Pinned history, fix-first review, isolated worktree, validation plan |
| Upstream review | Existing | [`gufo-upstream-review`](../.agents/skills/gufo-upstream-review/SKILL.md) | none | Simplest suitable local route, risks, dependencies and validation |
| Upstream fixes | Existing | [`gufo-upstream-fixes`](../.agents/skills/gufo-upstream-fixes/SKILL.md) | none | Backport suitability with Windows risk and validation |
| Benchmark a model | Existing | [`benchmark-model`](../.agents/skills/benchmark-model/SKILL.md) | [`tools/bench/model-bench.py`](../tools/bench/model-bench.py) | Every retained cell measured, compared and rendered |
| Optimize a kernel | Existing | [`optimize-kernel`](../.agents/skills/optimize-kernel/SKILL.md) | profiling and microbenchmarks | One mechanism per change, matched A/B, quality gates, retained/rejected record |

## 1. Review a change

Status: implemented and fixture-tested; context tool exercised on real commit
ranges. End-to-end agent review validation remains to be recorded.

Goal: a read-only, evidence-based review of a diff or pull request.

Delivered:

- [x] `.agents/skills/review-change/SKILL.md`
- [x] `tools/ci/review-context.py`
- [x] `tests/tools/review_context_test.py` (13 focused tests as of 2026-10-18)
- [x] CI hooks in both the Windows CPU and Ubuntu repository-check jobs
- [x] stack classification, required-check selection, risk flags and
      `--run-checks` for format/docs/dependencies
- [x] smoke-tested against the current checkout and a real commit range

The tool classifies changed paths by ownership area and recommends the checks
required by [testing](TESTING.md) and [AGENTS.md](../AGENTS.md). It also flags
repo-specific hazards:

- arithmetic-sensitive kernels and quantization paths without logit evidence
- state/cache/snapshot changes without a payload-version decision
- API changes without buffered/streaming/replay coverage
- Windows-affecting shared code without Linux preservation
- new model packages missing the four documents, index row or checker entry
- formatting churn, large/binary artifacts, untracked files and dirty trees

The skill reviews intent, correctness, state, API, concurrency, platform,
build/dependencies, tests, docs and performance claims. It must distinguish
"verified by command/artifact" from "inspected" and "unverified".

Acceptance:

- A diff that changes `src/core/quant/*` recommends the dequant/GEMM tests and
  a matched-token logit/PPL gate.
- A diff that changes `src/cli/serve/openai_chat.cpp` recommends
  `openai_chat_test` and `http_server_test`, plus streaming/replay coverage.
- A diff that adds `src/models/<name>/` flags missing model docs and dispatch
  wiring.
- A whitespace-only reformat is flagged as churn.
- A clean tree with no code changes does not recommend GPU or logit work.
- The tool never modifies the worktree.

Next validation: complete one bounded real review through the skill, retaining
the context packet, findings, executed checks and unverified items. Evaluate
whether the agent reads the connected implementation and distinguishes a
check recommendation from proof that the check passed.

## 2. Qualify an inference change

Status: implemented and fixture-tested; Flash-Next GPU control exercised.
End-to-end validation on a real changed build remains.

Goal: make the numerical gate a single, repeatable workflow.

The repository requires matched-token logits and perplexity for changes to
kernels, loading, tokenization, prefill/decode, sampling, caching and
speculation. That requirement is now executable instead of spread across
prose:

- `.agents/skills/qualify-inference-change/SKILL.md`
- `tools/models/qualify.py` (implementation under `tools/gufo/qualify/`)
- `docs/models/<model>/artifacts/qualify.json` for the three text models
- `tests/tools/qualify_test.py` (21 focused tests as of 2026-10-18, no model
  or GPU required)
- CI hooks in both the Windows CPU and Ubuntu repository-check jobs

Delivered shape:

- a declaration per model: file/tool roles, path-prefix triggers,
  scope-to-check/capture requirements, fast/full/reference/logit profiles,
  capture commands, comparators, declared metric limits and required outputs
- `plan` delegates diff collection to `review-context.py`, selects affected
  models and scopes, and reports uncovered inference paths
- `record` runs one side, hashes the binary, model/sidecar shards, corpus,
  config and tools, captures source revision/fingerprint, and writes logs plus
  a resumable JSON record
- `compare` refuses mismatched config/model/corpus identities and fingerprints,
  enforces plan coverage, compares complete raw-logit rows with the existing
  `tools/bench/logit-eval.py`, evaluates declared metric limits, and writes
  `qualification.json` plus `qualification.md`
- `report` re-renders retained evidence

Exercise: a real Flash-Next UD-Q4_K_XL run on Windows/gfx1151 recorded both
sides of the `1:2,4,8:p8` teacher-forced schedule (225 positions, 248,320-token
vocabulary), hashed all four shards and the Q8_0 MTP sidecar, and compared all
three raw-row files. The same-binary control reported `CONTROL` with three
matches. This validates the harness; it is not evidence that a changed build is
qualified.

`gufo bench --logit-eval` is implemented for Flash-Next. Qwen27B uses explicit
`--capture-logits`/`--capture-prefill-logits` file comparisons. DeepSeek keeps
its model-owned DSpark/AR replay and independent reference suites; it has no
two-build logit capture. Do not invent a weaker gate for it.

Coverage is limited to these declarations and their implemented captures.
ASR, TTS, image and audiovisual models do not yet have `qualify.json` gates;
their model-owned procedures still apply. Surface uncovered requirements rather
than treating declaration presence or a passing profile as full coverage.

Next validation: use the runner on an actual bounded inference change, with
distinct baseline/candidate builds and matched artifacts/toolchains. Retain the
plan, required model/state checks, comparisons and verdict. If arithmetic is
unchanged, require exact full-logit equality; intentional changes need the
model-owned limits and reference evidence required by [testing](TESTING.md).
Use the next suitable real change rather than inventing an engine change to
exercise the workflow. Existing manual qualification records are useful
precedents but do not validate this runner end to end.

## 3. Qualify a quantization variant

Status: proposed; next implementation after selecting a bounded pilot.

Goal: make a reproducible support and quality decision for a new Q4/Q5/Q6/Q8/IQ
variant of an already-supported model. A successful load alone is insufficient.

### Comparison contract

Keep two questions separate:

- **Engine regression:** different builds, identical model artifacts and
  matched settings. Workflow #2 owns this; unchanged arithmetic requires exact
  logits.
- **Variant quality:** normally the same build, different weight artifacts
  from the same identified source checkpoint, with matched token histories and declared
  numerical limits. Exact logits across quantizations are not expected.

The current `qualify.py compare` deliberately rejects changed model identities;
identical binaries are allowed only for a control. Preserve those rules. Reuse its
provenance, capture and reporting helpers where suitable, with a separate
variant comparison contract. If variant support also requires engine changes,
qualify those changes separately; do not change both axes without controls.

Declare the reference's purpose: an independent implementation of the same
quantized artifact checks engine execution; a full-quality teacher measures
quantization degradation; a retained Gufo build checks regression. One does
not establish the others. Gufo BF16 comparisons must state that they do not
establish original-checkpoint or GGUF-conversion parity.

### Pilot and inputs

Start with one model and one candidate. Qwen27B is the preferred first pilot:
its [target test](../tests/models/qwen27b/target_test.cpp) already measures
full-vocabulary distribution differences against BF16. Existing
[quantization evidence](models/qwen3.8-27b/artifacts/quantization-quality.json)
is a measurement precedent, not an acceptance policy. The current comparison
reports metrics without enforcing quantization quality limits; the 24-label
continuation NLL difference is not a corpus perplexity score.

Before a real qualification run, establish:

- candidate and teacher/reference artifacts, checkpoint lineage, all shards,
  tokenizer/template identity, required sidecars and their immutable hashes
- model-owned limits and rationale for distribution drift, matched-token
  NLL/perplexity and a small relevant capability fixture, chosen before judging
  the candidate; do not turn historical measurements into arbitrary thresholds
- fixed corpus/token histories, context, batching, sampling and speculative
  settings, with enough workload coverage to exercise the claimed routes
- validated build/toolchain, GPU, memory/disk capacity and campaign scope;
  load large reference and candidate models sequentially where needed

The inspector and CPU fixtures can be implemented before those large artifacts
are available. Real-task validation cannot. A Gufo BF16-only pilot must retain
its reference limitations; missing required teacher or capability evidence
prevents a full quality pass. Flash-Next is a useful second pilot for sharding,
mixed types, expert routes and dense re-encoding.

### Deliverables and stages

1. **Inspect.** Reuse [GGUF helpers](../tools/gufo/gguf.py) where suitable to
   enumerate all shards and tensor names, roles, shapes and stored types.
   Detect missing/duplicate tensors, incompatible metadata and invalid storage.
2. **Resolve support.** Produce a model-owned matrix of loader acceptance,
   runtime type, prefill/decode/verification route, fallback or re-encoding,
   and memory expansion. Headers alone cannot prove kernel support. Tie rules
   to actual loader/dispatch policy and test their agreement; unknown roles,
   shapes or types remain unsupported or unverified. Avoid a second unchecked
   Python list of purported engine capabilities. For example, Flash-Next
   re-encodes some dense K/I-quant matrices to Q8_0 while experts retain native
   quantized routes.
3. **Record and check.** Retain exact artifact/build identities and confirm
   loading and execution routes. Run the fast suite plus the candidate's
   affected operator/model checks. State, rollback, speculation and vision
   checks apply to the modes being claimed. A CPU-only fast suite does not
   qualify the candidate weights. Prevent state/cache reuse across mismatched
   model identities.
4. **Compare quality.** Compare complete vocabulary distributions on identical
   token histories and matched-token NLL/perplexity, then the declared capability
   fixture. Use an explicit comparator and model-owned limits; reject missing,
   nonfinite or out-of-limit metrics. Report reference limitations and any
   unresolved model-quality failures without relaxing gates.
5. **Decide and retain.** Write JSON and Markdown evidence with support status,
   quality verdict, per-check coverage, reference identity, commands and gaps.
   Only a passing quality gate permits retained performance measurements via
   `benchmark-model`. Update `QUALITY.md`, the model card and `BENCHMARKS.md`
   for claims actually measured; record default changes in `EXPERIMENTS.md`.

Deliver a short `qualify-quant-variant` skill, an artifact-driven inspector and
runner, and a model-owned declaration for the pilot. No source diff should be
required. Reuse existing suites and recording helpers; do not build a general
experiment framework or implement every model in the first change.

### Acceptance

- CPU fixtures cover malformed/truncated GGUFs, missing/duplicate shards or
  tensors, unsupported role/type/shape combinations and conversion reporting.
- A matrix claim is checked against loader/dispatch behavior. Recognizing a
  `GgmlType` or accepting a tensor is not enough to claim all runtime modes.
- Comparison fixtures allow the intended artifact differences while rejecting
  checkpoint/tokenizer/corpus/settings mismatches and undeclared sidecar changes.
- Missing/skipped checks, absent comparators or limits, nonfinite values and
  exceeded limits prevent a pass. Tests cover both acceptance and refusal.
- One real candidate completes the declared workflow with retained evidence.
  Agent evaluation checks that load-only or short-NLL-only evidence cannot
  become a full support claim or trigger benchmark publication.
- Inspection/declaration fixture checks run in Windows and Linux CPU CI.
  Real GPU qualification stays local and bounded.

A new quantization *format* (a `GgmlType` the engine does not implement) needs
the separate byte-layout, scalar dequant/dot, dispatch and kernel workflow.
A recognized format used in an unsupported tensor role or route can also need
engine work; report that gap rather than treating every variant as load-only.

## 4. Add a model

Status: proposed; follows the routine maintenance workflows.

Goal: staged model bring-up with explicit evidence at each boundary.

The stages are: intake and delta map, scaffold, config/loader, independent
numerical parity, generation/API dialect, state/speculation/vision, quality and
docs. Each stage ends with a report and an evidence gate. Continue within the
authorized scope when the gate passes; stop for unresolved requirements or
decisions outside that scope. Independent reference parity comes before
generation/API qualification and performance work.

Build the completeness checker first: model package, CMake wiring, dispatch
branches, tests, four documents, model index row, checker list entry, notices
and tracked files. Add scaffolding only after a real bring-up demonstrates
which repetitive steps are stable enough to generate.

Do not automate the delta analysis or the numerical judgment. That is the
actual engineering.

## 5. Other workflows

- **Triage a bug:** reproduce -> minimize -> bisect -> fix -> regression test.
  Bisect when it helps establish the cause. Include device fault, OOM, hang
  and cancellation branches. Reuse review and inference qualification when
  applicable; preserve a failing reproducer before changing behavior.
- **CI failure triage:** classify infrastructure versus source failure,
  reproduce the smallest failing target using the matching environment, and
  retain the original error and resolution. Do not hide failures with retries,
  tolerance changes or disabled tests.
- **Investigate a regression:** distinct from optimization. Reproduce with
  matched settings, bisect, profile, attribute, then fix. Reuse benchmark
  artifacts; add a comparison helper only for a concrete gap in identity,
  workload matching or result comparison.
- **API contract change:** Chat Completions/Responses, tool calls, reasoning,
  streaming, history replay and capability reporting.
- **Windows port change:** POSIX shims, unified-memory placement, DLL staging,
  GUI settings and Linux preservation. Fold the cheap checks into review.
- **State format change:** explicit version and migration or invalidation
  decision, cache/model identity, snapshot/restore, rollback and restart/replay
  fixtures. Reuse affected model suites.
- **Add a quantization type:** correctness first; native kernels second.
- **Dependency/toolchain upgrade:** pinned versions, kernel resource budgets,
  Linux CI and Windows runtime validation.

## What not to automate

- Unbounded model bring-up, downloads or full sweeps on routine changes.
- Automatic acceptance of kernel algorithms or quantization trade-offs from
  performance alone.
- Publication, pushes or pull-request creation outside the user's scope.
  Authorized preparation and local evidence generation should continue.
- Generic review without stated intent, connected-code inspection and evidence.
- Separate qualification/benchmark engines for every task-specific skill.
- Duplicating `AGENTS.md` or canonical procedures inside skills.

## Workflow health and agent evaluation

Add cheap CI checks for skill discovery, local links, referenced repository
entrypoints and declaration validity. Existing documentation checks already
cover Markdown links; add only missing checks. Commands needing external tools
or large artifacts need declared prerequisites, not mandatory execution in CPU CI.

Start agent evaluation with small scenarios tied to the implemented workflows:
an arithmetic-sensitive diff with missing logit evidence, an uncovered model
path, a same-binary control presented as qualification, and a GGUF that loads
but lacks quality evidence. The expected result includes the selected procedure,
missing evidence and a refusal to claim completion. Record scenario, agent
configuration, actions and outcome separately from tool unit-test results.

Require a successful real-task record before marking a workflow validated.
Expand evaluation when a real failure reveals a gap; do not require a broad
agent benchmark or recurring automation to finish an individual workflow.

## Rollout

1. **Close foundation evidence gaps.** Retain a complete real review through
   `review-change` and a changed-build run through `qualify-inference-change`.
   Add cheap workflow health checks and the first agent scenarios. These tasks
   can proceed alongside the CPU implementation of #3.
2. **Qualify one quantization variant.** Select pilot artifacts and declare
   limits; implement inspection, support resolution and variant comparison;
   run fixtures and one real qualification before extending to another model.
3. **Support routine maintenance.** Add bug and CI triage, then focused API
   contract and Windows-port guidance using existing tests and scripts.
4. **Improve diagnosis and change control.** Add regression comparison where
   needed, state-format and dependency/toolchain workflows. Use concrete tasks
   to decide where new tooling is justified.
5. **Expand model support.** Build the model completeness checker, then a
   scaffold based on observed bring-up work. Add quantization-format guidance
   when an actual unsupported format is requested.
6. **Automate proven local workflows.** Consider PR bots and scheduled benchmark
   refresh only after representative local tasks reliably produce evidence.
   Repository/account writes and recurring compute require explicit scope.

Priority follows maintenance needs. A requested task may bring a later workflow
forward; it does not require completing the catalog first.

## Progress log

- 2026-10-17: Roadmap created.
- 2026-10-17: `review-change` workflow implemented. `review-context.py`
  classifies paths, maps them to the checks in `TESTING.md`, flags arithmetic,
  state, API, platform, model-doc and churn hazards, and can run the cheap
  format/docs/dependency checks. Eight focused tests pass; `check-docs.py`
  passes; the tool was smoke-tested on a real commit range.
- 2026-10-17: `qualify-inference-change` workflow implemented. Declarative
  `qualify.json` files cover Flash-Next, Qwen3.8 27B and DeepSeek V4 Flash;
  `plan`/`record`/`compare`/`report` retain provenance and refuse mismatched
  artifacts, coverage gaps and exact-logit differences. Thirteen fixture tests
  pass; the plan was exercised on real commit ranges and the logit path was
  exercised on Windows/gfx1151 with the real Q4_K_XL target and Q8_0 MTP
  sidecar (control run, three exact row comparisons).
- 2026-10-18: Roadmap reassessed for fork maintenance. Replaced blanket "done"
  labels with scoped maturity and remaining real-task validation. Defined #3's
  separate artifact comparison contract, pilot inputs, runtime support matrix,
  quality gates and acceptance fixtures. Moved bug/CI/API/Windows maintenance
  ahead of model scaffolding and deferred bots/scheduling. Local reassessment
  verified 13 review-context tests, 21 qualification tests and documentation
  integrity; it did not run new GPU qualification. Earlier test counts above
  describe the original delivery dates.

## Open questions

- **Resolved:** inference qualification uses declarative `qualify.json` gates;
  `QUALITY.md` remains the measured record and explanation. Existing exact
  build-comparison rules remain intact when adding variant qualification.
- **Pilot decision:** which Qwen27B candidate and full-quality reference are
  available, what acceptance limits and capability fixture are justified, and
  what Windows/GPU budget is authorized? These block a real #3 quality pass,
  not inspector/fixture implementation.
- **Support policy:** what is the smallest way to expose or verify model-owned
  loader/dispatch rules without duplicating them in an unchecked support table?
- **Coverage:** which missing model/platform qualification should be added next,
  based on actual maintenance work and available independent references?
- **Windows profiling:** what supported tools can attribute kernel and wall-time
  costs here? Keep microbenchmark and end-to-end measurements distinct; neither
  alone substitutes for a GPU timeline when attribution requires one.
- **Discovery:** is the catalog sufficient, or does the agent setup need a small
  skill index? Add one only if it improves discovery without a second manually
  maintained workflow registry.
