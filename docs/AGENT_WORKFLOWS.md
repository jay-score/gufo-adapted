# Agent development workflows

Status: active roadmap. Last updated: 2026-10-17.

This document tracks agent-facing development workflows for this fork. It is the
planning and progress record; the canonical procedures stay in the linked
documents. A workflow is only "done" when its skill exists, its deterministic
tooling exists, and its checks have been exercised on a real change.

## Purpose

The repository already contains deep procedures in
[adding models](ADDING_MODELS.md), [testing](TESTING.md),
[benchmarking](BENCHMARKS.md), [performance](PERFORMANCE.md) and the existing
skills under [`.agents/skills/`](../.agents/skills/). The missing layer is
orchestration: a small number of gated workflows that tell an agent how to
start, what to verify, when to stop, and what evidence to leave behind.

The best existing model is the benchmark system: declarative `bench.json`, one
driver, retained artifacts and rendered tables. New workflows should copy that
pattern rather than add more prose.

## Principles

1. **Skill = intake -> stages -> gates -> evidence -> stop.** Keep skills short
   and link to canonical docs instead of duplicating them.
2. **Pair every skill with deterministic tooling.** A skill is a prompt, not a
   gate. If a rule matters, a script or CI check should enforce it.
3. **Definition of done is executable, or explicitly unverified.** Never accept
   "looks right" for numerical, state or platform work.
4. **Resumable state for multi-day work.** Model bring-up and performance
   campaigns need a small ledger: stage, decisions, artifacts, open questions,
   next step.
5. **Human checkpoints for expensive or irreversible actions.** Model
   downloads, full sweeps, toolchain changes, kernel rewrites and pushes.
6. **Platform-aware.** Windows is the primary fork target; Linux CI is still
   required for shared code. Do not assume Nix, `rocprofv3`, `gh` or a GPU.
7. **No subagent requirement.** The workflow must work in a single session;
   local setups may not support multi-session agents.
8. **Evals prevent rot.** Each skill needs a small fixture or scenario that
   should make it fail loudly when evidence is missing.

## Workflow catalog

| Workflow | Status | Skill | Tooling | Acceptance |
| --- | --- | --- | --- | --- |
| Review a change | Done | [`review-change`](../.agents/skills/review-change/SKILL.md) | [`tools/ci/review-context.py`](../tools/ci/review-context.py) | Classifies a diff, selects required checks, flags hazards, runs cheap checks, produces a severity-ranked report with an explicit unverified list |
| Qualify an inference change | Next | proposed | proposed `tools/models/qualify.py` | Selects and runs the affected model's fast suite, matched-token logits/PPL and state checks against a recorded baseline |
| Qualify a quantization variant | Next | proposed | proposed GGUF support-matrix tool | Enumerates tensor-role types, maps them to loaders/kernels, runs quality gates, then benchmarks and docs |
| Add a model | Proposed | proposed | proposed scaffold + completeness checker | Staged bring-up with an independent-reference parity gate before generation, API or performance work |
| Triage a bug | Proposed | proposed | none initially | Reproduces, minimizes, bisects, fixes and adds a regression test; handles device fault/OOM/hang branches |
| Investigate a regression | Proposed | proposed | proposed `tools/bench/compare.py` | Reproduces with matched settings, bisects, profiles and attributes the mechanism |
| Change the API contract | Proposed | proposed | existing server tests | Covers Chat Completions/Responses, tool calls, reasoning, streaming, cache/replay and capability reporting |
| Windows port change | Proposed | proposed | existing Windows scripts | Checks POSIX shims, memory placement, DLL staging, GUI settings and Linux preservation |
| Add a quantization type | Proposed | proposed | existing dequant/GEMM tests | Byte-layout parity, scalar dequant/dot correctness, then native kernels with declared arithmetic limits |
| Dependency or toolchain upgrade | Proposed | proposed | existing checks | Validated pins, kernel resource budgets, Linux CI and Windows runtime checks |
| CI failure triage | Proposed | proposed | existing presets | Separates infrastructure from code, reproduces the smallest failing check locally |
| State format change | Proposed | proposed | existing state tests | Snapshot version, migration, replay and rollback compatibility |
| Upstream sync | Existing | [`gufo-upstream-sync`](../.agents/skills/gufo-upstream-sync/SKILL.md) | [`tools/upstream-sync.py`](../tools/upstream-sync.py) | Pinned history, fix-first review, isolated worktree, validation plan |
| Upstream review | Existing | [`gufo-upstream-review`](../.agents/skills/gufo-upstream-review/SKILL.md) | none | Simplest suitable local route, risks, dependencies and validation |
| Upstream fixes | Existing | [`gufo-upstream-fixes`](../.agents/skills/gufo-upstream-fixes/SKILL.md) | none | Backport suitability with Windows risk and validation |
| Benchmark a model | Existing | [`benchmark-model`](../.agents/skills/benchmark-model/SKILL.md) | [`tools/bench/model-bench.py`](../tools/bench/model-bench.py) | Every retained cell measured, compared and rendered |
| Optimize a kernel | Existing | [`optimize-kernel`](../.agents/skills/optimize-kernel/SKILL.md) | profiling and microbenchmarks | One mechanism per change, matched A/B, quality gates, retained/rejected record |

## 1. Review a change

Status: done (2026-10-17).

Goal: a read-only, evidence-based review of a diff or pull request.

Delivered:

- [x] `.agents/skills/review-change/SKILL.md`
- [x] `tools/ci/review-context.py`
- [x] `tests/tools/review_context_test.py` (8 focused tests)
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

## 2. Qualify an inference change

Goal: make the numerical gate a single, repeatable workflow.

The repository requires matched-token logits and perplexity for changes to
kernels, loading, tokenization, prefill/decode, sampling, caching and
speculation. That requirement is currently spread across
[testing](TESTING.md) and per-model `QUALITY.md` files.

Proposed shape:

- a declarative `qualify.json` per model (fast suite, full suite, logit
  commands, tolerances, required artifacts)
- a runner that selects the affected model from the diff, captures baseline
  and candidate provenance, and writes an evidence artifact
- a report that lists exact commands, revisions, binary hashes, model/sidecar
  identities, corpus hashes and limits

This should reuse the benchmark driver's provenance discipline rather than
invent a second format.

## 3. Qualify a quantization variant

Goal: turn "this GGUF size loads" into a tracked decision.

This is the high-frequency quantization workflow: a new Q4/Q5/Q6/Q8/IQ variant
of an already-supported model. It is mostly qualification, not kernel work.

Steps:

1. Build a support matrix from the GGUF headers: tensor role, `GgmlType`,
   loader acceptance, native kernel, fallback/re-encoding.
2. Load the artifact and record the exact shard/sidecar identity.
3. Run the model's fast quality suite and matched-token logits/PPL against the
   qualified baseline.
4. Only then measure benchmarks and update `BENCHMARKS.md`, `QUALITY.md` and
   the model card.
5. Record the decision in `EXPERIMENTS.md` when it changes a default.

Do not treat a successful load as support. Do not publish benchmark numbers
before quality gates pass.

A new quantization *format* (a `GgmlType` the engine does not implement) is a
separate, deeper workflow: byte-layout parity, scalar dequant/dot, dispatch,
then native kernels. It must not be conflated with a new variant.

## 4. Add a model

Goal: a staged bring-up copilot, not a one-shot generator.

The stages are: intake and delta map, scaffold, config/loader, independent
numerical parity, generation/API dialect, state/speculation/vision, quality and
docs. Each stage ends with a report and an explicit stop. The independent
reference parity gate comes before any performance work.

Back it with a scaffold tool and a completeness checker. The checker matters
more: it verifies the model package, CMake wiring, dispatch branches, tests,
four documents, model index row, checker list entry, notices and tracked files.

Do not automate the delta analysis or the numerical judgment. That is the
actual engineering.

## 5. Other workflows

- **Triage a bug:** reproduce -> minimize -> bisect -> fix -> regression test.
  Include device fault, OOM, hang and cancellation branches.
- **Investigate a regression:** distinct from optimization. Reproduce with
  matched settings, bisect, profile, attribute, then fix.
- **API contract change:** Chat Completions/Responses, tool calls, reasoning,
  streaming, history replay and capability reporting.
- **Windows port change:** POSIX shims, unified-memory placement, DLL staging,
  GUI settings and Linux preservation. Fold the cheap checks into review.
- **Add a quantization type:** correctness first; native kernels second.
- **Dependency/toolchain upgrade:** pinned versions, kernel resource budgets,
  Linux CI and Windows runtime validation.

## What not to automate

- Full model bring-up without human checkpoints.
- Inventing kernel algorithms or choosing quantization trade-offs.
- Publishing benchmark numbers or opening/pushing pull requests.
- Full model sweeps or multi-gigabyte downloads on every change.
- Generic review without stated intent and evidence.
- Duplicating `AGENTS.md` or canonical docs inside skills.

## Rollout

- Phase 1: `review-change`; `qualify-inference-change`;
  `qualify-quant-variant`.
- Phase 2: `add-model` scaffold/checker; `triage-bug`; regression comparison.
- Phase 3: API contract, Windows hazards, dependency/state-format workflows.
- Phase 4: PR bot and scheduled benchmark refresh after local workflows are
  proven.

## Progress log

- 2026-10-17: Roadmap created.
- 2026-10-17: `review-change` workflow implemented. `review-context.py`
  classifies paths, maps them to the checks in `TESTING.md`, flags arithmetic,
  state, API, platform, model-doc and churn hazards, and can run the cheap
  format/docs/dependency checks. Eight focused tests pass; `check-docs.py`
  passes; the tool was smoke-tested on a real commit range.

## Open questions

- Should model qualification be declarative (`qualify.json`) or inferred from
  each `QUALITY.md`?
- What is the Windows substitute for `rocprofv3`-based profiling: kernel-bench,
  end-to-end model-bench, or both?
- Should skills be indexed in `.agents/skills/README.md` and checked for dead
  links/tools in CI?
- Which model and fixture should be the first skill eval?
