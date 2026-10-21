# Windows upstream sync log

Track merges from [official Gufo](https://github.com/gufo-org/gufo) into the
Windows integration branch. This log starts after the October 2026 forward-port;
it is not a retroactive inventory of the old fork's selective imports.
An imported PR is not automatically qualified on Windows.

## Repeatable procedure

1. Follow the [sync skill](../.agents/skills/sync-windows-upstream/SKILL.md).
   Verify remote roles, worktree status and local versus remote `windows-main`
   before fetching. Preserve unpublished Windows work.
2. Record the Windows base and fetched `official/main` SHA before merging.
   List **all** incoming commits with `git log --reverse --oneline
   WINDOWS_BASE..OFFICIAL_HEAD`; resolve PR numbers and merge SHAs with
   `gh pr view NUMBER --repo gufo-org/gufo --json number,title,url,mergedAt,mergeCommit`.
   Include release commits and dependencies in the entry.
3. Merge that exact upstream SHA on an isolated candidate descended from the
   Windows base. An already isolated T3 task branch is a suitable candidate.
   Preserve official ancestry through a merge commit. Review cleanly merged
   Windows call sites as well as conflicts; retain both platforms' evidence.
4. Review all six [compatibility contracts](WINDOWS-COMPATIBILITY.md).
   Audit new callers and moved implementations. Run the bounded Windows CPU
   gate, affected GPU/model checks, documentation/formatting checks and the
   available Linux gate. Preserve individual timing results and failures.
5. Append an entry below with the PR table, base/head identities, resolutions,
   six contract outcomes, exact checks, evidence location and pending limits.
   Verify each PR merge SHA and the recorded official head with
   `git merge-base --is-ancestor SHA HEAD`. Do not replace old entries.
6. Integrate a qualified candidate into `windows-main` only after rechecking
   its base. Publication follows the requested endpoint; a sync PR must use
   merge merging to retain ancestry. Never squash/rebase the upstream merge
   or force-push the maintained branch.

For the next sync, copy the shape of the latest entry and replace its evidence.
Use **validated**, **reviewed/unchanged**, or **pending (reason)** per contract;
do not count missing models, excluded disk persistence or noisy timings as passes.

## October 21, 2026

Status: **candidate validation in progress** on `t3/sync-upstream-oct-21-prs`.

- Windows base: `db4f6fbcea0514db1456abf8571f20cb525de30e` (also
  `origin/windows-main` at fetch time; no unpublished divergence).
- Previous official ancestor: `f020bd3a`.
- Imported official head: `e060aeed923730b74ea3f4116c3c39de55741370`.
- Integration: ordinary merge of the recorded official head; use
  `git log --merges --oneline -- docs/WINDOWS-UPSTREAM-SYNC.md` to locate it.

| PR | Upstream change | Merge SHA |
| --- | --- | --- |
| [#692](https://github.com/gufo-org/gufo/pull/692) | Compare the bound publication waiter with cold by answer | `0e4f987b8145bc37fca25b601881abd369982c36` |
| [#540](https://github.com/gufo-org/gufo/pull/540) | Qwen 27B AR and DFlash2 cache continuation | `da5d5a2b237c654bc0523e33fdee10465240da0c` |
| [#690](https://github.com/gufo-org/gufo/pull/690) | Wide Flash-Next routes for prefill passes above eight rows | `53f73b5653b32caf1d8cb3113c3f2f05f0e88d0a` |
| [#699](https://github.com/gufo-org/gufo/pull/699) | Reduce Flash-Next sampled decoding overhead | `e060aeed923730b74ea3f4116c3c39de55741370` |

The only additional incoming commit is upstream's v0.20.0 release
`0c03748a`. Its version, manifest and changelog changes are retained as published.
No version is manually assigned by this sync.

### Merge review

Only `docs/models/qwen3.8-flash-next/EXPERIMENTS.md` and `QUALITY.md`
conflicted: both branches inserted independent evidence at the top. Retained
the October 21 upstream sections and the October 20 Windows n-gram evidence.
No source file required conflict resolution.

Reviewed the clean merges in `inference_backend.cpp`,
`text_model_runner.cpp`, `text_generation_scheduler.cpp` and the Flash-Next
ROCm executor: WDDM-aware resource claims, native host-memory accounting,
live prefill metrics and snapshot admission remain present. Shared snapshot
copy kernels moved to `src/core/hip/snapshot_copy.*`; their consumers and
both model build lists follow the move. Windows disk-store guards remain.

The shared transfer pool now supports a bound execution stream. Native pinned
transfer coverage additionally checks real copies on that stream, invalid and
repeated binding, and stream-owner retention after pool/lease destruction until
the completion releases it. Linux fault-injection harnesses remain unchanged.
The first native run also exposed a stale pre-sync fixture: committed backing
now registers anonymous pages, but the test still queried `hipHostMalloc`
flags. It now checks the host registration and device mapping; staging still
checks `hipHostMallocCoherent`. The allocator is unchanged by this sync.
See [AMD's memory API contract](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___memory.html).

### Compatibility and validation

Validation results and pending qualifications are recorded here before handoff.
Evidence is retained under `build/sync-2026-10-21/` in the task checkout.
The previous Windows executable and colocated DLLs remain in the maintained
checkout; the candidate builds into this task's own `build/` directories.

| Contract | Outcome |
| --- | --- |
| Cached concurrent n-gram reads | Native independent row/hash/offset/descriptor checks pass. Windows still uses `O_CONCURRENT_RANDOM`; fresh-process AR/MTP controls below. |
| WDDM admission budgets | Validated: native live DXGI/HIP oracle and allocation/fallback checks pass. Existing admission helpers survive the merge. Multi-GPU switching remains unqualified. |
| Pinned memory and transfer lifetime | Validated: corrected/expanded native test passes, including bound-stream copies/ownership. Flash-Next AR/MTP 4K cross-slot probes reproduce every captured component, logits, next-step tokens and actual MTP drafts. Fault injection and physical memory pressure remain unqualified. |
| Sockets and shutdown | Implementation unchanged; all bounded CPU transport and real console-shutdown checks pass. |
| Filesystem identity, privacy and failures | Implementation and both checkpoint-store guards unchanged; bounded CPU identity/privacy/failure checks pass. Disk persistence remains deferred. |
| GPU queue policy | Windows boundary unchanged; CPU and HIP startup checks pass. No Windows queue-residency/performance claim. |

Completed checks (TheRock 10.0.0/Clang 23.0.0, MSVC 14.51.36231,
SDK 10.0.26100.0, CMake 4.4.0; compiler/cache details in the retained logs):

- `tools/windows/build.ps1 -Preset windows-cpu -Vcpkg VCPKG_PATH` and
  `ctest --preset windows-cpu --no-tests=error`: **15/15 pass** (17.83 seconds).
- `tools/windows/build.ps1 -Preset windows-gpu -Vcpkg VCPKG_PATH`: production
  HIP executable builds. Both builds use this checkout's pinned vcpkg manifest.
- Affected extra CPU checks: sampling, idle spill, Qwen continuation layout,
  Flash-Next n-gram, MTP sampling and prompt lookup: **6/6 pass**.
- Focused GPU gate: WDDM, pinned transfers, queue policy, speculative verifier,
  Qwen attention policy/long-context operators, Flash-Next projections,
  routed experts, snapshot copies and sampled-verification oracle: **10/10
  pass across the initial run and the focused pinned-fixture rerun**. The
  original failed pinned-fixture result is retained, not overwritten.
- Flash-Next continuation probe: `--mode ar` and `--mode mtp`, each with
  `--context 4096 --sessions 2 --backing-gib 1 --round-trip`: **pass**.
- Shared `tools/ci/check-format.py` with clang-format 21.1.8,
  `tools/ci/check-docs.py`, `tools/ci/check-dependencies.py`, and
  `git diff --check`: **pass**.
- Functional-runner unit tests under WSL: **80/80 pass**; new Qwen recovery
  evidence-refusal tests: **6/6 pass** on Windows and WSL. The initial Windows
  runner invocation hit a default-encoding failure (the affected case passes
  with Python UTF-8 enabled) and the existing POSIX-only `SIGHUP` harness.

Additional Windows harness exclusions, retained as failures in the evidence:
`cache_slot_test` requires POSIX `fork`/`SIGABRT`; `component_text_cache_test`
and `text_model_runner_test` reach the deliberately disabled disk stores.
These broader tests are outside the bounded Windows preset. They do not pass
on Windows and were not made to pass by weakening the store guards.

### Live checks and limits

Both Flash-Next AR and MTP pass the SDK `long-context` and `cache-depth`
suites with 32K contexts, two sessions and an explicit 8 GiB RAM budget.
They exercise multi-turn recall, exact warm/cold answers, deep RAM reuse,
rewritten branches and peer rotation. MTP responses contain actual proposed
and accepted drafts. Each suite retains all request/phase timings; these
correctness runs do not by themselves qualify speed.

The additional `sampling-defaults` suite passes its explicit/default sampler
output checks but fails its zero-prefill replay assertion after the deep-cache
history: 15 of 22 prompt tokens are reused, and seven are re-prefilled.
The exact pre-sync AR baseline reproduces the same output and failure under
the same history. A fresh MTP baseline passes this suite. Further MTP controls
and the short first-decode timing investigation are retained with the live
evidence; no blanket sampling/cache or timing pass is claimed.

The varied fresh-process prompt contains 11,036 tokens. Each process sends it
once with default prompt reuse and then with `cache_prompt: false`; both
requests report zero reused tokens. The initial AR prefill times are
6,533.72/6,093.59 ms (base) and 6,298.96/6,072.70 ms (candidate), with identical
completions and prompt/cache/draft counts. The repeated AR baseline also passes
the strict request-phase limits against the candidate. Its cumulative idle-copy
counter covers only 25.69 MiB at the second response, versus 72.48 MiB on the
candidate; the initial baseline covers 30.67 MiB. Those asynchronous background
totals are different work, remain visibly inconclusive in the raw comparison,
and do not qualify background-copy performance.

Controls use the exact Windows base built separately in the retained detached
`baseline-source` worktree, TheRock 10.0.0 for both revisions, byte-identical
runtime DLLs and one common `run-bin/gufo.exe` path. Builds finish before these
HTTP timing runs. Binary SHA-256 values, commands, loaded `/props`, complete
responses, phase counters, individual timings and graceful shutdown results
are in `identities.json` and the per-arm JSON/log files. Model identity uses
path/size/mtime; full model hashes were not collected. The tested candidate
binary reports `db4f6fbcea05-dirty`; the retained source patch identifies its
merged source, and its version is upstream's 0.20.0.

Full Windows inference/performance qualification, Linux checks and unavailable
model profiles must remain explicitly pending until evidence establishes them.
The upstream PRs' Linux measurements do not qualify Windows. In particular,
#699's upstream greedy mixed-C2 timing remains inconclusive, as recorded in the
model's quality document.
