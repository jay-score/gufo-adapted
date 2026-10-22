# Windows upstream sync log

Track merges from [official Gufo](https://github.com/gufo-org/gufo) into the
Windows integration branch. This log starts after the October 2026 forward-port;
it is not a retroactive inventory of the old fork's selective imports.
An imported PR is not automatically qualified on Windows.

## Windows acceptance policy

This is a Windows-specific Gufo port. Preserve Windows compatibility,
correctness, stability and performance when accepting upstream changes.
Upstream inclusion or passing Linux checks does not override a Windows
regression.

- Check the known-regression entries before every sync. Skip or defer the
  code from PRs with confirmed Windows regressions until a fix passes the
  affected Windows checks. A focused compatibility fix may be retained with
  the PR when it is validated and documented.
- Preserve official ancestry: normally merge the recorded upstream head,
  then explicitly revert the regressing PR on the candidate. Record it as
  **reverted/deferred**, rather than imported and validated. If dependencies
  prevent a safe isolated revert, defer the dependent changes or the candidate
  until the compatible combination is validated.
- Confirm attribution against the previous Windows base with matched inputs
  and toolchain. Keep inherited failures, missing coverage and noisy results
  visibly pending; they are not automatically evidence that an incoming PR
  caused a regression.
- Record every skipped, reverted or deferred PR in the sync entry and known
  regressions section. Include its URL and merge SHA, affected Windows behavior,
  reproduction/toolchain details, baseline and candidate evidence, disposition
  and revert/fix SHA, affected dependencies, and required reintroduction checks.
  Keep failed observations and the original entry when a later fix resolves it.
- Preserve these exclusions during future conflict resolution, backports and
  related upstream changes. Reapply a deferred PR only after reviewing its fix
  and passing the recorded Windows checks against the previous Windows base.
  Append the fixing SHA, validation evidence and new status to its record.

## Known upstream regressions

[`known-regressions.json`](../tools/windows/known-regressions.json) mirrors
these entries for the sync tools: PR, merge/revert SHAs, identifiers that must
stay absent while reverted, and reintroduction checks. Update both together.

### PR #699: Windows first-request MTP decode regression

**Avoid reapplying #699 unchanged on Windows.** Its code is reverted by
`161035d8`; upstream ancestry is retained by merge `54fb4b4e`. PRs #692,
#540 and #690 remain included. This is a Windows/TheRock observation, not
an assertion about Linux performance.

With the same model, toolchain, runtime DLLs, server path and 11,036-token
fresh-process request, pre-sync first MTP decode measures 57.51, 53.86 and
51.32 ms in separate controls. The full import measures 60.90 and 63.35 ms;
removing #699 alone measures 53.55 ms. The rebuilt explicit revert measures
53.25/51.52 ms on first/repeat decode, with identical output and cache/draft
work. Its 6,557.75/6,284.98 ms prefill also remains within the baseline gate.
The focused import/base pair exceeds the unchanged 5% and 3 ms request-phase gate. Warm repeats are about 51-53 ms;
outputs and prompt/cache/draft counts agree. See the model's
[individual observations](models/qwen3.8-flash-next/WINDOWS.md#sync-first-decode-control-october-21-2026).

Temporary host-stage traces locate roughly 7-9 ms in the first neural
predictor's Q8 MoE enqueue after KV-only prefill. The baseline pays this
first-use cost during prefill instead. This suggests lazy HIP setup moving
into decode; it does not establish a complete root cause. Removing eager
vision setup does not fix it (59.60 ms). Restoring full scalar predictor
catch-up measures 52.75 ms once, but the combined scalar/batch fallback
measures 66.46 ms. Explicit startup function preparation also fails (67.02 ms).
None of those experimental changes is retained. Profiling measurements are
diagnostic and are not qualification timings.

**Reintroduction requirement:** review the upstream fix, then run
[the first-request control](#reproduce-the-first-request-control) against the
previous Windows binary. Retain every
phase, output and cache/draft count; a warmed benchmark alone is insufficient.
A later merge will not automatically undo the revert because #699 is already
an ancestor. Any deliberate reapplication must cite this entry and pass the
focused cold-start control plus affected correctness checks.

## Repeatable procedure

1. Follow the [sync skill](../.agents/skills/sync-windows-upstream/SKILL.md).
   Verify [repository roles](WINDOWS-COMPATIBILITY.md#repository-roles),
   worktree status and local versus remote `windows-main` before fetching. Preserve unpublished Windows work.
2. Record the Windows base and fetched `official/main` SHA before merging.
   Run `tools/windows/sync-inventory.ps1 -WindowsBase WINDOWS_BASE
   -OfficialHead OFFICIAL_HEAD -OutFile build/sync-YYYY-MM-DD/inventory.md`.
   It lists **all** incoming commits, resolves PR titles and merge SHAs with
   `gh`, lists release commits separately, reports files changed on both sides
   and upstream work on reverted regions, and emits an entry skeleton. Exit
   code 1 means the sync imports a known regression.
3. Apply the Windows acceptance policy and known-regression entries above.
   Merge that exact upstream SHA on an isolated candidate descended from the
   Windows base. An already isolated T3 task branch is a suitable candidate.
   Preserve official ancestry through a merge commit. Review cleanly merged
   Windows call sites as well as conflicts; retain both platforms' evidence.
4. Review every [compatibility contract](WINDOWS-COMPATIBILITY.md).
   Audit new callers and moved implementations. Run the bounded Windows CPU
   gate, affected GPU/model checks, documentation/formatting checks and the
   available Linux gate. Preserve individual timing results and failures.
   Revert/defer confirmed Windows regressions and validate the resulting
   combination before accepting the sync. Rerun the inventory with
   `-Candidate HEAD` (it fails if reverted code is present), run
   `tools/windows/diff-guard.ps1` (see the
   [maintained difference](WINDOWS-COMPATIBILITY.md#maintained-difference)),
   and open a draft PR against `windows-main`:
   pushing a `sync/official-*` branch alone triggers no hosted checks. Watch
   the Linux and Windows CPU runs with `gh run watch`.
5. Append an entry below with the PR table, base/head identities, resolutions,
   per-contract outcomes, exact checks, evidence location and pending limits.
   Put model measurements in the model's `WINDOWS.md` sidecar, not in the
   upstream model documents.
   List skipped/reverted/deferred PRs and their reintroduction requirements
   explicitly; do not silently omit them from the inventory. Record new
   reverts in `known-regressions.json`.
   Verify each PR merge SHA and the recorded official head with
   `git merge-base --is-ancestor SHA HEAD`. Do not replace old entries.
6. Integrate a qualified candidate into `windows-main` only after rechecking
   its base. Publication follows the requested endpoint and the
   [merge policy](WINDOWS-COMPATIBILITY.md#repository-roles).

For the next sync, copy the shape of the latest entry and replace its evidence.
Use **validated**, **reviewed/unchanged**, or **pending (reason)** per contract;
do not count missing models, excluded disk persistence or noisy timings as passes.

### Reproduce the first-request control

Finish builds first and stop nothing you do not own. Run
[`first-prompt-control.py`](../tools/windows/first-prompt-control.py):

```powershell
python tools/windows/first-prompt-control.py --base-binary PREVIOUS\gufo.exe `
  --head-binary CANDIDATE\gufo.exe --repeats 2 --output build/sync-DATE/first-prompt
```

For each mode (AR with `--speculative off`, MTP with the shared-Q8 sidecar),
it alternates base/head in ABBA order. Each arm stages the executable, its DLLs
and `hipblaslt/` at the same path, then starts a fresh private server with
`--context 32768 --sessions 2 --think off --cache-ram-bytes 8589934592`. It
POSTs the retained 11,000-word body below to `/v1/chat/completions`, first
with `cache_prompt: true` and then with `cache_prompt: false`, and stops the
server with a console break. Models default to `Models` in
`tools/windows/local.json`. Omit `--head-binary` for an unchanged-binary
control.

The output directory must be new. It keeps `control.json` with settings,
binary/DLL/model identities, commands, `/props`, complete responses and
wall/phase timings. It also keeps one server log per arm and
`comparison.json`. [`compare-phases.py`](../tools/windows/compare-phases.py)
reproduces the comparison from `control.json`. Correctness problems fail the
run: wrong prompt reuse, a wrong answer, an HTTP error or an unclean stop.
Each phase is compared on its own with the repository's 5% and 3 ms rule; it
counts only when every repeat moves the same way, and it is compared only when
both arms did the same prompt/completion/draft work. Exit status 0 means
passed, 1 failed and 2 inconclusive. An unchanged-binary control with any
moved phase is inconclusive and measures noise. Investigate flags with more
repeats or a focused control, and save failed observations as well as
successes. For these artifacts, expect 11,036 prompt tokens, zero reused
tokens and `ACK`.

On this host, the first GPU submission after the device idles for about 5 s
stalls for 0.3–0.9 s in the driver. This happens in AR and MTP, with any
binary, and also mid-process after an idle gap. A fresh arm's first request
hits it at random (7 of 16 short MTP arms, October 22, 2026), inside slot
reassignment. A settle delay makes it certain rather than removing it, and a
warm-up request would defeat the control. The comparison therefore derives
each request's `slot_wait_ms` from the server's cumulative
`component_cache.global_spill.residual_wait_ms`, subtracts it from the wall,
queue and TTFT phases that contain it, and reports `slot_wait_ms` without
gating on it. A slot wait that is slower in every repeat gets a note: check the
server logs before calling it noise. Evidence that lacks the field is compared
unadjusted.

The retained request body, for reference:

```python
import random

words = ("river town family valley stone bridge boats timber salt wool market "
         "wheat children forest quiet garden road lake tree green water blue").split()
passage = " ".join(random.Random(813).choices(words, k=11000))
body = {
    "model": "gufo",
    "messages": [
        {"role": "system", "content": "Read the archive. Follow the final user instruction exactly."},
        {"role": "user", "content": passage + "
Reply with exactly the word ACK."},
    ],
    "max_tokens": 128, "temperature": 0, "seed": 42,
    "chat_template_kwargs": {"enable_thinking": False},
}
```

## October 22, 2026: PR #350 candidate

Status: **unqualified candidate; timing controls and extended-context/API checks are in progress**.

- Windows base: `d10a852f0f094d926128ecf2e9dd112cae0c4a8e` (PR #74 merged).
- Previous official ancestor: `53ebb51ed0c0b24dbe7c1e35dce6c97d82740bfe`.
- Imported official commit: `2947624b9e65f52d4b4fdd96297f39e040a0ed24`.
- Integration merge: `8170cfc855484da939df61cbcf4c31d125cde772`.
- Scope: #350 only. The later #709, #688, #710, #677 and #711 commits will
  follow in separately qualified sync PRs.
- Evidence: `build/sync-350-2026-10-22/` in the task worktree.

| PR | Upstream change | Merge SHA |
| --- | --- | --- |
| [#350](https://github.com/gufo-org/gufo/pull/350) | Opt-in Flash-Next YaRN context extension | `2947624b9e65f52d4b4fdd96297f39e040a0ed24` |

### Merge review

One conflict, in `projection_ops_test.cpp`: preserve the null vision descriptor
from the #699 revert and pass the new YaRN argument. Clean merges of the
executor, kernels, engine and serving options were also reviewed. #699 remains
reverted; its six forbidden identifiers stay absent.

Windows' MSVC library prints padded hexadecimal fractions in `std::hexfloat`,
breaking upstream's pinned YaRN cache identity. The Windows-only normalization
in `rope_scaling.cpp` removes that padding and passes the original test
unchanged. Linux formatting is unchanged. The shared edit is marked and listed.

The Windows-owned `win32_yarn_snapshot_test.cpp` covers checkpoint replay
without the upstream snapshot test's Linux ELF fault hooks. It checks exact
logits, tokens and actual MTP draft work, serialized replay, old executor-format
rejection/reset/recovery, and rejection of a different scaling factor.
Bitwise replay controls use matching prefill boundaries. The current executor
format changes from `QFNSNAP5` to `QFNSNAP6`; the outer session version stays 18.

### Compatibility and validation

| Contract | Outcome |
| --- | --- |
| Cached concurrent n-gram reads | Windows reopening/read flags unchanged; native n-gram checks pass. Fresh-process inference timing remains inconclusive pending controls. |
| WDDM admission budgets | Helper unchanged; scaled-context allocation/admission still needs the extended-context run. |
| Pinned memory and transfer lifetime | Native snapshot-copy operator and AR/MTP 4K cross-slot component/continuation checks pass. Extended-context capture remains pending. |
| Sockets and shutdown | Implementation unchanged; bounded native HTTP/WebSocket and real console-shutdown checks pass. |
| Filesystem identity, privacy and failures | Native implementation checks pass; new model payload/factor compatibility passes. API disk restart remains pending. |
| GPU queue policy | Boundary unchanged; bounded native startup policy checks pass. |

Completed local checks and current limits:

- Native Windows CPU gate: **16/16 pass** (`final-cpu-ctest.txt`).
- Selected operators: eight pass initially; the original rope-policy test fails
  on MSVC padding, then passes after the normalization (`head-operator-ctest.txt`,
  `fixed-rope-ctest.txt`). No GPU kernel change was needed after those passes.
- Full-model numerical/continuation evidence and per-request timing details are
  recorded in the [Windows model sidecar](models/qwen3.8-flash-next/WINDOWS.md).
- Final bounded YaRN fixture: AR/MTP exact replay and scaling-factor rejection
  pass (`final-yarn-snapshot-3.log`). Earlier fixture assumptions and their
  failed observations remain retained; the production restore/reset behavior
  is byte-identical to the Windows base.
- Shared formatting passes for 618 C++ files. Documentation, dependencies,
  maintained-difference and reverted-identifier checks pass.
- First-prompt control: correctness passes in all eight fresh processes;
  snapshot timing and first MTP decode have noisy cells. Overall timing is
  **inconclusive**, not passed (`cold-control-1/`). An unchanged-base MTP
  control is in progress for the flagged histories.
- Standard benchmark, extended-context execution, API continuation/cache checks
  and hosted Linux/Windows checks remain pending. This candidate is not ready
  to merge until the sync PR's qualification record is complete.

## October 22, 2026: PR #702

Status: **candidate local checks pass; hosted qualification is recorded on the sync PR**.

- Windows base: `e416eac00b1450eb03d82165d76ebd1970ac0364` (includes PR #73).
- Previous official ancestor: `e060aeed923730b74ea3f4116c3c39de55741370`.
- Imported official commit: `53ebb51ed0c0b24dbe7c1e35dce6c97d82740bfe`.
- Integration: ordinary merge, `45bb338c2e5e42b5d99a985c65be22e05792c1fb`.
- Fetched `official/main`: `7c819652`. This import intentionally stops at
  #702 as requested; the six newer upstream commits are not included.
- Evidence: `build/sync-702-2026-10-22/` in the task worktree.

| PR | Upstream change | Merge SHA |
| --- | --- | --- |
| [#702](https://github.com/gufo-org/gufo/pull/702) | Stop the benchmark server when startup is interrupted | `53ebb51ed0c0b24dbe7c1e35dce6c97d82740bfe` |

### Merge review

No conflicts or files changed on both sides. The two imported files,
`tools/gufo/model_bench/servers.py` and `tests/tools/serving_bench_test.py`,
remain byte-identical to the recorded upstream commit. The helper now stops
its server and re-raises when readiness checking exits through an exception,
including `KeyboardInterrupt`. Review covered early process exit, readiness
timeout and the existing process-group shutdown. No engine, model, CMake,
dependency or Windows platform code changes.

The inventory finds no incoming change to #699's reverted regions. Its revert
and reintroduction requirements remain intact; the candidate check confirms
that its six forbidden identifiers remain absent.

### Compatibility and validation

| Contract | Outcome |
| --- | --- |
| Cached concurrent n-gram reads | Reviewed/unchanged: no model or I/O path changes. |
| WDDM admission budgets | Reviewed/unchanged: no admission or device-memory changes. |
| Pinned memory and transfer lifetime | Reviewed/unchanged: no allocation, transfer or snapshot changes. |
| Sockets and shutdown | Native engine paths unchanged; bounded HTTP/WebSocket and real console-shutdown checks pass. Benchmark-helper interruption cleanup passes separately under WSL. |
| Filesystem identity, privacy and failures | Implementation unchanged; bounded identity, privacy and failure checks pass. |
| GPU queue policy | Implementation unchanged; bounded native startup policy checks pass. |

Completed checks:

- `tools/windows/build.ps1 -Preset windows-cpu -Vcpkg VCPKG_PATH
  -Ninja NINJA_PATH -Jobs 4`: pass with TheRock's compiler, MSVC 14.51.36231,
  SDK 10.0.26100.0, CMake 4.4.0 and the existing pinned vcpkg manifest.
- `ctest --preset windows-cpu --no-tests=error --output-on-failure`:
  **16/16 pass**, 19.49 seconds (`cpu-build.txt`, `cpu-ctest.txt`).
- `python3 -I tests/tools/serving_bench_test.py` under WSL/Python 3.12.3:
  **pass**, including the new real-process interruption test
  (`serving-bench-wsl.txt`). This helper uses POSIX process groups; this is
  Linux lifecycle evidence, not a claim of native Windows benchmark support.
- Matched interrupted-start control with the exact Windows-base helper and
  candidate helper under WSL: both propagate `KeyboardInterrupt`; the base
  leaves its real `sleep` child alive, while the candidate has already reaped
  it. The control explicitly cleans up both test-owned processes
  (`startup-control.py`, `base-servers.py`, `startup-control.txt`).
- Inventory before and after merging, including `-Candidate HEAD`: pass.
  `diff-guard.ps1`: pass, with 72 changed upstream files, unchanged by this
  import (`inventory.md`, `candidate-inventory.md`).
- `check-docs.py`, `check-dependencies.py`, `git diff --check`, and the
  imported-commit ancestry check: pass. No C++ edits require formatting.
- Hosted Linux/Nix and Windows CPU gates run on the draft sync PR; their
  results qualify that PR's exact head.

### Live checks and limits

No inference or shared runtime paths changed, so no GPU/model rebuild,
numerical sweep or inference timing comparison is required for this import.
The existing Windows qualification limits remain unchanged. The benchmark
cleanup result is supported by a real child process, without loading a model.
Preserve official ancestry by merging the sync PR with a merge commit.

## October 21, 2026

Status: **local handoff complete; #699 reverted; qualification limited below**
on `t3/sync-upstream-oct-21-prs`. The candidate is retained locally for review.
The maintained `windows-main` remains at the recorded base; it has not been
promoted past the outstanding qualification limits.

- Windows base: `db4f6fbcea0514db1456abf8571f20cb525de30e` (also
  `origin/windows-main` at fetch time; no unpublished divergence).
- Previous official ancestor: `f020bd3a`.
- Imported official head: `e060aeed923730b74ea3f4116c3c39de55741370`.
- Integration: ordinary merge of the recorded official head,
  merge commit `54fb4b4e`, followed by the explicit #699 revert `161035d8`.

| PR | Upstream change | Merge SHA |
| --- | --- | --- |
| [#692](https://github.com/gufo-org/gufo/pull/692) | Compare the bound publication waiter with cold by answer | `0e4f987b8145bc37fca25b601881abd369982c36` |
| [#540](https://github.com/gufo-org/gufo/pull/540) | Qwen 27B AR and DFlash2 cache continuation | `da5d5a2b237c654bc0523e33fdee10465240da0c` |
| [#690](https://github.com/gufo-org/gufo/pull/690) | Wide Flash-Next routes for prefill passes above eight rows | `53f73b5653b32caf1d8cb3113c3f2f05f0e88d0a` |
| [#699](https://github.com/gufo-org/gufo/pull/699) | Reduce Flash-Next sampled decoding overhead; **reverted/deferred on Windows** | `e060aeed923730b74ea3f4116c3c39de55741370` |

The only additional incoming commit is upstream's v0.20.0 release
`0c03748a`. Its version, manifest and changelog changes are retained as published.
No version is manually assigned by this sync.

### Merge review

Only `docs/models/qwen3.8-flash-next/EXPERIMENTS.md` and `QUALITY.md`
conflicted: both branches inserted independent evidence at the top. Retained
the October 21 upstream sections and the October 20 Windows n-gram evidence.
No source file required conflict resolution. The later #699 revert conflicts
only in `QUALITY.md`; its resolution removes the reverted claims and preserves
the Windows October 20 n-gram evidence.

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
The initial import's native run also exposed a stale pre-sync fixture: committed backing
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
  `ctest --preset windows-cpu --no-tests=error`: **15/15 pass** on the import (17.83 seconds) and again after the revert
  (16.91 seconds).
- `tools/windows/build.ps1 -Preset windows-gpu -Vcpkg VCPKG_PATH`: production
  HIP executable builds. Both builds use this checkout's pinned vcpkg manifest.
- Affected extra CPU checks: sampling, idle spill, Qwen continuation layout,
  Flash-Next n-gram, MTP sampling and prompt lookup: **6/6 pass**.
- Focused GPU gate: WDDM, pinned transfers, queue policy, speculative verifier,
  Qwen attention policy/long-context operators, Flash-Next projections,
  routed experts, snapshot copies and sampled-verification oracle: **10/10
  pass across the initial run and the focused pinned-fixture rerun**. The
  original failed pinned-fixture result is retained, not overwritten. After
  reverting #699, the remaining focused gate passes **9/9** (71.69 seconds);
  #699's removed sampled-verification test is not counted as a current pass.
- Qwen 27B continuation probe at 32K: **pass**, including component bytes,
  restoration and next-step logits/tokens. Native `--prefill-cancel-only` with
  the official DFlash2 sidecar: **AR and DFlash2 pass** exact resume against
  direct cold execution, with actual drafting. The broader
  `--continuation-only` includes Windows-disabled disk persistence and remains
  outside this qualification.
- Flash-Next continuation probe: `--mode ar` and `--mode mtp`, each with
  `--context 4096 --sessions 2 --backing-gib 1 --round-trip`: **pass**.
- Post-revert Flash-Next `qwen38_flash_next_session_test` with
  `--sampling-only` and `--cache-only`: **both pass**. Coverage includes graph
  capture/execution, EOS and sampling variants, 200-token checkpoint replay,
  and learned-policy greedy/thinking/nonthinking continuation with exact
  tokens, sampler state, drafts and final logits after RAM/serialized restores.
- Standard post-revert `gufo bench -p 2048 -n 128 -r 1 -v` with
  `--speculative off`, then `--speculative mtp --mtp-model MTP_PATH`:
  **both complete with exit zero**. AR pp/tg: **1,807.04/28.88 tok/s**;
  MTP pp/tg: **1,753.56/38.65 tok/s**. Greedy tg128 hashes match:
  `5177a75da6988d5ce9a14e1a744ed16829784eda18d40139d8a1e76469af8412`.
  MTP proposes 109 and accepts 58 drafts. These single-repetition spots do
  not qualify a branch speedup. Commands/results are retained in
  `final-bench-*-command.json`, logs and `final-bench-summary.json`.
  Windows PowerShell wraps verbose native stderr in `NativeCommandError`
  records; the captured benchmark exits are zero.
- Shared `tools/ci/check-format.py` with clang-format 21.1.8,
  `tools/ci/check-docs.py`, `tools/ci/check-dependencies.py`, and
  `git diff --check`: **pass**.
- Functional-runner unit tests under WSL: **80/80 pass**; new Qwen recovery
  evidence-refusal tests: **6/6 pass** on Windows and WSL. The initial Windows
  runner invocation hit a default-encoding failure (the affected case passes
  with Python UTF-8 enabled) and the existing POSIX-only `SIGHUP` harness.

The additional independent `qwen38_flash_next_gpu_probe --mtp-audit` fails
on image sequence 1, step 0: head-mixer relative RMS `0.0000695` exceeds the
unchanged `0.00005` scalar-oracle limit (maximum absolute error `0.0047131`).
The exact pre-sync Windows base, rebuilt with the same toolchain and the
byte-identical audit fixture, reproduces every printed value and failure.
This is an inherited Windows numerical qualification gap, not a new sync
failure. The oracle is not passed, its tolerance is unchanged, and both
`final-mtp-audit.log` and `baseline-mtp-audit.log` are retained.

Additional Windows harness exclusions, retained as failures in the evidence:
`cache_slot_test` requires POSIX `fork`/`SIGABRT`; `component_text_cache_test`
and `text_model_runner_test` reach the deliberately disabled disk stores.
These broader tests are outside the bounded Windows preset. They do not pass
on Windows and were not made to pass by weakening the store guards.

### Live checks and limits

The initial import's Flash-Next AR and MTP pass the SDK `long-context` and `cache-depth`
suites with 32K contexts, two sessions and an explicit 8 GiB RAM budget.
They exercise multi-turn recall, exact warm/cold answers, deep RAM reuse,
rewritten branches and peer rotation. MTP responses contain actual proposed
and accepted drafts. Each suite retains all request/phase timings; these
correctness runs do not by themselves qualify speed.

After the #699 revert, Qwen 27B AR and official DFlash2 also pass both API
suites at 32K/two sessions/8 GiB. DFlash2 responses contain actual drafts
(up to 115 proposed tokens in the long-context observations); both private
test servers exit cleanly with code zero. These runs exercise the #540
component-cache path rather than Windows-disabled disk persistence.

The additional `sampling-defaults` suite passes its explicit/default sampler
output checks but fails its zero-prefill replay assertion after the deep-cache
history: 15 of 22 prompt tokens are reused, and seven are re-prefilled.
Both exact pre-sync AR and MTP baselines reproduce the same output and
seven-token failure under the same deep history. Fresh MTP baseline and
import processes pass this suite. This is an inherited history-sensitive
cache assertion, not evidence of a new sampling regression; no blanket
sampling/cache or timing pass is claimed. The baseline MTP long-context
control also flags a reasoning-signature replay despite a correct answer;
the import passes that check. All raw observations remain retained.

The varied fresh-process prompt contains 11,036 tokens. Each process sends it
once with default prompt reuse and then with `cache_prompt: false`; both
requests report zero reused tokens. The initial AR prefill times are
6,533.72/6,093.59 ms (base) and 6,298.96/6,072.70 ms (initial import), with identical
completions and prompt/cache/draft counts. The repeated AR baseline also passes
the strict request-phase limits against that import. Its cumulative idle-copy
counter covers only 25.69 MiB at the second response, versus 72.48 MiB on the
import; the initial baseline covers 30.67 MiB. Those asynchronous background
totals are different work, remain visibly inconclusive in the raw comparison,
and do not qualify background-copy performance.

Controls use the exact Windows base built separately in the retained detached
`baseline-source` worktree, TheRock 10.0.0 for both revisions, byte-identical
runtime DLLs and one common `run-bin/gufo.exe` path. Builds finish before these
HTTP timing runs. Binary SHA-256 values, commands, loaded `/props`, complete
responses, phase counters, individual timings and graceful shutdown results
are in `identities.json` and the per-arm JSON/log files. Model identity uses
path/size/mtime; full model hashes were not collected. The initial import reports `db4f6fbcea05-dirty`; the final tested code is
commit `161035d8` (with documentation edits present during the build). Its
SHA-256 is `447196666e8f2e1158e78ac59ab2b464ee58469d2a84226ed1dec619e1419298`.
Both use upstream's 0.20.0 version; retained patches and binary hashes
distinguish every arm.

The rebuilt revert passes the affected MTP first-decode gate. Final AR
prefill measures 6,309.95/6,109.18 ms and decode 42.65/35.24 ms; these phases
also pass against the focused base control, with identical output and work.
Snapshot timing remains unqualified: final MTP first snapshot is 67.92 ms
versus 61.66 ms, and final AR snapshots are 53.51/40.56 ms versus 42.07/35.57 ms.
Those flags and asynchronous copy totals remain in
`reverted699-comparison.json` and `final-ar-comparison.json`; no complete
performance pass is claimed from these pairs.

Full Windows inference/performance qualification remains pending. Linux/Nix
checks could not run: WSL has Python but no Nix/CMake/compiler toolchain, and
no hosted check was triggered for this local-only handoff. Qwen Q8_K_XL and
the DSpark/DeepSeek profiles are unavailable locally and remain unqualified.
Multi-GPU, physical memory pressure and Windows disk persistence also remain
outside the evidence above. The upstream PRs' Linux measurements do not
qualify Windows. PR #699 is explicitly reverted for the Windows regression;
its prior Linux evidence does not override this disposition.
