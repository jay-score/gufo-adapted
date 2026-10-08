# Flash-Next prefill adaptation plan for PR #470

Status: proposed; planning only. No implementation, builds, model loads,
benchmark runs or integration commits have been performed for this plan.

The intended outcome is faster long-prompt Flash-Next prefill with consistent
attention behavior across supported prefill boundaries, while short requests,
decode, cache replay and Windows memory placement retain their contracts.
Implement the useful outcomes as independently reviewable changes. The larger
chunk default is the final decision, based on local measurements.

## Pinned scope

Fork base: `dce753fc36aa8a027b4728dfcd36c33e42150c5c`, branch `windows-port`.
The checkout was clean before writing this plan. Reconfirm HEAD and local work
before execution; if affected code changes, refresh the comparison.

Source: [PR #470](https://github.com/gufo-org/gufo/pull/470), reviewed through
[5c6ebb4660b6712b18c6a2c20b3ee3d3e1c3cf2a](https://github.com/gufo-org/gufo/commit/5c6ebb4660b6712b18c6a2c20b3ee3d3e1c3cf2a),
base `7701ba769f454a580ffcbaa29538d65e7f33b38c`, squash merge
[47b639159315fcdba17e6a144d67e273de9ead6e](https://github.com/gufo-org/gufo/commit/47b639159315fcdba17e6a144d67e273de9ead6e).
Official main at review was `f17e37b8bb7df5fb83ea7ce6d4dc4ef6d6677253`.
Later PR #459 is a separate optimization, not a prerequisite.

The final source includes the scratch/scheduling follow-up
[def2ed1e904f912e69550344aa7c13a99f535d0e](https://github.com/gufo-org/gufo/commit/def2ed1e904f912e69550344aa7c13a99f535d0e).
Its code supersedes the original PR body's Linux reader count and scratch cost.

## Design boundaries

Use the existing engine, scheduler, NgramTable and allocator. Preserve:

- TheRock 10.0.0, current CMake/vcpkg pins and Linux compatibility.
- Windows cached overlapped n-gram handles, 128 readers and current row-cache
  size initially; selective wakeups can change independently of pool size.
- Capture-safe Windows HIP submission before blocking on n-gram I/O.
- Existing image-prefix identity, stateless API history, snapshot payload format,
  request sampling/reasoning overrides and sampled MTP acceptance/rollback.
- Stable scratch addresses for captured graph replay and existing process
  ownership, launcher configuration and the running production executable.

No new dependency, user switch, memory manager, generic task framework or
additional I/O pool is planned. Upstream tail-chunk/SyncThrough APIs, snapshot
lineage machinery, prompt-lookup changes, Linux cache sizing, benchmark-skill
changes and unrelated functional SDK infrastructure are outside this adaptation.

## Execution order and comparison points

Preparation records a baseline before implementation. Then use six focused
changes, with the 2048 default unchanged in changes 1 through 5:

1. Attention boundary correctness.
2. Scheduler fairness for short arrivals.
3. Selective n-gram worker wakeups.
4. Bounded next-chunk n-gram prefetch.
5. Lifetime-based scratch reuse.
6. Qualified 4096-token default, only if it passes the local gates.

Changes 1, 2, 3 and 5 are independently useful. Prefetch does not require scratch
reuse or a larger chunk. The default change depends on attention correctness
and qualified serving/memory behavior. Failure of a later optimization does not
invalidate a completed correctness or fairness fix.

For arithmetic-changing boundary corrections, establish a corrected 2048
reference after change 1. Compare each subsequent candidate both with its
immediate predecessor and with that corrected reference. Preserve the original
baseline for documenting the bug; never compare with a moving executable.

## Preparation: establish reproducible evidence

When implementation is authorized, use an isolated branch/worktree following
the app's worktree workflow. Keep production binaries and user configuration
untouched. Build a matched baseline with the same dependencies as candidates.

Record source SHA, executable SHA-256 and --version, compiler/runtime versions,
model/shard hashes, Q8_0 MTP and BF16 projector identities, corpus/token hashes,
context, draft limits, sampling, cache policy, concurrency and power settings.
Use local IQ4_XS artifacts for the initial qualification; do not infer that an
upstream UD-Q4_K_XL result qualifies all quantizations.

Use a fixed public/local corpus with reproducible token counts. Capture focused
baseline logits/perplexity and short/long serving timings before any edits.
Baseline configurations include AR and MTP at C1, plus C2 when exercising
scheduler interleaving. Context 262144 for C1 and 131072 for C2 are proposed
large-context controls, conditional on recorded Windows budget headroom.

Implementation outputs and large dumps belong in ignored
`build/pr470/<stage>/`. Keep exact commands and concise retained evidence in
this plan/qualification document.

## Change 1: attention independent of the budget-crossing batch

Source attribution:
[f5401508e0352afbfc9d769414b15486ea100cdf](https://github.com/gufo-org/gufo/commit/f5401508e0352afbfc9d769414b15486ea100cdf).

Affected areas: `kernels/rocm/executor.cpp: Executor::Attention`, attention
operator fixtures and existing session/probe checks under
`tests/models/qwen38_flash_next`.

Current Forward selects sparse attention for the entire batch when
start_pos + n exceeds indexer_top_k. Split the attention launch by absolute
query position: pre-budget queries use the existing dense tiled path; remaining
queries use the existing masked path. Offset Q, gates, masks and output rows
consistently; project the combined output once.

Check the sparse launch's supported geometry before emitting a dense-prefix
launch. Preserve fallback behavior if the tiled route refuses the shape, and
preserve last_only/MTP catch-up semantics. Do not change projection arithmetic,
indexer ranking or attention kernels unless a focused regression proves the
dispatch-only correction insufficient.

Acceptance:

- Reproduce the crossing failure before the fix using bulk versus split prefill,
  including a nonzero retained prefix; confirm it disappears after the fix.
- Cover positions around 2047/2048/2049, non-tile-aligned prefixes and a batch
  starting below and ending above the budget.
- Equivalent prefill schedules match complete logits exactly; unchanged
  schedules match the recorded original baseline.
- Verify 4096/8192 diagnostic batches using probe-sized scratch, without changing
  the production default.
- Run attention operator checks and focused session prefill checks with AR/MTP.

The numerical contract is equivalent prefill arithmetic. The current quality
document explicitly records scalar-versus-bulk differences; scalar decode is
not an exact-equality oracle for every bulk schedule. Operator formulas and
schedule-equivalent comparisons serve different purposes.

## Change 2: short requests progress at the next scheduling opportunity

Source attribution: scheduling portion of
[def2ed1e904f912e69550344aa7c13a99f535d0e](https://github.com/gufo-org/gufo/commit/def2ed1e904f912e69550344aa7c13a99f535d0e).

Affected areas: `src/cli/serve/text_generation_scheduler.cpp`,
`tests/cli/text_generation_scheduler_test.cpp`; a focused HTTP check using
existing local transport helpers where suitable.

Use one last-served prefill request ID to place arrivals before that request's
repeated turn, while retaining the order of peers already waiting. Extend the
existing bounded-prefill decision to short waiting peers as well as runnable
decoders and pending captures. Use the existing decode-active token budget;
do not introduce a new priority queue or fixed elapsed-time budget.

Only delay a ready decoder to assemble its initial batch when the pending peer
can finish within the bounded prefill budget. Two long prefills retain wide,
alternating turns while no latency-sensitive peer is ready.

Acceptance:

- Port the three behavioral fixtures: short arrival during a long chunk,
  two long prefills retaining fair wide turns, and short arrival behind another
  waiting long peer.
- Check both AR and speculative/batched execution. Add a repeated-arrival
  ordering fixture if needed to prove older peers cannot starve.
- A short arrival starts at the next eligible turn and emits its first token
  before the long prompt finishes, without waiting for another full long chunk.
- Older long requests progress; active decode/capture budgets and cancellation
  remain correct.
- Fresh scheduler/runner CPU tests pass, including device-loss coverage.
- AR/MTP HTTP overlap confirms unchanged per-request output semantics.

This guarantee begins after the current forward finishes. It does not promise
mid-kernel preemption or bypass an outstanding read.

## Change 3: wake only useful n-gram readers

Source attribution:
[2d6bc73ea2da0d326523857458945ecb361a4e5d](https://github.com/gufo-org/gufo/commit/2d6bc73ea2da0d326523857458945ecb361a4e5d),
using the final worker-count behavior as context rather than importing it.

Affected areas: `src/models/qwen38_flash_next/ngram.cpp` and
`tests/models/qwen38_flash_next/ngram_test.cpp`.

After deduplication/cache hits determine pending work, notify one worker per
job for small gathers and notify all for gathers at least the pool size.
Keep the existing queue batching, zero-job completion and worker counts.

Acceptance: cached/duplicate/empty gathers, small misses, large batches and
read failures retain completion and row values. Verify the synchronization
contract directly; avoid brittle tests that assert OS thread wake ordering.
Full logits/perplexity remain exact. Retain the change if it reduces wake
overhead without a repeatable served decode/prefill regression.

## Change 4: prefetch one predicted chunk's n-gram rows

Source attribution:
[0610c8f1a0513364b5920337460705b2febfb172](https://github.com/gufo-org/gufo/commit/0610c8f1a0513364b5920337460705b2febfb172).

Affected areas: backend Flash-Next Prefill, engine.hpp/cpp Sync and Feed,
executor.hpp/cpp PleFetch/WaitPle/Ple/Forward/destructor, and batch.cpp
ForwardBatch. Existing probe/session fixtures provide coverage.

Pass an optional bounded next-token hint through the current
Prefill -> Sync -> Feed -> Forward route. Feed prefers its own known next
internal chunk, otherwise the following Sync's hint. Clip to actual remaining
prompt/context and max_batch; no tail-capacity or checkpoint API is needed.

Keep one executor-owned prefetch slot: copied tokens, before/after NgramHistory,
pageable row storage and explicit pending/ready/claimed state as needed.
Hash from a copy; start only after the current demand gather is complete.
Read into pageable storage while the current GPU forward runs.

A normal non-speculative gather can claim only an exact token/history match.
Copy successful rows into the existing pinned upload buffer at the normal PLE
boundary. Do not point captured graph nodes at the pageable prefetch allocation.
An interleaved demand drains outstanding reads before starting its own read;
completed unclaimed rows may remain available until replaced.

Failure/lifetime rules:

- An unstarted or failed unclaimed prediction is discarded; demand uses its
  original history and normal read path.
- A claim's history advances exactly once. If read completion fails, abandon
  that prediction and reread the actual demand from the saved original history;
  a demand failure uses existing session invalidation.
- No prediction mutates live session history before demand owns its result.
- Drain readers before reuse/free, on error exits and at executor destruction.
- Apply capture-safe Windows submission before every newly introduced blocking
  wait that would otherwise leave queued GPU work unsubmitted.
- Keep speculative snapshots and rollback independent of prediction state.

Acceptance: identical/mismatched tokens, identical tokens with different history,
reset/restore, interleaved decode/batch, read failure, cancellation and destruction
all retain baseline behavior. Extend existing fixtures or a model-local test;
add only the narrow test hook needed if deterministic I/O failure cannot be
triggered with existing temporary files.

Require exact full logits/perplexity and sampled replay at matched request
budgets. Profile the gather's overlap with GPU work and measure served results.
A slow unused prediction must not create a repeatable short-request/decode
regression. If the single-read design fails that gate, revise or reject this
optimization before considering a larger I/O redesign.

## Change 5: scratch sized by live stages

Source attribution: allocation portion of
[def2ed1e904f912e69550344aa7c13a99f535d0e](https://github.com/gufo-org/gufo/commit/def2ed1e904f912e69550344aa7c13a99f535d0e).

Affected areas: executor Create/Scratch, with local batch.cpp and MTP/fused
kernel consumers inspected for lifetime proof.

Write a model-local lifetime inventory before aliasing: linear attention,
full attention, routed experts and mixer/PLE intermediates. Use one allocation
sized to the maximum proven-live stage and fixed slices for each stage.
Reuse the existing allocation ownership. Keep cross-stage residuals, routed
results consumed by deferred Combine, shared-expert intermediates and pinned
buffers separate whenever their lifetimes overlap.

Account for max_batch, routed slots, convolution tail, mask storage,
alignment, local fused projection writes and batched rows. Preserve stable
addresses for graph replay; do not allocate/repoint scratch during Forward.

Acceptance: exact operator outputs and full logits; graph/eager, scalar/batched,
AR/MTP catch-up, sampled rollback, image-prefix and snapshot tests pass.
Measured allocated bytes decrease at 2048 without a sustained latency regression.
At candidate 4096, record net engine allocations, pageable prefetch storage,
available host memory and WDDM/DXGI headroom. RSS and GPU counters overlap on UMA;
do not add them as independent RAM costs.

## Change 6: choose the production chunk default from measured outcomes

Source attribution:
[f9a57deb0b4870eebffa982e3098eb60bd25553a](https://github.com/gufo-org/gufo/commit/f9a57deb0b4870eebffa982e3098eb60bd25553a).

Affected area: `engine.cpp: kPrefillChunkTokens`, plus local evidence/docs.
Use separate matched production builds for the 2048/4096 comparison.
No runtime toggle or retained duplicate path is needed.

Proposed qualification gates, declared before measurements:

- Exact equivalent-schedule full logits and matched teacher-forced perplexity;
  no non-finite values or unexplained differences.
- At least 3% median served prefill throughput improvement on both an uncached
  roughly 26K prompt and one deeper long-prompt control, confirmed in interleaved
  A/B/B/A rounds. This is a proposed usefulness threshold, not a measured gain.
- No reproducible latency regression greater than both 5% and 3 ms for an
  individual short-request/TTFT/decode phase. Aggregate averages cannot clear
  a slow request or scenario. Investigate inconclusive samples before landing.
- C2 arrivals and ongoing streams retain bounded prefill and avoid an additional
  full long-prefill turn; paired worst SSE gaps must not regress repeatably.
- Intended C1/C2 context limits still load and execute within recorded Windows
  budgets, without allocation/eviction failures or materially reduced usable
  cache/session capacity. Do not lower context or disable cache to hide costs.

Use short prompts, pp2048/tg128, 26K and one deeper workload first; expand depth
only when the memory/performance question needs it. Retest the upstream
text -> repeated warmed image -> arithmetic sequence under default and bounded
cache budgets, with fixed draft/sampling settings.

If the gates fail, retain 2048 and the independently successful changes.
Record the rejected default in EXPERIMENTS.md. Do not land 4096 just because a
microbenchmark or the upstream Linux run improved.

## Validation commands and evidence

Build only the stage's affected targets. Windows commands use build.ps1 with
detected tool paths; native paths belong in run records, not tracked defaults.

Scheduler CPU checks:

```powershell
powershell -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset cpu-test -Target text_generation_scheduler_test -Jobs 4
powershell -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset cpu-test -Target text_model_runner_test -Jobs 4
ctest --test-dir build/cpu-test -R '^(text_generation_scheduler_test|text_generation_device_loss_test|text_model_runner_test)$' --output-on-failure --no-tests=error
```

N-gram/operator checks build the corresponding
`qwen38_flash_next_ngram_test`, `qwen38_flash_next_attention_ops_test`,
`qwen38_flash_next_projection_ops_test`, `qwen38_flash_next_gdn_ops_test`
and relevant MoE/MTP targets in gpu-test. CTest names are
`qwen38_flash_next.ngram`, `qwen38_flash_next.attention_ops`, etc.
Check their CMake registration and build fresh before running.

Explicit model tools: `qwen38_flash_next_session_test --prefill-only`,
then relevant batch/sampling modes, `qwen38_flash_next_snapshot_test`,
`qwen38_flash_next_rollback_test` and
`qwen38_flash_next_image_prefix_test`. Preserve their documented CLI and model
arguments. Use gpu_probe for 2048/4096/8192 diagnostic batch comparisons; extend
its existing dump mode only if needed. These are targeted checks, not a full
model sweep.

Teacher-forced validation follows [TESTING.md](../TESTING.md#matched-token-and-layer-comparisons):
`gufo bench --logit-eval corpus.txt --logit-schedules "1:1:2,4,8"`,
separate --logit-out files and complete raw-logit hash comparisons using
`tools/bench/logit-eval.py`. Sampling/rejection, graph capture and snapshot
interleaving need their separate fixtures.

Build production gufo for performance. Run C1/C2 HTTP checks against isolated
test-owned processes/ports; include cache growth/history edits and image-prefix
continuation after prefetch/scratch changes. Retain exact requests, cold/warm
state, per-request timings, stream gaps, draft counts and memory records.
Use existing tools under tests/tools where they fit; create one focused stdlib
arrival/transition harness only for coverage currently absent.

Shared scheduler/API behavior also requires fresh openai_chat_test and
http_server_test; template tests only if rendering changes. Linux hosted
CPU/repository CI and focused HIP correctness remain landing requirements,
reported separately from Windows results.

Run repository-compatible formatting before committing C++ changes,
`python tools/ci/check-docs.py` and `git diff --check`. Planning-only edits
require the documentation/whitespace checks.

## Commit, documentation and rollback policy

One focused Conventional Commit per retained improvement, with immutable
Upstream-Commit attribution and a matching UPSTREAM.md entry added alongside
implementation. Update this document with actual commands/results, model
QUALITY/EXPERIMENTS evidence and only qualified benchmark figures.
Do not create integration ledger entries for this proposed plan.

No cache/config format migration is planned. Each optimization is reversible
independently; a larger default can revert to 2048 without reverting the
attention/scheduler fixes. If implementation proves a persistent format change
necessary, revise the plan before introducing it.

Completion means every retained stage meets its gates and unrun/blocked checks
are explicitly identified. A missing-model skip or successful build alone
does not qualify correctness or speed.

