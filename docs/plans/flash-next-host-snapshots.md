# Flash-Next incremental host snapshots

This fork implements host-side prefix sharing independently of
[upstream PR #445](https://github.com/gufo-org/gufo/pull/445). No upstream GPU
allocator, attention implementation or in-forward checkpoint path is imported.
The implementation changes checkpoint ownership and transfers, with no kernel
arithmetic changes or new runtime switches.

## State and ownership contract

Attention K/V rows and completed pooled indexer keys share immutable host blocks.
New rows and any partial block crossed by a rewind are copied. Every capture copies recurrent state, convolution and
PLE history, unpooled raw-indexer rows, MTP residual/frontier state and the kept
hidden rows afresh. Host tokens, logits, image identity and policy metadata also
remain private to each checkpoint.

The source session remembers a weak reference to its last completed capture,
with independently valid trunk and draft prefix lengths. Each forward operation
limits those lengths to its write position, including batched execution. Reset
and contiguous-byte restore clear provenance. A successful native restore binds
the restored checkpoint as the source. Token equality alone cannot enable sharing.

Each region owns its block references directly, rather than retaining a whole
ancestor snapshot. Immutable allocations are bounded to 1 MiB and only whole
allocations are shared. A rewind may recopy one partial block per immutable
region, preventing unused tails from accumulating across repeated branches. Copies use a
separate stream, finish before publication, and drain before cancellation or
exceptions can free their destinations. Capturing an already frozen session is
serialized for its provenance metadata; unrelated sessions remain independent.

Native restore transfers the complete state directly from its host blocks, with
one final stream synchronization. `CopyTo` exports the existing complete version-15
payload into caller-owned storage. Disk export streams the blocks directly,
without allocating a contiguous temporary. The diagnostic `bytes()` view materializes
that payload once, lazily. Disk format, cache identity, logical byte admission
and eviction policy are unchanged. A first capture, an expired predecessor or
a raw disk restore still requires a full capture.

## Windows qualification

October 6, 2026, native Windows 11, gfx1151, TheRock ROCm 10.0.0/clang 23,
MSVC 14.51 headers/SDK, Ninja and the unchanged manifest-pinned x64-windows
dependencies. Production timing uses the `release` preset; model tests use
`gpu-test`. GUI settings were not changed; disk tests use private cache directories.

Fork base: `a3458e5b58c4dae4d3ffa8253baaf0d0ae54f6e6`. The original production
baseline is `e91eea6c7213`, whose source, CMake and dependencies match the base;
the intervening change is README-only. Baseline executable SHA-256:
`464277d5a4407fe3ec50ae8c34cb4ecfb557a21b759f756b546344a638292a5e`.
Candidate executable `a3458e5b58c4-dirty` SHA-256:
`e1c1910f644c7350ee2426889746bf3b874922973ad9ff39fc44a855670b397a`.

Artifacts are UD-IQ4_XS (three target shards), shared Q8_0 MTP and the BF16
projector. Fresh hashes match the [previous recorded identities](device-loss-pr390.md).
The checked corpus is
[`host-snapshot-corpus.txt`](../models/qwen3.8-flash-next/artifacts/host-snapshot-corpus.txt),
SHA-256 `e29a67f814ab3fe409f9c845669e420746d3c5ab55196e0085efbde4f1fae49e`.

The full and incremental payload comparison passes in AR and MTP at
2047 -> 4095 tokens. AR shares 50,307,072 device bytes and copies 171,531,264;
MTP shares 54,497,280 and copies 176,357,120. Their exported payloads match
independent full captures byte for byte. The fixture also checks unchanged
capture, rewind with a changed suffix, reset with identical tokens, cancelled
capture after transfers are queued, weak-reference expiry, concurrent export,
source destruction and both native and persistent restoration. Existing sampled
snapshot tests preserve continuation tokens, acceptance and deferred rejection
state. The snapshot test's timings are diagnostic, not production benchmarks.

All 675 complete matched-token logit rows match their baseline SHA-256 hashes:
225 positions each at schedules `1`, `1`, and `2,4,8`. Perplexity is exactly
21.579505 in every schedule. Prefill chunk checks and peer snapshot/graph-capture
interleaving pass. All 105 explicit rollback-prefix continuations pass. The
image-prefix fixture passes in AR/MTP, including warm/cold/restored full-logit
and perplexity equality, partial-image restoration and cancellation recovery.

The session test now respects the existing `keep_rollback_rows` tuning when
checking restore at a full context: Windows may retain warm graph scratch;
untuned execution must trim it. Reset follows the same scratch-retention
policy, while always clearing checkpoint provenance.

C2/C4/C6/C8 batched AR/MTP logits, sampling/RNG, residual rejection and peer
cancellation pass with both Windows defaults and `GUFO_PLATFORM_TUNING=none`.
HTTP passes 27 cache-growth checks, 15 history-edit checks and two streamed
cancellation/resume cases. Existing baseline version-15 disk checkpoints restore
correctly in the candidate. The final candidate's direct streamed checkpoints
also restore in the old baseline with identical seeded output, a real disk hit
and zero prefill on the 2371-token restored prompt. Its subsequent turn matches
too. Disk tests use private caches and stop only their own server processes.

## Production measurements

One matched growing-history sequence per build, C1, IQ4_XS/Q8_0 MTP, greedy,
seed 441, thinking off, context 49152, prefill chunk 2048, disk off. Each
background contains 16384 or 32768 repetitions of the single-token text
` archive`; each new user message contains 2048 ` note` repetitions and asks
for repeated `beta` output. Warm turns process 2079 new prompt tokens and return
126-128 actual output tokens under a 128-token limit. Replay the complete
returned assistant message before appending the next user turn.

The table is the median of four corresponding warm turns, excluding model load
and the initial cold turn. Cached histories span 18615-25234 and 34999-41615
tokens, respectively. All ten replies, finish reasons, token counts, draft counts
and accepted-draft counts match exactly. Both builds retain six snapshot entries;
their available-memory-derived byte budgets vary slightly, with no byte-admission
skips in the measured sequences. The initial baseline overlapped model-file
hashing and is excluded; the retained baseline repeat has no hashing workload.

| Initial background | Snapshot elapsed, baseline -> candidate | Warm-turn wall time, baseline -> candidate | Reduction |
| --- | ---: | ---: | ---: |
| 16K tokens | 552.00 -> 243.36 ms | 5167.66 -> 4747.35 ms | 8.1% |
| 32K tokens | 896.99 -> 266.25 ms | 5666.96 -> 4869.30 ms | 14.1% |

Snapshot elapsed falls 55.9% / 70.3%; time to first token falls
3201.59 -> 2937.98 ms / 3554.83 -> 3050.23 ms. These are workload-specific
HTTP latencies, not decode-kernel throughput or general long-context results.
`cache_snapshot_ms` aggregates capture elapsed time and can include asynchronous
waiting; it is not a pure device-copy timer. Logical `cache_snapshot_bytes`
remain identical despite the reduced physical copies.
[Measurements, requests and identities](../models/qwen3.8-flash-next/artifacts/incremental-host-snapshots.json).

## Reproduction

Build the candidate with `tools/windows/build.ps1`, using the same toolchain and
dependency paths as the baseline. For the qualified model and MTP paths:

```powershell
powershell -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset gpu-test -Target qwen38_flash_next_model_tests -Jobs 4
powershell -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset release -Target gufo -Jobs 4
& $gpuBin/qwen38_flash_next_snapshot_test.exe --model $model --mtp-model $mtp
& $gpuBin/qwen38_flash_next_session_test.exe --model $model --mtp-model $mtp --prefill-only
& $gpuBin/qwen38_flash_next_session_test.exe --model $model --mtp-model $mtp --batch-only
& $gpuBin/qwen38_flash_next_rollback_test.exe --model $model --mtp-model $mtp
& $gpuBin/qwen38_flash_next_image_prefix_test.exe $model $mtp $projector $imageA $imageB
```

The long worktree path makes hipBLASLt's staged lazy-library paths reach the
Windows path-length limit, coinciding with a crash before capture. Launching
through a short `subst` drive fixes the candidate's startup
without changing engine code or the toolchain. No hipBLASLt override is used in
the successful runs. Remove task-owned drive mappings after testing.

Run the identical command for both production executables, changing only
`$exe` and `$out`:

```powershell
& $exe bench --model $model --mtp-model $mtp --speculative mtp --temperature 0 --seed 441 --logit-eval docs/models/qwen3.8-flash-next/artifacts/host-snapshot-corpus.txt --logit-out $out --logit-schedules '1:1:2,4,8'
python tools/bench/logit-eval.py "$baseline-s0.bin.sha256" "$candidate-s0.bin.sha256"
python tools/bench/logit-eval.py "$baseline-s1.bin.sha256" "$candidate-s1.bin.sha256"
python tools/bench/logit-eval.py "$baseline-s2.bin.sha256" "$candidate-s2.bin.sha256"
```

Commands, raw dumps, test logs, HTTP responses and model hashes are retained
locally in the implementation worktree's ignored `build/validation/` directory.
Linux build/runtime and performance have not been exercised here. This is
exact consistency with the recorded fork baseline, not independent upstream
model qualification.
