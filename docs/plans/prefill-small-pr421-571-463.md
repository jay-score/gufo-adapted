# Focused upstream prefill changes

Worktree branch: `perf/upstream-prefill-small-pr421-571-463`, based on
`6985ca53b04d0115cde735812d86662bde529e9d`. Official review tip:
`f6ca085a5091eed23541cde6fa1ff282cbd4c68b` (October 15, 2026).

This campaign kept the original 2048-token chunk. The subsequent independent
[4096-token experiment](prefill-chunk-4096.md) records the later fork-default
chunk-size change and measurements.

## Scope and acceptance

Apply only the eager selector grid, dense projection launch order and attention
LDS stage from #421, the wide unquantized F16 tile from #571, and bounded selector
scratch from #463. Keep the 2048-token chunk default, Windows reader count,
MTP catch-up mask range and captured decode capacity. BF16 library replacement,
LDS-only barriers, residual fusions and final-row pruning are outside this series.
The #470 n-gram prefetch is deferred in [todo.md](../../todo.md).

Preserve complete logits and perplexity exactly. Use existing independent
operators during each step and compare the complete series against preserved
baseline binaries. Keep intermediate binaries for diagnosis. No performance
claim follows from a successful build or removal of scratch alone.

## Baseline

Fresh native Windows release and assertion-enabled `gpu-test` builds use
TheRock 10.0.0, AMD clang 23.0.0git
`8f497e0992fb7513f7f78a6f6b6f1056c375e961`, and pinned vcpkg commit
`9e2895bf6afb246396d85232ba70fcfa1fa67ba1`.
The unchanged builder environment uses MSVC 14.51.36231 headers/libraries,
Windows SDK 10.0.26100.0, CMake 3.29.2 and Ninja 1.12.0. No toolchain update
was performed.
The original binaries remain under ignored `build/release/gufo-baseline.exe`
and `build/gpu-test/baseline-session.exe`, beside their unchanged runtime assets.
The existing user executable and GUI settings are untouched.

The session diagnostic adds a 32768-token prefix to its existing
96/1023/1024/2048/4096 cases, using the same harness for both builds. Each format
records 324 complete vocabulary rows over three fixtures, AR/MTP and eight
teacher-forced continuations, plus continuation NLL/perplexity. This is fork
regression qualification, not new unquantized-model or GGUF conversion parity.

## Incremental checks

- Baseline: fresh release and GPU builds pass. `select_ops`, `projection_ops`,
  `attention_ops` and `gdn_ops` pass (4/4). The existing resource contract and
  all nine dense F16 variants have zero scratch/register spills.
- Eager grid: fresh release/GPU builds, cleared-output full/bounded-grid
  selection operator, original resource guard and nine zero-scratch/spill dense
  F16 variants pass. Formatting passes. Final matched-model qualification is
  pending with the rest of the series.
- Launch order: fresh release/GPU builds and projection/attention/GDN/F16
  replay operators pass (4/4), including ragged projection and fused epilogue
  checks. All eleven dense F16 variants retain zero scratch/register spills;
  the original resource guard and formatting pass.
- Attention LDS stage: fresh release/GPU builds, projection and attention
  operators pass (2/2). Formatting and the original resource guard pass;
  eleven dense F16 variants have zero scratch/register spills.
- F16 router tile: fresh release/GPU builds and projection/F16 replay/GDN
  operators pass (3/3). The existing F16 check compares a 2048-token projection
  with 1025/1023-token pieces exactly and retains its independent F64 bounds.
  Formatting, original resource guard and twelve scratch/spill-free dense
  F16 variants pass.
- Selector scratch: fresh release/GPU builds and full/bounded/compact selector
  masks pass. The compact fixture reuses an exactly-sized buffer over 31-row
  chunks, including 131K positions and ties, and clears masks before comparing.
  Formatting, original resource guard and twelve scratch/spill-free dense
  F16 variants pass. The score buffer is 20 MiB instead of 128 MiB at the
  model's 262K context; this is not a process committed-memory measurement.
- Final matched comparison: all 648 complete vocabulary rows (160,911,360
  F32 values) and both entire capture files are byte-identical. Both
  continuation NLL/perplexity summaries are exact. The diagnostic source
  SHA-256 is identical between baseline and candidate.
- Both formats pass the full session, serialized snapshot and rollback suites
  (6/6). These include graph replay, private C2/C4/C6/C8 logits/RNG, sampled
  rejection/residual correction, fresh loads, 23 serving sampler strategies,
  cancellation and 105 rollback prefixes per format. Snapshot tests also cover
  the default prefix draft head; the session diagnostic uses the full head.
- HTTP: all 92 baseline/candidate responses have identical text and prompt/
  output token counts, with zero cached tokens. Each arm covers greedy
  C1/C2/C4/C6/C8, one seeded-sampled C1 and one 256-by-256 PNG attachment using
  the pinned BF16 projector, in AR and full-head MTP modes. Every request emits
  all 128 budgeted tokens. The image increases the prompt from 2271 to 2339
  tokens and retains the same output hash across builds in both modes.
  These paired runs establish functional regression parity, not full HTTP
  latency non-regression.

## Native production measurements

The final production executable reports `c4447eedb977`; the baseline reports
`6985ca53b04d`. All generated-token hashes agree across the ABBA arms. Each cell
below is the mean of two fresh-process observations, using pp2048 and tg128,
greedy AR and depths 0/32768. Loading and depth preparation are outside the
timed regions. The existing 2048-token engine chunk is unchanged. The native
fixture repeats a small token pattern and is not an SSD-heavy serving workload.

| Format | Test / depth | Baseline tok/s | Candidate tok/s | Change |
| --- | --- | ---: | ---: | ---: |
| IQ4_XS | pp2048 / 0 | 993.45 | 995.50 | +0.21% |
| IQ4_XS | pp2048 / 32768 | 934.58 | 927.01 | -0.81% |
| IQ4_XS | tg128 / 0 | 28.77 | 28.52 | -0.85% |
| IQ4_XS | tg128 / 32768 | 26.66 | 26.55 | -0.43% |
| UD-Q4_K_XL | pp2048 / 0 | 1588.80 | 1595.37 | +0.41% |
| UD-Q4_K_XL | pp2048 / 32768 | 1429.74 | 1431.38 | +0.11% |
| UD-Q4_K_XL | tg128 / 0 | 29.12 | 28.75 | -1.25% |
| UD-Q4_K_XL | tg128 / 32768 | 26.79 | 26.52 | -0.99% |

These differences are within the observed between-arm variation (up to 2.54%).
There is no demonstrated native throughput gain at the current chunk size;
strict latency non-regression is not established either. The confirmed gains
are the smaller score allocation and the compiled attention projection's
48-to-24-KiB LDS reduction, with VGPRs 221 to 207 and zero spills. Performance
numbers in imported kernel comments describe upstream controls, not these
Windows measurements.

[Portable build/model identities, quality and individual observations](../models/qwen3.8-flash-next/artifacts/prefill-small-pr421-571-463.json).

## Evidence and commands

Task-owned raw commands, logs, model SHA-256 identities, preserved executable
hashes, captures and resource reports live under ignored `build/prefill-small/`.
Builds use `tools/windows/build.ps1 -Target gufo -Jobs 4` and the `gpu-test`
preset for the affected tests, with the same local dependency paths throughout.
The worktree reuses the parent's pinned vcpkg installation; no dependencies or
toolchain versions were updated.

Full-row capture:

```powershell
build/gpu-test/baseline-session.exe --model FIRST.gguf --mtp-model MTP.gguf --capture-prefill-logits baseline.bin
build/gpu-test/qwen38_flash_next_session_test.exe --model FIRST.gguf --mtp-model MTP.gguf --capture-prefill-logits candidate.bin
```

Compare SHA-256 for the entire captures, every reported vocabulary row and the
NLL/perplexity summaries. Production timing uses matched pp2048/tg128 with
depths 0 and 32768, greedy AR, in ABBA order with one repetition per arm
(two independent observations per binary/point). Small differences inside the
observed arm spread remain inconclusive. Timings run separately from builds,
hashing and correctness jobs.

Linux build/CI and GPU timeline profiling have not been run.

The final binaries and all intermediate candidates remain in this worktree.
The five implementation commits update `UPSTREAM.md` individually; the #470
follow-up is a separate TODO commit. The original checkout, running-user
configuration and 2048-token engine default are unchanged. Nothing was pushed.
