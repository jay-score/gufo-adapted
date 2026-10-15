# Focused upstream prefill changes

Worktree branch: `perf/upstream-prefill-small-pr421-571-463`, based on
`6985ca53b04d0115cde735812d86662bde529e9d`. Official review tip:
`f6ca085a5091eed23541cde6fa1ff282cbd4c68b` (October 15, 2026).

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
- F16 tile and selector scratch: pending.
- Final matched logits/perplexity, state checks and timings: pending.

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
depths 0 and 32768, greedy AR, and three repetitions per point. Run timings
separately from builds, hashing and correctness jobs.

Linux build/CI and GPU timeline profiling have not been run.
