# AGENTS.md

## Platform and Build (Linux Primary Target — AMD Strix Halo `gfx1151`)

The primary development and runtime target is **Linux x86_64 on AMD Strix Halo (`gfx1151`)** with unified memory (54–60 GiB budget) and NVMe SSD streaming. Preserve cross-platform (`_WIN32` / `compat/win32`) compilation compatibility in shared code, but optimize and validate natively on Linux.

CMake owns compiler flags, dependencies, and installation; Nix supplies the pinned Linux toolchain (including the patched ROCr runtime).

```sh
nix build                              # production package, no tests/tools
./result/bin/gufo diagnose
nix build .#checks.x86_64-linux.pr      # bounded hosted CPU/repository checks
nix develop                            # GPU development and reference tools

# Iterative development builds inside Nix environment:
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target gufo qwen38_flash_next_tests expert_cache_manager_test

# Production release binary for benchmarking / serve:
nix develop -c cmake --preset release
nix develop -c cmake --build --preset release --target gufo --parallel 8
```

Stage only task-owned paths before Nix builds; flakes include tracked files. Measure performance with `build/release/gufo` or `result/bin/gufo`. Preserve compiler/dependency versions when comparing results.

## Out-of-Core MoE Streaming Architecture (`Qwen3.8-Flash-Next`)

### Key File Map
- `src/models/qwen38_flash_next/kernels/rocm/executor.cpp`: Core MoE execution (`StreamedMoeRows` for decode, `StreamedMoeRowsDeskRoom` for whole-chunk prefill).
- `src/models/qwen38_flash_next/kernels/rocm/streamed_moe_policy.hpp`: Thresholds (`kDeskRoomMinTokens`) and pass-sizing math for Desk/Room streaming.
- `src/storage/expert_cache_manager.{hpp,cpp}`: Expert slot state machine, `Resolve`, speculative `Prefetch`, `FinishPrefetch`, and eviction policies (`lru` default, `hotness`, `decayed`).
- `src/storage/ssd_batch_reader.{hpp,cpp}`: Linux `O_DIRECT` + 4 KiB page-aligned range coalescing and 3-tier I/O priority (`Demand -> Room -> Speculative`).
- `src/models/qwen38_flash_next/engine.{hpp,cpp}`: Memory budget planner (`--memory-budget-gb`), session KV/snapshot reservation, and model initialization.
- `src/models/qwen38_flash_next/ngram.{hpp,cpp}`: Per-Layer Embedding (`NgramTable` / `--ple-shelf`) lookup and SSD read path.
- `MAPPING.md` & `docs/models/qwen3.8-flash-next/EXPERIMENTS.md`: Ground-truth engineering log and verified invariants.

### Verified Architectural Invariants (Do Not Regress)
1. **Default Cache Policy (`lru`):** Keep `lru` as the default `--cache-policy`. Live GPU benchmarks with speculative prefetch (`--speculative mtp --prompt-lookup`) proved `lru` beats `hotness` by reducing decode cold misses by 34% (high prefill frequency counts under `hotness` cause premature eviction of newly prefetched Stage A/B experts).
2. **Prefetch Lifecycle Ordering:** `FinishPrefetch(layer.layer_index)` MUST execute *after* `Resolve` and layer execution complete (via scope guard on early returns) and *before* submitting next-layer lookahead. Never call `FinishPrefetch` before `Resolve`, or it will cancel queued prefetches and zero out `prefetched_hit_async` accounting.
3. **3-Tier I/O Priority:** `Demand` (`Resolve`) -> `Room` (prefill pass streaming) -> `Speculative` (decode lookahead, capped at `kSpeculativeTransactionCap = 2`).
4. **Deferred Epilogue Exactness:** Maintain 100% bit-exact FP32 logit equivalence on deterministic changes via deferred `Combine()` (`moe_pending_`) across multi-pass windows.

## Focused Tests & Verification

Before committing C++ changes, run the shared CI format check:

```sh
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
# Add --fix to apply formatting, then rerun the check.
```

Run the smallest check covering the change (`gpu-test` is `RelWithDebInfo` with assertions enabled):

```sh
nix develop -c cmake --build --preset gpu-test --target <test-target>
nix develop -c ctest --preset gpu-full -R '^<test-name>$' --output-on-failure
```

Key focused tests for Qwen3.8-Flash-Next and server changes:
- MoE & cache unit tests: `ctest --preset gpu-full -R '^(expert_cache_manager_test|qwen38_flash_next\.(iq_experts_ops|moe_ids|moe_epilogue_ops|routed_wmma_ops))$' --output-on-failure`
- Offline cache replay test: `nix develop -c python3 tests/tools/test_expert_cache_replay.py`
- Server / API tests: `ctest --preset gpu-full -R '^(openai_chat_test|http_server_test)$' --output-on-failure`

For code changes that could affect inference correctness, run matched-token logit and perplexity checks against a recorded baseline (`docs/TESTING.md#matched-token-and-layer-comparisons`). Require exact full-logit equality when arithmetic should be unchanged, or declared numerical limits when it intentionally changes.

## Profiling and Kernels

Apply [.agents/skills/optimize-kernel/SKILL.md](.agents/skills/optimize-kernel/SKILL.md).
Use `tools/bench/build.sh` for standalone HIP experiments, `tools/bench/gfx1151_peak.hip` for measured hardware ceilings, `tools/prof/prof.py` for pipeline/wall-time profiles, and `tools/prof/isa_mix.py` for instruction analysis. Production paths must retain quality; successful optimizations become the default, without extra switches.

## Agent Operating Rules (Token & Hardware Efficiency)

- **Keep command output concise:** Never dump thousands of lines of build logs or raw traces into context. Pipe verbose commands through `tail -n 40` or filter for errors.
- **Avoid unprompted full-model prewarm sweeps inside the agent loop:** Do not run multi-minute 26+ GiB full-model prewarm benchmarks repeatedly inside automated loops; validate logic with unit tests first, then run targeted `build/release/gufo` benchmarks when ready.
- **CodeGraph:** If `.codegraph/` exists, use CodeGraph before searching or reading code to locate an implementation.
- **Git & Commits:** Confirm the checkout, branch, and working tree before editing. Follow Conventional Commits with a single-line message. Keep improvements small and independently reviewable; do not push unless requested.