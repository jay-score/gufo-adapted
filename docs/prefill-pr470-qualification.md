# PR #470 retained qualification

October 8, 2026. Base: `1b9c75b1560f8677ed951d306f9b81efecb1c2f4`.
Windows 11 x64, Strix Halo gfx1151, 128 GB unified memory. This record covers
attention boundary correctness and short-request scheduler fairness. The
engine's default remains 2048 tokens. Prefetch, reader tuning, allocator scratch
reuse and the 4096-token default remain deferred.
Immutable sources and implementation attribution are in [UPSTREAM.md](../UPSTREAM.md).

## Identities

Fresh `release` and assertion-enabled `gpu-test` builds use TheRock 10.0.0,
clang 23.0.0git / LLVM `8f497e0992fb7513f7f78a6f6b6f1056c375e961`,
MSVC headers 14.51.36231, Windows SDK 10.0.26100.0, CMake 4.4.0 and Ninja 1.12.0.
Dependencies retain the base manifest, SHA-256
`4fd0225963923472102dc9d97eff370f52cc8f4e99d763946487b63dc71cdf01`.
The baseline CLI reports `1b9c75b1560f`; the attention candidate reports
`1b9c75b1560f-dirty`. Binaries were preserved beside their matching runtime assets.

| Binary | SHA-256 |
| --- | --- |
| Baseline release Gufo | `083c8aee6171d4230c5dcd6e4d866f564005aa8be746fe9e783f130384205f38` |
| Attention release Gufo | `2ab75d90682d9a0e19f36ff22aedf6f3ce03a9af9ebd0e97d4a503f2307962e3` |
| Baseline session test | `3b74492ec2b5ca4250c5d5b5ca39550a729e6ffa942543bac191453301dc4dff` |
| Clean attention session test | `acab0067de28d652a9c51f2583f92875fb32baf10d07e077a5e17ce98a11db5b` |
| Final release Gufo, `29a26847a4be-dirty` | `9bb9b47c098d2a5190290f72fe1b28aa997cbe258c8bb4d4af1d7e5e77479281` |
| Final session test | `651a4c79a6927b9f2deaa70b3f2b0445ac034b7f96cad0891a4f347cec9ad4cb` |
| Assertion-build Gufo, `29a26847a4be-dirty` | `3f50fb58b56aeb0fbef922aefe1ecd10c96c218be272b666c96db585d5a97932` |

Target shards are `Qwen3.8-Flash-Next-UD-IQ4_XS-0000N-of-00003.gguf`.
All five artifacts were freshly hashed before implementation:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Target shard 1 | 10946624 | `5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4` |
| Target shard 2 | 49835229856 | `577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7` |
| Target shard 3 | 43836407744 | `d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833` |
| `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2786568256 | `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6` |
| `mmproj-BF16.gguf` | 907542944 | `2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355` |

## Attention results

Before the fix, a 4096-token Sync after a retained 1024-token prefix fails exact
logit equality against bulk prefill at continuation row zero. The earlier eight
chunk cases pass. The fix keeps every pre-budget query on dense attention,
launches the sparse tail first to preserve geometry refusal, offsets all row
buffers consistently and projects once. Last-only predictor catch-up retains
its original route. Snapshot payload version 16 rejects older computed state.

| Check | Actual result |
| --- | --- |
| Existing eight chunk cases plus 4096/1024 and 3072/1024, AR and MTP, four frontier rows each | All 80 full-logit comparisons byte-identical; no tolerance changes |
| Existing three-fixture capture, prefixes 96/1023/1024/2048/4096, AR/MTP, eight Evaluate continuations | All 270 full rows byte-identical to baseline; repeated baseline also identical |
| Capture NLL / perplexity, 240 labels | `0.25759154328132261` / `1.2938102451945588`, unchanged |
| Teacher-forced schedules `1:2,4,8:p8`, temperature 0, seed 459 | All 549 raw-logit hashes match within corresponding schedules; perplexity `1.343238 / 1.343238 / 1.361810`, unchanged |
| Existing analytic attention operators | Pass with existing numerical envelopes and geometry checks |
| Snapshot and rollback targets | Exact round trips, old-version rejection, deferred residual/RNG replay; all 105 rollback-prefix continuations exact |
| Image-prefix target, BF16 projector, two 512x512 screenshots | AR/MTP extension, snapshot, seeded/greedy replay and cancellation checks pass |
| Session `--batch-only` and `--sampling-only` | C2/C4/C6/C8 logits/RNG/state, sampled acceptance/rejection and all 23 serving strategies pass |
| Matching-bundler resource guard, baseline and release candidate | 57 affected gfx1151 kernels, zero scratch/spill failures |

The 268185600-byte capture has SHA-256
`dfc7d5b755688838732f806676ce238af0b1ad461c6ced5d8aa3963ea38f1731`.
The 897-byte teacher-forcing corpus has SHA-256
`7a552ae620229f69188b6018e693b726976f468a273b867e3e48f0d7a584eb39`:
eight repetitions of `The quick brown fox jumps over the lazy dog. A triangle
has three sides. Water freezes at zero degrees Celsius. `, followed by LF.
Screenshot hashes are `d701019e7fc79fdb8a5066c07718957dd513b0ce02432dede29fd8d9989966fb`
and `dc19ae2e21a9ba616a208f1944aad7cc47c2045ff200972bbcf2bfc7672d4efd`.

An additional 3072/1025 diagnostic still differs: all mixer rows match through
layer 19, and that layer's full query and gate matrices also match, but its
untouched indexer produces a different mask at token 2906. That is the first
attention difference; recurrent state propagates
it afterward. A 4096/1025 diagnostic also fails equality. These are retained
limitations, not passed checks or relaxed tolerances. This adaptation establishes
the dense/sparse dispatch boundary, not arbitrary chunk-shape or scalar/bulk
equality. A wider indexer numerical rewrite is outside this change.

Temporary tracing caused excluded capture failures when a timestamp-preserving
source restore left an instrumented object in the incremental build. Removing
that object forced recompilation; diagnostic strings are absent from the clean
binary above. All reported qualification runs used that clean binary.
Logs, raw rows, commands and diagnostic comparisons remain in ignored `build/pr470`.
Linux build/CI and additional model quantizations are unrun. These are fork
consistency checks; independent full-model quality is not newly established.

## Scheduler fairness

The pre-fix scheduler fails the new deterministic arrival fixture: it repeats a
long prefill before the new request's first prefill. The adaptation inserts an
arrival before that repeated turn while preserving older waiting peers, bounds
work ahead of short waiting prompts using the existing decode budget, and forms
an initial speculative batch only when its pending peer fits that budget.
It hands off after the current forward; it does not preempt a running kernel.

Fresh CPU builds pass `text_generation_scheduler_test`,
`text_generation_device_loss_test`, `text_model_runner_test`, `openai_chat_test`
and `http_server_test` (5/5). Four new fixtures run in both AR and batched MTP
modes: short arrival during long prefill, wide alternating long prefills, short
arrival behind an older waiting peer, and repeated arrivals with long-request
progress. The existing suite retains cancellation, asynchronous capture,
device-loss and fallback coverage.

The rebuilt session target passes all 23 serving sampling strategies, including
24 batched C2 requests with exact replay and token budgets. The final release
repeats all 549 teacher-forced raw-logit matches and unchanged perplexity above.
Its production resource guard again checks 57 kernels with zero scratch/spill
failures; `kernel_resources_test` also passes on freshly built assertion-build
Gufo. Formatting with clang-format 21.1.8, documentation links, Python syntax
and `git diff --check` pass. Linux CI and real-model qualification of other
models/quantizations remain unrun; repeated arrivals are deterministic CPU
coverage, not a sustained HTTP load measurement.

The explicit real-model check uses the existing standard-library benchmark
transport. Both baseline and candidate run at C2, context 16384, MTP,
`--prefill-chunk 512`, temperature 0, seed 470, reasoning disabled and cache
reuse disabled. The long prompt is 10818 tokens; the short prompt is 18 tokens.
Arrival follows the first observed 2048-token prefill chunk. Both short and long
completion hashes match the baseline. The ongoing-decode phase emits 256 tokens
while the long peer advances through multiple prefill chunks. These measurements
are individual observations, not medians or a throughput claim.

| HTTP observation | Baseline | Final candidate |
| --- | ---: | ---: |
| Short request client first-token latency | 7774 ms | 3631 ms |
| Long request client first-token latency | 12392 ms | 12365 ms |
| Ongoing decoder's largest server token gap | 1455 ms | 821 ms |

Both runs pass the broad HTTP progress contract; the deterministic fixtures
isolate the extra repeated-turn bug. An earlier candidate observation gave
3630 ms / 12374 ms / 821 ms respectively. The final column uses the binary
identity above. The focused HTTP script checks short-output/control equality,
early short emission, incremental long progress and ongoing decode; it does
not impose a fragile timing threshold.

## Reproduction

Use the pinned dependency paths with `tools/windows/build.ps1`. Build the session,
attention-ops, snapshot, rollback and image-prefix targets in `gpu-test`, and Gufo
in both `release` and `gpu-test`. Set `$MODEL`, `$MTP`, `$PROJECTOR`, `$GPU` and
the screenshot paths to the artifacts above. Run GPU workloads sequentially:

```powershell
$sessionArgs = @('--model', $MODEL, '--mtp-model', $MTP)
& "$GPU/qwen38_flash_next_session_test.exe" @sessionArgs --prefill-only
& "$GPU/qwen38_flash_next_session_test.exe" @sessionArgs --capture-prefill-logits build/pr470/candidate-prefill.bin
& "$GPU/qwen38_flash_next_session_test.exe" @sessionArgs --batch-only
& "$GPU/qwen38_flash_next_session_test.exe" @sessionArgs --sampling-only
& "$GPU/qwen38_flash_next_snapshot_test.exe" @sessionArgs
& "$GPU/qwen38_flash_next_rollback_test.exe" @sessionArgs
& "$GPU/qwen38_flash_next_image_prefix_test.exe" $MODEL $MTP $PROJECTOR $IMAGE_A $IMAGE_B
& "$GPU/qwen38_flash_next_attention_ops_test.exe"
& build/release/gufo.exe bench @sessionArgs --speculative mtp --temperature 0 --seed 459 --logit-eval build/pr470/corpus.txt --logit-out build/pr470/candidate --logit-schedules '1:2,4,8:p8'
python tools/bench/logit-eval.py build/pr470/baseline-s0.bin.sha256 build/pr470/candidate-s0.bin.sha256
# Repeat the comparison for s1 and s2; compare the complete prefill captures too.
python tools/ci/check-kernel-resources.py build/release/gufo.exe --bundler "$ROCM/lib/llvm/bin/clang-offload-bundler.exe"
```

Build the four CPU targets above before running their five CTest entries. For
the real-model check, start an isolated server using the model identities above:

```powershell
& build/release/gufo.exe serve --host 127.0.0.1 --port 18080 --sessions 2 llm --served-model-name gufo @sessionArgs --speculative mtp --context 16384 --prefill-chunk 512
# In another shell, then stop only this test server:
python tests/tools/prefill_fairness_test.py --url http://127.0.0.1:18080 --out build/pr470/candidate-http.json
ctest --test-dir build/cpu-test -R '^(text_generation_scheduler_test|text_generation_device_loss_test|text_model_runner_test|openai_chat_test|http_server_test)$' --output-on-failure --no-tests=error
ctest --test-dir build/gpu-test -R '^kernel_resources_test$' --output-on-failure --no-tests=error
```
