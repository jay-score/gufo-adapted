# Flash-Next 4096-token prefill chunk experiment

October 15, 2026, native Windows/gfx1151. Flash-Next's original internal prefill
capacity was `min(2048, max_context)`. After this experiment and the paired
memory comparison below, 4096 is the fork default. This is separate from HTTP's `--prefill-chunk`,
which budgets prefill between active decode steps. There is no CLI override
for this internal model constant.

The experiment changes only `kPrefillChunkTokens` from 2048 to 4096 in
`src/models/qwen38_flash_next/engine.cpp`. Both builds contain the five selected
#421/#571/#463 changes. The 2048 binary is the preserved `c4447eedb977` build;
the 4096 binary is `ef549cd80daf-dirty` with that single-line patch. Those
commits otherwise have identical C++ and test sources. The model diagnostic
source is unchanged. No #470 prefetch, kernel, precision, model artifact,
dependency, power setting or GUI setting changes accompany this experiment.

## Method

Use the same TheRock 10.0.0/clang 23 Windows release toolchain, runtime assets,
UD-IQ4_XS and UD-Q4_K_XL models as the
[preceding qualification](prefill-small-pr421-571-463.md). Preserve both
production binaries, assertion-enabled session diagnostics and intermediate
objects in this isolated worktree. Candidate binaries remain beside their
runtime assets as `build/release/gufo-chunk4096.exe` and
`build/gpu-test/chunk4096-session.exe`.

For each format, use fresh processes in ABBA order: chunk 2048, 4096, 4096,
2048. Each process runs:

```powershell
.\build\release\gufo-chunk2048.exe bench --model "$MODEL" --speculative off -p 2048,4096 -n 128 -d 0,32768 -r 1 -v
# Repeat with gufo-chunk4096.exe for the B arms.
```

The existing benchmark warms each shallow shape with an untimed pass; aligned
32K prefixes warm full-chunk deep shapes. The 4096 candidate's deep pp2048
control also receives an untimed pass. Prefill throughput times `Session::Sync`
for new prompt tokens, excluding model loading and preparation of the cached
prefix. Both binaries use context capacity 36865. Decode times 128 generated
tokens. Two observations per binary/point are arithmetic-mean summarized;
arm spread is `(max - min) / mean`. No builds, model hashing or other GPU jobs
overlap timing. The native fixed repeated token pattern has limited distinct
n-gram traffic; these results do not describe diverse disk-backed prompts.

## Numerical and state checks

Both formats retain all 324 complete vocabulary rows each, including three
prose/code fixtures, AR/MTP, prefixes 96/1023/1024/2048/4096/32768 and eight
teacher-forced continuations. All 160,911,360 F32 values, both whole capture
files and continuation NLL/perplexity are byte-identical to the preserved
2048-chunk captures. This is a regression comparison, not independent
unquantized-teacher or conversion qualification.

Both assertion-enabled full session suites pass with the Q8_0 MTP sidecar and
full draft head. They cover bulk/split boundaries, private C2/C4/C6/C8 state,
graph capture/replay, cancelled mutation/restore recovery, image-identity
snapshot attachment, greedy/seeded interleaving, sampled p/q rejection and
residual correction, fresh loads, and 23 serving sampler strategies in AR/MTP.
The compiled resource guard passes, including all twelve dense F16 variants
without scratch/register spills. Formatting passes all 489 C++ files.

```powershell
.\build\gpu-test\chunk4096-session.exe --model "$MODEL" --mtp-model "$MTP" --capture-prefill-logits "$OUT"
.\build\gpu-test\chunk4096-session.exe --model "$MODEL" --mtp-model "$MTP"
```

The paired IQ4_XS AR HTTP campaign passes all **46 responses**: greedy
C1/C2/C4/C6/C8, seeded-sampled C1 and a real 256-by-256 four-color image with
the BF16 projector. Each request emits 16 tokens, reuses zero prompt tokens,
and preserves prompt/text hashes and counts across builds. Both arms receive
the same untimed C1 warmup. The plain fixture has 4498 prompt tokens; the image
case has 4566. Use eight sessions, context 65536, thinking off, cache RAM capped
at one byte, the default 512 active-prefill budget and per-client queue limit
16. All test servers are stopped afterward.

HTTP C1 prefill is **863.85 → 864.95 tok/s** (5206.91 → 5200.33 ms) in this
single pair: no useful speed difference is established. The fixture encounters
the existing 2048/4096 checkpoint boundaries described below. The full native
gains therefore should not be assumed for this server request. Raw per-request
latencies are retained; this bounded campaign is functional validation rather
than general HTTP latency non-regression qualification.

## Timing and decision

All rates are tokens/second; positive change favors chunk 4096. `pp4096` is a
4096-token prompt, independent of the internal chunk limit. Depth is the
already-populated prefix. Spread lists chunk 2048 / chunk 4096.

UD-IQ4_XS:

| Workload / depth | Chunk 2048 | Chunk 4096 | Change | Arm spread |
| --- | ---: | ---: | ---: | ---: |
| pp2048 / 0 | 999.10 | 993.00 | -0.61% | 0.29% / 0.66% |
| pp4096 / 0 | 984.65 | 1169.67 | +18.79% | 0.14% / 0.02% |
| tg128 / 0 | 28.71 | 28.55 | -0.57% | 2.75% / 1.19% |
| pp2048 / 32768 | 935.78 | 932.66 | -0.33% | 0.91% / 0.44% |
| pp4096 / 32768 | 928.97 | 1101.49 | +18.57% | 1.35% / 0.04% |
| tg128 / 32768 | 26.64 | 26.55 | -0.32% | 1.58% / 0.19% |

UD-Q4_K_XL:

| Workload / depth | Chunk 2048 | Chunk 4096 | Change | Arm spread |
| --- | ---: | ---: | ---: | ---: |
| pp2048 / 0 | 1607.03 | 1610.91 | +0.24% | 0.35% / 0.24% |
| pp4096 / 0 | 1588.27 | 1691.59 | +6.51% | 0.89% / 0.12% |
| tg128 / 0 | 28.69 | 29.47 | +2.72% | 0.07% / 0.14% |
| pp2048 / 32768 | 1441.41 | 1438.89 | -0.17% | 0.22% / 0.22% |
| pp4096 / 32768 | 1434.18 | 1533.82 | +6.95% | 0.45% / 0.69% |
| tg128 / 32768 | 27.43 | 27.43 | +0.00% | 0.04% / 0.26% |

All generated-token hashes match across the four arms of each format.
The pp4096 gain is larger than observed arm variation. Small-prompt controls
remain within 1%; decode observations do not establish a general decode gain.

**Decision:** adopt 4096 as the Flash-Next default for this fork. The prefill
gain on prompts over 2048 tokens outweighs a fixed ~1.36 GiB extra GPU
reservation. Full 262K context with MTP still fits a 96 GiB carve-out, as
measured below. The measured `ef549cd80daf-dirty` executable denotes the
recorded one-line experiment patch; the later comment and documentation do not
change its behavior.

Many activation buffers scale with maximum batch capacity, so the larger limit
increases scratch reservation; see the paired memory comparison below. Linux,
MTP serving latency, sustained mixed-arrival fairness and GPU timeline profiling
remain unqualified. No #470 prefetch is included.

HTTP's [intermediate checkpoint grid](../../src/cli/serve/text_model_runner.cpp)
still has a 2048-token interval. It can split a short cold prompt into
2048-token forwards even with a 4096 model capacity; concurrent work also uses
the existing 512-token scheduling budget. The larger cap therefore does not
force 4096-token forwards on every HTTP request. Neither policy is retuned here.

The portable [artifact](../models/qwen3.8-flash-next/artifacts/prefill-chunk-4096.json)
retains binary/model/toolchain identities, individual native observations,
quality hashes and the bounded HTTP results. Raw captures, command records and
logs remain under ignored `build/prefill-4096/`.

## Brief memory check

A current-build-only IQ4_XS AR run uses pp4096/tg16, depth 0, context 4097,
one repetition plus the benchmark's untimed prefill warmup. No older build is
run for comparison. The successful sample lasts 28.5 seconds, with Windows
process/system memory and GPU Process Memory PDH counters sampled every 200 ms.

- GPU committed peak: **64.989 GiB**, comprising **64.615 GiB dedicated** and
  **0.374 GiB shared** in Windows accounting.
- Process private commit peak: **1.188 GiB**; peak working set **1.165 GiB**.
- System-wide commit: **27.576 GiB before launch**, **93.895 GiB peak**.
  This includes other applications and the driver, and is not Gufo-only memory.
- Lowest OS-reported available physical memory: **11.549 GiB**. This is
  OS-visible available memory, not all memory available to the UMA GPU.
  Dedicated/shared GPU labels are Windows accounting on UMA.

These counters overlap; do not add GPU/process figures to system-wide commit.
They describe the whole model/runtime run, not the incremental cost of chunk
4096. Buffered stdout prevents a reliable separate load/prefill peak. This is
a short-context AR check, without MTP or a vision request; larger contexts and
other serving configurations are not measured. No code or binary changes are
made for this check. Raw samples and logs are in `build/prefill-4096/memory-*`;
the compact summary is in the portable artifact above.

## Paired 2048/4096 memory comparison

Same day, worktree `t3-28bc9e40`: `gufo-chunk2048.exe` is `6ca4c1a1c109`;
`gufo-chunk4096.exe` is `6ca4c1a1c109-dirty` with only the
`kPrefillChunkTokens` change. Both use the TheRock 10.0.0 release toolchain and
the UD-Q4_K_XL model. One fresh process per arm, sampled every 100 ms with
the PDH counters from the brief check above, and no other model processes.

- **Bench:** AR pp4096/tg16 at depth 0, as in the brief check.
- **Serve:** context 200000, one session, Q8_0 MTP sidecar,
  `mmproj-BF16.gguf`, thinking off and cache RAM capped at one byte.
  One cold 12,911-token chat request with 32 output tokens.

| GPU committed, GiB | Chunk 2048 | Chunk 4096 | Change |
| --- | ---: | ---: | ---: |
| Bench, post-load plateau | 79.711 | 81.069 | +1.358 |
| Bench, whole-run peak | 79.962 | 81.319 | +1.357 |
| Serve, idle after load | 87.481 | 88.841 | +1.360 |
| Serve, peak during request | 87.878 | 89.237 | +1.359 |

The server's own `gpu_device_used_mib` after load is 89320 → 90709 MiB
(+1389 MiB) of 112.6 GiB. The extra memory is a fixed reservation made at
load. The transient rise during a request (about 0.25 GiB in bench, 0.40 GiB
in serve) is the same for both chunk sizes. Process private commit
differs by under 0.03 GiB. The served request used 8 forwards with a
2048-token maximum versus 6 with a 4096 maximum. Its single-sample prefill was
1201 → 1343 tok/s; this is not a qualified speed comparison. Bench pp4096 was
1623 → 1686 tok/s.

At full context 262144, with otherwise identical serve settings, the 4096 build
peaks at **90.83 GiB** GPU committed (2048: 89.47). Of that, **90.45 GiB** is
dedicated within the 96.00 GiB carve-out (2048: 89.11). Shared usage stays at
0.38 GiB (2048: 0.36), so nothing spills beyond the carve-out. Load time is
24.4 s for both. `gpu_device_used_mib` after load is 90948 → 92273 MiB.

Flash-Next snapshots serialize only unpooled indexer rows in chronological
order, and restore accepts any destination ring large enough. Disk checkpoints
written by 2048-capacity builds therefore remain loadable.

Only one session is measured. Activation scratch belongs to the model's single
executor and is shared by all sessions. Each session's indexer ring also
depends on the batch capacity (`bit_ceil(top_k + batch)`), adding about 26 MiB
per AR/MTP session at long context. Raw samples and logs are in ignored `build/mem-4096/`.
