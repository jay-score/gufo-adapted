# PR #518 stages 0 and 1: implementation record

Worktree: `gufo-decode-pr518-stages01`, branch
`perf/flash-next-decode-pr518-stages01`, based on
`ab4d98189a6d8103fce29d2537e4113da3da2420`.
The [integration plan](decode-kernels-pr518.md) defines the boundaries.
This work covers baseline preservation and local kernel improvements only.
Producer Q8_1 fusion, attention read batching and expert-routing changes remain
outside this change.

## Baseline identity

Fresh native Windows builds use TheRock 10.0.0, AMD clang 23.0.0git
(`8f497e0992fb7513f7f78a6f6b6f1056c375e961`), MSVC 14.51.36231 and
the repository's pinned vcpkg manifest. The Balanced power profile and existing
96 GiB dedicated graphics carve-out are retained. No clocks, GUI settings,
running user executable or platform defaults were changed.

The preserved production executable reports `gufo version ab4d98189a6d`.
Its DLLs, ROCm kernel assets and fresh operator/native test executables are
kept together in ignored `build/pr518/baseline-bin/`.
Model shards, shared Q8_0 MTP sidecars and BF16 projectors were SHA-256 hashed
from the two user-provided directories. Both sidecars have SHA-256
`5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6`.
The IQ4_XS model has three shards; UD-Q4_K_XL has four. Native IQ experts retain
their original encoding; only the existing loader converts eligible dense tensors.

Raw identities, exact commands, toolchain logs and results are retained under
ignored `build/pr518/`, including `model-identities.json`,
`baseline-record.json`, build logs and task-local harnesses.
Paths and hashes in those files identify the tested artifacts; these are
matched-build consistency checks, not new model-conversion qualification.

## Stage 0 evidence

- Fresh baseline release and `gpu-test` builds pass.
- All 15 Flash-Next CPU/operator tests pass: four CPU and eleven GPU tests.
- The compiled resource contract checks 57 affected prefill kernels, with
  zero failures.
- The fixed prose/code corpus has 118 tokens and SHA-256
  `343375225644235c8c33395219e176868c31c2076af078bb23919ea0ded9b528`.
  Scalar decode, cycling verification widths 1-8 and eight-row prefill record
  116 complete vocabulary rows each: 348 rows per quantization.
- Normal prefill captures 270 complete vocabulary rows per quantization at
  prefixes 96/1023/1024/2048/4096, in AR and MTP modes with eight continuations.
  IQ4_XS capture SHA-256:
  `dfc7d5b755688838732f806676ce238af0b1ad461c6ced5d8aa3963ea38f1731`.
  UD-Q4_K_XL capture SHA-256:
  `d04cbfdbfef624fe63b4f1b0c0efb141415f494ffba2721b30df52475419ef66`.

| Baseline perplexity | Scalar | Verification 1-8 | Eight-row prefill | Normal-prefill continuations |
| --- | ---: | ---: | ---: | ---: |
| IQ4_XS | 11.731620 | 11.731620 | 12.379730 | 1.2938102451945588 |
| UD-Q4_K_XL | 11.318354 | 11.318354 | 11.757455 | 1.1907444613614468 |

The completed numerical/state checks and release timing controls are recorded
below. Baseline failures remain qualification gaps rather than relaxed limits.

## Stage 1 decisions and validation

### DPP transport

Fresh production and operator builds pass. All 15 model/operator tests pass.
The IQ4_XS and UD-Q4_K_XL teacher-forced captures retain all 696 complete raw
vocabulary rows and exact per-schedule perplexities. The 57-kernel prefill
resource contract passes. Additional metadata inspection of 38 decode/attention
variants finds zero scratch or spills in both builds; several variants use
fewer VGPRs. This resource result is not a speedup measurement.

Artifacts: `dpp-quality/`, `dpp-operators.log`, `dpp-resources.log`,
`dpp-decode-resources.json`, `dpp.patch` and `dpp-binary.json` under the evidence
directory. The candidate is preserved as `baseline-bin/gufo-dpp.exe` beside
the same runtime assets as the baseline.

The complete retained group receives normal-prefill, state/rejection/rollback,
snapshot and matched release checks below. Linux execution remains unrun.

### Branch-free Q4 scales

The final combined helper retains both Q4 scale packings. Fresh production and
operator builds pass, with projection and routed-WMMA checks passing. All 696
matched full-logit rows and the six per-schedule perplexities remain exact.
The separate integer packing check covers 8,072 cases and four sub-block indices;
compiled operator checks provide the actual GPU validation.

Artifacts: `q4-scales-quality/`, `q4-scales-operators.log`,
`q4-scales-comparison.json`, `q4-scale-packing.json` and `q4-scales.patch`.
The preserved candidate is `baseline-bin/gufo-q4-scales.exe`.

### Small-kernel load scheduling

HC, MoE and small projections issue bounded reads ahead of the unchanged sums.
The residual register cache is limited to complete 2560-element rows; other
geometries retain guarded reads. The existing fused HC-down/SiLU route remains
in place. Only the materialization helper is reused from the producer-fusion
commit; producer fusion itself is excluded.

Four affected operator tests pass, including the new ragged F32 fixture and
inputs ending at the allocation boundary for every tested batch width.
Both quantizations retain all 696 complete matched logit rows and perplexities.
The 57-kernel prefill resource contract and 32 inspected changed-family variants
have zero scratch/spills. Artifacts: `loads-quality/`, `loads-operators.log`,
`loads-comparison.json`, `loads-decode-resources.json`, `loads.patch` and the
preserved `baseline-bin/gufo-loads.exe`.

### Aligned F32 vector reads

The new fast path loads complete aligned F32 passes in 16-byte pieces and keeps
the original FMA chain. Ragged and deliberately four-byte-misaligned fixtures
exercise the generic fallback. Fresh builds, the expanded projection test and
the 57-kernel prefill resource contract pass. All 696 matched logit rows and
perplexities remain exact. Artifacts use the `f32-vectors-` prefix and
`f32-vectors-quality/`; the candidate is `baseline-bin/gufo-f32-vectors.exe`.

### Cached norms and MTP row vectorization

Fresh builds, three affected operators and the 57-kernel resource check pass.
All 696 target rows and perplexities match the original baseline exactly.
The added MTP hash diagnostic was also rebuilt with the three original baseline
kernel files, then the candidate sources were restored and SHA-256 verified.
Both quantizations retain all 128 fixed-state stage hashes, including 16 full
vocabulary rows. The mode also passes isolated batching, image-ID handling and
ragged/full catch-up through 2048 rows. Artifacts: `baseline-mtp-identities.json`,
`baseline-mtp-hashes-*.log`, `norm-mtp-hashes-*.log`, their comparison JSON files,
`norm-mtp-quality/` and the preserved baseline/candidate probe executables.

The independent `--mtp-audit` remains separate and unchanged in its limits.
On the original baseline, IQ4_XS fails attention relative RMS 0.0093689 against
0.002; UD-Q4_K_XL passes the complete independent audit. The IQ4 failure also
occurs before the cached-norm change. Exact baseline regression agreement does
not resolve that pre-existing model/oracle gap, which remains unqualified.

### Few-row Q8 read-ahead

The narrow one-input path retains the original scalar Q8 arithmetic and requires
complete twenty-step groups. Fresh production/operator builds and the expanded
projection test pass, including the MTP 4×10240 geometry, every supported batch
width, gated outputs and fallback widths. All 696 matched target logit rows and
perplexities remain exact. Artifacts: `q8-ahead-quality/`,
`q8-ahead-operators.log`, `q8-ahead-comparison.json` and the preserved
`baseline-bin/gufo-q8-ahead.exe`.

### Q8 reduce-scatter

The narrow batched dispatch now distributes the original descending butterfly
among lanes. The scalar, wide/ragged and wave64 expert routes remain unchanged.
Fresh builds and strict projection operators pass for gated/ungated widths 1–8
and wider controls. All 696 complete matched target rows and perplexities remain
exact. The 57-kernel resource guard passes; inspection of 83 relevant decode
variants finds zero scratch/spills (maximum 150 VGPRs). Artifacts:
`scatter-quality/`, `scatter-operators.log`, `scatter-comparison.json`,
`final-resources.log` and `final-decode-resources.json`. The complete release
candidate is preserved as `baseline-bin/gufo-final.exe`.

## Final numerical and state checks

The final kernel commit is `76656a5befc12b2a9171895654c24c4a2524a42a`.
Fresh production and assertion-enabled native/probe/operator builds use the
same pinned toolchain. The preserved production candidate reports
`gufo version 76656a5befc1`, SHA-256
`8a3bb7517c72c7f6a2faa63dc42610d2ec32e0ea6e4b85c2a09f657ff480d9cf`.
The baseline SHA-256 is
`32f84f70f1cafa51597acd7a57560360fc07498ead7e9687825a76b5d5d970d5`.
Runtime DLLs and ROCm assets remain together with the preserved binaries.

- All 15 final Flash-Next CPU/GPU operators pass, including wave64 routed and
  native IQ controls. The 57-kernel resource guard and ten extraction fixtures
  pass; 83 inspected decode variants have zero scratch/spills.
- Both formats retain all 696 teacher-forced full-row hashes and six
  per-schedule perplexities. The 540 normal-prefill rows also match byte for
  byte, with exact mean NLL and perplexity. Total: 1,236 unchanged target rows.
- All 128 fixed MTP stage hashes match the original rebuilt baseline, including
  16 complete vocabulary rows. Isolated batching, image-ID and full/ragged
  catch-up comparisons pass through 2048 rows. UD-Q4_K_XL also passes the
  complete independent scalar MTP audit with unchanged limits.
- Both original and candidate formats pass native session, persistent snapshot,
  rollback and EOS checks. Coverage includes C2/C4/C6/C8 private state/RNG,
  sampled rejection and residual correction, graph replay, 23 serving sampler
  strategies, request budgets, cancellation and cache replay. Each candidate
  passes 105 rollback prefixes/trunk continuations with normal Windows tuning
  and separately with `GUFO_PLATFORM_TUNING=none`.
- With the compatible BF16 projector and two synthetic 256×256 images, IQ4_XS
  passes live/cold/snapshot/partial-image/cancelled continuation and greedy/seeded
  replay. All 24 complete-vector hashes and perplexities match the baseline.

Two additional baseline limitations remain explicit. The original IQ4_XS
independent MTP attention audit fails relative RMS 0.0093689 against 0.002.
The UD-Q4_K_XL two-image fixture fails its MTP stage-2 cached-versus-cold
greedy/seeded decode assertion on both original and candidate builds. All 20
complete-vector hashes and perplexities captured before that assertion match
exactly; the full Q4 image replay suite is unqualified. No assertion or numerical
limit was relaxed. These checks establish consistency with the original fork;
they do not resolve its independent model/encoder quality gaps.

Linux build/CPU/GPU execution remains unrun. Nix and rocprofv3 are unavailable
on this host, so GPU timeline/launch-count profiling is also missing. Resource
metadata and unprofiled wall timings do not replace those checks.

Artifacts: `final-build-identities.json`, `final-comparison.json`,
`final-quality/`, `final-operators.log`, `final-mtp-*.log`,
`final-image-comparison.json`, `baseline-image-*.log`, `final-image-*.log` and
`final-rollback-none-*.log` under ignored `build/pr518/`.

## Release measurements

One complete ABBA sequence per quantization records two baseline and two
candidate runs. This bounded start follows the kernel workflow's repeat-only-
to-resolve-noise rule; it replaces the review plan's proposed four full rounds.
Flagged cells receive focused paired repeats before a retention decision.
Builds, GPU tests and profilers run outside timing phases; model jobs are
serialized. The early IQ4 CLI pilot remains raw evidence, excluded from the
paired results. Its queued continuation was stopped before any HTTP work.

CLI controls use pp2048/tg128 at d0/d32768, greedy seed 518, adaptive MTP max7
and a max1 fixed-width control. Verbose generated-token hashes and draft counts
are retained. HTTP uses context 4096, four sessions, thinking off, greedy seed
518, a 128-output budget and private 2 GiB RAM/disk caches. Each C1/C2/C4 shape
receives a full-request warmup. Per-request phase timings, completed counts,
draft statistics, response hashes and observed SSE content-chunk gaps are kept.
Identical concurrent peers are compared by latency rank, including the worst
peer; their admission order is variable and every raw result remains available.

The functional controls include streaming tool cancellation with cached/cold
replay assertions, a one-output continuation, a two-output tool-history turn and
disk restart. Restart must report a disk hit and positive cached tokens. The
recorded disk tail reuses 1751 tokens, prefills 260 new tokens and emits six
tokens before EOS (budget 16). These are local representative controls, not
copies of upstream's private failed fixtures. Individual losses exceeding both
5% and 3 ms remain review flags; throughput gains cannot hide them.

### Measured decode result

These are median rates from two release runs per build, at each mode/depth.
The actual formats/artifacts/settings above are authoritative. Historical
Linux/reference benchmark tables are unchanged.

| Format / mode | Depth | Baseline tg128 (tok/s) | Candidate | Gain |
| --- | ---: | ---: | ---: | ---: |
| IQ4_XS AR | 0 | 27.35 | 28.03 | +2.5% |
| IQ4_XS AR | 32768 | 25.15 | 26.16 | +4.0% |
| IQ4_XS MTP | 0 | 42.45 | 43.34 | +2.1% |
| IQ4_XS MTP | 32768 | 65.31 | 65.41 | +0.2% |
| UD-Q4_K_XL AR | 0 | 27.89 | 28.87 | +3.5% |
| UD-Q4_K_XL AR | 32768 | 25.52 | 26.87 | +5.3% |
| UD-Q4_K_XL MTP | 0 | 35.95 | 38.44 | +6.9% |
| UD-Q4_K_XL MTP | 32768 | 66.48 | 68.35 | +2.8% |

All CLI generated-token hashes and HTTP response hashes match, with matching
prompt/cache/prefill/completed-token counts. Adaptive draft counts are recorded
separately. The max1 fixed-width controls also retain hashes, with tg gains
1.2–4.6%. CLI prefill control medians vary from -1.3% to +1.3%. The near-zero
IQ4 deep-MTP result is not claimed as a repeatable improvement.

### Supplementary batched and repetitive timing

The primary HTTP series stops at C4. The Q8 reduce-scatter changes batched
dispatch for two through eight inputs, and the review plan also asks for a
repetitive MTP workload, so a second ABBA sequence per quantization adds them.
Each mode uses a fresh 8-session server with the primary context, sampling,
budget and private caches. Cohorts of identical peers run once to warm, then
once for measurement. The mixed case reuses the primary prose/code prompt at
C6/C8. The repetition case copies a fixed 40-function Python module at
C1/C4/C8, with 97–99% MTP draft acceptance. C6/C8 execute as `batched-w6` and
`batched-w8`. Harness: `extra_coverage.py`; results: `extra-comparison.json`.

| Format / mode | Mixed C6 | Mixed C8 | Repetition C1 | Repetition C4 | Repetition C8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ4_XS AR | +3.5% | +3.7% | +3.9% | +3.5% | +4.2% |
| IQ4_XS MTP | +2.9% | +3.8% | +2.4% | +6.9% | +3.3% |
| UD-Q4_K_XL AR | +5.3% | +3.4% | +0.6% | +2.0% | +4.9% |
| UD-Q4_K_XL MTP | +4.1% | +6.7% | +1.1% | +6.4% | +5.9% |

These are median aggregate completed-token rates per cohort. Every response
hash matches. Prompt, cached and completed counts and execution plans also
match. Adaptive accepted-draft totals vary between runs of both builds.
Mean inter-token latency is lower on the candidate in all 20 cohorts.

The gate flags 37 of 452 cells; two exceed it in both adjacent pairs: IQ4 MTP
repetition C4 and Q4 AR mixed C8 best-peer first token. Twenty-eight flags are
first-token cells. Every measured request is a cache hit with zero prefill
tokens, and the server restores peers one at a time, synchronizing the stream
and copying a 172–179 MB snapshot for each, so the worst peer's first token
follows the serialized restore sum. In all 20 cohorts, the candidate's change
in worst-peer first-token time equals its change in summed restore time within
2.0 ms. Restore sums vary in both directions between runs of one build, by up
to 81 ms. The restore copy and admission code is unchanged by this port.
Best-peer first-token cells also absorb client submission skew; for example,
the baseline's last C4 peer arrived 11–14 ms late in both runs. The other nine
flags are maximum inter-token gaps. In Q4 AR repetition C1/C4, one candidate
run (3b) has a 58–67 ms stall, while the other candidate run beats both
baseline runs. In IQ4 MTP repetition C4, all peers share one stall of 192 ms
in both candidate runs, between the baseline runs' 164 and 199 ms. This
attributes the first-token flags to restore/admission variance; it does not
replace GPU timeline profiling.

### Functional timing limits and retention status

**The full functional-timing gate is not cleanly qualified.** The implementation
and Windows numerical/state gates are complete; the group remains in this
isolated review worktree. It is not a blanket recommendation to land every
specialization based on decode throughput alone.

The primary quartets flag 16 of 244 latency/rate cells at the unchanged 5% and
3 ms threshold. Focused ABBA repeats use four distinct histories per server,
first and steady one-output continuations, two-output tool turns, repeated C4
cohorts and streaming controls. All focused outputs/work counts match. Most
initial scalar short-turn losses do not reproduce in their pooled controls,
but 17 of 320 focused cells remain flagged, including per-history cells.
Five primary flags exceed the threshold in both adjacent pairs; two focused
per-history IQ4 MTP inter-token gaps do likewise. The supplementary C6/C8 and
repetition series adds 37 of 452 flagged cells, two in both pairs. These flags
are retained in the artifact and are not averaged away or relabeled as passes.

Stream logs separate server TTFT from client first-content delivery. Several
remaining first-token differences occur in cache restore/admission or client
delivery, with zero new prompt work. Those paths are unchanged by the kernel
port. This identifies where the time occurs; it does not prove every observed
loss is harmless noise. Native GPU timeline/host-cache profiling is unavailable
here, so a clean broad functional-latency claim remains outstanding.

A final IQ4 MTP ablation uses preserved original, DPP-only, pre-Q8 cached-norm
and final binaries on the same four short histories. Final decode phases are
46.3–47.5 ms against original 47.9–48.9 ms. The history-2 total token gap is
127.4 ms original, 139.7 ms DPP-only, 130.4 ms pre-Q8 and 139.0 ms final; the
late Q8 changes are not uniquely isolated as the cause. This bounded ablation
does not establish a successful specialization-level latency gate.

No inference tolerance, assertion, runtime format or public performance switch
was changed to obtain these results. No main-checkout change, GUI deployment,
push or new PR was performed. The preserved baseline remains available for
rollback and further matched checks.

[Portable identities, full row hashes and timing evidence](../models/qwen3.8-flash-next/artifacts/decode-pr518-stages01.json)
retain primary/focused/supplementary comparisons, every review flag and the
ablation's per-request timings, counts and output hashes. Request bodies, raw
responses, exact command argv/environment, logs, full-vector captures and
runtime assets remain in ignored `build/pr518/`.
