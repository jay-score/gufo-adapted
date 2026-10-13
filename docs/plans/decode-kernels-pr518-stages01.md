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

Timing controls and implementation validation are in progress. Results below
will distinguish completed checks from remaining qualification.

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

Remaining mechanisms and final checks are in progress. Final checks cover normal
prefill, state/rejection/rollback, snapshots and matched release timings. Linux
execution remains outstanding until recorded.

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
