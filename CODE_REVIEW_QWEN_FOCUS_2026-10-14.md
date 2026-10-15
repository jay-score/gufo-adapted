# Gufo Fork Code Review - Qwen Focus (2026-10-14)

What this change does: rechecks the original review against the current code,
callers, format restrictions, tests and documented numerical contracts. It
corrects unsupported claims and records the smaller follow-ups that remain.

Verified checkout: `C:/P2027/gufo`, branch `windows-port`, commit
`8f09bad1dda6c807ee222f787ebec7b0cd59a624`. Only this report is edited.
The original report was an inspection-only, multi-agent review; this verification
used CodeGraph followed by targeted source reads and freshly built CPU tests.

Scope: verification of the reported findings, with model-specific inspection
limited to Qwen3.8-27B and Qwen4exp / Qwen3.8-Flash-Next. This is not a complete
program audit. Expected use is native Windows/gfx1151 local serving, including
concurrent requests, with Linux compatibility retained. CPU reference code and
production GPU execution must be assessed separately.

## Verdict

**None of the original P0 claims establishes a production blocker.** The
"256x SSM hot path" headline is contradicted by loader restrictions. The MTP
exact-probability check is an intentional numerical invariant. Several proposed
simplifications would change determinism, numerical behavior or I/O policy
without evidence that the replacement preserves the required behavior.

Twelve rebuilt CPU CTest checks and all 35 GUI unit tests pass. These results
qualify the exercised contracts; they do not establish GPU correctness,
independent model parity or performance. Detailed dispositions below replace
the original P0/P1/P2 classifications and line-saving estimates.

## Confirmed follow-ups

### Should fix

1. **The Responses history harness omits an available cold-cache control**
   (`tests/tools/history_replay_test.py:145-150`).

   - **What this is:** The harness compares replayed tool histories through
     Chat Completions and Responses.
   - **Problem:** It says Responses has no `cache_prompt` override and runs
     only warm/retry phases for Responses. `ParseOpenAiResponseChat` already
     accepts that override at `src/cli/serve/openai_chat.cpp:2486-2493`.
   - **Fix:** Correct the comment and include the Responses cold phase with
     `cache_prompt=false`, retaining the existing content and usage checks.
   - **If we skip it:** This harness cannot detect a Responses replay error
     masked by reuse of warm state. This is a test gap, not a reproduced
     generation defect. The real-model harness was not run in this verification.

### Nice to have

2. **Some malformed GGUF errors lose the failing key and reason**
   (`src/core/gguf_reader.cpp:439-440,448-668`).

   - **What this is:** Metadata parsing rejects unknown types, truncated
     values and invalid arrays.
   - **Problem:** For example, a metadata type greater than `kFloat64` returns
     false without setting a key-specific error. With an initially empty
     error string, `OpenSingleFile` / `OpenMemory` replace it with
     `Malformed or truncated GGUF` (`:181-185`, `:342-346`).
   - **Fix:** Add the metadata key and failure reason at the failing branch.
   - **If we skip it:** Rejection remains safe, but diagnosing a broken
     artifact is harder. The original wording overstated this as missing
     outward error reporting; a generic fallback already exists.

3. **Windows physical-core reporting assumes two threads per core**
   (`src/core/diagnostics/linux_sysfs.cpp:96-98`).

   - **What this is:** The native Windows diagnostics fallback gets online
     logical processors, then divides by two.
   - **Problem:** In a 16-core/16-thread configuration with SMT disabled,
     it reports eight physical cores. This follows directly from the code;
     SMT-disabled hardware was not exercised here.
   - **Fix:** Count Windows processor-core topology records, rather than
     inferring physical cores from logical threads.
   - **If we skip it:** Diagnostics can be wrong outside the usual SMT-enabled
     configuration. No inference or throughput failure was established.

## Routing: confirmed, with corrected locations

`InferenceBackend::load` opens GGUF at
`src/cli/serve/inference_backend.cpp:2905`, selects `deepseek4` at `:2925`,
and selects `qwen4exp` at `:2980`. Flash-Next loads through its own
`Model::Load` at `:3011`; the other Qwen route constructs
`hip::QwenGpuModel` at `:3072`. The corresponding runner is
`QwenTextRunner`, not the scalar CPU `ForwardModel`.

`--speculative` and the sidecar options are validated within these routes
(`:2981-3000`, `:3122-3129`); they do not select the model architecture.
A CPU-only HTTP build reports that inference requires HIP (`:3087-3098`).

## Qwen3.8-27B: disposition of every original item

- **Block dequantization: helper behavior confirmed; reported production
  impact rejected.** `QwenTensorRef::Get` really expands 256 values for a
  Q3/4/5/6_K scalar access (`src/models/qwen/state.hpp:78-110`). But the
  quoted SSM reads are convolution, `dt` and `a` parameters. The loader
  accepts only F32 for SSM parameters and normalization weights
  (`weights.cpp:73-77,315-331`). Those reads never enter the packed-block
  branch for a successfully loaded model. CPU matrix projections use
  `TensorGEMV` / `quant::Dot`, and embedding lookup already dequantizes a row
  (`forward.cpp:97-102,128-130`). Generic or synthetic packed scalar access
  could be improved if measured; neither a 256x serving slowdown nor a 256x
  recoverable speedup was demonstrated.

- **DFlash controller and cost arrays: factual observation; P0 rejected.**
  `dflash_policy.hpp:49-120,127-218` selects a width using acceptance,
  context, quantization and cohort costs. Its acceptance estimate already
  uses an exponentially weighted update (`:220-234`). Replacing it with
  "an EWMA" alone does not preserve the batch-cost decision. The
  gfx1151-specific constants merit measured recalibration when the model or
  execution path changes; no failing workload or equivalent shorter
  implementation was supplied. Controller tests exist in
  `tests/models/qwen27b/dflash_test.cpp`; they were not run here.

- **Draft scatter: O(n squared) confirmed; P0 and the atomic-add fix rejected.**
  `hip/kernels/sample.hip:21-44` scans duplicates, deliberately accumulating
  each token's probabilities in their original F32 order. Production
  DFlash selector support is capped at 16 candidates per token by
  `dflash_weights.cpp:247-254`. This is a small bounded list, followed by
  full-vocabulary work (`sample.hip:1283-1293`).
  `TestDuplicateDraftCandidates` at
  `tests/models/qwen27b/sampling_test.hip:446` checks duplicate handling.
  Atomic accumulation can change addition order; sorting also needs to
  preserve per-token order. Keep this as an unmeasured optimization idea,
  and require matched distribution/replay checks before replacing it.

- **SSM/attention heap allocations: confirmed in the CPU reference only.**
  `ssm.cpp:198-209,286` allocates temporary vectors per invocation/head, and
  `forward.cpp:185` allocates attention context. This contradicts an
  unqualified reading of the scratch arena's "zero-heap-allocation decode"
  comment (`state.hpp:209-230`). Clarifying that comment is reasonable;
  moving buffers is optional reference-code work unless profiling establishes
  a material cost. This does not show per-token allocation in the GPU route.

- **CPU double accumulators and parity: observation confirmed; alleged
  bit-equality implication rejected.** `forward.cpp:182-229` uses double
  intermediates. The cited target tests compare GPU scalar, verification,
  batching and replay routes against each other, not CPU doubles against GPU
  FP32. [The quality contract](docs/models/qwen3.8-27b/QUALITY.md) explicitly
  distinguishes execution consistency from original-target/conversion
  qualification, and states sampler tolerances. Independent target parity
  remains a declared limitation; independent draft-operator evidence also
  exists, so "self-consistency only" is too broad for the whole model suite.

- **Tensor aliases and layer schedule: aliases confirmed; silent rejection
  claim rejected.** `weights.cpp:176-244` accepts alternate names. At
  `:250-259` an unexpected layer kind returns a specific error naming the
  block and expected kind. The report provides no artifact with a different
  ordering that belongs to the supported 27B profile. A documented alias
  matrix could help maintenance; relaxing the schedule is not justified by
  the evidence.

- **Verification commit/replay: empty-input no-op is normal; "untested"
  rejected.** `hip/prefill_chunk.cpp:685-687` returns only when there are no
  tokens to commit. Its replay/scalar fallback explicitly preserves decode
  arithmetic (`:688-714`). `hip/ssm_replay.cpp:50-52` states the attention-KV
  overwrite invariant. `CheckMixedContextBatch` in
  `tests/models/qwen27b/target_test.cpp:253-329` restores state, commits replayed
  prefixes and compares subsequent full logits; kernel-level recurrent-state
  tests also exist in `tests/kernels/persistent_deltanet_test.cpp`.
  These GPU tests were inspected, not rerun. No bad commit sequence was found.

- **Module seams: refactoring preference, not a verified defect.**
  `modules/sample.cpp:7-11` is a small CPU wrapper, but normalization and FFN
  also have HIP callers in `hip/decode_step.cpp:80-88,227-235` alongside the
  CPU composition in `forward.cpp`. Thus "no second backend" does not
  describe all the cited seams. No measured indirection overhead was shown.

- **Comments and constants: no numerical defect established.** Qwen3.5
  naming survives in the `qwen35` GGUF/runtime contract; renaming a comment
  may clarify marketing versus architecture names. `ssm.cpp:262-264` uses
  20 as a stable softplus approximation threshold, not an output clamp.
  The `-0.05` fallback is overridden by the required loaded `ssm_a` tensor;
  it matters to partial/synthetic CPU parameter sets, not missing loaded
  production weights.

## Qwen4exp / Flash-Next: disposition of every original item

- **Speculation stack: complexity confirmed; missing-win-threshold claim
  rejected.** `MtpLengthController::BestLength`
  (`mtp_policy.hpp:133-155`) compares expected tokens per relative cycle
  cost against the AR score of one. `MtpBatchController::Choose`
  (`:174-219`) uses measured physical-batch timings and a 1.01 hysteresis
  margin. Its observations update an EWMA and exclude width transitions
  (`:222-251`). Sampled width decisions avoid request timing to preserve
  reproducibility. `CheckLengthController`, `CheckCalibratedCosts` and
  `CheckBatchProfitability` exercise these distinctions in
  `tests/models/qwen38_flash_next/mtp_sampling_test.cpp:22-257` and passed.

  The dated table in `mtp_costs.hpp:11-16` names gfx1151, target/sidecar
  formats, date and a cost-audit reproduction command. The survival table
  has measured provenance in `engine.cpp:791-806`; survival and prompt lookup
  are opt-in (`engine.hpp:62-70`). Their current end-to-end benefit was not
  remeasured. There is no evidence for deleting a controller based on line
  count, nor a claim here that every opt-in policy wins on every workload.

- **Integer proposal masses and exact-equality throw: correctness finding
  rejected; preserve the invariant.** `mtp_sampling.hpp:32-68` assigns
  integer masses totaling 2^24 and exports each mass divided by 2^24 as
  F32. Those fractions are exactly representable. For generated unique
  candidate lists, their FP64 sum is exactly one, and the selected mass
  equals the mass used for the draw. `:97-98` therefore checks an exact
  internal representation, rather than comparing approximate softmax sums.
  Verification already uses standard p/q acceptance and the positive
  target-minus-proposal residual (`:99-106`).

  `CheckCompactProposals` checks exact mass, replay and target-mass
  conservation across sampling configurations. `CheckSampledOutputFrequencies`
  checks 100,000 draws at each of two temperatures, and
  `CheckVerificationBoundaries` checks filtering/RNG boundaries. All passed
  in the rebuilt CPU test. Direct sampling can only replace this scheme if
  the exported proposal remains the distribution actually sampled; removing
  the check alone is not a demonstrated fix.

- **Snapshots, pending decode and graph keys: structural observation;
  fragility/collapse claims unproven.** `engine.hpp:238-254` explicitly
  versions both payload and inference arithmetic. Invalidating cached state
  when arithmetic changes is part of the compatibility contract.
  `PendingDecode` (`engine.cpp:740-758`) carries distinct sampled, greedy,
  GPU and prompt-lookup state. No invalid flag combination was demonstrated.
  The graph key at `kernels/rocm/executor.cpp:2108-2125` is a 64-bit packing
  of token/logit counts and flags, including a prefix flag; calling it a
  "6-bit key" is misleading. No collision was established.

  `tests/models/qwen38_flash_next/session_test.cpp` contains graph-capture,
  cancellation, rollback-reuse and seeded-replay checks; `snapshot_test.cpp`
  covers incremental/persistent snapshots. Their presence contradicts an
  implied absence of safeguards. They were not executed here, and reducing
  this machinery to roughly 50 lines has not been shown to preserve it.

- **Kernel variants: count is not evidence of accidental duplication.**
  Scalar/vector and F32/F16 variants have different storage/alignment and
  execution requirements. For the named examples, actual comparison tests
  exist in `hc_mix_ops_test.cpp`, `moe_epilogue_ops_test.cpp`,
  `projection_ops_test.cpp` and `routed_wmma_ops_test.cpp` under
  `tests/models/qwen38_flash_next/`. These are evidence of variant correctness
  coverage, not an exhaustive proof or an automatic performance gate for
  every kernel. GPU execution/performance was not rerun; no unused or
  behavior-equivalent variant was identified for deletion.

- **Reference "not F32": rejected.** `reference.hpp:23-28` declares both
  `Storage::kFloat32` and `Storage::kDecode`, with Float32 as the default.
  `reference.cpp:97-105` explicitly bypasses decode-storage emulation in
  Float32 mode. Optional `stage_inputs` are documented isolation controls,
  not proof that all reference calculations consume GPU outputs. The
  [quality document](docs/models/qwen3.8-flash-next/QUALITY.md) states the
  scope and numerical limits of measured operator/consistency checks, while
  leaving original unquantized-target/conversion qualification open. A
  complete F32-versus-decode target qualification was not performed here.

- **Configuration validation: intentional layers, with possible small
  redundancy.** `config.cpp:227-245` checks structural consistency;
  `:246-265` restricts shapes to specialized kernel geometry. Rejecting
  other profiles before launch is deliberate. Later checks can be redundant
  under the fixed profile (for example the 48-layer/interval-four check),
  but that does not make the whole structural validation path dead or unsafe.
  The rebuilt config tests, including malformed metadata, passed.

- **N-gram reader: proposed mmap replacement is not established as equivalent.**
  `ngram.hpp:38-40` and `ngram.cpp:98-128` explain the bounded compressed-row
  cache, Linux page-cache avoidance and the measured Windows cached-overlapped
  exception. [Windows I/O notes](docs/WINDOWS.md#how-the-port-works) also record the
  32-versus-128-reader measurements and prefill impact. mmap plus prefetch
  would change the memory/I/O policy and needs its own matched measurement.
  `/proc/self/fd/N` is emulated with `ReOpenFile` on Windows; it is not a
  missing literal `/proc` filesystem dependency.

  The native Windows `qwen38_flash_next.ngram` test passed. Its fixtures cover
  unaligned BF16 and IQ4_NL rows, duplicates, async gathers, large batches,
  bounds, failed reads and recovery. "No ngram-win tests" is incorrect.

- **Vision reuse: confirmed, with a stated geometry contract and an existing
  quality limit.** The shared encoder and `DeviceRope` are reused. The
  `11/11/10/0` sections are explicitly enforced in `config.cpp:253` and
  invalid section arrays are rejected by `config_test.cpp:179-189` (passed).
  `qwen/vision/rope.hpp:13-27` documents and implements the shared mapping.
  This establishes the checked geometry, not numerical encoder parity.
  The quality document already reports one encoder comparison at **6.47%
  relative L2 versus a 5% limit**. That pre-existing, documented limitation
  was absent from the original positive note; it was not rerun here.

## Shared engine: split the bundled claims

- **Model dependencies in core: confirmed architecture coupling.**
  `src/core/speculative/speculative_verifier.hpp:14-20` includes Qwen policy,
  generator, tokenizer and HIP executor types; `draft_backend.hpp:14` includes
  the Qwen tokenizer. Moving these dependencies is a possible maintenance
  project. `ToolFormat` is in the serving interface and selects model-specific
  output parsing. Neither observation demonstrates a broken request.

- **Runner interface and snapshot state machine: no concrete failure found.**
  Virtual-method/flag counts and file length do not establish a defect.
  `text_model_runner.cpp:571-605,628-633` already explains warm-frontier,
  immutable-checkpoint, image-context and async-worker lifetime rules.
  Scheduler/cache tests exercise the connected path. This verification did
  not establish that collapsing its states preserves those rules.

- **GGUF diagnostics: confirmed only at the narrower severity of follow-up 2.**
  Invalid data is rejected, and public open paths provide a generic error.

- **Multiple sampling paths and thresholds: unmeasured optimization choices.**
  Fast paths and full-distribution fallbacks are tested in
  `tests/cli/logit_sampler_test.cpp`, including filtered/full-sort and
  deferred-residual replay comparisons. That rebuilt test passed. The
  constants `64`, `256` and `1024` are not a correctness finding without an
  input that changes the required distribution; their optimality was not
  benchmarked here.

- **"No scheduler fairness bounds": contradicted for work-unit scheduling.**
  `text_generation_scheduler.cpp:1166-1195` limits initial batch assembly and
  marks decoders due after a prefill turn. `:1198-1247` rotates work queues.
  `TestDecodeActivePrefillIsBounded`,
  `TestPrefillYieldsToEveryDueDecoder` and
  `TestPendingClientsAreRoundRobinAndIndividuallyBounded` explicitly test the
  policy in `tests/cli/text_generation_scheduler_test.cpp`. The rebuilt
  scheduler test passed. A wall-clock latency guarantee under arbitrary GPU
  execution times is a separate, unmeasured property.

- **Host snapshot budget: fallback is functional on Windows.**
  `text_model_runner.cpp:21-60` obtains available pages through `sysconf`
  before trying Linux-specific refinements. The Windows shim at
  `compat/win32/posix.cpp:913-930` supplies physical-memory availability;
  failed `/proc` reads simply leave that fallback intact. Half the available
  memory is a conservative automatic budget, with an explicit override
  (`text_model_runner.cpp:365-382`). No bad budget case was demonstrated.

## API / server / GUI

- **Duplicated LLM option registration: confirmed maintenance opportunity.**
  `src/cli/serve/serve.cpp` contains corresponding registrations in explicit
  `llm` and default-LLM paths (`:484+`, `:960+`). Shared registration could
  prevent future drift. No currently mismatched option was identified;
  neither a P1 bug nor the original deletion estimate is established.

- **OpenAI module size / helper extraction: preference without a failing case.**
  `ResponsesOutput` and type-hint logic could be reviewed as separate
  components, but size alone is not evidence that extraction is necessary.
  At `openai_chat.cpp:559-566`, `thinking/chat/auto` are accepted modes and
  `on/off/none` are additional accepted aliases. The error names canonical
  accepted values; it does not reject those values or invert their meaning.
  Listing aliases would be a wording choice, not a demonstrated API defect.

- **History replay comment/control: confirmed; see follow-up 1.**

- **Vision-discovery assertions disabled: rejected.**
  `tests/cli/http_server_test.cpp:243-270` uses `assert`, but
  `CMakeLists.txt:349-351` applies `-UNDEBUG` for test targets. The freshly
  generated compile command contains `-DNDEBUG` followed by `-UNDEBUG`.
  `http_server_test` passed after rebuilding; this is not a vacuous release
  assertion test in the supported build.

- **GUI sampler defaults: deliberate, documented behavior.**
  `tools/gui/settings.py:22,34-37` and `command.py:9-12,35-36` set/forward
  editable defaults. [The GUI contract](tools/gui/README.md) explicitly
  documents the Flash-Next thinking sampler and per-request API overrides.
  Making the GUI greedy is a product change, not a verified correction.

- **GUI reasoning menu: no missing Qwen capability.**
  `models.py:3` exposes Auto/Low/Medium/Xhigh. The actual formatter maps
  minimal to low and high/max to xhigh
  (`src/models/qwen/chat_template.cpp:30-49`). The shorter native menu is
  intentional; broader CLI vocabulary does not imply distinct missing
  Qwen behaviors. Generic-family presentation could be a separate UX
  discussion. All 35 existing GUI unit tests passed.

## Windows fork

- **Conditional portability work: confirmed structure, limited conclusion.**
  Windows-specific CMake, socket, loader, I/O and diagnostic branches keep
  their Linux alternatives. This inspection and native CPU build are not
  proof that Linux still builds or runs correctly. Linux validation was not
  performed here.

- **Crash reporter and socket drain: rationale exists; no failure shown.**
  `compat/win32/posix.cpp:71-168` explains Windows exception reporting and
  MSVC exception metadata. `src/core/platform/socket.hpp:72-99` explains why
  unread-input close behavior needs a bounded Windows drain. Neither code
  size nor the 50-round/10-ms limits establish an error. Dedicated malformed
  exception-record and socket-drain boundary tests remain unverified;
  the general HTTP test pass is not exhaustive coverage of those cases.

- **128 n-gram readers: measured rationale already documented.** See the
  n-gram disposition above. Current hardware/toolchain optimality was not
  remeasured; do not remove this on an assumed equivalence with Linux I/O.

- **Platform tuning switches: tested parsing, separate platform qualification.**
  `tests/core/platform_tuning_test.cpp` tests defaults, ordered overrides,
  unknown names and all-switch selection. `tuning.hpp:4-18` states why defaults
  differ by platform. These facts correct a blanket "untested matrix"
  interpretation. They do not qualify every flag combination or all-on
  Linux execution. No promotion/deletion criterion was evaluated here.

- **Physical core count: conditional diagnostic bug; see follow-up 3.**

- **Compatibility coverage gap: retain narrowly.** No dedicated exhaustive
  open/mmap/flock/fmemopen/spawn or `DeviceMemoryInfo` unit suite was identified.
  There is indirect native coverage from GGUF, n-gram, HTTP and scheduler
  tests, so "no tests" is too strong. The mapped-file shrink case at
  `tests/core/gguf_reader_test.cpp:657-660` is explicitly Linux-only because
  Windows refuses that operation while the view is mapped; the skip is not
  evidence that Windows silently accepts the same operation.

## Validation performed

Native Windows x64, TheRock ROCm **10.0.0**, AMD clang **23.0.0git**
(`8f497e0992fb7513f7f78a6f6b6f1056c375e961`), `cpu-test` / RelWithDebInfo,
`ENGINE_ENABLE_HIP=OFF`. The configured source revision was `8f09bad1dda6`.
All named targets were built before their CTest invocation.

Passed 12 CTest checks:

- `openai_chat_test`, `http_server_test`, `logit_sampler_test`.
- `text_generation_scheduler_test`, `gguf_reader_test`.
- `qwen_forward_test`, `qwen_ssm_test`, `qwen_gemm_route_test`.
- `qwen38_flash_next.config`, `.ngram`, `.mtp_sampling`, `.prompt_lookup`.

Also passed all **35 GUI unit tests** via the existing GUI environment.
For reproduction, the separate build/test invocations used during this pass
are consolidated below:

```powershell
.\tools\windows\build.ps1 -Preset cpu-test -Target qwen38_flash_next_mtp_sampling_test -Jobs 4
cmake --build build/cpu-test --target qwen_forward_test qwen_ssm_test qwen_gemm_route_test qwen38_flash_next_config_test qwen38_flash_next_ngram_test qwen38_flash_next_prompt_lookup_test openai_chat_test http_server_test logit_sampler_test text_generation_scheduler_test gguf_reader_test --parallel 4
ctest --test-dir build/cpu-test -R '^(qwen_forward_test|qwen_ssm_test|qwen_gemm_route_test|qwen38_flash_next\.(config|ngram|mtp_sampling|prompt_lookup)|openai_chat_test|http_server_test|logit_sampler_test|text_generation_scheduler_test|gguf_reader_test)$' --output-on-failure --no-tests=error
.\build\gui-env\Scripts\python.exe -m unittest discover -s tests/tools -p 'gui_*_test.py'
python tools/ci/check-docs.py
```

Documentation check: `python tools/ci/check-docs.py` passed after correcting
the Windows documentation anchor.

## Not checked / next actions

No engine implementation was changed. GPU operator/model suites, real-model
history replay, matched-token full-logit/perplexity runs, current cost-table
benchmarks and Linux builds were not run. Existing quality/benchmark records
were inspected as evidence of declared contracts, not treated as freshly
reproduced results. SMT-disabled diagnostics were derived from source, not
reproduced by changing hardware settings.

The original deferred areas remain deferred: comprehensive continuation-disk
checksum/atomic/LRU audit, all inference load/error paths, the full speculative
verifier, HTTP/WebSocket/Anthropic transport bodies, all GPU kernel arithmetic,
Windows mmap/overlapped-I/O edge semantics, dependency-download pin audits and
complete GUI/browser behavior. The documented encoder-parity limit also remains
open. No full-program correctness or performance certification is implied.

Prioritize the missing Responses cold control. Diagnostic improvements can be
small independent changes. Any CPU/GPU optimization or controller replacement
needs an observed workload, a matched baseline, the applicable numerical/replay
checks and measured end-to-end benefit; the original speculative line-saving
and speedup estimates are withdrawn.
