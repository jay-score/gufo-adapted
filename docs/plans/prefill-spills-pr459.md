# PR #459: Windows prefill spill review and adaptation plan

Review date: October 7, 2026. Status: implemented and qualified on Windows;
shared Linux and additional full-model coverage are listed below. The source
change is introduced by the same commit as this record.

## Finding and comparison

This fork needs the affected prefill resource fixes. Actual Windows code-object
metadata confirms both register spills in Qwen's blocked quantized GEMMs and
private-array scratch in Flash-Next's paired routed GEMMs. The qualification
below measures the recovered Windows short-prefill throughput against a fresh
baseline, with exact numerical comparisons and unaffected controls.

- Checkout: `t3/review-pr-459-regression-fix`, initially clean.
- Fork HEAD: `dce753fc36aa8a027b4728dfcd36c33e42150c5c`.
- Fresh official main and merged PR:
  [f17e37b8bb7df5fb83ea7ce6d4dc4ef6d6677253](https://github.com/gufo-org/gufo/commit/f17e37b8bb7df5fb83ea7ce6d4dc4ef6d6677253).
- Reviewed PR head:
  [9bccebc5e1d42a702820700736b2601cb1533e90](https://github.com/gufo-org/gufo/commit/9bccebc5e1d42a702820700736b2601cb1533e90).
- PR base: `47b639159315fcdba17e6a144d67e273de9ead6e`.
- Shared merge base: `d9a84f13f35d1f98da22886a12eb25dc7062e392`.
- Scope: the complete net diff of [PR #459](https://github.com/gufo-org/gufo/pull/459),
  its three constituent commits, relevant local kernels, dispatch, build settings
  and tests. This is not a general upstream divergence audit.

The merged fix is not an ancestor of this fork; inspection finds its Qwen fixes
and routed cache selection fix absent. Flash-Next already has a clang-23-gated
scheduling barrier in its separate W8A8 kernel. That existing fix is distinct
from the missing Qwen fix.

## Windows evidence

Read-only inspection used `C:\P2027\gufo\build\release\gufo.exe`, version
`9abda11670ba`, SHA-256
`5292be3f368fc97823932d5cd1879009bf84324bb9446271a08025774aca90ee`.
The affected kernel files, Qwen CMake settings, root CMake/presets and Windows
build script have no changes between that binary's revision and reviewed HEAD.
This is an existing production binary, not a fresh HEAD build.

Installed compiler: TheRock ROCm 10.0.0, AMD clang 23.0.0git,
LLVM revision `8f497e0992fb7513f7f78a6f6b6f1056c375e961`, gfx1151.
The existing production compile database specifies the same compiler, gfx1151,
Qwen wave32/wave64 units, and Flash-Next's `-O3 -ffast-math` settings.

The diagnostic decoded the PE `.hip_fat` section, then reused the pinned PR
head's offload-bundle and AMDGPU metadata readers. It found 67 bundles and
2,737 unique target/name kernel records. Among 57 instantiations of the four
guarded kernel families, 17 use scratch:

| Local kernel / representative tile | Scratch bytes per work-item | VGPR spill count | Implication |
| --- | ---: | ---: | --- |
| Qwen W8A8, 128x128, BK=2 | 480 | 153 | The exact large-prefill spill identified upstream is present |
| Qwen W8A8 fused SwiGLU, 128x128, BK=2 | 480 | 151 | The guard must include the fused epilogue specialization |
| Qwen W8A8, 128x64, BK=4 | 428 | 144 | Short-prompt tile also spills |
| Qwen Q3_K, 128x128, wave32 | 456 | 133 | K-quant vectorization issue is relevant |
| Qwen Q6_K, 128x64, BK=4 | 624 | 197 | K-quant short-prompt pressure is relevant |
| Qwen Q6_K, 128x128, wave64 | 268 | 88 | Both Qwen translation units need coverage |
| Qwen IQ4_XS, 128x64, BK=4 | 52 | 24 | Coverage must include IQ formats |
| Flash-Next paired Q4_K/Q5_K, BN=64/128 | 80 each | 0 | Private cache lives in scratch despite zero reported VGPR spills |

The remaining scratch cases are Qwen Q3_K/Q6_K narrow tiles and IQ4_NL/IQ3_S.
All four inspected Flash-Next W8A8 instantiations have zero scratch. The paired
Flash-Next finding concerns Q4_K/Q5_K routes; it does not establish a speedup
for the user's IQ4_XS expert route.

Full selected metadata is retained locally in ignored
`build/review-pr459/baseline-resources.json`. No server was stopped, rebuilt or
replaced, and no GPU workload was launched.

## Intended outcome and preferred route

Restore spill-free affected prefill kernels under the existing Windows
toolchain, preserve their numerical results and existing dispatch, and make a
compiler-induced recurrence fail a GPU-independent build check. A broad GEMM
rewrite adds unnecessary arithmetic and tuning risk. The small upstream source
constraints are a suitable starting point; adapt the regression guard locally.

| Improvement / source commits | Pros | Cons and Windows risks | Recommendation | Dependencies and validation |
| --- | --- | --- | --- | --- |
| Qwen bounded K-block unroll, W8A8 scheduling barrier, local SLP suppression; [2f223d79](https://github.com/gufo-org/gufo/commit/2f223d79a146a4ebf26a168f39b89dde8a96c050) | Targets confirmed wave32/wave64 and fused Q8 spills without new buffers or dispatch | Source scheduling and SLP changes can affect occupancy, contraction or other kernels in those units | **Adapt:** preserve BK<=2 unroll; do not unroll BK=4; add W8A8 barrier; append `-fno-slp-vectorize` to the two Qwen prefill units | No new dependencies or toolchain. Operator checks, resource inspection, exact logits and matched timing |
| Paired routed cache selection; [2f223d79](https://github.com/gufo-org/gufo/commit/2f223d79a146a4ebf26a168f39b89dde8a96c050) | Removes confirmed private-array scratch while retaining fetched codes | Extra integer mask operations may affect occupancy; Q4/Q5 applicability is narrower than all expert formats | **Adapt:** fixed-slot integer mask blend inside `kPair`, preserving dequantization and accumulation | Existing HIP builtins only; Q4_K/Q5_K paired BN=64/128 controls and full-model checks |
| Resource check; [2f223d79](https://github.com/gufo-org/gufo/commit/2f223d79a146a4ebf26a168f39b89dde8a96c050), [53912b8f](https://github.com/gufo-org/gufo/commit/53912b8ffe45c9ce16fc04555718c85409bc32e1), [9bccebc5](https://github.com/gufo-org/gufo/commit/9bccebc5e1d42a702820700736b2601cb1533e90) | Detects future compiler regressions without a GPU; rejects empty extraction and uses matching bundler | Upstream host reader is ELF-only; Linux allowance names and budgets are not a Windows baseline | **Adapt:** PE/ELF extraction and focused zero-scratch policy for affected families | Standard-library Python reader plus compiler-adjacent bundler; parser, compressed-bundle and policy fixtures |
| Published Linux measurements and broad scratch allowance file | Useful external evidence and diagnostic context | Different host ABI, compiler build, artifacts and fork kernels; unsafe to call these local qualification | **Skip** verbatim import; retain attribution and measure locally | Record actual Windows results; independent Linux CI remains required |

Do not alter Flash-Next's existing gated W8A8 barrier as part of this fix. It is
already effective in the inspected binary. Do not add a fence to the generalized
K-quant kernel merely for symmetry: upstream found that it made a clang-22 Q3_K
tile spill. Preserve current tile layouts, wave modes, shared-memory barriers,
prefetch, output epilogues, allocation, samplers and cache formats.

## Regression guard design

Reuse the useful offload/metadata reader rather than introduce an LLVM parsing
dependency. Add a small PE section reader for `.hip_fat` while retaining ELF
`.hip_fatbin`; distinguish the Windows `.hipFatB` wrapper section from the payload.
Pass the bundler adjacent to `CMAKE_HIP_COMPILER`, resolving `.exe` on Windows.
Retain the corrected one-output-per-target compressed-bundle fallback.

Initially enforce zero scratch and zero VGPR/SGPR spill counts for every
instantiation of the affected Qwen W8A8/K-quant and Flash-Next routed-F16 families.
Require gfx1151 and expected family/variant coverage, including Qwen fused Q8,
wave64 Q6_K and paired routed Q4_K/Q5_K tiles. Missing kernels or empty extraction
must fail. Use architecture plus symbol as the identity; do not let another
architecture overwrite a record. Keep occupancy informational until measured.

This intentionally narrower policy catches the regression without importing
hundreds of unrelated Linux scratch allowances or blessing current broken
Windows scratch. A global all-model budget is a separate, reviewed extension.
Windows uses a different `size_t` ABI encoding in these kernel symbols, so the
Linux mangled-name allowance file cannot simply serve as its baseline.

Parser/policy fixtures must cover valid PE and ELF payloads, compressed bundles,
compiler-adjacent bundler selection, malformed input, zero extracted kernels,
missing required variants, wrong architecture, nonzero scratch with zero spill
count, and spills in a fused specialization.

## Implementation sequence and acceptance

1. Preserve a fresh baseline and build identities on the reviewed fork SHA.
   Implement the PE/ELF resource guard and demonstrate that baseline fails on
   the confirmed kernels. This guard has no runtime performance cost.
2. Apply the Qwen scheduling/unroll constraints and the two-unit SLP setting.
   Build only affected targets during iteration. Check every instantiated
   affected kernel, including fused Q8 and wave64 Q6_K, against zero scratch.
3. Apply the paired routed cache blend. Verify exact selected packed codes for
   all four groups, then zero scratch in Q4_K/Q5_K BN=64/128 specializations.
4. Qualify baseline versus candidate on identical Windows toolchain, artifacts,
   corpus and settings. Retain the source constraints and their focused resource
   contract as one regression fix, qualified against the same baseline. Record
   integrated upstream attribution in `UPSTREAM.md` alongside the implementation.

Build and run the existing assertion-enabled targets:
`qwen_prefill_quant_gemm_ops_test`, `qwen_q4kxl_quant_ops_test`, and
`qwen38_flash_next_routed_wmma_ops_test`. Include the existing fused SwiGLU
whole-buffer comparison. Add missing boundary cases around batches 8/9 and
95/96, partial rows/tokens, BK tails and unsaturated inputs. Include controls for
unaffected kernels in the two SLP-disabled units.

Require exact baseline/candidate full-logit equality and unchanged perplexity
at matched histories; unchanged greedy text alone is insufficient. Exercise
Qwen Q8 and mixed K/IQ weights plus Flash-Next paired Q4_K/Q5_K weights, and use
the user's IQ4_XS/Q8_0 MTP configuration as a control. Cover affected prefill
boundaries and relevant verification widths. Any mismatch must be investigated;
do not relax quality tolerances to accept the optimization.

Measure release pp8/pp16/pp64/pp2048 and decode controls with loading excluded,
warm shapes, identical memory placement and alternating repeated builds. Include
Qwen DFlash2 and Flash-Next MTP controls. Resource success is necessary, but only
matched local measurements can establish a useful speed recovery and rule out
material decode/other-unit regressions. Confirm shared Linux build/tests and
clang-22 controls; upstream reports a small short-prompt cost on that compiler.

Expected scope: a few kernel edits, one source-local CMake setting, a portable
resource-reader adaptation and focused tests. No prerequisite upstream feature
series, compiler upgrade, model migration or persisted-data migration is needed.
Main landing gates are compiled zero-scratch results, exact numerical parity,
measured Windows benefit/control stability and shared Linux validation.

## Initial review validation

- Fresh authenticated GitHub PR metadata/discussion and immutable Git patches.
- Fork/official ancestry, current source, dispatch, CMake and focused-test review.
- Installed TheRock/compiler identity and existing binary revision/hash checks.
- Successful read-only Windows PE/offload metadata extraction described above.
- No candidate build, CPU/GPU correctness suite, matched-logit/perplexity run,
  throughput comparison or Linux runtime validation was performed during review.

## Implementation and Windows qualification

The implementation retains the reviewed constraints in Qwen, including the
fused Q8 specialization, and the fixed-slot integer blend in Flash-Next's paired
cache. Existing Flash-Next W8A8 barriers and fork dispatch stay intact.
`-fno-slp-vectorize` is appended to the two Qwen source files, preserving wave64
and the unrelated small-batch ILP settings. There are no new runtime switches,
buffers, dependency pins, model/cache formats or Windows memory-policy changes.

The adapted resource reader reuses upstream offload/AMDGPU metadata decoding and
compressed-bundle extraction, with PE payload support and a focused zero-scratch
contract. Fifty-seven required template prefixes cover current production
dispatch without host argument ABI suffixes. Every affected family variant is
checked; additional variants must also be spill-free. Architecture/name identities
remain separate, and repeated code objects retain the worst resource counts.
Empty/truncated payloads and missing variants fail. Unlike the upstream general
budget, this does not whitelist existing scratch in unrelated models.

The ten parser/policy fixtures cover PE/ELF section extraction, AMDGPU notes,
uncompressed and zlib/compressed-bundler paths, empty/truncated input, required
variants, Windows/Linux argument ABI suffixes, wrong architecture, duplicate
records, private-array scratch and fused register spills. CTest uses the
compiler-adjacent bundler with the host executable suffix. The CPU-only parser
test is included in the bounded PR selection.

### Builds and resource regression

Native Windows 11/gfx1151; TheRock ROCm 10.0.0 and AMD clang 23.0.0git
`8f497e0992fb7513f7f78a6f6b6f1056c375e961`, MSVC 14.51.36231 headers/SDK,
CMake 4.4.0, Ninja and the existing pinned vcpkg manifest. Dependencies were
copied into this worktree's ignored build directory without package upgrades.
The main checkout's executable and GUI configuration were preserved.

The production baseline was built freshly at the reviewed fork HEAD and retained
as `build/release/gufo-pr459-baseline.exe`, version `dce753fc36aa`, SHA-256
`e117e2f016b78d692606197db3852d700b233b0086cc373ad23b9ea3bf5a410d`.
The production candidate is `build/release/gufo.exe`, version
`dce753fc36aa-dirty`, SHA-256
`e75c7504b971aef4485f19c08c80e8f8225298f94100df86b0d5124164d7593e`.
These identities refer to the tested pre-commit binaries, not a later rebuild.

Resource checks decode all 57 required variants. The fresh baseline fails on
17 variants; the intermediate cache-blend-only binary removes all four paired
scratch cases, leaving 13 Qwen failures. The complete production candidate and
fresh GPU-test executable both pass with zero scratch and zero reported
VGPR/SGPR spills in every checked variant. Occupancy is reported but not used as
a substitute for measured throughput.

Commands from the isolated worktree (`$Rocm`, `$Vcpkg` and `$Ninja` identify the
existing installations, without changing their versions):

```powershell
$BuildArgs = @('-Jobs', '4', '-Rocm', $Rocm, '-Vcpkg', $Vcpkg, '-Ninja', $Ninja)
powershell -NoProfile -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset release -Target gufo @BuildArgs
powershell -NoProfile -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset gpu-test -Target qwen27b_target_test @BuildArgs
powershell -NoProfile -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset gpu-test -Target qwen_prefill_quant_gemm_ops_test @BuildArgs
# In the same imported MSVC/HIP environment, build the other affected targets:
cmake --build build/gpu-test --target qwen_q4kxl_quant_ops_test qwen38_flash_next_routed_wmma_ops_test qwen38_flash_next_session_test gufo --parallel 4
python tools/ci/check-kernel-resources.py build/release/gufo.exe --bundler "$Rocm/lib/llvm/bin/clang-offload-bundler.exe" --report
ctest --test-dir build/gpu-test -j 1 -R '^(kernel_resources_parser_test|kernel_resources_test|qwen_prefill_quant_gemm_ops_test|qwen_q4kxl_quant_ops_test|qwen38_flash_next\.routed_wmma_ops)$' --output-on-failure --no-tests=error --timeout 600
```

The five fresh CTest checks pass in 56.20 seconds. The operator fixtures include
all supported K/IQ formats, wave64, partial row/token tiles, BK stage tails,
8/9 and 95/96 dispatch boundaries, Q8 fused SwiGLU buffer equality, paired
Q4_K/Q5_K cache groups and BN=64/128, and unrelated projection/quantizer controls.
No quality tolerances were relaxed. Native `qwen27b_target_test MODEL` also passes
its full target checks, including verification widths, replay/rollback,
concurrency, wide cache and prefill fingerprints. Flash-Next's session check
passes peer snapshot/graph replay, execution-mode isolation and split-prefill
logit equality through 4096 tokens:

```powershell
./build/gpu-test/qwen27b_target_test.exe $QwenModel
./build/gpu-test/qwen38_flash_next_session_test.exe --model $FlashModel --mtp-model $MtpModel --prefill-only
```

The full Qwen target check was run from a byte-identical copy of its fresh test
executable beside the release DLLs while unrelated GPU-test targets compiled.
An earlier operator-build attempt failed when staging DLLs during a numerical
capture from that directory. The capture finished, then the build was rerun and
all focused checks passed. No user-owned process was stopped.

### Numerical comparison and identities

Qwen's capture diagnostic was extended separately from normal checks to cover
8/9/64/95/96/129/2048-token prefixes of three fixed prose/code token histories,
plus eight scalar continuation tokens per prefix. The identical test source
SHA-256 is `317a463c5b3e3c646a424d01d5f6d5b508c7a8c38d5061ba126b643d7987c86c`.
It was built against baseline and candidate model code in the same GPU-test
preset. Preserved baseline test executable SHA-256:
`c7d71e90bea1266258713d3331b4ae47bc8968cf71f8de54f08e2c6bc9f4db4c`;
candidate: `dd44295910b0290d7ccb17dabb4dad546899bda99d3d6abfc567266e917b9c37`.

All **189 complete FP32 logit rows** (248,320 logits each, 187,729,920 bytes)
are byte-identical. Both raw files have SHA-256
`67b3dbb19e1b7d5b06cef6e9fb2443d4f111fe23079614f8358ba11d2aa73a93`.
On 168 continuation labels, both have mean NLL **0.61468373729780568** and
perplexity **1.8490717146575339**.

Flash-Next uses the same IQ4_XS three-shard checkpoint and Q8_0 MTP sidecar as
the local control, greedy sampling and seed 459. The 897-byte ASCII corpus is
eight repetitions of the following sentence group, including a trailing space
per repetition, followed by LF:

```text
The quick brown fox jumps over the lazy dog. A triangle has three sides. Water freezes at zero degrees Celsius.
```

Corpus SHA-256: `7a552ae620229f69188b6018e693b726976f468a273b867e3e48f0d7a584eb39`.
Schedules `1:2,4,8:p8` produce 183 positions each. All **549 full raw-logit
hashes** match baseline/candidate, and the top-log-probability dump binaries also
remain unchanged. Perplexity is **1.343238 / 1.343238 / 1.361810** for those
schedules in both builds. Equality is across builds within each schedule;
different prefill/decode arithmetic is not asserted to match each other.

```powershell
# Baseline and candidate Qwen test executables, respectively:
& $QwenTest $QwenModel --capture-prefill-logits build/review-pr459/qwen-logits.bin
# Repeat with separate baseline/candidate output prefixes:
& $Exe bench --model $FlashModel --mtp-model $MtpModel --speculative mtp --temperature 0 --seed 459 --logit-eval build/review-pr459/corpus.txt --logit-out build/review-pr459/flash --logit-schedules '1:2,4,8:p8'
python tools/bench/logit-eval.py build/review-pr459/flash-baseline-s0.bin.sha256 build/review-pr459/flash-candidate-s0.bin.sha256
python tools/bench/logit-eval.py build/review-pr459/flash-baseline-s1.bin.sha256 build/review-pr459/flash-candidate-s1.bin.sha256
python tools/bench/logit-eval.py build/review-pr459/flash-baseline-s2.bin.sha256 build/review-pr459/flash-candidate-s2.bin.sha256
```

An initial Flash-Next diagnostic invocation included `p9` and wider schedules;
the baseline rejected `p9` because requested logit rows exceed its existing
capture limit. It was rerun successfully within the limit. Wider prefill is
covered by the operator/session checks above, not by those teacher-forced dumps.
Full-accept teacher forcing does not establish sampled rejection parity.

Model SHA-256 values verified from the actual local files:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Qwen3.8-27B UD-Q4_K_XL | 17559178144 | `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e` |
| Official Qwen27B DFlash2 Q4_K_M | 1143006816 | `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` |
| Flash-Next IQ4_XS shard 1 | 10946624 | `5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4` |
| Flash-Next IQ4_XS shard 2 | 49835229856 | `577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7` |
| Flash-Next IQ4_XS shard 3 | 43836407744 | `d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833` |
| Flash-Next shared Q8_0 MTP | 2786568256 | `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6` |

These are matched-build consistency checks, not new original-model or GGUF
conversion parity claims. Model quality contracts remain in their existing
[Qwen](../models/qwen3.8-27b/QUALITY.md) and
[Flash-Next](../models/qwen3.8-flash-next/QUALITY.md) documents.

### Matched production throughput

The retained release binaries above ran native C1 `gufo bench` at depth zero,
capacity 4096, temperature zero and seed 459. Each mode has two rounds, three
repetitions per process, in baseline/candidate then candidate/baseline order.
Native prompt-shape warmup precedes measurement; model loading is excluded from
the reported rates. The unchanged native deterministic token fixtures are used,
not the HTTP prose benchmarks or a llama.cpp comparison. DFlash2 uses the
official Q4_K_M draft and fixed seven-proposal blocks; Flash-Next MTP uses the
shared Q8_0 sidecar with maximum seven drafts and its existing greedy policy.

```powershell
& $Exe bench --model $QwenModel -p 8,16,64,2048 -n 128 -d 0 -r 3 --temperature 0 --seed 459 --speculative off
& $Exe bench --model $QwenModel -p 8,16,64,2048 -n 128 -d 0 -r 3 --temperature 0 --seed 459 --speculative dflash2 --dflash-model $DraftModel --draft-policy fixed --draft-tokens 7
& $Exe bench --model $FlashModel -p 8,16,64,2048 -n 128 -d 0 -r 3 --temperature 0 --seed 459 --speculative off
& $Exe bench --model $FlashModel -p 8,16,64,2048 -n 128 -d 0 -r 3 --temperature 0 --seed 459 --speculative mtp --mtp-model $MtpModel --draft-tokens 7
```

Qwen UD-Q4_K_XL short prefill recovers consistently in both paired rounds:
AR pp16 **61.89 → 92.27 tok/s (+49.1%)**, pp64 **240.55 → 349.36 (+45.2%)**;
DFlash2 pp16 **59.33 → 86.93 (+46.5%)**, pp64 **225.74 → 321.75 (+42.5%)**.
These are arithmetic means of two process means. AR/DFlash2 pp2048 are
**−0.3% / −0.2%**, and tg128 **+1.7% / +0.2%**. Small control differences
are not promoted as speedups. Full measurements and per-process deviations are
retained with the [Qwen model](../models/qwen3.8-27b/artifacts/prefill-spills-pr459.json).

Flash-Next IQ4_XS AR controls range from −1.1% to +0.3%. Initial MTP pp2048
averages −2.5%, including a candidate run with 34.75 tok/s standard deviation;
tg128 averages −1.8%. Those two MTP controls received a separate five-repetition
confirmation in the same reversed order: pp2048 **967.54 → 971.77 (+0.4%)**,
tg128 **43.10 → 43.00 (−0.2%)**. The initial deficit did not persist; neither
the small gains nor losses establish a control regression or improvement.
Both sets of measurements are retained rather than selecting the fastest run.
The [Flash-Next record](../models/qwen3.8-flash-next/artifacts/prefill-spills-pr459.json)
retains the measured controls. No full-model paired Q4/Q5 or Qwen Q8 speedup is
claimed from IQ4_XS/mixed-Q4 timings or from resource counts alone.

```powershell
& $Exe bench --model $FlashModel -p 2048 -n 128 -d 0 -r 5 --temperature 0 --seed 459 --speculative mtp --mtp-model $MtpModel --draft-tokens 7
```

### Remaining coverage

The available full models were Qwen UD-Q4_K_XL and Flash-Next UD-IQ4_XS; Qwen
Q8 and paired Flash-Next Q4_K/Q5_K full-model artifacts were unavailable. Their
affected kernels have compiled resource and independent operator coverage, not
new full-model numerical/performance qualification. No model sweep or additional
model download was undertaken. The Flash diagnostic's wider row limit and the
absence of sampled-rejection cross-build capture are described above.

Linux/Nix build, Linux GPU execution and clang-22 runtime controls are unrun on
this Windows host. Shared code retains the upstream constraints, portable ELF
fixtures pass, and there are no new platform-only kernel APIs or dependencies;
these inspections do not replace Linux CI before shared-platform landing.
Only the task-owned isolated worktree is modified; no main installation, GUI
settings, remote branch or running user server is replaced.

Final repository checks pass: clang-format 21.1.8 checks all 485 C++ files,
the explicit HIP formatting check passes, documentation validates 76 Markdown
files and 432 local links/anchors, and `git diff --check` is clean. Provenance is
recorded in [UPSTREAM.md](../../UPSTREAM.md). The change is committed locally;
no push or deployment is part of this qualification.
