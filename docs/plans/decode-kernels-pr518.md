# Flash-Next decode integration plan: upstream PR #518

Recommend a staged **adaptation** of the useful decode mechanisms, keeping this
fork's Windows HC-down/SiLU fusion, IQ weight support, batched execution and
compact rollback. Reuse upstream kernel implementations where their arithmetic
and dispatch match. Do not cherry-pick the aggregate PR or import its benchmark
tables. The outcome is faster default decode with unchanged numerical and
session behavior, rather than reproducing all fifteen patches.

## Review identity and evidence

- Review date: October 13, 2026.
- Fork: `windows-port`, `ab4d98189a6d8103fce29d2537e4113da3da2420`.
- Official main fetched during review:
  `a33e989cf49e533e3cb591a9f0483c67c590ba50`.
- Shared ancestor: `d9a84f13f35d1f98da22886a12eb25dc7062e392`.
- [PR #518](https://github.com/gufo-org/gufo/pull/518), merged October 10:
  [7e4621c37079ab1bc62c1fd07ca03c37b85831cc](https://github.com/gufo-org/gufo/commit/7e4621c37079ab1bc62c1fd07ca03c37b85831cc).
  Its squash parent is `a41e88da229441f31b49cef5ea1cbfc5f2704dc3`;
  reviewed PR head is `37adfae453666ce94855a08ff61da6020d5c8792`.
- Scope: all fifteen implementation patches, the net source diff, relevant
  fork callers/tests, PR discussion and merged validation evidence. Follow-up
  history in the affected ROCm directory through the fetched tip was checked;
  it contains #536, #549, #460 and #571, with no identified required decode
  correction to #518. This is not a general upstream-sync audit.
- The merge is not an ancestor of the fork. The reviewed mechanisms remain
  absent except for overlapping local optimizations described below.

The author's original ABBA comparisons on two gfx1151 machines report roughly
6-7% single-session decode and 4% aggregate C2/C4 improvement with UD-Q4_K_XL
and shared Q8_0 MTP. These are upstream results, not Windows or IQ4_XS results.
An independent comment reported an AR C1 regression; the author could not
reproduce it consistently and observed capped-clock slow runs on both builds.

The merged [quality record](https://github.com/gufo-org/gufo/blob/7e4621c37079ab1bc62c1fd07ca03c37b85831cc/docs/models/qwen3.8-flash-next/QUALITY.md)
and [review artifact](https://github.com/gufo-org/gufo/blob/7e4621c37079ab1bc62c1fd07ca03c37b85831cc/docs/models/qwen3.8-flash-next/artifacts/decode-kernels-review.json)
provide stronger correctness evidence: fixed-history full logits at 64K/128K,
widths 1-8, partial rollback, operators and session/snapshot checks passed.
However, the functional timing gate **failed** for two request phases:

- AR single-token decode after tool/cancellation history: about 44.63 ms versus
  38.88-38.92 ms in the later comparison.
- MTP disk-restored 289-token continuation prefill: about 557.21 ms versus
  516.68-526.89 ms.

Other concurrency/queue timing flags were inconclusive. A separate fresh-prefix
C1 prefill retry passed, but did not clear those two functional failures.
The original review file's blanket “prefill unchanged” description therefore
needs this qualification. No upstream timing regression is assumed to occur on
Windows; reproduce relevant cases locally before landing.

## Integration boundaries

The net source patch touches thirteen files, all under
`src/models/qwen38_flash_next/kernels/rocm`. A read-only `git apply --check`
against the current fork succeeds for ten file patches; `executor.cpp`,
`kernels.hip.cpp` and `kernels.hpp` require adaptation. Patch cleanliness is
not compile or runtime validation: individual commits also depend on helpers
and signatures introduced by earlier patches.

Important fork differences:

1. `Executor::HcMix` already uses `qfn_mmq_q8_0_hc_down_silu` with
   `Tune().fused_hc_down` on Windows. Keep its deeper HC weight prefetch and
   fused activation. Adding upstream's standalone SiLU pass would surrender
   an existing optimization.
2. `GdnRowReduce` already uses DPP with its own ascending reduction tree.
   Preserve that tree. Add faster transport only where shuffles remain.
3. Above eight routed input rows, Q4 batching uses the separate wave64 kernel.
   The narrow anchor-list optimization must not replace that route. Changes
   to `mmq/common.hpp` and `vecdotq.hpp` also reach the wave64 translation unit;
   audit logical 32-lane reductions, active lanes and generated code there.
4. `DeviceModel::CopyDense` converts supported IQ/K dense tensors to Q8_0.
   Routed experts are copied in their native encoding. Q8 dense changes are
   relevant to IQ4_XS, but Q4_K grouped expert changes do not automatically
   accelerate its native IQ experts. Keep all IQ dispatch cases.
5. The fork's [experiment record](../models/qwen3.8-flash-next/EXPERIMENTS.md)
   reports a related Q5 down grouping/reordering experiment losing on mixed
   routing. This does not reject #518's different method, but makes it
   conditional on local projection-chain and serving measurements.
6. TheRock 10.0.0/clang 23 uses fast math. Upstream's attention patch explicitly
   spells an eight-product FMA permutation inferred from its compiler. That
   permutation must match the current Windows binary, not just the formula.
   Likewise qualify cached norms, small GEMMs and materialization barriers.

No other upstream PR, loader change, weight re-encoding, toolchain update,
snapshot migration or new user option is required. Omit upstream's refreshed
README/charts/tables; keep their evidence linked as upstream evidence.

## Source decisions

Sources below are the original PR commits; retain the squash SHA as the overall
reference. “Take” means reuse the narrow implementation after validation, not
that it has been integrated or tested locally.

| Source / mechanism | Benefit | Risk / Windows fit | Route and validation |
| --- | --- | --- | --- |
| [11827a9](https://github.com/gufo-org/gufo/commit/11827a9a809cab06ab447054b56c122770b3f25b): producer-written Q8_1 | Removes separate quantization launches and rereads | Adaptation required: scratch lifetimes, graph replay, padding, local HC fusion | **Adapt** in two steps: direct GDN/attention outputs, then HC/mixed reuse. Check F32 outputs and all consumed Q8 bytes against standalone quantization; full logits and state replay. |
| [6e52e0f](https://github.com/gufo-org/gufo/commit/6e52e0f8a30fa609b8e2a0a817c2e48b2875a963): small GEMM/HC reads ahead | Hides dependent memory latency | Register pressure and altered compiler contraction | **Adapt** only affected loops; projection and HC operator checks plus cached/replayed execution. |
| [28b3c03](https://github.com/gufo-org/gufo/commit/28b3c03b43f6951a5d13c2031cf1b6ced1ae6b52): DPP/permlanex16 | Removes LDS-crossbar shuffle traffic | Full participating lane groups required; shared helper reaches prefill and wave64 | **Adapt** remaining shuffle sites, preserving each site's tree and architecture fallback. Independent operators and prefill controls. |
| [80a9a8a](https://github.com/gufo-org/gufo/commit/80a9a8ab510908a80cd07f84c81a2432533769e3) + [0ecb6f6](https://github.com/gufo-org/gufo/commit/0ecb6f68598c7b2b23fd3bb3022d2e63aa3091f1): Q4 anchor lists/LDS ballot grouping | Avoids idle blocks and repeated routing reads at narrow verification widths | Scratch size and stable slot ownership; limited benefit for IQ experts | **Take** as one complete narrow-routing improvement, including enlarged group scratch. Keep wide wave64 dispatch. Test shared/disjoint experts, inactive IDs and 2-8 rows. |
| [19cbf32](https://github.com/gufo-org/gufo/commit/19cbf32fabea093c161fc018f9c36a0bfba8db0a) + [773fcfb](https://github.com/gufo-org/gufo/commit/773fcfb3147876c9cc52d8d53954f57c142d5544): MoE/HC epilogue loads | Loads precede serial sums | More registers; must retain slot/partial add order and fallback tails | **Adapt** alongside load scheduling; HC/MoE independent operator formulas and exact baseline comparison. |
| [479dea2](https://github.com/gufo-org/gufo/commit/479dea29254df29f72c2fcda4a6fd3bf8e582ae3): expert-ordered down quantization | Groups slots for better weight reuse without changing row values | Related local mixed-routing loss; Q5_1/Q8_0, K=640 eligibility only | **Defer** until profiling justifies it. Trial separately, preserving original output-slot map; reject if mixed serving loses. No expansion to IQ formats in this change. |
| [ead001b](https://github.com/gufo-org/gufo/commit/ead001b89494a7a75ddf2a753758ed21f1cd97b0): eight K/V reads in flight | Reduces decode-attention load latency, especially at depth | TheRock arithmetic order, register pressure, sparse tails; local boundary fixes | **Adapt** only narrow attention. First capture/replay operator inputs against the old binary; retain current compiler's accumulation order and #470/#485 boundary behavior. |
| [a17f9ea](https://github.com/gufo-org/gufo/commit/a17f9ea148de29a6b2b95afed89a5411c530f799) + [d1a8a8e](https://github.com/gufo-org/gufo/commit/d1a8a8e26311b366a8cfa374af9db59cf1e3be3e): branch-free Q4 scales | Removes divergent packing reads | Reaches both scalar/grouped and wave64 Q4 dots | **Take** final combined helper, not the earlier intermediate packing alone. Exhaust scale packing/sub-block indices and run projection/routed operators. |
| [482fd02](https://github.com/gufo-org/gufo/commit/482fd027d109c160dc97f307202f717c05fc5bc5): F32 vector loads | Faster router/alpha-beta/indexer small projections | Alignment, ragged tails and FMA specialization | **Adapt** aligned whole-pass fast path; preserve general fallback. Depends on small-GEMM load scheduling. |
| [e1b20ba](https://github.com/gufo-org/gufo/commit/e1b20ba9e11540e817116ae196b51696cc80d71a): cached norms/MTP rows | Avoids rereads and vectorizes row copy/add | 10240-wide norm register cost; materialization helper prerequisite | **Take/Adapt** selected kernels; retain exact square/FMA tree and alias behavior. HC/MTP operators, full-logit and prefill checks. |
| [0aa6f75](https://github.com/gufo-org/gufo/commit/0aa6f75af25639504a6aaaa14946d6f47f31f3dd): few-row Q8 loads ahead | Helps draft HC inject's long, narrow projection | Specific dispatch geometry; not a replacement for local HC-down kernel | **Take** restricted few-row path; scalar/gated projection and allocation-boundary checks. |
| [c1f0131](https://github.com/gufo-org/gufo/commit/c1f01311318e6c677fd25c268b51a4e94ca324cc): Q8 reduce-scatter | Fewer reductions for 2-8 dense inputs | Same operands/tree required; keep ragged/multi-wave/wider WMMA routes | **Take** after transport helpers; exact scalar-versus-batch projection checks at every width 1-8. |

The two original documentation commits are **Skip** for import: this fork needs
its own measured results and provenance, not replacement upstream tables.

## Dependency-ordered execution

### 0. Preserve a reproducible baseline

On implementation approval, recheck HEAD and use the T3 worktree workflow for
an isolated branch. Leave the running GUI/engine alone. Build a baseline release
and keep its executable, runtime DLLs, kernel assets and source identity intact.
Build candidate with the same TheRock, MSVC/SDK and vcpkg versions. A build from
the current checkout must not overwrite a binary in use.

Prioritize the exercised IQ4_XS + shared Q8_0 MTP configuration, and retain
UD-Q4_K_XL + that sidecar as the Q4/Q5 kernel qualification configuration.
Record shard/sidecar hashes, compiler versions, `--version`, binary hash,
carve-out/weight placement, power profile, context, sampling, draft limit,
platform tuning and cache history. Numerical comparisons use the same prefill
boundaries. Compare each tuning configuration against its own baseline;
`GUFO_PLATFORM_TUNING=none` can also change prompt checkpointing.

Capture short code/prose full-logit baselines for scalar and every verification
width, and the existing normal-prefill capture. Establish C1 AR/MTP d0/d32K
pp2048/tg128 and HTTP C1/C2/C4 controls. Also preserve short-output/tool-cancel
and disk-restored short-continuation requests with their exact cache work.

### 1. Apply local kernel improvements

Start with DPP transport helpers and the final Q4 scale helper. Then implement
small-GEMM/HC/MoE load scheduling, aligned F32 loads, cached norms/MTP row
vectorization and few-row Q8 loads. Follow with Q8 reduce-scatter.

Keep one focused commit per mechanism or inseparable pair. Build only affected
targets, run the relevant operators and a bounded logit comparison after each
change. A clean patch that needs `Materialize` is not an independent import:
introduce the helper with its first consumer and validate its compiled arithmetic.
Measure suspect register-heavy specializations immediately; reject losing ones.

### 2. Fuse decode activation output

First pass GDN and gated-attention Q8_1 outputs directly to the existing
`Dense(..., Q8Input, ...)` overload. Their single consumer can use an explicit
descriptor, without introducing a general activation cache.

Then adapt HC combine and mix epilogue producers. Reuse the upstream bounded
two-descriptor idea for `xn` and `mixed` if needed by existing generic callers:
source pointer, row count, width and dedicated Q8 destination. No global cache
or new public API. Audit every source write and invalidate before overwrites,
fallbacks, forward/MTP entry and batched entry. Reuse `mixed` for shared/routed
experts only within the matching eligible lifetime. Reset metadata correctly
when graph execution bypasses host-side producer calls.

Preserve local fused HC-down/SiLU on the default Windows path. That path may
continue using standalone quantization of `lo`: the HC GEMV owns one output row
per wave, so emitting a 32-element Q8 block there is not a trivial fusion.
Use upstream's SiLU+Q8 producer on the existing non-fused fallback only if useful.
Do not add a standalone SiLU launch on the fused path just to match upstream.

Require the original Q8_1 padding/layout and quantizer tie behavior. Initialize
padding that is consumed, and compare those bytes too. Keep decode Q8_1 separate
from prefill tiled Q8/F16 caches. Size dedicated storage for at most eight rows
per applicable producer; record the small added scratch allocation rather than
repeating the upstream “memory unchanged” claim.

### 3. Adapt narrow attention

Batch eight existing K/V row reads, preserving the current tile, key order,
softmax sum tree, tail masking and split merge. First reproduce the old kernel's
compiled product/FMA sequence on TheRock with unsaturated and captured inputs.
Do not assume upstream's `7,3,5,1,6,2,4,0` expression order matches Windows.
If that order differs, implement the scheduling outcome using the current order.
Any full-logit mismatch blocks retention; it is not grounds to loosen tolerances.

Check both dense and sparse attention around the indexer budget, split tiles,
empty splits, compression tails and the retained MTP catch-up boundary fixes.

### 4. Add narrow Q4 routing improvements; trial down ordering separately

Import anchor lists plus LDS/ballot grouping as a complete unit, including
scratch sizing and barriers before list-entry reuse. Leave >8-row wave64 and
native IQ expert routes in place. Benchmark shared and disjoint routing.

Only trial expert-order down quantization if this remains a meaningful cost in
the profile. Its trial needs its own commit and A/B controls; remove the trial
if mixed-routing request time regresses. Omitting a losing specialization is a
deliberate adaptation, not an incomplete port.

## Validation and measurement gates

Use the [testing contract](../TESTING.md#matched-token-and-layer-comparisons)
and [model quality checks](../models/qwen3.8-flash-next/QUALITY.md).
Do not duplicate all model suites during each small iteration.

Build relevant `qwen38_flash_next_<name>_test` targets with
`tools/windows/build.ps1 -Preset gpu-test -Target <target> -Jobs 4`, then CTest
with `--output-on-failure --no-tests=error`. Operator names are `projection_ops`,
`hc_mix_ops`, `gdn_ops`, `attention_ops`, `moe_epilogue_ops`, `moe_ids_ops`,
`routed_wmma_ops`, `iq_experts_ops`, `mtp_ops` and `select_ops`; select those
covering the current mechanism. Shared reduction changes need the broader
affected operator set, including prefill controls.

Extend existing operator fixtures only for actual gaps: producer Q8 byte equality,
padding/guards, every width 1-8, unsaturated inputs, aligned/misaligned fallback,
ragged rows and routing ownership. Keep independent CPU/FP64 formula checks as
well as exact baseline comparison. End input tails at the allocation boundary.

For an initial bounded full-logit comparison, run both release binaries with
identical `$Model`, `$Mtp`, `$Corpus`, settings and different `$Out` prefixes:

```powershell
& $Exe bench --model $Model --mtp-model $Mtp --speculative mtp --temperature 0 --seed 518 --logit-eval $Corpus --logit-out $Out --logit-schedules '1:1,2,3,4,5,6,7,8:p8'
python tools/bench/logit-eval.py "$Baseline-s0.bin.sha256" "$Candidate-s0.bin.sha256"
python tools/bench/logit-eval.py "$Baseline-s1.bin.sha256" "$Candidate-s1.bin.sha256"
python tools/bench/logit-eval.py "$Baseline-s2.bin.sha256" "$Candidate-s2.bin.sha256"
```

All full-row hashes, finite outputs and per-schedule perplexity must remain exact.
Reuse `qwen38_flash_next_session_test --capture-prefill-logits OUTPUT` for normal
prefill/chunk controls. Its existing model context is 6145, so it does **not**
establish deep sparse-attention equality. For the attention change, add a small
explicit mode to that existing diagnostic: prefill a fixed 32K/64K code/prose
prefix once, then capture 128 fixed-token continuation logits at widths 1-8 and
partial rollback. Set the corresponding model/session context in that mode.
This avoids dumping every prefix row or importing upstream's private probe.

Build and run the explicit native `qwen38_flash_next_session_test`,
`qwen38_flash_next_snapshot_test`, `qwen38_flash_next_rollback_test` and
`qwen38_flash_next_eos_test` with `--model FIRST.gguf --mtp-model MTP.gguf` for
the final candidate. Include repeated shape switches, graph capture/replay,
all kept prefixes, sampled rejection/residual correction, seeded snapshot replay,
C2/C4/C6/C8 ragged peers and cancellation. Run focused rollback checks under
both normal Windows defaults and `GUFO_PLATFORM_TUNING=none`. Include one BF16
projector image-prefix continuation since these kernels run after image prefill.
Do not interpret teacher-forced equality as a rejection or independent-quality pass.

Profile separately from wall-time measurements. Inspect launch count, GPU time,
VGPR/private scratch and graph execution. Run `kernel_resources_test` after
shared helper/epilogue changes; its bounded prefill resource contract is a guard,
not a resource audit of every new decode kernel.

For final release timing, start with four interleaved ABBA rounds at d0/d32K,
pp2048/tg128, AR and MTP. Compare both quantizations and HTTP C1/C2/C4 with
identical prompt/cache histories. Include mixed prose/code and repetitive MTP;
extend C6/C8 when batching dispatch is touched, and d64K when attention is touched.
No model loading, builds or profilers may run during timing. Record individual
request latency, TTFT, maximum token gap, prefill work and draft statistics,
not only summed decode rates. If adaptive greedy draft widths change because
execution is faster, retain fixed-width controls for arithmetic/acceptance
comparison and report adaptive completed-token rate separately.

The target is a repeatable decode improvement beyond baseline variability on the
fork's actual model/settings; the upstream 6-7% is a hypothesis, not a minimum
or promise. Reject reproducible per-request regressions exceeding both 5% and
3 ms, including the two upstream failed patterns. Use unchanged-baseline repeats
to distinguish noise, and resolve unexplained regressions before claiming a pass.
No new performance switch should be needed for a retained optimization.

Run repository-compatible formatting and documentation checks, plus Linux
build/CPU CI. Shared HIP helper changes also require Linux gfx1151 operator
validation; CPU CI alone cannot establish Linux GPU correctness. If unavailable,
record that gap explicitly and leave cross-platform qualification outstanding.

## Delivery and stopping rules

The first useful delivery is the measured local-kernel group, followed by fused
activation production and attention, then useful Q4-specific improvements.
Estimated effort is several engineering days with the existing toolchain/models;
arithmetic drift or noisy Windows timings can extend validation. There is no
known prerequisite-series import. The main unresolved issues are TheRock
arithmetic parity, scratch/graph lifetime correctness and whether down ordering
is worthwhile on this fork.

For each retained improvement, update root `UPSTREAM.md` in the implementation
commit with immutable source links, adaptation/omissions, areas and actual
validation. Use `Upstream-Commit:` attribution for adaptations. Update model
`EXPERIMENTS.md`, `QUALITY.md` and measured Windows benchmark evidence after
validation; preserve historical measurements and their scopes. No ledger import
entry is created by this review-only plan.

An exactness failure blocks the affected change. A measured performance loss
rejects the specialization. Keep the already validated independent improvements;
do not force all fifteen patches into the final result. Rollback consists of
reverting the affected focused commit and selecting the preserved baseline
build; no persistent format migration is planned.

Review performed: source/diff/caller inspection, immutable upstream fetch,
discussion/artifact inspection and read-only patch applicability checks. No
engine source was changed and no builds, GPU correctness tests or performance
measurements were run as part of this plan.
