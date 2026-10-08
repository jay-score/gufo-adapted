# Qwen3.8 Flash-Next quality

**All 63 measured tg128 requests match fresh AR completions:** 21 AR,
21 mixed MTP and 21 repetitive MTP at C1/2/4/6/8. Unsloth UD-Q4_K_XL target,
shared Q8_0 MTP; [identities](artifacts/model-identities.json).
These are consistency checks, not original unquantized-model or GGUF-conversion
qualification. Latest measurements: September 20–23, 2026.

| Check | Result |
| --- | --- |
| MTP versus scalar CPU formulas, eight text/image states | Fusion/attention relative RMS <0.0008 (limit 0.002); full-width normalization, split projections, recursive carry and full Q8 head checked |
| Batched MTP/AR, C2/C4/C6/C8 | Logits, tokens, acceptance, RNG and every 1–8-token rollback prefix match isolated execution |
| Sampling | 23 AR/MTP configurations; shared FP64 target filtering/CDF, p/q acceptance and residual correction pass |
| Prefill and cache | Full logits match across tested chunk boundaries, short tails and restored state through 4096 tokens |
| Scalar versus bulk prefill, 2176 tokens | Same top-1; logit RMSE 0.18, not bit-identical |
| Serving | Cancellation, three-turn continuation, reasoning/tool history, concurrent image/text isolation and disk restart pass |

Sampled MTP can consume different RNG draws from AR. Seeded replay requires
the same build, request budget, capacity and sampling configuration; live cost
timings only steer greedy decoding. Draft sampling uses the full Q8 head's
top 64 logits; upstream draft-sampler equivalence is not claimed.

## Vision

September 29 Windows image-prefix cache qualification: a text prefix followed
by two 512x512 synthetic screenshots retains byte-identical full logits and
teacher-forced perplexity versus the recorded pre-change baseline, with AR and
MTP. The focused test checks live extension, snapshot restore, greedy/seeded
decoding and rejection of a changed earlier image. This is cache consistency
validation, not additional encoder quality qualification.
[Raw hashes and HTTP measurements](artifacts/image-prefix-cache.json).

```powershell
powershell -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset gpu-test -Target qwen38_flash_next_image_prefix_test -Jobs 4
.\build\gpu-test\qwen38_flash_next_image_prefix_test.exe "$MODEL" "$MTP" "$MMPROJ" screenshot-a.png screenshot-b.png
```

**One Gufo encoder comparison fails:** relative L2 **6.47%** versus official
Transformers BF16, above the **5%** limit, on a 1024×1024 synthetic texture.
Both use the same converted GGUF weights. Against FP32, Gufo and Transformers
BF16 differ by **7.83% / 8.10%**, respectively. This is numerical drift, not an
image-answer score; **llama.cpp was not tested**. Optimizations retain native
embedding bytes but do not resolve this gap. [Evidence](artifacts/vision-parity.json).

## Reproduce

### Windows prefill spill regression

October 7, 2026: TheRock 10.0.0/clang 23, UD-IQ4_XS and shared Q8_0 MTP,
matched baseline/candidate for [PR #459](../../plans/prefill-spills-pr459.md).
Schedules `1:2,4,8:p8` retain **549 identical full raw-logit hashes** and
perplexity **1.343238 / 1.343238 / 1.361810**, respectively. Equality is across
builds within each schedule. The prefill/session test passes chunk boundaries,
graph/snapshot replay and isolation through 4096 tokens. Independent paired
Q4_K/Q5_K routed operators pass and their former 80-byte private cache is gone.
The IQ4_XS model is a control for this paired-cache fix. Additional UD-Q4_K_XL
validation has 47 Q4_K gate/up layer pairs and one Q5_K pair (layer 2),
with eligible Q5_1/Q8_0 down projections. The four-shard target and shared Q8_0
sidecar load successfully on this Windows host. The new explicit session capture
retains **270 byte-identical full-logit rows** in AR/MTP modes after
96/1023/1024/2048/4096-token prefixes of three fixtures and eight continuations.
All 240 labels retain mean NLL **0.17457870930144928** and perplexity
**1.1907444613614468**. The model's snapshot/graph replay, execution-mode isolation
and split-prefill checks through 4096 tokens also pass. This closes the paired
Q4/Q5 full-model execution-consistency gap; Linux and sampled rejection parity
remain unrun. The separate teacher-forcing diagnostic still has its eight-row
limit; the session capture records each normal prefill's frontier instead.
[Exact identities and captured row hashes](artifacts/prefill-spills-pr459-q4.json).

Tests live in [`tests/models/qwen38_flash_next`](../../../tests/models/qwen38_flash_next).
Use `--batch-only`, `--sampling-only` or `--prefill-only` on the session test;
the snapshot test covers persistent image/text state. For independent MTP checks:

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen38_flash_next_model_tests qwen38_flash_next_gpu_probe
nix develop -c build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model "$MODEL" --mtp-model "$MTP" --mtp-audit
```

The [vLLM](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/models/qwen4_exp/amd/mtp.py)
and [SGLang](https://github.com/sgl-project/sglang/blob/993d1fccbaafe3e79d91567d2fc1d665cc94fa50/python/sglang/srt/models/qwen4_exp_mtp.py)
formulas supply independent predictor checks; pinned Transformers ignores MTP
weights. [Vision reproduction](../qwen3.8-27b/QUALITY.md#vision).

For focused Windows rollback validation, build and run the explicit model test:

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset gpu-test -Target qwen38_flash_next_rollback_test
.\build\gpu-test\qwen38_flash_next_rollback_test.exe --model "$MODEL" --mtp-model "$MTP"
```

It checks every kept prefix of verification widths 2–8, repeated shapes,
restoration of the starting snapshot, reset, and subsequent full trunk logits.
Keeping one row means retaining the target anchor and rejecting all drafts.
Run once with normal Windows defaults and once with
`$env:GUFO_PLATFORM_TUNING = "none"`; remove the variable afterward. This uses
teacher forcing without advancing the MTP head. The separate snapshot test
checks real speculative snapshots, including deferred rejection replay. Neither
test establishes quality against an independent model implementation.

The explicit `qwen38_flash_next_eos_test` keeps EOS enabled and checks physical
position, committed tokens, full frontier logits against target replay, sampled
residual correction, snapshots, and serving stops/cancellation/continuation.
The [October 3 Windows audit](../../plans/flash-next-eos.md) records the tested
IQ4_XS/Q8_0 artifacts, cases, results and remaining coverage limits.

## Incremental host checkpoints on Windows

October 6, 2026, IQ4_XS target, Q8_0 MTP and BF16 projector: all 675 matched-token
full-logit hashes match the unchanged fork baseline, with perplexity 21.579505
in each schedule. Incremental and full snapshot exports are byte-identical in
AR/MTP, including direct disk streaming, branch overwrite, reset, cancellation,
eviction and source destruction. C2/C4/C6/C8 sampling/logits/RNG and residual
replay pass with Windows defaults and disabled platform tuning; all 105 explicit
rollback-prefix continuations and the AR/MTP image-prefix checks pass.

HTTP passes 27 growth checks, 15 history-edit checks and two cancellation/resume
cases. All ten matched timing replies and token/draft counts agree. Existing
version-15 disk checkpoints load correctly, and the old executable restores the
final candidate's streamed checkpoints with identical continuation output.
These are fork consistency checks; Linux validation is still required.
[Commands, scope and evidence](../../plans/flash-next-host-snapshots.md).

## Benchmark method

September 22-23, 2026; one warmed sample per point, greedy, thinking off.
Single-user uses pp2048/tg128; MTP pp is the maximum across mixed/repetitive
workloads. C1/2/4/6/8 use the same d0 prompts; every session prefills before
measured tg128, with at most four prompt-tail tokens reevaluated. Rates sum
individual decode rates. Gufo d0/C1 agree within 0.4% with matching drafts/output.
AR reference is llama.cpp b11069; MTP uses pinned `6fcaa16f`.
Loading: cold files, C1/MTP/capacity 262144. Memory: C1/AR/capacity 133121,
peak global HIP allocation including idle memory. Full commands, counts and
identities remain in [artifacts](artifacts/bench.json) and the
[benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).
