# Qwen3.8 27B quality

**Greedy DFlash2 matches isolated AR on all 160 measured mixed/repetitive
requests across Q4/Q8 and C1/2/4/6/8.** Targets are Unsloth UD-Q4_K_XL /
UD-Q8_K_XL; the draft is Q4_K_M. [Identities and dated evidence](artifacts/model-identities.json).
These are execution-consistency checks; original-target and GGUF-conversion
parity remain unqualified.

| Check | Q4_K_XL | Q8_K_XL |
| --- | ---: | ---: |
| AR/DFlash2 mixed completions, eight depths 0–128K | 8/8 match | 8/8 match |
| AR repetitive C2/C4/C6/C8 requests versus C1 | 20/20 match | 20/20 match |
| DFlash2 mixed requests versus isolated AR, C1–C8 | 59/59 match | 59/59 match |
| DFlash2 repetitive requests versus isolated AR, C1–C8 | 21/21 match | 21/21 match |
| Target verification, widths 2–8 and ragged cohorts | Full logits/features match scalar | Full logits/features match scalar |
| Serving and state | Cancellation, continuation, RAM/disk restore and seeded replay pass | Same; also eight 32K continuation forks |

## Quantization comparison

September 24, 2026: short teacher-forced comparison against **Gufo's BF16 target**, with identical
prefixes: three prose/code fixtures, nine logit rows each, full 248,320-token
vocabulary, temperature 1 without filtering. This measures quantization and
native arithmetic together; BF16 here is not an independent upstream oracle.
Results and reproducible identities are in [the compact record](artifacts/quantization-quality.json).

| Target versus BF16 | Mean KL divergence (nats) | Mean total variation | Top-1 agreement |
| --- | ---: | ---: | ---: |
| Q4_K_XL | 0.001650 | 0.01566 | 27/27 |
| Q8_K_XL | 0.00006537 | 0.003128 | 27/27 |

## DFlash2 and sampling

The pinned [original DFlash2 operators](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a)
were checked with independently decoded GGUF weights: worst stage relative
RMSE **5.83e-6**, full-logit maximum error **7.72e-5**, proposal total variation
**1.60e-5** on a 24-token prefix/seven proposals. Limits are 1e-4, 1e-3 and 1e-4.
This qualifies draft execution, not weight conversion.

AR and verification apply penalties → temperature → top-k → top-p → min-p.
Tests cover 216 GPU sampling cases and 23 served strategies, including p/q
acceptance, full-vocabulary residual correction, nonfinite inputs and replay.
**Sampled DFlash2 need not match AR's same-seed sequence**; within-mode seeded
replay requires the same configuration. GPU/CPU distributions have numerical
tolerances, not bit-identical upstream sampler parity.

## Vision

Q4/Q8 AR and DFlash2 pass multi-image, sampled/greedy C4, cancellation,
generated-history continuation and disk-restart checks. Q8 also requires exact
cold/live/restored continuation logits. Encoder controls use official
Transformers operators over the **same converted BF16 GGUF**; they cannot detect
conversion errors. Flash-Next has a separate [vision limit](../qwen3.8-flash-next/QUALITY.md#vision).

## Reproduce

### Windows device upload

September 28, 2026: native Windows/gfx1151 with TheRock 10.0.0, UD-Q4_K_XL,
BF16 projector and Q4_K_M DFlash2. The Windows loader uses bounded uploads
into device memory to avoid the [large host-registration failure](../../WINDOWS.md#qwen27b-weight-memory).

- Device-allocation checks and 1,878,568,960 bytes of embedding, output and
  first/last-layer weights match the original GGUF exactly.
- Three fixed prose/code histories produce **27/27 byte-identical full-logit
  rows** (248,320 logits per row) against a synchronous device-copy control.
  Both have continuation perplexity **3.6879261563836652** on 24 labels.
  This short check isolates upload changes; the original registered-host
  Windows loader cannot supply logits because it hangs. The control is not
  an independent model-accuracy oracle.
- Twelve HTTP smoke cases pass in each of AR and DFlash2: text, streaming,
  sampled output, reasoning override, single/multiple images, image cache
  replay/history, function calls, and Responses image/tool-result input.
- The existing upload test passes byte equality, partial EOF, range bounds
  and failed-read draining. Linux GPU validation was not run on this host.

[Identities, settings and results](artifacts/windows-device-upload.json).
Raw logs and logits remain in the local ignored `build/qwen27b-windows-test/`
directory. This is bounded Q4 qualification, not a context/concurrency sweep
or validation of the other local quantizations.

Capture the same small numerical fixture on two builds (Windows example):

```powershell
.\build\gpu-test\qwen27b_target_test.exe $Model --capture-logits baseline.bin
# Rebuild the candidate with the same toolchain and model, then:
.\build\gpu-test\qwen27b_target_test.exe $Model --capture-logits candidate.bin
Get-FileHash baseline.bin,candidate.bin -Algorithm SHA256
```

The command prints token histories, row hashes and continuation perplexity;
the binary contains all 27 FP32 logit rows in case/row/vocabulary order.
Require identical token histories and full-file hashes for an upload-only
change. The normal target-test invocation additionally checks replay,
verification and concurrency; capture mode does not run that larger suite.

### Model suites

Run only the affected [model suite](../../../tools/qwen27b/check.py):

```sh
nix develop -c python3 tools/qwen27b/check.py fast
nix develop -c python3 tools/qwen27b/check.py reference \
  --model "$MODEL" --reference-model "$BF16_REFERENCE"
nix develop -c python3 tools/qwen27b/check.py serving \
  --model "$MODEL" --dflash-model "$DRAFT"
```

[Draft oracle](../../../tools/qwen27b/dflash_reference.py),
[vision oracle](../../../tools/qwen27b/vision_reference.py) and
[vision serving checks](../../../tools/qwen27b/vision_check.py) retain source
pins and compare operators, full logits, RNG, image/state isolation and replay.
Full logits stay outside Git. A missing-weight skip is not a pass.

## Meaning of exact

Matching Gufo AR/speculative tokens or completion hashes tests consistency.
Matching llama.cpp hashes also does not establish original-model accuracy;
quantized implementations can disagree. Keep the reference, weights and history
fixed before interpreting any numerical difference.

## Benchmark method

September 23, 2026; one warmed sample per point, greedy, thinking off.
Single-user rows use pp2048/tg128 and cached depths 0–128K. Each mode creates
its own eight-token prefix reply; histories can differ across engines.
Concurrency uses the older short-prompt corpus, not matching pp2048 prompts;
its C1 is not directly comparable to single-user d0. AR uses repetitive text;
DFlash2 uses mixed/repetitive cases. Rates sum individual decode rates.
Loading uses cold files, C1/DFlash2/capacity 262144; memory uses C1/AR peak HIP
allocation. Commands, hashes and counts remain in [artifacts](artifacts/bench.json)
and the [benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).
