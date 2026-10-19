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

## SSM FP16 epilogue correction

October 19, 2026: the synthetic SSM failure exposed compiler-dependent
square/add contraction and regrouping of the four norm partial sums. The direct
FP16 epilogue now explicitly preserves the original FP32 producer's first FMA,
sum order and rounded mean. The exact output/state checks and their tolerances
remain intact; the 31-token reproducer is included in the default test.

On Windows/TheRock 10.0.0/gfx1151, both recurrent-state formats have zero FP16
bit mismatches through the 2,080-token case. All 13 bounded CPU/operator checks
and the Q4_K_XL target's verification, rollback, snapshot, concurrency and
prefill suite pass. Matched short logits remain exact on all 27 rows. Large
prefill intentionally changes 27 of 189 rows relative to the faulty build;
all 189 rows match a temporary reference build using the original FP32
epilogue followed by a standalone FP16 cast, byte for byte. The reference is
an execution-contract oracle, not an independent checkpoint-quality teacher.

[Compact validation and identities](artifacts/qualification-ssm-fp16-20261019.json)
retain the original-build differences, commands and exact reference comparison.
Linux execution and performance are unverified. The Q5 quantization campaign
below retains its original failed verdict; its separate corpus-quality limit
has not been resolved or rerun by this correction.

## Quantization comparison

### Q5 variant workflow pilot

October 19, 2026: native Windows/gfx1151 pilot for Unsloth UD-Q5_K_M against
the same conversion family's BF16 GGUF. The
[declaration](artifacts/variant-qualify.json) pins source revision and artifact
hashes and sets budgets before candidate measurement. These budgets apply to
each capture separately: mean KL <= 0.01 nats, mean total variation <= 0.05,
worst-row total variation <= 0.15 and candidate/reference perplexity <= 1.05.
They are bounded pilot acceptance policy, not universal model limits or
thresholds inferred from the September measurements below.

The three captures contain 27 short-fixture rows/24 labels, 189 normal-prefill
rows/168 labels and 512 corpus rows/512 labels. The corpus is the first 513
tokens of the local WikiText-2 raw-v1 test file; its full-file hash is pinned.
This is sample perplexity, not a full WikiText-2 evaluation. The dataset is
[WikiText](https://huggingface.co/datasets/Salesforce/wikitext), released under
CC BY-SA 3.0/GFDL; its text is not vendored here.

The native probe binds 851 AR tensors with no re-encoding. Fifteen embedded MTP
tensors are explicitly excluded. Three exact thinking-off greedy capability
fixtures check arithmetic, JSON reproduction and a Python expression. Same-Q5
llama.cpp execution uses the complete vocabulary on the 27 short-fixture
prefixes with separately declared tighter limits. Neither reference establishes
original checkpoint or conversion parity. Draft, vision, HTTP API and performance
qualification are outside this pilot.

**Outcome: FAIL.** All eight fast checks, four of five GPU operator checks, the
full native target state/verification suite and all three capability cases on
both models pass. `qwen_ssm_ops_test` fails the existing direct-FP16 epilogue bit
comparison, reproducibly with unchanged inference sources and synthetic inputs;
it does not load the candidate weights. Separately, corpus worst-row TV **0.2790**
exceeds the **0.15** budget. Neither failure is waived.

| Capture versus Gufo BF16 | Complete rows / labels | Mean KL (nats) | Mean TV | Maximum row TV | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short prose/code | 27 / 24 | 0.001337 | 0.01381 | 0.04967 | 0.98355 |
| Normal prefill | 189 / 168 | 0.001944 | 0.006232 | 0.11781 | 0.99901 |
| WikiText-2 sample | 512 / 512 | 0.002041 | 0.01484 | **0.27903** | 1.00324 |

The 512-label sample perplexities are **5.74492** for Q5 and **5.72638** for BF16.
The **0.324%** aggregate increase is within budget, but does not cancel the
worst-row failure at the 319-token prefix (zero-based row 318).

Same-Q5 llama.cpp b11731 (`42b70032b`) passes the 27 short-prefix execution
budgets: mean KL **0.0001717**, mean TV **0.005079**, maximum TV **0.01412**,
27/27 top-1 matches. At the additional worst-corpus-prefix diagnostic, Gufo Q5
versus llama.cpp Q5 has TV **0.02776**, and llama.cpp Q5 versus Gufo BF16 has TV
**0.25400**. Thus the large BF16 comparison drift is not solely a Gufo Q5
discrepancy; this does not establish independent BF16 accuracy or broad
cross-engine parity. The diagnostic is separate from the declared short-prefix
reference gate.

[Compact identities, checks, measurements and reproduction](artifacts/qualification-q5-k-m-20261019.json)
retain the failed verdict. Raw evidence remains under ignored
`artifacts/variant/qwen27b-q5-20261019/`. The initial reference adapter's redundant
FP32 probability-sum check was corrected to stable full-vector FP64 normalization;
only that reference phase was rerun, with both attempts retained. No quality
budgets, model defaults or inference arithmetic were changed to admit Q5, and
no performance measurements were published. A failed pilot is not a general
claim that this quantization is unusable.

### Earlier Q4/Q8 measurements

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

### Windows prefill spill regression

October 7, 2026: TheRock 10.0.0/clang 23, UD-Q4_K_XL, matched baseline/candidate
for [PR #459](../../plans/prefill-spills-pr459.md). Three fixed histories at
8/9/64/95/96/129/2048-token prefixes plus eight scalar continuations produce
**189 byte-identical full FP32 logit rows**. All 168 continuation labels retain
mean NLL **0.61468373729780568** and perplexity **1.8490717146575339**.
The full native target test passes verification, replay, rollback and wide-cache
checks; focused Q8/K/IQ prefill operators cover the changed dispatch boundaries.
The newly supplied full Q8_0 target also produces **189 byte-identical rows**
and unchanged mean NLL **0.6197359249209301** / perplexity
**1.8584372100877764** across the same fixtures.
The full Q8 native target suite also passes verification, replay, rollback,
C1-C8 isolation, wide-cache and 8K prefill consistency checks.
[Q8 identities and row hashes](artifacts/prefill-spills-pr459-q8.json) retain
the matched evidence. Linux validation remains unrun. This establishes execution
consistency for the compiler adaptation, not original-model parity.

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
