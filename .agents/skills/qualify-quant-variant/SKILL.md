---
name: qualify-quant-variant
description: Inspect and qualify a new weight quantization of an existing Gufo model, with native loader/routes, matched-token quality and independent execution evidence. Use qualify-inference-change for changed engine builds with identical weights.
metadata:
  origin: gufo
---

# Qualify a quantization variant

Read [AGENTS.md](../../../AGENTS.md), [workflow #3](../../../docs/AGENT_WORKFLOWS.md#3-qualify-a-quantization-variant)
and the model's `QUALITY.md`. For new campaigns read the versioned
[quantization policy](../../../docs/QUANTIZATION_POLICY.md). The bounded version 1
pilot is Qwen3.8-27B text AR, declared
in [variant-qualify.json](../../../docs/models/qwen3.8-27b/artifacts/variant-qualify.json).

## Intake and support

Establish candidate and teacher lineage, immutable hashes, all shards, corpus,
toolchain, GPU and the modes being claimed. Read actual tensor types: a file's
quantization suffix can hide unsupported mixed formats. NVFP4/MXFP8, IQ2_S,
IQ3_XXS and Q2_0 have already exposed gaps in this fork; do not promise support
from model family or a generic IQ label.

Run `python tools/models/qualify-variant.py inspect MODEL --out INVENTORY.json`.
This validates container/shard inventory, not engine execution or quality.
The native `qwen27b_inspect` calls the production loader and GEMM resolver;
the Python tool has no independent list of supported runtime formats.
Unknown layouts/roles/routes require an explicit support decision. New engine
formats belong in a separate implementation, with workflow #2 regression
qualification. Preserve `qualify.py compare`'s identical-weight safeguards.

## Gate

Build `qwen27b_variant_checks` in `gpu-test` using the documented toolchain
and Windows build wrapper or Linux CMake presets. The runner checks tools are
inside the selected checkout's build, retains its CMake/compiler/runtime
identities, and runs CTest with per-test skip and coverage enforcement.

```sh
python tools/models/qualify-variant.py run \
  --candidate Q5.gguf --teacher BF16_FIRST_SHARD.gguf \
  --corpus wiki.test.raw --repo BUILD_CHECKOUT \
  --inspect-tool BUILD_CHECKOUT/build/gpu-test/qwen27b_inspect \
  --target-test BUILD_CHECKOUT/build/gpu-test/qwen27b_target_test \
  --llama-server LLAMA_BIN/llama-server \
  --out artifacts/variant/NEW_RUN
```

Add `.exe` on Windows. The default declaration reproduces version 1; select a
new predeclared version 2 campaign with `--declaration CAMPAIGN.json`. Keep the
original declaration/verdict intact. Use a fresh output directory. Raw logs,
captures and per-stage JSON remain there even when a gate fails. A successful
inspect or build cannot become a quality pass.

The declaration pins this pilot's artifacts and corpus and sets numerical
budgets before candidate measurements. It requires all fast/operator tests,
the native state/verification suite, three exact capability fixtures on both
models, three complete-vocabulary captures, and independent llama.cpp execution
on all short-fixture prefixes. Models load sequentially. The reference server
is a temporary process owned and stopped by the runner; never replace a user's
server or reuse its cache. No projector or draft is loaded in this AR pilot;
embedded MTP tensors are explicitly excluded.

Missing teachers/references, skipped checks, mismatched checkpoint/tokenizer/
histories/settings, nonfinite metrics or exceeded hard budgets prevent `PASS`.
Version 2 records distribution-trigger exceedances as visible `alerts`, with
tail percentiles/counts; a maximum-TV outlier alone is not a quality failure.
Likelihood/functional requirements and separate same-weight execution budgets
remain hard gates. Declare fresh held-out acceptance inputs and scoring before
measurement, separately from calibration and known smoke/outlier cases. Verify
the design and supplemental task evidence; the current runner's three fixed
capability checks do not prove broad held-out quality. Do not relax a budget or
regenerate expected replies after a failure. Attribute the mismatch and report
it; unexplained cross-engine drift is not proof either engine is wrong. Gufo BF16
measures quantization plus native arithmetic; same-GGUF llama.cpp checks
execution, neither proves original-checkpoint parity.

## Retain

Retain JSON/Markdown status, hashes, exact commands, per-mode coverage and gaps.
The 512-label corpus sample and short/repeated-prefill fixtures have separate
metrics; none represents full-dataset perplexity or broad capability quality.
Keep large outputs ignored; update the model's `QUALITY.md` and workflow maturity
with compact evidence. Do not publish performance until all declared quality
gates pass. Permission already granted for bounded work persists; large extra
downloads, engine-format implementation and publication need their own scope.
