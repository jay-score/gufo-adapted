# Qwen27B cross-engine difference investigation

Status: planned and deferred, October 19, 2026. This document authorizes no
background run; resume when the user chooses to investigate and the GPU is free.

## Goal and existing evidence

Explain the residual same-Q5 distribution difference between Gufo and the pinned
llama.cpp build. Determine whether it comes from valid arithmetic choices,
a Gufo defect, a reference defect or a combination. Preserve native precision
unless evidence justifies a correction; matching an approximate reference is
not itself the goal.

The [retained follow-up](../models/qwen3.8-27b/artifacts/quantization-q5-investigation-20261019.json)
pins model, executable, DLL, source, corpus and request identities. Its
[quality discussion](../models/qwen3.8-27b/QUALITY.md#q5-follow-up-corpus-tails-and-reference-execution)
distinguishes quantization sensitivity from execution uncertainty:

- On five selected outliers, Gufo/llama BF16 TV is 0.24--1.18%; same-Q5 TV
  reaches 9.00%. These rows were selected for diagnostics, not average quality.
- With llama logical batch 512 and physical microbatch 1, the three tested Q5
  histories retain up to 6.87% Gufo/llama TV, with matching top choices.
  Llama's own batched/scalar-like results differ by up to 10.06% TV.
- Gufo's scalar Q5 dot consumes FP32 activations; the pinned llama GPU matvec
  uses Q8_1 activation blocks. This is a candidate contributor, not attribution.
  The synthetic Q5 dequant-dot CPU oracle passes (maximum relative error 2.23e-7).
- `-b 1 -ub 1` failed in the installed reference loader. Preserve that failed
  attempt; use the successful `-b 512 -ub 1` control when reproducing.

Raw baseline: ignored `artifacts/variant/qwen27b-q5-followup-20261019/`.
Reference: llama.cpp b11731 / `42b70032b`; Gufo source at `697d3ac8`
contains the validated SSM correction. Full executable/toolchain hashes in the
record, rather than revision labels alone, identify the measured binaries.

## Scope and prerequisites when resumed

Use the existing local Q5 and BF16 shards, WikiText token histories and Strix
Halo gfx1151/TheRock 10.0.0 toolchain. No new model download is needed. Confirm
artifact hashes, free GPU, source/build identity, available disk for traces and
that all servers are owned by this investigation. Run models sequentially.

Use an isolated checkout/build for instrumentation. Keep the serving executable
and production defaults intact. Retain exact commands, patches, tool/DLL hashes,
environment, input activation bytes and finite full-vocabulary rows. If existing
binaries cannot be reproduced, establish a new matched baseline and explain the
identity change before attributing numerical differences.

Exclude broad quality evaluation, performance tuning, speculative decoding,
vision and API changes. The original pilot failure and version 2 quality-policy
adoption stay separate. Do not widen numerical tolerances to end this task.

## Investigation stages

1. **Reproduce and align execution.** Replay the three successful scalar-like
   controls: window/row 0/318 (319-token prefix), 3/52 (53 tokens), and 4/267
   (268 tokens), using their recorded explicit token IDs. Start with one
   repeat, then repeat only if results drift. Check vocabulary, BOS handling,
   prefix equality, output normalization, reset/cache behavior, batch and
   microbatch schedules, KV type, recurrent-state type, attention mode,
   activation precision and actual dispatched kernels. Keep the five-row
   batched results as a scheduling control. Retain the existing 27 short
   fixtures and BF16 controls. Exit: reproduce the gap or identify the protocol
   difference that explains it; do not proceed on unmatched inputs/settings.
2. **Find the first differing stage.** Add bounded, model-owned trace hooks
   in the isolated builds: embeddings, block inputs/outputs, normalization,
   projections, attention or recurrent update, FFN and final logits. Start at
   block boundaries, then narrow to one operator. Align layouts and compare
   the same token and intermediate input; propagated end-of-model error is
   not a local operator error. Prefer the shortest failing history to reduce
   trace size. BF16 and short-prefix controls distinguish generic precision
   differences from Q5-specific dispatch. Exit: the earliest meaningful
   divergence is localized, with input/output bytes and the arithmetic
   contracts of both paths recorded.
3. **Test one mechanism at a time.** Build a minimal reproducer from that
   operator's actual activation and weight bytes. Independently decode Q5
   blocks and compute a host FP64 dequantized-dot oracle. Compare Gufo FP32
   activations with the same inputs passed through the pinned Q8_1 rounding
   and dot rules; inspect scale handling and reduction order. If the first
   divergence is elsewhere, test the measured mechanism there instead:
   KV/state rounding, prefill route, norm/reduction or recurrence. Establish
   local tolerances from the independent oracle and documented precision
   before judging a changed result. An experimental precision-matched route
   is a diagnostic control, not a production switch. Exit: changing one
   mechanism predicts and measurably changes the local and final difference,
   or the hypothesis is rejected with retained evidence.
4. **Validate the explanation.** Recheck all five outlier histories and the
   27 short prefixes, plus a small predeclared fresh set of ordinary histories
   not selected for large drift. Pair BF16 controls where relevant. Quantify
   the explained portion and remaining local error and full-logit KL/TV/NLL;
   record activation, KV/state and scheduling contracts for each result.
   Do not infer general parity from selected tails or matching top tokens.
   If operator effects interact, measure those interactions before claiming
   a complete explanation. Exit: state exactly what is explained and what
   remains unverified, with bounded coverage.
5. **Decide the production outcome.** A demonstrated Gufo defect gets a small
   fix, real-input regression fixture and workflow #2 qualification against
   the recorded baseline and correct oracle. A reference defect gets a
   retained reproducer and local limitation; external reporting is a separate
   action. Valid precision differences may justify a documented comparison
   contract for future campaigns, with predeclared fresh validation, while
   preserving the original execution results. An unexplained gap remains
   open; no expanded parity pass or tolerance relaxation.

## Evidence and completion

Keep large traces and model bytes in a fresh ignored artifact directory. Retain
a compact model-owned JSON record with commands, hashes, per-history metrics,
oracle results, accepted/rejected hypotheses, remaining questions and next action.
Update this plan and `QUALITY.md` with the supported explanation and coverage.

The investigation is complete when the residual is attributed and checked
against an independent arithmetic oracle, or a concrete unresolved mechanism
and bounded reproducer are recorded for a scoped follow-up. The latter is an
investigation report, not a correctness/parity pass. Any production correction
must complete its own checks, including matched-token regression qualification
and Linux validation status. Do not publish performance from diagnostic builds.
