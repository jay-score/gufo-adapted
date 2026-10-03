# History replay fixes

The implementation is split into typed argument formatting, literal control
token handling, and DeepSeek separator framing. Flash-Next speculative EOS and
checkpoint accounting remain a separate investigation; no speculative policy
or arithmetic changes are included.

Typed historical values use the reference template's JSON separators (`", "`
and `": "`). Formatting happens in the model renderer, keeping wire JSON
compact and covering internal callers. Object order, UTF-8, escaping and string
argument bytes are preserved. Persistent formatter identities change whenever
the rendered prompt or its tokenization changes; existing disk entries remain
in their previous compatibility partition.

## Validation

Windows uses TheRock 10.0.0/AMD clang 23, MSVC 14.51.36231 headers, Ninja and
the existing pinned vcpkg dependencies. Formatting uses clang-format 21.1.8.
Fresh JSON, Qwen/DeepSeek template, API and HTTP CPU targets pass for typed
formatting. Linux execution and real DeepSeek model validation are unavailable
on this host.

The pre-edit source is `e4072b410552e8e2fc905118de6efd26ce381477`. The baseline
engine reports `c2fd8121424b-dirty`, SHA-256
`d7dbd6f1bc835087d96667d2e0d2562c32e0a90805c70e7cddb5f2a2188bb76e`;
its source matches the preceding parser commit. The Qwen27B target capture
contains 27 full-logit rows, 248320 logits per row and 24 teacher-forced labels,
with perplexity 3.6879261563836652. Candidate comparison is pending.

The isolated HTTP harness uses Qwen3.8-27B UD-Q4_K_XL, matching BF16 projector,
context 8192, one session, thinking off, greedy sampling and seed 47. It checks
actual typed calls, identity-preserving replay, terminal status, continuation
text, complete retry reuse and Chat cold controls for both response modes.
Responses currently lacks the Chat `cache_prompt` override, so its cold
controls are covered through the shared Chat runner. Logs, exact commands and
raw responses are retained under ignored `build/history-replay/`.

An initial baseline request containing literal vocabulary control spellings
produced a corrupt `literal` tool argument. Deterministic tokenizer fixtures
are used to establish the actual token contract independently of model output.
