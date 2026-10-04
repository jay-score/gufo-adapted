# Consolidated tool output parsing

Implemented in the commit introducing this record, against fork base
`c2fd8121424bf10db9867706134fbe48de8e53ed`. The review pinned official main at
`8bdde807e57fadfe57f4a1005707559ae6afc82f` on 2026-10-03.

## Sources and scope

| Upstream source | Adapted behavior |
| --- | --- |
| [#393, c33e050](https://github.com/gufo-org/gufo/commit/c33e050eced6389852617994fe7349367df4c900) | Tool format belongs to the admitted model request. |
| [#396, 2c6a106](https://github.com/gufo-org/gufo/commit/2c6a1064f39a4d3ea0b8d92efea0beedf18150f1) | Unfinished JSON strings retain ownership of quoted tool openers. |
| [#397 draft, d91674a](https://github.com/gufo-org/gufo/commit/d91674a4dd8479a6e6c7044e0b784099ff25b31f) | Bound DeepSeek parsing to an outer envelope and retain suffix prose. |
| [#391 draft, 6718dba](https://github.com/gufo-org/gufo/commit/6718dba3293a0e89696029ccf036bfb4646dd251) | Explicit content mode preserves literal thinking tags. |

The fork uses one incremental `GeneratedTextParser` for both APIs and response
modes. It retains an active envelope and JSON quote/escape state across chunks;
completed envelopes are decoded with the existing argument/type rules. It
replaces the separate streaming filter and whole-output reparse. JSON envelope
decoding uses its unquoted outer boundary once. Backtick examples stay text,
and ordinary prose can resume after each envelope. Reasoning bytes are now
preserved consistently by buffered output and streaming deltas.

The scheduled request owns the admitted model state, whose runner supplies the
native format: Qwen and Flash-Next use Qwen; DeepSeek uses canonical DSML.
Default deferred backends capture format at admission; unknown format retains
legacy wrappers. No grammar engine, schema validator or automatic format
switching was imported. Existing required-choice, typed-parameter, duplicate,
stop, cancellation and malformed-call contracts remain covered.

A complete DSML envelope with unexpected body text or an unfinished invoke
fails at EOS, even if an earlier invoke was valid. An interrupted envelope can
still retain its complete invokes. Bare invokes after the outer closing tag
remain ordinary content.

## Validation

The pre-change adapter probe reproduced the format-selection and JSON recovery
bugs in both APIs, buffered and streaming. Ignored investigation artifacts live
in `build/pr393-396-review/`. Maintained regression fixtures exercise whole
output, every possible two-piece split, byte-at-a-time JSON arguments, UTF-8,
escaped quotes, mismatched/legacy wrappers, empty-envelope examples, code
examples, trailing prose, admitted-format stability and reasoning modes.
Responses fixtures also compare actual deltas to terminal text.
An additional regression rejected a complete, malformed JSON envelope whose
quoted argument contained a valid-looking native call. Before restricting the
Qwen decoder to one frame, this fixture emitted that inner call at a token
limit; it now emits none. Both unfinished and rejected complete JSON envelopes
retain string ownership.

Fresh Windows `openai_chat_test` and `http_server_test` pass. The production
`gufo` build passes with TheRock 10.0.0, AMD clang 23, MSVC 14.51.36231 headers,
Ninja and the existing pinned x64-windows vcpkg dependencies. No dependency
versions changed. The isolated worktree reuses the existing dependency directory
through an ignored junction; the first staging attempt failed before this local
setup was supplied, then the build script passed. Formatting uses CI-compatible
clang-format 21.1.8. Formatting and documentation checks pass.

The baseline executable is `gufo version c2fd8121424b`, SHA-256
`e06c922c4874662c3813caea21f64c7e0f8f6c4335e6ae89a2e3076bc0170e14`.
The candidate is `gufo version c2fd8121424b-dirty`, SHA-256
`d7dbd6f1bc835087d96667d2e0d2562c32e0a90805c70e7cddb5f2a2188bb76e`.
Its source diff and both builds' exact commands, identities and results are
retained in `build/pr393-396-review/`.

Matched Flash-Next UD-IQ4_XS, shared Q8_0 MTP and BF16 projector artifacts have
the hashes listed in the [previous model validation](device-loss-pr390.md#validation).
The recorded sizes and modification times remain identical; those hashes were
reused. The corpus is the same 60-token UTF-8 corpus, SHA-256
`5100c8ab6e0ec059b2dfaf7d0d488f1c157f97b33a910b4a711b36ef6cb904d7`.
Both runs use the same toolchain, libraries, artifacts and environment, with
`GUFO_PLATFORM_TUNING` and `GPU_MAX_HW_QUEUES` unset.

```powershell
& $exe bench --model $model --speculative mtp --mtp-model $mtp --draft-tokens 7 --logit-eval build/pr393-396-review/corpus.txt --logit-schedules '1:1:2,4,8' --logit-out build/pr393-396-review/$tag
python tools/bench/logit-eval.py build/pr393-396-review/baseline-s0.bin.sha256 build/pr393-396-review/candidate-s0.bin.sha256
python tools/bench/logit-eval.py build/pr393-396-review/baseline-s1.bin.sha256 build/pr393-396-review/candidate-s1.bin.sha256
python tools/bench/logit-eval.py build/pr393-396-review/baseline-s2.bin.sha256 build/pr393-396-review/candidate-s2.bin.sha256
```

All 174 full-logit rows match exactly; the three full binary dumps are byte
identical. Each schedule has 58 positions and perplexity 63.369153 on both
builds. Eight matched HTTP cases also pass with identical text, token counts
and finish behavior: Chat cold/retry/advancing history, streaming Chat,
buffered/streaming Responses and concurrent Chat. These use IQ4_XS, Q8_0 MTP,
BF16 projector, context 4096, two sessions, maximum 32 tokens, thinking off,
greedy sampling and seed 31 on a test-owned port 18090. This is regression
evidence, not a performance measurement or full model qualification.

Four additional candidate Flash-Next tool cases pass: buffered/streaming Chat
and Responses each produce exactly one declared `add` call with integer
arguments `{"a":2,"b":3}` under required choice. These use the same IQ4_XS/MTP
artifacts, context 4096, one session, maximum 128 tokens, thinking off, greedy
sampling, seed 31 and test-owned port 18091; no projector is loaded for these
text-only requests. Commands and normalized results are in
`tool-model-results.json` beside the local validation artifacts.

## Remaining work

History replay was subsequently addressed in the
[separate history fixes](history-replay.md): Qwen typed argument rendering,
literal prompt controls and DeepSeek separator handling. Flash-Next
speculative/history accounting investigated in
[#400](https://github.com/gufo-org/gufo/pull/400). No templates, prompt tokens,
samplers, cache formats or speculative accounting changed here.

Native raw string syntax still reserves its parameter closer; it cannot safely
encode every literal framing delimiter. Qwen recovery from nested raw openers
retains the fork's existing ambiguity policy. Unmatched or orphan delimiters
outside recognized envelopes remain text; broad output sanitization was not
imported. Linux execution and real DeepSeek/Qwen27B model checks remain unrun.
