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

Literal ranges distinguish data from formatter-owned control tokens. The
tokenizer skips special matches overlapping these ranges and keeps ordinary
BPE intact across the ranges' boundaries. This covers the direct template path
and `vision::Prepare`, which also handles plain HTTP requests. Only actual image
attachments create structural image placeholders and expanded image tokens.
Qwen3.8's final formatter identity is `qwen38-reasoning-compiled-v6`; DeepSeek's
typed formatter identity is `deepseek-v4-flash-0731-compiled-v4`.

DeepSeek's formatter adds exactly two newlines before its call block. The
shared output parser holds a maximum two-byte newline suffix and removes it
when the following canonical opener becomes an attempted call. Ordinary
mentions, empty examples, code, suffix prose and other whitespace are retained.
This prevents the same separator from being replayed as content and added again
by the formatter. Buffered and streamed output use the same parser.

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
with perplexity 3.6879261563836652. All rows and token histories match the
candidate exactly. The 26818560-byte binary dumps are identical, SHA-256
`4a4972a1669fea578b24575b4b7f8a05bdc38b8ce9a0169ac5d82441ecd0da88`.
Capture uses the existing `tests/models/qwen27b/target_test.cpp`, linked to
`gufo_core` in the release build, context 128, no speculation or vision and
predetermined teacher-forced histories. This isolates unchanged inference
arithmetic; corrected prompt tokens are established by deterministic fixtures,
not expected to match the old broken prompts.

The production candidate for typed/literal handling reports
`3865833a12a7-dirty`, SHA-256
`827289b3646cda75b3e082547d3aa1214a16e4e1571d40b91871fd553255f955`.
The final separator candidate reports `1f8e70791309-dirty`, SHA-256
`88f163be3a63241af05796e211ee06e8cb6240a117fb1e3855f401a6212772d5`.
Its numerical core is unchanged from the capture candidate. The final
five-target CPU suite and GPU production build pass. API fixtures exercise
every two-piece split and byte-by-byte chunks, including separator-only chunks
and multibyte DSML delimiters. Actual HTTP preparation fixtures cover plain
requests, data in every role, schemas, reasoning, typed/string arguments, real
image expansion, image order, stable checkpoints and unchanged ordinary BPE.

Ordinary typed replay passes all 14 HTTP checks on baseline AR, baseline
DFlash2 and both candidate modes. The first Chat warm replay reuses 413 tokens on the
candidate, including the generated typed call, versus baseline's 353 prompt
tokens; replay prompt lengths are respectively 439 and 433. Retries reuse their
entire prompts and cold Chat controls reuse zero tokens. Canonical template
spacing permits this reuse; arbitrary generated whitespace is not guaranteed
to replay with an identical token prefix.

The isolated HTTP harness uses Qwen3.8-27B UD-Q4_K_XL, matching BF16 projector,
context 8192, one session, thinking off, greedy sampling, seed 47 and up to 192
new tokens. DFlash2 uses the Q4_K_M sidecar and seven draft tokens with the
existing adaptive policy; accepted draft tokens confirm it is active. It checks
actual typed calls, identity-preserving replay, terminal status, continuation
text, complete retry reuse and Chat cold controls for both response modes.
Responses currently lacks the Chat `cache_prompt` override, so its cold
controls are covered through the shared Chat runner. Logs, exact commands and
raw responses are retained under ignored `build/history-replay/`.

Candidate validation passes 70 HTTP checks: ordinary typed replay in AR and
DFlash2, literal history in both modes, and image replay in AR. Forty-two
baseline checks cover ordinary typed replay in both modes plus image replay.
All 42 matched signatures retain exact text, arguments, finish/status and
completion-token counts. Vision checks include user images through both APIs
and Responses image tool results, buffered and streamed. Chat's existing API
accepts images only in user messages; this change preserves that contract.
Literal-history fixtures put vocabulary control spellings in user content,
historical arguments and tool results. They assert an actual generated typed
call first, then replay the constructed history and require the BETA continuation,
successful termination and full-prompt retry reuse. They do not require a model
to generate its own EOS spelling as literal text.

Fresh model hashes match the
[recorded artifact identities](../models/qwen3.8-27b/artifacts/windows-device-upload.json):
UD-Q4_K_XL weights, DFlash2 Q4_K_M and BF16 projector. The same files, toolchain
and runtime settings are used throughout; `GUFO_PLATFORM_TUNING` and
`GPU_MAX_HW_QUEUES` are unset. No GUI configuration or user cache was changed.

```powershell
# Same target source and release toolchain for both captures:
& build/release/qwen27b_replay_logits.exe $model --capture-logits build/history-replay/baseline-logits.bin
& build/release/qwen27b_replay_logits.exe $model --capture-logits build/history-replay/candidate-logits.bin
# Against an isolated server, repeat in AR and DFlash2:
python tests/tools/history_replay_test.py --url http://127.0.0.1:18193/v1 --out build/history-replay/typed-results.json
python tests/tools/history_replay_test.py --url http://127.0.0.1:18193/v1 --literal --out build/history-replay/literal-results.json
# Vision smoke, AR with the matching BF16 projector:
python tests/tools/history_replay_test.py --url http://127.0.0.1:18193/v1 --images --out build/history-replay/image-results.json
```

These are focused regression checks, not performance measurements or full
model qualification. DeepSeek has CPU template/parser coverage only because
no model is available locally. Linux execution remains unrun. Flash-Next
MTP/EOS accounting is covered in the [separate EOS audit](flash-next-eos.md).
Native raw string delimiters still follow the existing
ambiguity policy; no blanket marker removal or output sanitization was added.

An initial baseline request containing literal vocabulary control spellings
produced a corrupt `literal` tool argument. Deterministic tokenizer fixtures
are used to establish the actual token contract independently of model output.
