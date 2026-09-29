# Two selected upstream changes: Windows validation

Date: 2026-09-29. Integration is limited to projector basename validation and
Qwen tokenizer performance. The remaining changes in
[PR #6](https://github.com/thomas9120/gufo/pull/6) are not included.

## Sources and environment

- Baseline: `a0e887c6702068e0a32b63a0c367e0cf9f650899`.
- Projector change: port commit
  [`0efa345`](https://github.com/pixmaate/gufo/commit/0efa345bd80d47524021bb43fed3407b3074a3b0),
  originating in official upstream
  [`f09c8bb`](https://github.com/gufo-org/gufo/commit/f09c8bbf0211f3af275922615a64c54ade26644e).
- Tokenizer change: port commit
  [`8d23a1b`](https://github.com/pixmaate/gufo/commit/8d23a1b32129b387b8b8a93a3b1afced1efe9670),
  originating in official upstream
  [`e866256`](https://github.com/gufo-org/gufo/commit/e8662563098dd810644c07977a1bf885e068bfb0).
- The production tokenizer patch has the same stable patch ID as the incoming
  commit: `e5cba659da72cc14911f8d88d44b11119a6046db`. Local additions cover merge
  ordering and special-token matching in the existing tokenizer unit test.
- Native Windows 11 x64, Radeon 8060S / gfx1151, TheRock ROCm 10.0.0 and its
  clang, MSVC headers/SDK, `gpu-test` RelWithDebInfo with assertions enabled.
  Baseline and candidate use the same installed vcpkg dependencies and flags.
- Formatting: clang-format 21.1.8, matching the previously validated repository
  formatter. No kernel arithmetic or compiler flags changed.

## Projector compatibility

The baseline and candidate were built from source in this checkout. A real
Qwen3.8-27B BF16 projector and a 256-by-256 red PNG were passed through
`qwen27b_vision_test`, with output width 5120. Two copies of the projector were
made under the ignored validation directory: one with an arbitrary basename
of the same length, and one with `general.basename` renamed to an unrelated
metadata key. The original model files were not changed.

The baseline accepts the original and rejects both altered copies. The
candidate accepts all three, with **all 29 captured FP32 vision tensor files
byte-identical to the original baseline**. It still rejects an F16 projector
and a BF16 projector requested at the wrong output width. Existing discovery
checks continue to reject explicit invalid sidecars and skip incompatible
automatically discovered sidecars.

Projector SHA-256:
`83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53`
(931,146,432 bytes). This qualifies the metadata relaxation for the tested
artifact; it does not establish compatibility with arbitrary projector types.

## Tokenizer equivalence and regressions

An isolated probe was linked against baseline and candidate `gufo_core` builds.
It compares serialized token IDs for 275 texts, each encoded with special-token
parsing disabled and enabled: **550 cases per model vocabulary**. Inputs include
prose, code/tool markup, multilingual and decomposed Unicode, malformed and
adjacent special markers, deterministic mixed strings, letter/space runs up to
32,768 bytes, and a 2,048-turn synthetic conversation. It reads vocabulary and
merge metadata from the real Qwen27B and Flash-Next GGUFs; it does not perform
Flash-Next inference.

For both Qwen27B and Flash-Next, baseline and candidate match exactly:
245,515 token IDs across 550 cases per model, or **1,100 comparisons total**.
Both model vocabularies produce the same serialized result for this corpus.
The serialized result, including per-case lengths, is 984,260 bytes
with SHA-256
`dd0115c4f6d3b6a0f4bd9327d9175c3c68b7774a41e3cc94383ab624ca38d03a`.

Flash-Next tokenizer metadata came from
`Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`, SHA-256
`5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4`
(10,946,624 bytes). Only its tokenizer was exercised in this comparison.

The checked-in unit tests additionally cover leftmost equal-rank merges,
invalidated candidate pairs, newly formed pairs, rank precedence, long-piece
tails, and a ranked pair whose result is absent from the vocabulary. Special
token tests cover repeated delimiters, skipping occurrences consumed inside a
longer match, and resetting search cursors between calls.

For custom vocabularies with overlapping special strings, upstream explicitly
prefers the longest match. This is a deliberate behavior change from the old
unordered-map tie handling, covered with an explicit expected-token fixture.
It is not described as universal equivalence for every custom vocabulary.

Probe source SHA-256:
`3826acaf945af619578c1114f72b5dcb5c2b861d5baddacf59db727a72d83715`.
The source, build commands, executables, logs and serialized tokens are retained
locally in `build/upstream-take-validation/` and `build/gpu-test/`. Diagnostic
timings were collected, but compilation overlapped some runs; no controlled
end-to-end performance claim is made here.

## Matched-token logits and perplexity

Both builds ran the existing `qwen27b_target_test --capture-logits` path on
`Qwen3.8-27B-UD-Q4_K_XL.gguf`, SHA-256
`3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e`
(17,559,178,144 bytes). Context is 128, speculation is off, vision is not loaded,
and sampling is not used: the three fixed prose/code histories in the target
test determine the token sequence. The test-source SHA-256 is
`90e1a42e27dbccc041cb7a9761578d53a8b9f10170a6c95a4d750372775f1fe9`.

Reproduce on each revision with the same model and toolchain:

```powershell
build/gpu-test/qwen27b_target_test.exe "$env:GUFO_QWEN27B_MODEL" --capture-logits build/upstream-take-validation/baseline-logits.bin
# After rebuilding the candidate, write candidate-logits.bin instead.
```

The required limit was exact token-history and full-logit byte equality.
**All 27 rows, each containing 248,320 FP32 logits, match exactly.** The
26,818,560-byte dump has SHA-256
`4a4972a1669fea578b24575b4b7f8a05bdc38b8ce9a0169ac5d82441ecd0da88`.
Both builds have mean NLL **1.305064282821349** and continuation perplexity
**3.6879261563836652** over 24 labels. This is a bounded regression check,
not an independent model-quality evaluation or a full-corpus perplexity result.

Exact capture commands and executable hashes are retained in
`build/upstream-take-validation/baseline-logits-command.json` and
`candidate-logits-command.json`, with corresponding logs and binary outputs.
The target-test executable SHA-256 values are:

- Baseline: `17751246fb3236c166996feaa62184012b8af2392550783da81f26b69c8df5c8`.
- Candidate: `3a2083ddd9f321f9758b219114e466469cdc292730cfe7aa483fca73c2cdb7ff`.

## Focused checks and limits

The complete Windows `gufo` target builds successfully. The shared formatting
check passes for all 483 C++ files, and the documentation check passes.

After rebuilding the affected targets, all six checks pass:

```powershell
ctest --test-dir build/gpu-test --output-on-failure --no-tests=error -R '^(qwen_vision_discovery_test|qwen27b_vision_test|qwen_tokenizer_test|qwen_chat_template_test|openai_chat_test|http_server_test)$'
```

No model skip is counted as a pass. Linux CI and Linux GPU validation have not
been run on this Windows host. Full context/concurrency sweeps and a running
GUI/server deployment are outside this change. The existing local image-prefix,
Responses, loader and GUI implementations are preserved.
