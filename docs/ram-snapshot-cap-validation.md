# RAM snapshot cap validation

October 8, 2026. Implementation base: `b589fd002238e7cb0388e21103bd8d95158f0405`.
Windows 11 x64 / Strix Halo gfx1151, TheRock 10.0.0 / clang 23, MSVC
14.51.36231 headers, pinned vcpkg dependencies, RelWithDebInfo. This change only
limits existing snapshot admission; it changes no model arithmetic or formats.

## Builds and contracts

The isolated production `gufo` target and focused CPU targets were freshly
built with `tools/windows/build.ps1` (four jobs, installed dependency paths
passed explicitly). Additional CPU targets used the same imported MSVC
environment and CMake cache.

- Six CPU targets pass: `text_model_runner_test`, `continuation_cache_test`,
  `continuation_disk_store_test`, `text_generation_scheduler_test`,
  `openai_chat_test`, `http_server_test`. New cases cover refusal before payload
  allocation, live-session reuse, byte-cap eviction/restore, default/oversized
  caps respecting the post-state model claim, and disk publication despite
  explicit RAM refusal.
- All 35 GUI Python tests and 17 jsdom tests pass. Coverage includes validation,
  byte conversion with disk caching off/on, older presets loading without a
  write, independent per-preset values, and an enabled RAM field with disk off.
- Formatting (clang-format 21.1.8), documentation and `git diff --check` pass.
- Release help advertises the flag; negative, fractional and overflowing CLI
  values are rejected before model loading. An isolated launcher serves the
  control and shuts down through its authenticated Exit endpoint. Native browser
  screenshot inspection was unavailable; GUI interaction coverage is jsdom.

## Matched numerical check

Baseline executable: `85d9139449830e5b23dac61d57bdb0946cc5dc71`, SHA-256
`6154ab077fdc9eb7948810f7b5914f139c7e73a3be15271f63b4ef18570f19cc`.
Candidate reports `b589fd002238-dirty`, SHA-256
`398da737ea80c656197fac0211a0ed467b33b3322b029912391a05d2fd32b29c`.
The baseline's model/core source is unchanged through the implementation base;
the intervening serving changes are outside this teacher-forced arithmetic.

Both use the same local Flash-Next UD-IQ4_XS three-shard model and shared-Q8_0
MTP sidecar, temperature 0, seed 459, and the same 41-token corpus (SHA-256
`4f9f84529a6cf5375729f7b75c07c336a3e7ab66c21d1d57a89c464925d845cb`).
Each schedule has 39 labels, vocabulary 248320. All 117 complete raw-logit rows
match exactly by SHA-256. Perplexity is unchanged: `10.759960`, `10.759960`,
`10.953727` for serial, mixed verification, and prefill schedules respectively.

```powershell
& $Executable bench --model $Model --mtp-model $Mtp --speculative mtp `
  --temperature 0 --seed 459 --logit-eval $Corpus --logit-out $Output `
  --logit-schedules '1:2,4,8:p8'
python tools/bench/logit-eval.py $BaselineHashes $CandidateHashes
```

Logs, corpus, full-logit hashes and the HTTP fixture/results are retained in
the ignored `build/ram-cap-validation/` directory. Bench does not use the
serving cache; these checks supplement the runner and HTTP coverage.

## HTTP cap check

Two sequential isolated servers use IQ4_XS + shared-Q8_0 MTP, context 4096,
one session, seven draft tokens, thinking off, disk caching off, and explicit
RAM caps of 1 byte and 1 GiB. Each receives France, Japan, then France again
through `/v1/completions`, temperature 0, seed 459, eight output tokens.
Both startup events report the requested effective cap. All six complete text/
finish signatures match between caps, including repeated France outputs; draft
proposal/acceptance counts also match for corresponding requests.

The 1-byte cap logs snapshot refusal before allocation for 119,441,636-byte
payloads and returns all three requests successfully with zero cached tokens.
The 1-GiB cap retains those snapshots and restores all five prompt tokens on
the repeated France request, with zero prefill tokens and `cache=memory`.
Both test-owned model processes are stopped afterward. This verifies cap
plumbing, graceful refusal and real MTP replay; it does not measure long-context
commit savings or establish general latency improvements.

```powershell
& $Executable serve --host 127.0.0.1 --port 8082 --sessions 1 llm `
  --model $Model --served-model-name gufo --context 4096 --think off `
  --speculative mtp --mtp-model $Mtp --draft-tokens 7 --cache-ram-bytes $Cap
```

Linux CI, other models/quantizations, 128K/150K workloads, and measured Windows
committed-memory savings are unrun. The numerical corpus is a bounded regression
check, not a general model-quality evaluation. Weights/live context and temporary
disk buffers remain outside this retained-snapshot cap.
