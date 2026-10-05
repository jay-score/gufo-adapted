# PR #441 parser integration and validation

Date: 2026-10-05. Three selective fixes are implemented on
`sync/pr441-parser-fixes`, in the sibling `gufo-pr441-parser-fixes` worktree.
Windows CPU contracts, the production build and bounded GPU numerical checks
passed. Real-model edge coverage remains partial; Linux CI is unrun.

## Scope and pinned revisions

The reviewed fork base is `0a2b9fd65d5927888b6146fb07dbf9619984c505`.
[PR #441](https://github.com/gufo-org/gufo/pull/441) was reviewed at final head
[b45567a918b281bdce63e80dec3e6184ccdb9c25](https://github.com/gufo-org/gufo/commit/b45567a918b281bdce63e80dec3e6184ccdb9c25),
including its predecessor
[29fb0b4eec70f49088bb5126074e6f10be17a0d1](https://github.com/gufo-org/gufo/commit/29fb0b4eec70f49088bb5126074e6f10be17a0d1).
The final official merge is
[21d6e64f137f6bcbc5a8bf63f900cab648188df7](https://github.com/gufo-org/gufo/commit/21d6e64f137f6bcbc5a8bf63f900cab648188df7).
The fetched official checkpoint was `bc87c34801fc4c3844046428ff090371aba61818`,
with shared merge base `d9a84f13f35d1f98da22886a12eb25dc7062e392`.
The ignored review report is
`build/upstream-sync/0a2b9fd65d59-bc87c34801fc.json`.

| Local commit | Adaptation |
| --- | --- |
| `ba1349e502ee45a83d8f42de4c1e2bcd2b9bb7b8` | Canonical parameter framing preserves inline closing tags and closing-tag-like lines, including split LF/CRLF suffixes |
| `e5dd53adea21de8af18766778d9f262b58b79985` | Exact declared parameter spelling, including surrounding whitespace and bounded root references, shared by scanner and decoder |
| `d7136e78b7d4417900bf1dab34df63bcc5ff2170` | Finite-value and implicit-shape type hints within the existing bounded resolver, retaining text preference for ambiguous unions |

The [server contract](../SERVER.md) and
[adaptation ledger](../../UPSTREAM.md) describe the resulting behavior.
Explicit reasoning boundaries, compact legacy calls, JSON string ownership,
interrupted-call recovery and Windows inference paths remain intact.
Upstream grammar, mask-cache, tokenizer and implicit reasoning-recovery changes
are deferred. No generation-time schema enforcement is added.

## CPU and repository checks

Each new regression failed before its corresponding implementation: canonical
literal delimiters, spaced names and numeric enum recovery. Their baseline logs
are retained in `build/delimiter-baseline.log`, `build/names-baseline.log` and
`build/types-baseline.log`.

Fresh Windows `cpu-test` executables passed `openai_chat_test` and
`http_server_test` after each fix. Final results are 2/2 passed, followed by
`qwen_chat_template_test` (1/1 passed). The API fixtures cover both Chat and
Responses, buffered and streamed output, byte-wise chunks, every two-piece split
for canonical framing, LF/CRLF, empty strings, exact and legacy names, local
references/cycles, finite JSON kinds, inferred containers, JSON-owned tags,
ambiguous/conflicting hints, applicators and lookup-budget exhaustion.

The CPU setup used the CI `cpu-test` preset, TheRock clang, the MSVC environment,
Ninja and the existing pinned `x64-windows` vcpkg installation. Local reproduction
helpers are retained under ignored `build/`:

```powershell
powershell -ExecutionPolicy Bypass -File build/validate.ps1 -Configure
powershell -ExecutionPolicy Bypass -File build/validate.ps1 -Targets qwen_chat_template_test
ctest --test-dir build/cpu-test -R '^(openai_chat_test|http_server_test|qwen_chat_template_test)$' --output-on-failure --no-tests=error
python tools/ci/check-docs.py
python tools/ci/check-format.py
```

Changed C++ files pass clang-format 21.1.8; documentation and `git diff --check`
pass. The initial whole-tree format check failed on two inherited violations,
`src/cli/serve/http_server.cpp:522` and `src/cli/serve/openai_chat.hpp:11`.
Both failures were reproduced from the pinned base. A separate formatting-only
cleanup subsequently joined the two wrapped lines; the full clang-format 21.1.8
check then passed for all 485 C++ files.
Linux CPU/repository CI has not been run.

## Production GPU qualification

The production executable was built with `tools/windows/build.ps1 -Target gufo
-Jobs 4` using TheRock ROCm 10.0.0 for gfx1151, AMD clang 23.0.0git revision
`8f497e0992fb7513f7f78a6f6b6f1056c375e961`, MSVC headers/SDK, CMake 4.4.0,
Ninja 1.12.0 and the existing manifest dependencies. Runtime DLLs and kernel
directories were staged beside it. The tested candidate reports
`gufo version d7136e78b7d4`; GPU checks preceded the separate whitespace cleanup.

The baseline reports `gufo version 80076e5b909c`. Its source and build inputs
are identical to the reviewed fork base; the intervening commits change docs:

```powershell
git diff --quiet 80076e5b909c 0a2b9fd65d5927888b6146fb07dbf9619984c505 -- src CMakeLists.txt cmake vcpkg.json vcpkg-configuration.json
```

Executable SHA-256:

- Baseline: `327c070cf1d3d2fc98b2966a82ec4cd36b114ba1e7e666212ee09d4968b450cb`.
- Candidate: `9ba63f45f575810d83d645a09c443cf795cac116ee10eabdd8fc3f2fbc7a7886`.

Both runs used the same local Qwen3.8-Flash-Next `UD-IQ4_XS` three-shard weights
(`Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`) and
`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, with greedy sampling, seed 441 and MTP.
The text corpus is the single line below, followed by CRLF:

```text
The quick brown fox jumps over the lazy dog. A triangle has three sides. Water freezes at zero degrees Celsius.
```

Corpus SHA-256 is
`21068ef2838f5caed82483797cfb63c33f5c962e50c1b881f848999ed778fe5c`.
Run the same command for each recorded executable, substituting artifact paths:

```powershell
& $exe bench --model $model --mtp-model $mtp --speculative mtp --temperature 0 --seed 441 --logit-eval build/corpus.txt --logit-out build/baseline --logit-schedules '1:1:2,4,8'
# Repeat with the candidate executable and --logit-out build/candidate.
python tools/bench/logit-eval.py build/baseline-s0.bin.sha256 build/candidate-s0.bin.sha256
python tools/bench/logit-eval.py build/baseline-s1.bin.sha256 build/candidate-s1.bin.sha256
python tools/bench/logit-eval.py build/baseline-s2.bin.sha256 build/candidate-s2.bin.sha256
```

All 22 complete raw-logit rows per schedule match exactly across the 248,320
vocabulary (66 comparisons). Perplexity is **6.238422** in every baseline and
candidate schedule. This is a bounded arithmetic regression check on a short
corpus, not a broad model-quality or performance benchmark. Logs, binary rows
and digest manifests remain in `build/baseline-*` and `build/candidate-*`.

## Real-model API checks and remaining limits

Private servers loaded the same weights with `mmproj-BF16.gguf`, context 8,192,
one session, thinking off, temperature 0 and seed 441. AR used `--speculative
off`; MTP used the Q8_0 sidecar and `--draft-tokens 7`. No persistent disk cache
was enabled; each server had a separate process and RAM cache. `/v1/models`
reported text/image input and `/props` reported loaded vision support.

The test declared `record` with one required parameter, requested one call,
replayed its returned call identity with a tool result, and requested exactly
`OK`. It then retried that identical continuation to check full prompt-cache
reuse. Both APIs and both streaming modes were exercised for each case:

| Parameter schema/value | AR | MTP |
| --- | --- | --- |
| `value: {"enum":[1,2]}`, requested `1` | 4/4 correct calls, continuations and cache retries | 4/4 correct calls, continuations and cache retries |
| `value: {"properties":{"n":{"type":"integer"}}}`, requested `{"n":7}` | 4/4 correct calls, continuations and cache retries | 4/4 correct calls, continuations and cache retries |
| `" value ": {"type":"integer"}`, requested `7` | Model emitted trimmed name; edge unqualified | Same model output; edge unqualified |
| String `alpha\n</parameter> is literal\nomega` | Model emitted only `alpha`; edge unqualified | Same model output; edge unqualified |

Raw completions of the reconstructed native prompts confirmed both runs' edge
failures originate in generated text: `<parameter=value>` and a complete call
containing only `alpha`. Prompt counts matched the API requests (321 and 328).
Those model outputs cannot exercise the new exact-name/delimiter paths; the
deterministic fixtures establish their parser behavior. There is no claim of
generation-level schema compliance or improved model accuracy.

The first smoke harness incorrectly required a cache hit after switching
`tool_choice` from `required` to `auto`, which changes the rendered system prompt.
The corrected check requires full reuse on an identical retry; first
continuations legitimately miss. Chat streaming requests include usage events.
Local helpers and complete requests/results are retained under ignored `build/`:

```powershell
python build/gpu-smoke.py --binary build/release/gufo.exe --speculative off --output smoke-ar-qualified
python build/gpu-smoke.py --binary build/release/gufo.exe --speculative mtp --output smoke-mtp-qualified
```

The scripts shut down only their own servers. The original checkout, GUI
configuration and selected executable are preserved. No branch was pushed or
promoted to `windows-port`.
