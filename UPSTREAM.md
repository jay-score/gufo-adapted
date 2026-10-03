# Upstream adaptations

Track changes imported or adapted from other Gufo repositories here, with one
entry per logical improvement. Identify sources by repository URL and immutable
commit SHA; remote names alone are insufficient. This fork uses `official` for
`gufo-org/gufo` and `upstream` for `pixmaate/gufo`, subject to local configuration.

## Updating this record

- Add an entry when an upstream change is integrated, in the same change as the
  implementation. Reviewed, deferred, or skipped candidates are not imports.
- Link every source commit and record the adaptation, omitted parts and reasons,
  affected areas, and validation actually performed. State failed or unrun checks
  explicitly; importing code does not establish that it passed validation.
- Follow [AGENTS.md](AGENTS.md) for `git cherry-pick -x` and `Upstream-Commit:`
  trailers. Once local implementation commits exist, record their SHAs. If this
  entry is committed with the implementation, identify that commit as "commit
  introducing this entry"; do not amend repeatedly to insert its own SHA.
- Keep entries newest first. Update status when a change is superseded or removed,
  linking the replacement or removal commit and retaining the original provenance.
- Link model-specific upstream contracts when relevant, without duplicating them.

Use this entry format:

```markdown
### YYYY-MM-DD — Improvement

- Source: [repository@full-SHA](https://github.com/owner/gufo/commit/full-SHA)
  (list every source commit, including imported prerequisites).
- Local: implementation commit SHA(s), or "commit introducing this entry".
- Adaptation: what was imported, changed or omitted, and why.
- Areas: affected paths; links to detailed model contracts when applicable.
- Validation: checks and results; failed or unrun checks and remaining limits.
- Status: retained / superseded / removed (with replacement or removal commit).
```

## Recorded adaptations

### 2026-10-02 — Exit immediately after confirmed text-device loss

- Source: [gufo-org/gufo@7d893a020372502757933ec80231927fc9d25a96](https://github.com/gufo-org/gufo/commit/7d893a020372502757933ec80231927fc9d25a96)
  (PR #390).
- Local: commit introducing this entry.
- Adaptation: retained the preallocated four-byte HIP probe and five-second
  polling deadline for Qwen, Flash-Next and DeepSeek. Runs detection before
  request invalidation; a serve-owned callback writes the original failure
  reason and exits 75 without model teardown. Pending probes and classified
  request errors keep their existing behavior. Omitted sticky loss state,
  HTTP/health status changes, new wire errors, signal shutdown and watchdog:
  immediate process exit closes clients and avoids dead-device cleanup.
- Areas: text runners/scheduler, serve process policy, CPU/process fixtures,
  Windows and Linux hosted CPU selections, and server documentation.
- Validation: fresh Windows GPU build and four focused CPU tests pass; the
  process control exits 76 at invalidation and the production handler exits 75
  first. GUI tests pass (34). All 174 full-logit rows and perplexity match the
  recorded baseline exactly; eight real-model HTTP cases match, including
  streaming, history reuse and concurrency. Formatting and docs checks pass;
  see the [validation record](docs/plans/device-loss-pr390.md). Actual
  driver-reset and Linux execution remain unrun.
- Status: retained.

### 2026-10-01 — Multiline edit boundaries and malformed-call diagnostics

- Source: [gufo-org/gufo@594a623913b4109e4499885e9f73ed4d4ad3698e](https://github.com/gufo-org/gufo/commit/594a623913b4109e4499885e9f73ed4d4ad3698e)
  (PR #373 multiline edit regressions).
- Local: 55ff2aff89dfa6c01a45a1b15c13f55fa09b9b8c.
- Adaptation: delimiters inside JSON strings are data in native Qwen/DeepSeek
  array/object arguments; thinking markers after an enabled tool opener do not
  start inline reasoning. DeepSeek raw string quotes retain native semantics.
  Added a fork-specific, non-retryable `malformed_tool_call` 502/SSE contract for
  complete framed attempts rejected at EOS. Stops, limits and cancellation omit
  incomplete calls normally, including required choice. Retains the fork's
  thinking boundary, duplicate rules and interrupted-call recovery. No blanket
  control repair, schema enforcement, automatic JSON routing or loop suppression.
- Areas: parser, backend error contract, API/HTTP fixtures and server contracts.
- Validation: multiline fixture failed before the boundary adaptation. Fresh
  Windows CPU `openai_chat_test`, `http_server_test`, `qwen_chat_template_test`
  and `ds4.template` passed. Covers both APIs buffered/streaming, mixed valid/bad
  output, escaped/raw newlines, literal Qwen/DSML/thinking delimiters, native
  string quotes, non-retryable buffered errors, SSE failure after headers and
  required calls at token limits. The [implementation record](docs/plans/tool-calling-pr373.md#implementation-record)
  retains the partial real-model results and remaining failures. Linux CI is
  outstanding.
- Status: retained.

### 2026-10-01 — Bounded Qwen declared-type recovery

- Source: [gufo-org/gufo@594a623913b4109e4499885e9f73ed4d4ad3698e](https://github.com/gufo-org/gufo/commit/594a623913b4109e4499885e9f73ed4d4ad3698e)
  (PR #373 typed wildcard regression and native-schema review).
- Local: 5935c53e99b451a1cd26a37c7cf47aa5391d036b.
- Adaptation: parser-only type hints from named properties, all matching
  patterns, additional-property schemas and bounded local references. Uses
  existing ICU with `ICU::i18n` linkage and a documented portable regex subset.
  Ambiguous/unsupported hints retain text. This is an independent adaptation,
  not upstream's wildcard JSON fallback or a schema validator. Does not import
  automatic protocol switching, constraints or inference changes.
- Areas: `CMakeLists.txt`, API parser/fixtures and server contracts.
- Validation: baseline fixture reproduced integer `x_count` becoming a string.
  Fresh Windows CPU `openai_chat_test` and `http_server_test` pass; coverage
  includes all JSON types, intersections, precedence, Unicode/escaped keys,
  reference chains/cycles, bounded lookup and guidance fallback. Pinned ICU 78.3
  Windows import libraries and `icuin78.dll` linked/staged successfully.
  Real-model wildcard probes pass in all four tested modes; the
  [implementation record](docs/plans/tool-calling-pr373.md#implementation-record)
  retains the partial qualification and remaining failures. Linux CI is
  outstanding.
- Status: retained.

### 2026-10-01 — Historical function-name preservation

- Source: [gufo-org/gufo@594a623913b4109e4499885e9f73ed4d4ad3698e](https://github.com/gufo-org/gufo/commit/594a623913b4109e4499885e9f73ed4d4ad3698e)
  (PR #373).
- Local: 5a95703c90afac4a448047e28044ed64a4d3bb2c.
- Adaptation: shared historical-function parsing for Chat Completions and
  Responses. Preserves non-empty string names except embedded NUL and requires
  JSON-object arguments. Declaration rules and generated-call allowlists remain
  intact. The native renderers retain historical names verbatim, including
  delimiter-bearing names; this is history preservation, not escaping or new
  invocation authorization. No upstream grammar or renderer changes imported.
- Areas: API parser, API fixtures, Qwen/DeepSeek template fixtures and server
  contracts.
- Validation: new history fixture failed against the previous parser. Fresh
  Windows CPU API and Qwen/DeepSeek template targets passed after adaptation;
  existing stateless grouping, call/result identity and image-order fixtures
  remain covered. Real-model Unicode history probes pass; the
  [implementation record](docs/plans/tool-calling-pr373.md#implementation-record)
  retains the failed exact-file agent goals. Linux CI is outstanding.
- Status: retained.

### 2026-10-01 — Disabled-tool marker delivery

- Source: [gufo-org/gufo@594a623913b4109e4499885e9f73ed4d4ad3698e](https://github.com/gufo-org/gufo/commit/594a623913b4109e4499885e9f73ed4d4ad3698e)
  (PR #373).
- Local: f997a39f14da75719a407ee99931736cdf1b4a4b.
- Adaptation: one recognition decision for buffered and streaming output in
  both APIs. Disabled tool markers remain text/reasoning and their prefixes
  stream immediately. Retains the fork's required thinking boundary and UTF-8
  decoding. Does not import upstream constraints, prompting or sampling changes.
- Areas: `src/cli/serve/openai_chat.cpp`, `tests/cli/openai_chat_test.cpp`,
  `docs/SERVER.md`.
- Validation: baseline Windows CPU `openai_chat_test` and `http_server_test`
  passed at `c4dcca536af5dc73998f24b28e548985e7f91efc`; new deterministic
  disabled-marker fixture reproduced delayed prefixes, then passed after the
  adaptation. Both APIs, buffered/streaming, disabled declarations, reasoning,
  byte-split markers/UTF-8 and immediate callback delivery are covered.
  Real-model literal probes pass; the
  [implementation record](docs/plans/tool-calling-pr373.md#implementation-record)
  retains the partial qualification. Linux CI is outstanding.
- Status: retained.

### 2026-10-01 — Conversation history-edit checkpoints

- Source: [gufo-org/gufo@0c350ed3ef5f8db9783385071ff05eec6bd687f1](https://github.com/gufo-org/gufo/commit/0c350ed3ef5f8db9783385071ff05eec6bd687f1)
  and [gufo-org/gufo@840d3736012ebeb123472b2dc8ca39411084b05e](https://github.com/gufo-org/gufo/commit/840d3736012ebeb123472b2dc8ca39411084b05e)
  (PR #362 branch work), plus grid deduplication from
  [gufo-org/gufo@d8475862a1872ddbadefa281e0aee54350f18e9f](https://github.com/gufo-org/gufo/commit/d8475862a1872ddbadefa281e0aee54350f18e9f)
  (PR #358).
- Local: 733aee8a44ba0ccaaae9b4448e8b316f6265611b.
- Adaptation: four bounded intermediate RAM checkpoints on a 2048-token grid
  and six retained entry slots per execution session, within the existing byte
  budget. Flash-Next snapshot pages are pre-faulted with bounded fault workers
  before huge-page advice via the existing Windows madvise layer. Preserves the
  fork's prefix-scoped image identity and the #358 warm-turn stable
  boundary/source protection. The upstream history-edit and rewind regression
  was adapted to a standalone stdlib HTTP check sharing the existing
  cache-growth transport; the unrelated upstream functional SDK framework was
  not imported.
- Areas: `src/cli/serve/text_model_runner.cpp`,
  `src/models/qwen38_flash_next/engine.cpp`,
  `tests/cli/text_model_runner_test.cpp`, `tests/tools/cache_edits_test.py`,
  `tests/tools/cache_growth_test.py`, `docs/KV-CACHE.md`, `docs/SERVER.md`,
  `docs/TESTING.md`, `docs/models/qwen3.8-27b/EXPERIMENTS.md`,
  `docs/models/qwen3.8-flash-next/EXPERIMENTS.md`.
- Validation: CPU assertions and the `cache_edits_test.py` HTTP check were
  added in-commit; run results were not verified while writing this entry.
- Status: retained.

### 2026-10-01 — Warm-turn stable checkpoints

- Source: [gufo-org/gufo@d8475862a1872ddbadefa281e0aee54350f18e9f](https://github.com/gufo-org/gufo/commit/d8475862a1872ddbadefa281e0aee54350f18e9f)
  including [gufo-org/gufo@492e5e1a948c5ac5456f850a7a128d0bee01e94c](https://github.com/gufo-org/gufo/commit/492e5e1a948c5ac5456f850a7a128d0bee01e94c)
  (PR #358 branch work), plus only the `PublishSnapshot` source-preservation
  API prerequisite from PR #362
  ([gufo-org/gufo@0c350ed3ef5f8db9783385071ff05eec6bd687f1](https://github.com/gufo-org/gufo/commit/0c350ed3ef5f8db9783385071ff05eec6bd687f1)).
- Local: 031d2891acf8ecc4a3027ffcccd84407c411b14b.
- Adaptation: retains the turn's own stable boundary after freezing its reused
  frontier, prefers replacement of incompatible branch tails, and guards
  against concurrently replaced sources. Uses three checkpoint entries per
  session (frontier, stable boundary, complete prompt) with unchanged byte
  budgets and execution-state count; preserves the fork's prefix image identity
  and Windows persistence behavior. Deferred the intermediate history
  checkpoints (later covered by `733aee8`), the six-entry layout and
  Flash-Next allocation changes. CPU regressions were ported and cache-growth
  functional coverage was adapted into an explicit stdlib-only real-model
  check instead of the absent upstream SDK framework.
- Areas: `src/cli/serve/continuation_cache.cpp`,
  `src/cli/serve/continuation_cache.hpp`,
  `src/cli/serve/text_model_runner.cpp`,
  `tests/cli/continuation_cache_test.cpp`,
  `tests/cli/text_model_runner_test.cpp`, `tests/tools/cache_growth_test.py`,
  `docs/KV-CACHE.md`, `docs/SERVER.md`, `docs/TESTING.md`.
- Validation: ported CPU regressions and the `cache_growth_test.py` check were
  added in-commit; run results were not verified while writing this entry.
- Status: retained; extended by `733aee8` (intermediate checkpoints).

### 2026-09-29 — Stable reasoning and image boundaries

- Source: [gufo-org/gufo@8783ccbb2c4ed6ff5fc2f6e19d774ffd04eeea6e](https://github.com/gufo-org/gufo/commit/8783ccbb2c4ed6ff5fc2f6e19d774ffd04eeea6e)
  (PR #301).
- Local: dbdc36cb2fb14f16227babe0528822832c257944.
- Adaptation: partial integration of the stable-boundary changes, preserving
  this fork's image-prefix reuse and Windows tuning. Omitted upstream disk
  image-prefix indexing and Qwen27B executor cancellation changes.
- Areas: `src/cli/serve/inference_backend.cpp`,
  `src/cli/serve/text_model_runner.cpp`, `src/models/qwen/chat_template.cpp`,
  `src/models/qwen/chat_template.hpp`, `src/models/qwen/vision/prompt.cpp`,
  `src/models/qwen/vision/prompt.hpp`, `tests/cli/text_model_runner_test.cpp`,
  `tests/models/qwen27b/vision_test.cpp`,
  `tools/serving/check-continuation.py`, `docs/SERVER.md`.
- Validation: assertions added in `tests/cli/text_model_runner_test.cpp` and
  `tests/models/qwen27b/vision_test.cpp` in-commit; run results were not
  verified while writing this entry.
- Status: retained.

### 2026-09-29 — Fallback and full-prompt checkpoints

- Source: [gufo-org/gufo@c362049c59e11a8fb998f5b2e0df2cfb8fac4c9c](https://github.com/gufo-org/gufo/commit/c362049c59e11a8fb998f5b2e0df2cfb8fac4c9c)
  (PR #281).
- Local: d25f30bf309cae50201495deba84c5991edbed22.
- Adaptation: retains the earlier fallback plus a complete prompt checkpoint;
  exact retries can avoid prefill without extra model sessions.
- Areas: `src/cli/serve/continuation_cache.cpp`,
  `src/cli/serve/continuation_cache.hpp`,
  `src/cli/serve/continuation_disk_store.cpp`,
  `src/cli/serve/continuation_disk_store.hpp`,
  `src/cli/serve/text_model_runner.cpp`,
  `tests/cli/text_model_runner_test.cpp`,
  `tests/models/qwen/tokenization/chat_template_test.cpp`,
  `tools/serving/check-continuation.py`, `docs/SERVER.md`.
- Validation: coverage added in `tests/cli/text_model_runner_test.cpp` and
  `tests/models/qwen/tokenization/chat_template_test.cpp` in-commit; run
  results were not verified while writing this entry.
- Status: retained.

### 2026-09-26 — Bounded disk staging with skipped-snapshot reporting

- Source: [gufo-org/gufo@d9a84f13f35d1f98da22886a12eb25dc7062e392](https://github.com/gufo-org/gufo/commit/d9a84f13f35d1f98da22886a12eb25dc7062e392)
  (PR #279, shared history with `official/main`).
- Local: same SHA (present verbatim, not an adaptation).
- Adaptation: none.
- Areas: `src/cli/serve/continuation_disk_store.cpp`,
  `src/cli/serve/continuation_disk_store.hpp`,
  `src/cli/serve/inference_backend.cpp`, `src/cli/serve/inference_backend.hpp`,
  `src/cli/serve/serve.cpp`, `src/cli/serve/text_model_runner.cpp`,
  `src/cli/serve/text_model_runner.hpp`,
  `tests/cli/continuation_disk_store_test.cpp`, `tests/cli/serve_test.py`,
  `docs/SERVER.md`.
- Validation: `tests/cli/continuation_disk_store_test.cpp` coverage added
  upstream in-commit; run results were not verified while writing this entry.
- Status: retained.

## Model-specific provenance

- [DeepSeek V4 Flash](src/models/deepseek_v4_flash/UPSTREAM.md): imported engine
  revision, integration boundaries, and update policy.
- [Qwen-Image-2.1](src/models/qwen_image_21/UPSTREAM.md): pinned model and reference
  implementation contracts.
