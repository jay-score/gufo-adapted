# TODO

## Memory follow-up — reviewed 2026-10-08

- [ ] [PR #445 — avoid Flash-Next prompt checkpoint copies](https://github.com/gufo-org/gufo/pull/445)
  (merged; source `82711d8c8ceb8bda9024914b6e7ac8495ba41742`). Review remaining
  in-pass checkpoint work against our existing immutable host-prefix sharing.
  Measure peak and retained Windows commit: upstream also pools released blocks
  and enlarges the n-gram cache, so net memory savings are unproven.
- [ ] [PR #477 — keep disk snapshots out of the page cache](https://github.com/gufo-org/gufo/pull/477)
  (open draft). Watch for results and a Windows-compatible approach. Current
  implementation uses Linux `posix_fadvise`; reduced filesystem cache pressure
  is not evidence of reduced Windows commit. Check disk-restore latency costs.
- [ ] [PR #425 — recycle snapshot payload buffers](https://github.com/gufo-org/gufo/pull/425)
  (open). Track capture-latency improvements, but defer for memory reduction:
  the pool retains up to 4 GiB of spare buffers. Re-evaluate after newer upstream
  checkpoint changes; preserve bounded retention and test pool accounting.

Recheck PR status before integration. These items are tracked, not imported.

## Upstream cache follow-up

Tracking the cache investigation for OpenCode and Pi on this Windows fork.
Upstream status below reflects the review on **2026-09-29**, rechecked on
**2026-10-01** for the items marked accordingly; recheck each remaining thread
before integrating more changes.

### Integrated locally

- [x] [PR #281 — fallback and full-prompt checkpoints](https://github.com/gufo-org/gufo/pull/281).
  Adapted in `d25f30b`. Retains an earlier fallback plus a complete prompt
  checkpoint; exact retries can avoid prefill without extra model sessions.
- [x] [PR #301 — stable reasoning and image boundaries](https://github.com/gufo-org/gufo/pull/301).
  Relevant stable-boundary changes adapted in `dbdc36c`, preserving this fork's
  image-prefix reuse and Windows tuning. This was a partial integration:
  upstream disk image-prefix indexing and Qwen27B executor cancellation changes
  were not imported. Arbitrary edits to older history can still cause misses.
- [x] [PR #358 — stable checkpoints on warm continuations](https://github.com/gufo-org/gufo/pull/358).
  Unmerged branch work at review time; adapted in `031d289` on 2026-10-01.
  Imports `d847586`/`492e5e1` plus only the `PublishSnapshot`
  source-preservation prerequisite from #362 (`0c350ed`); defers intermediate
  history checkpoints (covered later by `733aee8`), the six-entry layout and
  Flash-Next allocation changes. Rechecked 2026-10-01.
- [x] [PR #362 — prefixes across conversation history edits](https://github.com/gufo-org/gufo/pull/362).
  Unmerged branch work at review time; adapted in `733aee8` on 2026-10-01.
  Imports four bounded intermediate RAM checkpoints on a 2048-token grid and
  six retained entry slots from `0c350ed`/`840d373`, plus grid-aligned
  deduplication from #358 (`d847586`); preserves the fork's prefix-scoped
  image identity. Rechecked 2026-10-01

Background: [issue #259](https://github.com/gufo-org/gufo/issues/259) discusses
Pi agent loops, side requests displacing the only live session, disk staging,
and client history rewriting. Its staging/logging fix,
[PR #279](https://github.com/gufo-org/gufo/pull/279), was already in this fork.

### Checks for the next cache change

- Repeat exact retries, one-session chat switching, changed tool history,
  reasoning preservation on/off, image history, cancellation, and disk restart
  tests. Distinguish cached tokens from newly processed tokens.
- For inference changes, compare matched-token full logits and perplexity with
  a recorded baseline; include seeded MTP replay/rollback and snapshot identity
  checks. Measure long-context copy cost as well as prefill savings.
- GUI controls are available for reasoning preservation (default on) and disk
  caching (default off), including disk and RAM staging limits. Settings save
  per preset. GUI Stop/Exit does not drain pending disk writes.

## Upstream review 2026-10-05 — integration candidates

Review of the latest 30 official `main` commits (`2c6a106` →
`e03bb9115abc6be469927fed76fddaf1ef0682b3`, fetched 2026-10-05) against fork
HEAD `2236188c`. Unfinished candidates re-evaluated against clean fork HEAD
`a243a51292bf5f039b86c8d663f8af5a41e7ac61` on 2026-10-05; a fresh fetch
confirmed the same official tip. Merge base:
`d9a84f13f35d1f98da22886a12eb25dc7062e392`. Cross-checked patches, current
implementations, fixtures and `UPSTREAM.md`, including adapted equivalents.
This follow-up was inspection-only: no new builds or runtime validation.
Recommended integration order follows below; keep one focused commit and an
`UPSTREAM.md` entry per imported improvement. Validation below is proposed
unless an integrated entry explicitly records a pass.

- [x] [PR #415 — sampling: skip vocabulary blocks](https://github.com/gufo-org/gufo/commit/b945d0afdb26e4b790b8cb260762105e167ee1a3) (**Take**, adapted `SampleGreedy` to fork's `AdjustedLogit`, landed as `e8074c37` + ledger `fac6c1f8`). Validate: `logit_sampler_test` (incl. new block-skip case), `text_generation_scheduler_test`, `qwen38_flash_next.mtp_sampling` pass on Windows `cpu-test`; Linux `pr` + real-model logit unrun.
- [x] [PR #394 — reclaim output capacity from abandoned streams](https://github.com/gufo-org/gufo/commit/d85fb0fcd3fb906ea845f6e1404222d686b74e07) (**Take**, landed verbatim as `49852db5` + ledger `ae7d5956`). Validate: `text_generation_scheduler_test` passes on Windows `cpu-test`; Linux `pr` unrun.
- [x] [PR #385 — streaming error messages](https://github.com/gufo-org/gufo/commit/bf605399477b694c19adb6313952ecbf0986d8be) (**Take**, manual port as `614ed35a` + ledger `8a345c4b`; completions-streaming hunk omitted — fork rejects streamed completions pre-admission). Validate: `http_server_test`, `openai_chat_test` pass on Windows `cpu-test`; Linux `pr` unrun.
- [x] [PR #405 — Messages reasoning fields](https://github.com/gufo-org/gufo/commit/56383be0718ea0ce203fc42a55581304f0bbc253) + [PR #428 — parser refactor](https://github.com/gufo-org/gufo/commit/c3f4a9c26031b04f34b84a96f4d8cfc8f4337ad2) (**Take**, manual port as single change `0ce11489` + ledger `b562a9d6`; request-side only — no thinking-block rendering, harness tweak omitted with functional suite). Validate: `http_server_test` (new compat cases), `openai_chat_test`, `check-docs.py` pass on Windows; Linux `pr` unrun.

### Recommended next integrations, in priority order

- [ ] [PR #407 — stable checkpoint before replaced trailing context](https://github.com/gufo-org/gufo/commit/3d732377e5ac257fa3f1a59f57a17163e65763b5) (**Adapt**, small).
  Retain a checkpoint before agent clients' replaced final user context.
  Preserve literal-text handling, tool-result images and reasoning boundaries;
  bump formatter identity. Clients retaining that final message may prefill it
  again. Validate: `qwen_chat_template_test`, `vision_test`,
  `text_model_runner_test`, real-model cache replay, exact matched full logits
  and perplexity, Linux `pr`.
- [ ] [PR #434 — Responses compatibility](https://github.com/gufo-org/gufo/commit/d921a4bd956424241e3e050cf981023b5b81e475) (**Adapt**, manual API port).
  Add namespace function tools and nullable reasoning replay fields. Preserve
  namespace identity in output/history, reject duplicate flattened function
  names and enforce a 128-function cap. Define nested-namespace routing rather
  than copying upstream's top-level mapping. Some `include`/summary support
  already exists; retain its validation and current strict-schema behavior.
  Keep rejecting `text.format`; selectively accept harmless compatibility
  hints such as verbosity. Validate: `openai_chat_test`, `http_server_test`,
  `check-format.py`, `check-docs.py`, Linux `pr`; no real-model smoke needed.
- [ ] [PR #406 — idle GPU loss detection](https://github.com/gufo-org/gufo/commit/6a9ea9f263de4598a4c49954f10c5d285bcf7635) (**Adapt**, phase 1).
  Add a nonblocking idle probe on the scheduler thread, invoking the existing
  `on_device_lost` lifecycle. Preserve the fork's recovery behavior; a verbatim
  import would restore sticky/watchdog/HTTP behavior deliberately omitted from
  #390. Validate: `text_generation_scheduler_test`, `http_server_test`,
  `openai_chat_test`, Windows device-loss recovery smoke, Linux `pr`.
- [ ] [PR #392 — preserve disk writes at startup](https://github.com/gufo-org/gufo/commit/207bb4e0bf6cd0c666674794ec1f0dcf0eb49cb9) (**Adapt**, Windows locking required).
  Protect active publishers from another store's temporary-file cleanup.
  `compat/win32` already implements `flock()`; the blocker is upstream's use of
  a directory descriptor, not a missing primitive. Establish a suitable
  Windows lock file/store helper with shared publication and nonblocking
  exclusive cleanup semantics. Validate: `continuation_disk_store_test`
  including active-publisher coverage, cross-process Windows restart and
  abandoned-file cleanup tests, Linux `pr`.
- [ ] [PR #400 — remaining output framing/quoting gaps](https://github.com/gufo-org/gufo/commit/d910b92f9d722ee4a0ed8770c16749dea472afe2) (**Adapt**, selective).
  Literal prompt control-token handling is already retained. Compare echoed
  closers, client envelopes and richer fence/quote decisions against current
  fixtures; port demonstrated gaps into `GeneratedTextParser`. Do not
  cherry-pick the parser rewrite or broadly strip XML, examples or file data.
  Validate: `openai_chat_test`, `http_server_test`, `qwen_chat_template_test`,
  Linux `pr`; if prompt tokenization changes, add exact matched-logit and
  perplexity/replay checks.

### Existing behavior to verify before closing

- [ ] [PR #420 — DeepSeek client markup as content](https://github.com/gufo-org/gufo/commit/a72fc1e58c8d4784e7b824b4aad93a76b5a197b1) (**Already present** by inspection).
  The consolidated parser preserves content outside native DSML envelopes and
  does not strip client markup. Compare upstream regression fixtures with the
  fork's tests before closing; no production port currently identified.
  Validate: freshly built `openai_chat_test`, `http_server_test`.
- [ ] [PR #397 — DeepSeek envelope handling](https://github.com/gufo-org/gufo/commit/0df6ba3e1b5eade79782aacac559a02ea5468c2b) (**Already present** substantially; **Skip** grammar).
  Existing framing adaptation covers envelope ownership, argument-contained
  delimiters, complete interrupted calls and suffix prose. Compare remaining
  upstream fixtures and adapt only demonstrated gaps. Preserve visible suffix
  content; the commit title is not a requirement to discard it. The fork has
  no `json_constraint.cpp` by design. Validate: `openai_chat_test` envelope
  fixtures, `http_server_test`.

### Deferred workstreams and behavior decisions

- [ ] [PR #389 — live sessions + llama.cpp metrics](https://github.com/gufo-org/gufo/commit/1b4e6825f2cf6fa2203af3b4b3596bf25e4bfa4e) + [PR #403 — speculative round metrics](https://github.com/gufo-org/gufo/commit/8bdde807e57fadfe57f4a1005707559ae6afc82f) (**Defer**, useful observability).
  #403 can land standalone with `TotalDraftRounds` counted in
  `RecordServerMetrics` instead of upstream's `RecordRequestMetrics`. #389
  needs session tracking adapted to the current scheduler; parked-session
  hunks are N/A and slots/failure-test hunks wait with #389. Preserve #351 live
  counters/gauges without double-counting. Validate when revived:
  `http_server_test`, `text_generation_scheduler_test`, Linux `pr`.
- [ ] [PR #404 — union decoding decision](https://github.com/gufo-org/gufo/commit/ea064189976242f33f54bac92be6e0cdbd33fc48) (**Defer** union policy; replay **Already present**; **Skip** grammar).
  Both renderers already apply `tojson()` spacing to typed arguments, and
  DSML separators are retained. Do not move serialization into
  `ParseArguments` without a demonstrated benefit. Upstream's typed-first
  union decoding changes the fork's deliberate string-first policy; require
  concrete failing cases and an explicit behavior decision before adapting it.
  Validate any change: `openai_chat_test`, `http_server_test`, `json_test`,
  both model template tests, real-model replay, matched-logit/perplexity checks
  where prompt tokens change, Linux `pr`.
- [ ] [PR #384 — explicit RAM limit over auto budget](https://github.com/gufo-org/gufo/commit/53c590649295edf63abcc06117231cdc890b69c7) (**Defer**, Windows resource policy).
  Reconcile host RAM, WDDM/DXGI budgets, model-state allocations and disk
  staging before allowing larger explicit limits. Upstream's available RAM
  minus 4 GiB is not a validated Windows ceiling. Preserve automatic defaults
  and the fork's resource accounting. Validate: `text_model_runner_test`,
  disk/cache tests, Windows memory-pressure checks, docs check, Linux `pr`.
- [ ] [PR #406 — deferred streaming success](https://github.com/gufo-org/gufo/commit/6a9ea9f263de4598a4c49954f10c5d285bcf7635) (**Defer**, phase 2 after idle probing).
  Adapt admission/start callbacks and HTTP/SSE timing as a separate change,
  preserving cancellation and pre-header versus post-header failure behavior.
  Validate: `text_generation_scheduler_test`, `http_server_test`,
  `openai_chat_test`, SSE admission/queue-timeout/disconnect timing smoke,
  Linux `pr`; no logit checks for lifecycle-only changes.
- [ ] [PR #382 — shared in-flight prefixes](https://github.com/gufo-org/gufo/commit/c1eba3de7ffc1f71c9ad6e395138f9bf89ddcc7b) then [PR #386 — learned shared-prefix boundaries](https://github.com/gufo-org/gufo/commit/2ba3be24039186e00d2cecdb969f99ad695e1161) (**Adapt**, dedicated cache workstream).
  #386 depends on #382's sharing/checkpoint machinery. Reconcile with the
  fork's four-checkpoint grid, six slots, stable-boundary protection and
  prefix-scoped image identity. Measure concurrent and sequential use
  separately: extra copies/waiting can hurt single-request latency. Validate:
  `continuation_cache_test`, `text_generation_scheduler_test`,
  `text_model_runner_test`, real-model concurrency/growth/history edits,
  cancellation, cache pressure, image identity and speculative rollback,
  exact matched-logit/perplexity checks, Linux `pr`.
- [ ] [PR #421 — Flash prefill/indexer kernels](https://github.com/gufo-org/gufo/commit/653a31830447be5068a448a0284e43669d52f436) (**Defer** landing; investigate performance).
  Fits Flash-Next/gfx1151, but qualify the algorithm-4438 BF16 gate, WMMA
  intrinsics and LDS synchronization on TheRock 10.0.0. Require Windows
  `dense_gemm_bench`/`attn_bench` sweeps, GPU operator tests, exact full logits
  and perplexity with pinned IQ4_XS/MTP/BF16 projector artifacts, separate
  speculative rejection coverage and matched end-to-end timings. Linux
  speedups are not Windows evidence; shared kernels still need Linux CI.

Reviewed, no action: already present (#393 `c33e050`, #396 `2c6a106`, #390 landed `ee2bff3` via PR-tip entry `7d893a0`, #391 landed `6c0d493` via draft entry `6718dba` — behavior + fixture confirmed); skipped release chores (0.6.0/0.7.0/0.7.1/0.8.0), test-only #398/#413, `.gitignore`-only #435.
