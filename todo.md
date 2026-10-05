# TODO

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
HEAD `2236188c`. Cross-checked against `UPSTREAM.md`: none of these SHAs is
recorded there. Land in order, one focused commit each, with an `UPSTREAM.md`
entry per improvement.

- [x] [PR #415 — sampling: skip vocabulary blocks](https://github.com/gufo-org/gufo/commit/b945d0afdb26e4b790b8cb260762105e167ee1a3) (**Take**, adapted `SampleGreedy` to fork's `AdjustedLogit`, landed as `e8074c37` + ledger `fac6c1f8`). Validate: `logit_sampler_test` (incl. new block-skip case), `text_generation_scheduler_test`, `qwen38_flash_next.mtp_sampling` pass on Windows `cpu-test`; Linux `pr` + real-model logit unrun.
- [x] [PR #394 — reclaim output capacity from abandoned streams](https://github.com/gufo-org/gufo/commit/d85fb0fcd3fb906ea845f6e1404222d686b74e07) (**Take**, landed verbatim as `49852db5` + ledger `ae7d5956`). Validate: `text_generation_scheduler_test` passes on Windows `cpu-test`; Linux `pr` unrun.
- [x] [PR #385 — streaming error messages](https://github.com/gufo-org/gufo/commit/bf605399477b694c19adb6313952ecbf0986d8be) (**Take**, manual port as `614ed35a` + ledger `8a345c4b`; completions-streaming hunk omitted — fork rejects streamed completions pre-admission). Validate: `http_server_test`, `openai_chat_test` pass on Windows `cpu-test`; Linux `pr` unrun.
- [x] [PR #405 — Messages reasoning fields](https://github.com/gufo-org/gufo/commit/56383be0718ea0ce203fc42a55581304f0bbc253) + [PR #428 — parser refactor](https://github.com/gufo-org/gufo/commit/c3f4a9c26031b04f34b84a96f4d8cfc8f4337ad2) (**Take**, manual port as single change `0ce11489` + ledger `b562a9d6`; request-side only — no thinking-block rendering, harness tweak omitted with functional suite). Validate: `http_server_test` (new compat cases), `openai_chat_test`, `check-docs.py` pass on Windows; Linux `pr` unrun.
- [ ] [PR #389 — live sessions + llama.cpp metrics](https://github.com/gufo-org/gufo/commit/1b4e6825f2cf6fa2203af3b4b3596bf25e4bfa4e) (**Take**) then [PR #403 — speculative round metrics](https://github.com/gufo-org/gufo/commit/8bdde807e57fadfe57f4a1005707559ae6afc82f) (**Take**). Additive; don't double-count fork's #351 gauges. Validate: `http_server_test`, `text_generation_scheduler_test`, Linux `pr`.
- [ ] [PR #407 — stable checkpoint before replaced trailing context](https://github.com/gufo-org/gufo/commit/3d732377e5ac257fa3f1a59f57a17163e65763b5) (**Adapt**, Qwen template-only + formatter-identity bump). Validate: `chat_template_test`, `vision_test`, `text_model_runner_test`, real-model replay + logit, Linux `pr`.
- [ ] [PR #392 — preserve disk writes at startup](https://github.com/gufo-org/gufo/commit/207bb4e0bf6cd0c666674794ec1f0dcf0eb49cb9) (**Adapt**, blocked on Windows lock primitive for `flock()` in `compat/win32` or store layer). Validate: `continuation_disk_store_test` + Windows restart test, Linux `pr`.
- [ ] [PR #420 — DeepSeek client markup as content](https://github.com/gufo-org/gufo/commit/a72fc1e58c8d4784e7b824b4aad93a76b5a197b1) (**Adapt**, small, parser-only). Validate: `openai_chat_test`, `http_server_test`, Linux `pr`.
- [ ] [PR #397 landed — end DeepSeek output after call block](https://github.com/gufo-org/gufo/commit/0df6ba3e1b5eade79782aacac559a02ea5468c2b) (**Adapt** parser half / **Skip** grammar half — fork has no `json_constraint.cpp` by design). Validate: `openai_chat_test` envelope fixtures, `http_server_test`.
- [ ] [PR #400 landed — framing out of assistant content](https://github.com/gufo-org/gufo/commit/d910b92f9d722ee4a0ed8770c16749dea472afe2) (**Adapt** selective, own work item — no cherry-pick; port `QuoteTracker` decisions into `GeneratedTextParser`). Validate: `openai_chat_test`, `http_server_test`, Qwen `chat_template_test`, Linux `pr`.
- [ ] [PR #404 landed — union + typed-arg replay remainder](https://github.com/gufo-org/gufo/commit/ea064189976242f33f54bac92be6e0cdbd33fc48) (**Adapt** `dump()`→`tojson()`, union-typed-first, DSML marker; **Skip** union grammar; separator + typed rendering already retained). Validate: `openai_chat_test`, `json_test`, real-model replay + matched-logit, Linux `pr`.
- [ ] [PR #434 — Responses compatibility](https://github.com/gufo-org/gufo/commit/d921a4bd956424241e3e050cf981023b5b81e475) (**Adapt** manual port; include uniqueness check + 128-cap; omit strict-normalization + `text.format`, keep rejecting it). Validate: `openai_chat_test`, `http_server_test`, `check-format.py`, `check-docs.py`, Linux CI. No real-model smoke needed.
- [ ] [PR #384 — explicit RAM limit over auto budget](https://github.com/gufo-org/gufo/commit/53c590649295edf63abcc06117231cdc890b69c7) (**Adapt**, blocked on WDDM/DXGI budgeting decision in `device_memory.cpp`). Validate: `text_model_runner_test`, disk/cache tests, docs check, Linux `pr`.
- [ ] [PR #406 — idle GPU loss + deferred streaming](https://github.com/gufo-org/gufo/commit/6a9ea9f263de4598a4c49954f10c5d285bcf7635) (**Adapt** phased: slice 1 idle probe calling existing `on_device_lost` now; slice 2 deferred streaming follow-up — verbatim take would reintroduce the sticky/watchdog/HTTP surface deliberately omitted from #390). Validate: `text_generation_scheduler_test`, `http_server_test`, `openai_chat_test`, SSE timing smoke; no logit checks; Linux `pr`.
- [ ] [PR #382 — shared in-flight prefixes](https://github.com/gufo-org/gufo/commit/c1eba3de7ffc1f71c9ad6e395138f9bf89ddcc7b) + [PR #386 — learned shared-prefix boundaries](https://github.com/gufo-org/gufo/commit/2ba3be24039186e00d2cecdb969f99ad695e1161) (**Adapt** phased workstream; reconcile with fork's 4-checkpoint grid + 6 slots + stable-boundary protection). Validate: `continuation_cache_test`, `text_generation_scheduler_test`, `text_model_runner_test`, `cache_concurrency`/`cache_growth`/`cache_edits` real-model + matched-logit, Linux `pr`.
- [ ] [PR #421 — Flash prefill/indexer kernels](https://github.com/gufo-org/gufo/commit/653a31830447be5068a448a0284e43669d52f436) (**Adapt**, landing gated on TheRock 10.0.0 bitwise evidence: algo-4438 gate + `dense_gemm_bench`/`attn_bench` sweeps on Windows). Validate: GPU operator tests, exact full-logit + perplexity (pinned IQ4_XS/MTP/mmproj), speculative/rejection cover separately, Linux CI.

Reviewed, no action: already present (#393 `c33e050`, #396 `2c6a106`, #390 landed `ee2bff3` via PR-tip entry `7d893a0`, #391 landed `6c0d493` via draft entry `6718dba` — behavior + fixture confirmed); skipped release chores (0.6.0/0.7.0/0.7.1/0.8.0), test-only #398/#413, `.gitignore`-only #435.
