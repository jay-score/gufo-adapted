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
