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
  image identity. Rechecked 2026-10-01.

### Follow up

- [ ] [Issue #267 — shared-prefix checkpoints in RAM](https://github.com/gufo-org/gufo/issues/267)
  and [issue #331 — intermediate conversation checkpoints](https://github.com/gufo-org/gufo/issues/331).
  Implementing work now exists on the `cache-conversation-retention` (#358)
  and `cache-history-edits` (#362) branches, adapted locally in `031d289` /
  `733aee8` (see Integrated locally). Remaining: track their merge to upstream
  `main` and reassess chat-switching misses and reprocessing after older
  messages/tool results are edited or compressed. Rechecked 2026-10-01.
- [ ] [PR #330 — seeded MTP replay across cache reuse](https://github.com/gufo-org/gufo/pull/330).
  **Merged** upstream in `a917b79` (rechecked 2026-10-01; was open at review
  time); fixes [issue #300](https://github.com/gufo-org/gufo/issues/300).
  Still not integrated here: it changes model arithmetic and advances the
  snapshot payload to version 16; this fork currently uses 15.
  Require Windows correctness and performance validation before adoption.
- [ ] [PR #321 — preserve the first exact snapshot](https://github.com/gufo-org/gufo/pull/321).
  Open at review time; the maintainer expected it to be superseded by #330.
  Now that #330 is merged, verify #321 was closed/superseded rather than
  integrating it. Rechecked 2026-10-01.
- [ ] [Issue #318 — checkpoint-copy latency](https://github.com/gufo-org/gufo/issues/318).
  Track first-token and streaming stalls from copying large snapshots. Our
  #281/#301 checks found increased copy cost on a long warm continuation, so
  successful cache hits do not guarantee lower latency.
- [ ] [Issue #275 — disk-cache eviction between conversations](https://github.com/gufo-org/gufo/issues/275).
  Track whether one long conversation can crowd other conversations out of the
  disk cache; reassess with multiple chats and realistic disk/staging limits.
- [ ] [Issue #313 — persistent system-prompt checkpoints](https://github.com/gufo-org/gufo/issues/313).
  Watch for reusable shared-prefix support across new sessions.
- [ ] Revisit the omitted portions of #301 if disk image-history reuse or
  Qwen27B cancellation becomes a priority; preserve the fork's existing vision
  and cache-identity behavior.

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
