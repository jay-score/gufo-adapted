# Bug tracker — 413 `payload_too_large` after many screenshots (opencode)

Status: open, deferred.
Date: 2026-10-05.
Upstream: `gufo-org/gufo`, no direct issue found (checked 2026-10-05).

## Symptoms

After a long opencode session with many screenshots, every
`POST /v1/chat/completions` fails with `413`:

```text
2026-10-04 22:57:24 [INFO] ... status=200 ... prompt_tokens=112406 ... cached_tokens=112383 ...
2026-10-04 22:57:25 [WARN] ... status=413 ... error_code=payload_too_large ...
2026-10-04 22:57:28 [WARN] ... status=413 ... error_code=payload_too_large ...
2026-10-04 22:57:32 [WARN] ... status=413 ...
2026-10-04 22:57:41 [WARN] ... status=413 ...
2026-10-04 22:57:52 [WARN] ... status=413 ...
2026-10-04 22:58:04 [WARN] ... status=413 ...
```

Signature: `duration_ms=0.1-0.2`, no `prompt_tokens/cache/*` fields on the
`413`s. Client retries the same oversized history, so all subsequent requests
fail until the session is compacted/restarted.

## Root cause

HTTP-layer early reject, before inference/cache:

* `src/cli/serve/http_server.cpp:1435-1436`: `Content-Length >
  options_.max_request_body_bytes` sets `payload_too_large`.
* `src/cli/serve/http_server.cpp:1482-1484`: returns
  `413 Payload Too Large / payload_too_large`.
* Default limit `8 MiB`: `src/cli/serve/http_server.hpp:87`,
  `src/cli/serve/serve.cpp:146-147,671-672` (`--max-request-bytes`,
  documented in `docs/SERVER.md:461-462`).

Opencode resends full message history each turn, with screenshots as
`data:image/...;base64` in `messages[].image_url`. Payload grows
monotonically (~1-3 MiB per PNG + ~33% base64 overhead) and eventually
exceeds `8 MiB`. Last success (`r2451`, `112k` prompt tokens, `112383`
cached) shows the cache was healthy; `413`s never reach cache lookup, so this
is not a cache bust/eviction.

`Connection: close` is always sent (`src/cli/serve/http_server.cpp:311-312`),
so this is not keep-alive corruption.

## Upstream check (2026-10-05, via GitHub API search)

No open/closed issue/PR for `payload_too_large`, `413`, or `screenshot`.

Related only:

* #285 (closed): `400 invalid_prompt: Rendered prompt ... exceeds maximum
  bound` — fixed 1 MiB rendered-prompt cap, independent of
  `--max-request-bytes`/`--context`. Next wall after `413` is raised.
* #288 (merged, `8a46cd9`): `RenderedPromptBoundBytes(context) =
  max(1 MiB, context*128)`, `QwenChatOptions(request,max_context)`.
  Fixes #285, not this `413`. Already present in this fork
  (`src/models/qwen/chat_template.cpp:24`,
  `src/cli/serve/inference_backend.cpp:68-76`;
  local struct default is `8 MiB` at
  `src/models/qwen/chat_template.hpp:107`, but serve overwrites it).
* #298 (closed): image-bearing turn re-prefills full prompt
  (`cached_tokens=0`).
* #336 (open tracking): agent-workload cache/continuation defects.
* #262 (closed): defaults too conservative (`max_body_bytes=8388608`,
  small context); maintainer agreed to update defaults.

## Deferred fix options (not implemented)

1. Raise default `--max-request-bytes` (e.g. `32 MiB` as in `flake.nix`,
   or `64 MiB` as in `tools/gufo/model_bench/llm.py:122`). Cost:
   `max_connections x limit` worst-case buffering.
2. Make `413` actionable: include attempted `Content-Length` and configured
   limit in JSON error + `log_details` (currently generic
   `"request body is too large"`, no size in log).
3. `gzip Content-Encoding` would not help screenshots much (PNG+base64
   already compressed).
4. Server cannot safely drop old images server-side; real fix needs client
   compaction (opencode `/compact`, new session, fewer/smaller screenshots).

## Workaround for now

Restart serve with e.g. `--max-request-bytes 33554432`, and compact/start a
new opencode session. Expect the next limit to be context/rendered-prompt,
not HTTP, for very long histories.
