# Windows upstream-friction TODO

Goal: keep `windows-main` descended from official Gufo history while making
each `official/main` merge cheaper and less error-prone. Source analysis:
`git diff e060aeed...cdf793e7` (official head at last sync vs `windows-main`)
is 190 files, 48 under `src/`.

Hygiene items are done (see commit: `official` remote, skill checkout/remote
corrections, `tools/windows/local.json` defaults, `WINDOWS.md` CPU-CI counts,
`ci.yml` intent comment). What remains is below, roughly ordered by
leverage-per-effort. Check off with PR links as each lands.

## 1. Upstream portable fixes to shrink the 48-file `src/` delta (defer this for now)

Several Windows touches are pure-portable and could disappear from our diff
entirely if accepted upstream. Propose each as an isolated upstream PR when
authorized, then drop it from the maintained difference on the next sync.

- [ ] `src/core/json.hpp`: `size_t` vs `unsigned long long` constructor
  template. No behavior change on Linux; unblocks MSVC ABI.
- [ ] Aligned allocation: aligned `new` instead of `std::aligned_alloc`
  (absent from Windows CRT). Prefer standard portable code per
  [plan](plans/windows-compatibility.md#keep-the-maintained-difference-small).
- [ ] `src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_runtime.hip.hpp`
  and `src/models/minimax_h3/media.cpp` one-liners — confirm portability,
  propose upstream.
- [ ] `src/core/platform/` (`device_memory.*`, `socket.hpp`,
  `cache_directory.hpp`): propose as a platform abstraction with Linux
  pass-through, so OS branches live in one place.
- [ ] Create `docs/WINDOWS-UPSTREAM-CANDIDATES.md` tracking file, owner,
  upstream issue/PR, and blocker for each candidate. The sync skill already
  says "propose that isolated fix upstream when authorized" but with no list,
  nothing ever leaves the diff.

## 2. Collapse `CMakeLists.txt` Windows hooks into `cmake/Windows.cmake`

`CMakeLists.txt` has ~8 `if (WIN32)` sites; every upstream refactor nearby
conflicts. Move each into a function in the already-isolated Windows module so
the root file keeps ~2 hooks (one early `include`, one late).

- [ ] `GUFO_WEBP_TARGET` (`WebP CONFIG` vs `PkgConfig::WEBP`) selection.
- [ ] `gufo-server` copy-vs-symlink target.
- [ ] `ROCTX` required-vs-optional (`GUFO_NO_ROCTX` fallback).
- [ ] Fault-library (`device_loss_faults`, `cache_disk_faults`) exclusion.
- [ ] `cmake/Checks.cmake` exclusion on Windows.
- [ ] Verify with `git diff --stat`: root `CMakeLists.txt` delta should be
  only the include hook plus comments.

## 3. Stop editing shared build/CI/test files

- [ ] `CMakePresets.json`: move `windows-cpu`/`windows-gpu` presets into an
  included file via the `"include"` field instead of editing the shared
  presets. Our additions are purely additive and would then never conflict.
- [ ] `tools/ci/check-docs.py`, `tools/ci/check-dependencies.py`,
  `tests/cache/boundary_test.py`: refactor to config
  (e.g. `tools/ci/windows-docs.json`, an extra allowed-deps list read by the
  unmodified script) instead of forking the scripts. Upstream edits here
  currently collide with ours every time.
- [ ] Shared tests (`http_server_test`, `audio_websocket_test`,
  `gguf_identity_test`, `sampling_test`, `session_test`, `run.py`,
  `test_runner.py`): prefer the existing pattern — new `win32_*` files, shared
  files untouched. Where a shared edit is unavoidable, wrap it in
  `// WINDOWS:` markers and add a diff-lint that fails the sync if a shared
  `src/`/`tests/` file changes without one.

## 4. De-conflict repeat-offender docs

- [ ] Top-appended model docs (`EXPERIMENTS.md`/`QUALITY.md`): the Oct-21 sync
  conflicted because both branches insert at the top. Switch Windows evidence
  to append-at-bottom with dated `##` headers, or keep Windows notes in
  `WINDOWS-*.md` sidecars. Alternative: a `union` merge driver for those
  paths.
- [ ] Single source of truth: the contract/remote/procedure list exists in 4
  places (sync skill, `WINDOWS-COMPATIBILITY.md`, plan §Future updates,
  `WINDOWS.md` intro) and they already disagree. Make
  `WINDOWS-COMPATIBILITY.md` canonical; the skill and plan link to it instead
  of duplicating remote tables, paths, and check counts.

## 5. Automate the manual sync procedure (~600 lines of copy-paste today)

- [ ] `tools/windows/sync-inventory.ps1 OFFICIAL_HEAD WINDOWS_BASE`: emit the
  PR table (number/title/URL/merge SHA via `gh`), `diff --stat` both
  directions, and a fresh sync-log entry skeleton with
  `validated / reviewed-unchanged / pending (reason)` slots per contract.
- [ ] `tools/windows/known-regressions.json`: machine-readable mirror of the
  sync log's known-regressions table (PR, merge SHA, revert SHA,
  reintroduction checks) plus a pre-merge guard that fails if
  `WINDOWS_BASE..OFFICIAL_HEAD` intersects it. Markdown stays
  human-readable; JSON is the gate.
- [ ] `tools/windows/diff-guard.ps1`: diff `official/main..HEAD`, fail if a
  shared `src/` file outside an allowlist changed or a `// WINDOWS:` marker
  is missing. Keeps the "small difference" contract enforced, not
  aspirational.
- [ ] `#699` revert watcher: on each fetch, report whether upstream touched
  the reverted regions (`engine.cpp` vision/catch-up/lookup paths,
  `sampling.cpp` compact distributions, `executor.cpp` verification
  workspace) so a landed fix is noticed without rediscovery. Do not reapply
  without citing the sync log entry and passing the cold-start control.
- [ ] Push the candidate branch and `gh run watch` the Linux + Windows-CPU
  gates instead of the current "use an available Linux/Nix environment"
  hand-wave (both workflows already trigger on `windows-main`).

## 6. Promote the validation harness out of ignored `build/`

- [ ] `tools/windows/first-prompt-control.py`: the versioned 10–16K-token
  cold-prompt generator (the `random.Random(813)` body in
  `WINDOWS-UPSTREAM-SYNC.md`) plus fresh-process server orchestration for
  AR/MTP, reuse-on then reuse-off. This check caught both the n-gram ~50%
  regression and #699; it must not live only in ignored `build/` evidence
  dirs that won't exist next quarter.
- [ ] `tools/windows/compare-phases.py`: per-request/phase comparison with
  the repo's 5% and 3 ms gates, no averaging, unchanged-binary controls,
  automatic retention of binary SHA + colocated DLL set + `/props` +
  timings. Today's flow depends on tribal `build/sync-*/` evidence.
- [ ] `tools/windows/local.json` follow-up: extend the new host-config
  mechanism with a `Models` path used by the harnesses above, so model-shard
  locations stop living in prose.
