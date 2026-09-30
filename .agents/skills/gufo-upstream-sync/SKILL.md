---
name: gufo-upstream-sync
description: Start an official Gufo upstream sync for this Windows fork by pinning history, reviewing fixes first, and preparing an isolated worktree with a validation plan. Use when asked to start or carry out an upstream sync; use gufo-upstream-review for review-only requests.
---

# Gufo Upstream Sync

Initiate the workflow in [UPSTREAM_SYNC.md](../../../docs/UPSTREAM_SYNC.md) using
the existing [sync helper](../../../tools/upstream-sync.py). **Prioritize fixes
over feature updates.** Preserve the fork behavior inventory and workflow policy
in that guide; use them as the maintained source rather than duplicating them.

A request to start a sync authorizes fetching, review and isolated worktree
preparation. Apply fixes when the user's request also authorizes integration;
existing authorization remains valid. Creating this skill does not itself start
a sync. Do not push, publish, enable GitHub Actions, change toolchains or replace
a running executable unless requested. Do not merge a whole checkpoint merely
because the helper prints its merge command.

## Establish the target

Read [AGENTS.md](../../../AGENTS.md) and the sync guide. Confirm the actual
checkout, HEAD, status and remotes before changing anything:

```powershell
git rev-parse --show-toplevel
git status --short --branch
git branch -vv
git remote -v
git config --get rerere.enabled
git config --get rerere.autoupdate
git config --get merge.conflictStyle
```

Run subsequent commands from that repository root. Use the user's target branch;
otherwise use `windows-port`, even if the calling checkout is a task branch.
Preserve unrelated edits and local configuration. If `.codegraph/` exists, use
CodeGraph before locating or reading implementation code; do not create an index.

The official repository is `gufo-org/gufo`, branch `main`. `origin` is the
personal fork; this fork's remote named `upstream` refers to `pixmaate/gufo`.
Add `official` only if absent; never repoint an existing remote. If its URL is
wrong, report the mismatch and use a separately named official source for review
until the remote configuration is resolved. The helper requires the verified
`official` remote.

For a clone without the setup, these are the PowerShell commands. Apply missing
settings only; preserve explicit user choices and do not repeat setup already
completed in the session:

```powershell
git remote add official https://github.com/gufo-org/gufo.git
git config --local rerere.enabled true
git config --local rerere.autoupdate false
git config --local merge.conflictStyle zdiff3
```

## Pin history, then prepare

Start with a fresh review report (replace the base for a user-selected target):

```powershell
python tools/upstream-sync.py --base windows-port
if ($LASTEXITCODE -ne 0) { throw "Upstream review failed" }
```

Read the JSON report at the path printed by the helper. Record its `base_sha`,
`official_sha`, `merge_base` and review date. Do not substitute a later moving
remote tip. If fetching fails, state the failure; use `--offline` only as an
explicitly labeled cached comparison. Shallow or unrelated history needs
resolution before a reliable integration.

Prepare at those immutable SHAs. Set `$reportPath` to the printed JSON path,
then choose an unused worktree path and branch; the dated names below are examples.
Use the app's managed worktree workflow when required by the user's environment,
creating it at `$forkSha`; otherwise use the helper:

```powershell
$syncReport = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
$forkSha = $syncReport.base_sha
$officialSha = $syncReport.official_sha
$syncStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$syncPath = "../gufo-sync-$syncStamp"
$syncBranch = "sync/upstream-fixes-$syncStamp"
python tools/upstream-sync.py --offline --base $forkSha --tip $officialSha --prepare $syncPath --branch $syncBranch
if ($LASTEXITCODE -ne 0) { throw "Sync preparation failed" }
```

The helper creates a branch and worktree, not an upstream integration. Never
reuse an occupied path or reset an existing sync branch. Existing `build/`, GUI
environment and local settings are not copied. For a later run, inspect any
previous sync branch before deciding to resume or create a new attempt.

The helper's lightweight checks are also available:

```powershell
python tools/upstream-sync.py --offline --base $forkSha --tip $officialSha --check
if ($LASTEXITCODE -ne 0) { throw "Repository checks failed" }
```

Without `--prepare`, these check the calling checkout, which may differ from the
pinned base. Use `--check` alongside `--prepare` to check the newly prepared tree.
Neither mode builds Gufo nor qualifies an upstream integration.

## Review fixes and integrate within scope

Use [gufo-upstream-review](../gufo-upstream-review/SKILL.md) to inspect candidate
patches, current equivalents, prerequisites, Windows adaptations and validation.
Pass the pinned target/range explicitly: a bounded recent-history review is not
qualification of an entire checkpoint. Prioritize correctness, crashes, leaks,
API compatibility and Windows regressions. Review performance changes only with
correctness evidence; assess feature updates separately. Commit titles, patch
equivalence and recorded SHA references are leads, not proof of complete behavior.

For authorized integration, apply selected fixes in dependency order in the
prepared worktree, keeping one improvement per commit and upstream attribution:

```powershell
git cherry-pick -x <reviewed-official-fix-SHA>
```

Resolve conflicts by preserving the documented fork behavior, then test each
improvement before proceeding. Review reused `rerere` resolutions. For a full
checkpoint merge, review every included change, including features and workflow
additions, before running `git merge --no-ff --no-commit $officialSha`. Prefer
selective fixes while a complete checkpoint remains unsuitable. Follow the guide
for merge completion, abort, promotion and ancestry preservation.

Build affected test targets before running CTest. For API fixes, from the prepared
worktree:

```powershell
powershell -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset gpu-test -Target openai_chat_test -Jobs 4
if ($LASTEXITCODE -ne 0) { throw "Chat test build failed" }
powershell -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset gpu-test -Target http_server_test -Jobs 4
if ($LASTEXITCODE -ne 0) { throw "HTTP test build failed" }
ctest --test-dir build/gpu-test -R '^(openai_chat_test|http_server_test)$' --output-on-failure --no-tests=error
if ($LASTEXITCODE -ne 0) { throw "API tests failed" }
```

Use the bounded CPU setup in CI when GPU tooling is unnecessary or unavailable.
Select other checks from the inventory and [TESTING.md](../../../docs/TESTING.md).
Inference changes need matched-token logits and perplexity against a recorded,
matched baseline, plus affected state/sampling/rollback checks. Preserve isolated
caches, model artifacts, the qualified toolchain and the running server. Report
missing coverage; missing-model skips are not passes. Shared changes need Linux CI.

## Deliver the result

Report the pinned SHAs, review/report path, prepared branch/worktree, fixes to
take/adapt/defer, prerequisites, workflow exclusions and relevant validation.
Distinguish preparation, source integration, successful builds, CPU/GPU quality
and deployment. For initiation-only requests, finish with a concrete prepared
worktree and fix-first plan. For authorized integration, continue through the
requested fixes and affected checks; do not stop to seek authorization already
given. Leave unresolved failures explicit and do not push unless requested.
