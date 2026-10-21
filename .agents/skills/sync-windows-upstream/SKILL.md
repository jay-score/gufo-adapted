---
name: sync-windows-upstream
description: Merge gufo-org updates from official/main into this fork's windows-main branch, preserving Windows compatibility and upstream ancestry through focused review and validation. Use for upstream syncs, conflict resolution during those syncs, and sync readiness reviews.
---

# Sync official Gufo into Windows

Maintain `windows-main` as the Windows integration branch descended from
official Gufo history. Read the repository's `AGENTS.md`, the
[maintenance plan](../../../docs/plans/windows-compatibility.md#future-official-updates)
and [Windows compatibility contracts](../../../docs/WINDOWS-COMPATIBILITY.md)
before starting. The checklist is the canonical implementation/test map; update
it when files move or contracts change rather than copying it into this skill.

## Establish the checkout and remote roles

The maintained checkout is currently `C:\P2027\windows-main`. Discover the actual
root with `git rev-parse --show-toplevel`; use that root for commands and logs.
It is a linked worktree with shared Git storage under `C:\P2027\gufo\.git`,
not an independent clone. Inspect `git worktree list` before switching branches
or creating another checkout. Preserve unrelated worktrees and their metadata.
The former temporary checkout path is a compatibility junction still used by
some build caches; do not remove it as part of a sync.

Verify `git remote -v`, the current branch, tracking configuration and worktree
status. Expected roles, subject to explicit user changes:

| Remote/ref | Purpose |
| --- | --- |
| `official` → `https://github.com/gufo-org/gufo.git` | Official source of updates; merge `official/main`. |
| `origin` → `https://github.com/thomas9120/gufo-windows.git` | User's fork; publish Windows work here when requested. |
| `upstream` → `https://github.com/pixmaate/gufo.git` | Different fork; not the official update source. |
| `windows-main` tracks `origin/windows-main` | Keep this tracking relationship; do not change it to `official/main`. |

If `official` is absent, adding it with the official URL is part of setting up
this workflow. If it exists at a different URL, investigate before changing it.
Do not repoint `upstream` or change the user's other remote configuration.
Prefer `jj` if available and usable for this checkout, otherwise Git; the
examples below use Git. Check `gh auth status` before GitHub operations.

## Prepare and merge a candidate

1. Inspect local changes and ahead/behind state. Preserve local commits and
   unrelated edits; do not reset, force checkout, automatically stash or silently
   commit them. If the current tree has unrelated edits, use a separate candidate
   worktree from the intended committed base. Resolve any essential base choice
   from the session before asking the user.
2. Fetch `official` and `origin`. Record the local Windows base SHA and fetched
   `official/main` SHA. Compare `windows-main` with `origin/windows-main` before
   deciding the integration base; do not pull blindly or discard unpublished
   Windows commits. If official is already an ancestor of the chosen Windows
   base, report it up to date rather than making an empty merge.
3. Inspect the incoming commits/diff and the maintained Windows difference:
   `git log windows-main..official/main --oneline`,
   `git diff --stat windows-main...official/main` and
   `git diff --stat official/main..windows-main`.
   Use CodeGraph first for code exploration when `.codegraph/` exists, per
   repository instructions. Include new callers of OS-sensitive operations,
   not just files with merge conflicts.
4. Create a unique `sync/official-YYYY-MM-DD` candidate from the chosen Windows
   base and merge the recorded official SHA. Do not routinely rebase the shared
   branch or cherry-pick only selected official fixes.

Example from a clean, confirmed `windows-main` base; run commands sequentially
and inspect exit status before proceeding, especially in Windows PowerShell
where native command failures do not automatically stop a command list:

```powershell
git fetch official
git fetch origin
git switch windows-main
$windowsBase = git rev-parse HEAD
$officialHead = git rev-parse official/main
$syncBranch = 'sync/official-' + (Get-Date -Format 'yyyy-MM-dd-HHmmss')
git switch -c $syncBranch
git merge --no-ff $officialHead -m "chore(sync): merge official/main"
```

Resolve conflicts at the OS boundary, retaining the current upstream algorithm
and Linux behavior alongside the narrow Windows adaptation. Do not use whole-
file ours/theirs resolutions without reviewing the discarded changes. Preserve
CMake-owned flags/dependencies, Linux/Nix configuration, official versioning,
public CLI/API behavior and model/cache formats. Keep the GUI an independent
launcher. Do not reintroduce old fork-only tuning switches or enable deferred
Windows disk checkpoint stores merely to make a merge build.

## Review and validate

Check off all six [compatibility contracts](../../../docs/WINDOWS-COMPATIBILITY.md):
cached concurrent n-gram reads, WDDM budgets, pinned memory, sockets/shutdown,
filesystem semantics and queue policy. Record each as unchanged/reviewed,
validated, or pending with its reason; a clean merge is not validation.

- Build and run the bounded Windows CPU gate using the documented toolchain:
  `./tools/windows/build.ps1 -Preset windows-cpu -Vcpkg PATH`, followed by
  `ctest --preset windows-cpu --no-tests=error`.
- Build the candidate Windows HIP executable and affected GPU/model test targets
  when inference or shared runtime paths change. Follow
  [Windows status](../../../docs/WINDOWS.md) for commands and current limits.
  Run the named WDDM, pinned-transfer and n-gram checks when their contracts are
  affected; snapshot changes also need the AR/MTP continuation probe.
- Use [functional checks](../../../tests/functional/README.md) for affected
  models/modes and histories. Shared text behavior calls for all four profiles;
  avoid unrelated full model sweeps. Retain numerical quality checks and the
  standard speed benchmark required by repository guidance.
- For prefill/n-gram/loader/I/O changes, compare the previous Windows binary
  and candidate with a varied 10–16K-token first HTTP prompt in fresh server
  processes and repeat with prompt reuse disabled. Include AR/MTP as affected.
  Warmed pp2048 alone previously missed a roughly 50% first-prompt regression
  caused by losing cached overlapped n-gram reads. Follow the checklist's
  [control record](../../../docs/models/qwen3.8-flash-next/BENCHMARKS.md#windows-cold-prompt-control-october-20-2026).
- Keep model, toolchain, launch settings, cache history and power conditions
  matched. Capture binary/revision identities, commands, individual request
  timings, output and actual cache/draft work. Apply repository timing limits
  per request/phase, investigate failures with alternating/unchanged controls,
  and retain failed/noisy observations. Do not average away or relabel a
  regression as a pass. Keep baseline binaries with their required runtime DLLs;
  do not overwrite an executable used by a live GUI/server. Stop only test
  processes owned by the run.
- Run `python tools/ci/check-docs.py`, `git diff --check` and required formatting
  for changed C++. Run the bounded Linux checks through an available Linux/Nix
  environment or hosted CI. Windows results do not qualify Linux, and missing
  hardware/models or skipped tests remain pending, not passed. Fix confirmed
  regressions before publishing a qualified sync.

Local hints, not portable defaults: the previous Windows validation used
TheRock 10.0.0 at `C:\TheRock\build`, with vcpkg at
`C:\Users\pegas\Documents\Projects-2026\Strix-halo-optimizations\vcpkg`.
The build helper defaults to `C:\vcpkg`, which was absent on this host.
Flash-Next model shards and the shared-Q8 MTP sidecar were under
`C:\Users\pegas\Documents\Llama\LLama-GUI\models\Qwenflash-next-4kxl`.
Verify paths and versions before reuse. Ignored `build/` investigation scripts
and old binaries may be missing in another checkout; use maintained test tools
or retain a reproducible workload with the new sync evidence.

## Integrate and hand back

Follow the user's requested endpoint: review only, local integration, or
publication. Creating/using this skill is not itself a request to perform a
sync, push branches, open a PR or merge on GitHub. Existing task authorization
continues to apply; do not ask again for actions already requested.

When local integration is requested and the Windows base is still unchanged,
fast-forward `windows-main` to the validated candidate. If it has advanced,
integrate the intervening changes on the candidate and rerun affected checks
before updating it. Preserve official ancestry. For an authorized GitHub sync
PR, target `windows-main` in the user's fork and use a merge commit, not squash
or rebase merging. This Windows maintenance policy differs from official
`main`'s usual squash release workflow. If repository settings prevent merge
commits, report the blocker rather than silently losing ancestry. Never force-
push the maintained branch. Use a Conventional Commit title and register the
PR with the current thread when the host provides PR-linking tools.

Report both base SHAs and final candidate SHA, conflicts/resolutions, the six
contract outcomes, exact checks/results, pending qualifications and any PR URL.
Verify the recorded official SHA is an ancestor of the final candidate with
`git merge-base --is-ancestor`. Leave an unqualified candidate clearly marked
as such. Do not create releases or manually bump versions/changelogs for a sync.
