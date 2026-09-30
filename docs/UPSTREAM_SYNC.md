# Keeping the Windows fork current

**Prioritize upstream fixes over feature updates.** Review correctness, crashes,
resource leaks, API compatibility and Windows regressions first. Review
performance changes only with correctness evidence. Feature updates are a
separate decision; being newer is not sufficient reason to import them.

The stable fork branch is `windows-port`. Keep its published history intact.
Prepare updates in a separate worktree, validate them, then promote through a
reviewed PR. The helper prepares a branch and report; it never merges,
cherry-picks, commits, pushes, changes remotes or starts Gufo.

## Sources and first-time setup

| Remote | Source | Use |
| --- | --- | --- |
| `origin` | `thomas9120/gufo` | Personal fork and PR destination |
| `official` | `gufo-org/gufo` | Official upstream fixes and checkpoints |
| `upstream` | `pixmaate/gufo` | Windows port; inspect its adaptations separately |

Verify with `git remote -v`; names alone do not identify a source. If `official`
is missing, add it once without replacing either existing remote:

```powershell
git remote add official https://github.com/gufo-org/gufo.git
git config --local rerere.enabled true
git config --local rerere.autoupdate false
git config --local merge.conflictStyle zdiff3
```

These Git settings belong to the local clone. Linked worktrees share the
repository configuration and recorded conflict resolutions. Reused resolutions
still need review and tests. Avoid blanket `ours`/`theirs` resolution: it can
silently discard one side's fixes.

## Review and prepare

Run from the repository root with Python 3.10+ and Git:

```powershell
python tools/upstream-sync.py
python tools/upstream-sync.py --offline --check
python tools/upstream-sync.py --prepare ../gufo-sync --branch sync/upstream-fixes
```

The default base is `windows-port`, even when the current checkout is a task
branch. Use `--base <branch-or-SHA>` deliberately for another target. Normal
runs fetch only official `main`; `--offline` uses the cached remote-tracking
revision and labels the result accordingly. `--tip <SHA>` selects an older
official checkpoint, which must be an ancestor of fetched/cached official main.

The JSON report under `build/upstream-sync/` records the date, immutable fork
and official SHAs, shared merge base, incoming commits, patch-equivalent imports,
recorded local SHA references, and overlapping paths. Overlap is not a conflict
prediction. A local reference may describe a partial adaptation, so inspect the
diff and current code before declaring an upstream change fully present. Worktree
preparation uses the recorded SHAs, not a moving branch name.

`--prepare` creates a new branch at the pinned fork base and an unused worktree
path. It preserves edits in the calling checkout. Build outputs, GUI environment
and local build settings are not copied; select the intended executable when
testing. `--check` runs the documentation and dependency checks in the prepared
worktree, or the calling checkout when no worktree is prepared. The report records
their exit codes and the checked tree's HEAD and working-tree status. The checked
tree may differ from the review base. Those checks do not build Gufo or qualify
an integration.

Review incoming fixes in dependency order before considering features. Read the
patch, prerequisites and later corrections, then check the
[fork behavior inventory](#fork-behavior-to-preserve) and
[workflow policy](#github-actions-during-upstream-updates). The existing
[upstream review skill](../.agents/skills/gufo-upstream-review/SKILL.md) describes
this review in more detail.

## Import fixes and advance the baseline

For the current divergent history, use small, focused imports of useful fixes:

```powershell
# In the prepared worktree, after reviewing a fix and its prerequisites:
git cherry-pick -x <official-fix-SHA>
```

If the fix needs adaptation, preserve its upstream SHA in the commit body,
describe the retained fork behavior, and keep one improvement per commit. Record
partially imported series explicitly. A reviewed SHA reference is not proof that
all behavior in the original commit was imported. Validate each improvement
before continuing. Cherry-picks preserve attribution but do not advance the
official ancestry baseline.

Periodically assess a checkpoint merge to reduce accumulated divergence, with
**fixes remaining the priority**. A merge includes every change reachable from
that checkpoint, including features. Review that complete range; use selective
fix imports while included features or toolchain changes remain unsuitable. Do
not create a fake merge claiming unimported changes are present.

The helper prints a merge command using the pinned checkpoint. When the full
range is accepted, run it in the prepared worktree:

```powershell
git merge --no-ff --no-commit <recorded-official-SHA>
# Resolve conflicts, review the complete result and run affected checks.
git commit -m "chore(sync): merge reviewed official checkpoint"
```

Use `git merge --abort` to abandon an in-progress merge. This retains the
baseline honestly when an update cannot be validated. Avoid unrelated edits
inside the merge; keep substantive follow-up adaptations independently
reviewable.

## Fork behavior to preserve

This inventory describes obligations, not a claim that all listed checks have
passed against an incoming update. Update it when fork behavior changes.

| Behavior and reason | Implementation | Regression evidence needed |
| --- | --- | --- |
| Native Windows build, pinned dependencies and complete runtime staging | [Windows build](../tools/windows/build.ps1), [compatibility layer](../compat/win32/posix.cpp), [spawn](../compat/win32/spawn.cpp), [CMake](../CMakeLists.txt), [manifest](../vcpkg.json) | Fresh Windows build and CPU CI; preserve TheRock 10.0.0 and Linux build; retain aligned I/O, mapping/locking, sockets, path and handle semantics |
| Windows memory budgeting and concurrent n-gram reads | [device memory](../src/core/platform/device_memory.cpp), [tuning](../src/core/platform/tuning.hpp), [n-grams](../src/models/qwen38_flash_next/ngram.cpp) | `platform_tuning_test`, affected model/loader checks; preserve DXGI/WDDM budgeting and cached overlapped random reads |
| Qwen27B device uploads avoid large registered-host-memory failures | [model loader](../src/models/qwen/hip/model_loader.cpp), [quality record](models/qwen3.8-27b/QUALITY.md#windows-device-upload) | Full-size model load and GPU access; matched-token full logits and perplexity; AR/DFlash2 and vision checks where affected |
| Responses tools, tool-result images, replay identity, streaming and request reasoning/sampling overrides | [chat adapter](../src/cli/serve/openai_chat.cpp), [HTTP server](../src/cli/serve/http_server.cpp), [backend](../src/cli/serve/inference_backend.cpp), [API contract](SERVER.md) | Fresh `openai_chat_test`, `http_server_test`, Qwen template tests when rendering changes; direct and tool-result images, stateless history, image order, streaming and cancellation |
| Actual loaded vision capability and compatible BF16 projector discovery | [encoder](../src/models/qwen/vision/encoder.hip), [runner](../src/cli/serve/text_model_runner.cpp) | Discovery/vision tests, `/v1/models` and `/props`; retain incompatible-sidecar rejection and BF16 qualification |
| Cache fallback/full checkpoints, mutable reasoning and reusable image prefixes | [cache](../src/cli/serve/continuation_cache.cpp), [runner](../src/cli/serve/text_model_runner.cpp), [Flash-Next engine](../src/models/qwen38_flash_next/engine.cpp) | Cache/disk-store/runner tests; cancellation, retry, image append/reorder, rollback and interleaved sessions; use separate disk caches for baseline and candidate |
| Fast sampling, speculation and platform tuning retain numerical correctness | [sampling](../src/core/sampling.cpp), [Flash-Next executor](../src/models/qwen38_flash_next/kernels/rocm/executor.cpp) | Distribution, rejection/residual, RNG, EOS, rollback and snapshot checks plus matched-token logits/perplexity; constrained output must cover local top-candidate paths |
| Simple GUI, explicit save/preset loading, existing config and owned process-tree shutdown | [GUI behavior](../tools/gui/README.md), [process ownership](../tools/gui/launcher_process.py), [Job Objects](../tools/gui/windows_job.py) | GUI Python and DOM checks; Stop/Exit, launcher death, retryable failures; opening the GUI must not load a model |
| Guided setup preserves consent, verified downloads and rerun recovery | [setup](../tools/windows/setup.ps1), [launcher](../tools/windows/gui.ps1) | `windows_setup_test.py`, `windows_launcher_test.py`; retain installed tools and existing launcher settings |
| Fork CI retains Windows and Linux coverage without importing official release automation by default | [CI](../.github/workflows/ci.yml), [workflow policy](#github-actions-during-upstream-updates) | Preserve both CPU jobs, launcher/setup and sync safety checks, and triggers targeting `windows-port`; review newly added workflows before promotion |

## GitHub Actions during upstream updates

Keep `ci.yml` and both its jobs: Windows CPU/launcher checks protect the port,
while Linux CPU/repository checks protect shared behavior and future upstream
integration. These jobs do not run GPU benchmarks or full model sweeps. Preserve
the fork's `windows-port` push and PR triggers when resolving upstream CI changes.

Inspect every added or changed file under `.github/workflows/` during upstream
review. Prioritize fixes to existing checks over adding release or feature
automation. Apply these decisions to incoming official workflows:

| Workflow | Fork policy | Reason and adaptation needed |
| --- | --- | --- |
| `ci.yml` | Keep and adapt fixes | Preserve both platforms' coverage and fork-specific checks rather than replacing the file wholesale with official CI |
| `release-please.yml` | Exclude by default; include only as a deliberately reviewed release setup | The inspected official version schedules daily release PR creation, automatically merges release PRs, publishes releases and dispatches builds to `gufo-org/toolboxes`. It requires GitHub App variables/secrets and access to `gufo` and `toolboxes`. A fork release setup must use its own destinations, credentials and validation requirements, including Windows checks |
| `pr-title.yml` | Optional | Lightweight Conventional Commit title validation. The inspected official trigger targets `main`; adapt it to `windows-port` if adopted, and include its checker script |

Recheck workflow contents at each selected official SHA; these descriptions are
not a guarantee that future versions behave identically. For checkpoint merges,
review workflow additions explicitly and document exclusions in the integration
record. Optional workflows can remain disabled individually until intentionally
configured. GitHub's enabled/disabled settings are separate from tracked YAML;
check those settings when enabling Actions or promoting an update.

## Validation and promotion

Use the smallest relevant checks in [AGENTS.md](../AGENTS.md),
[Windows guidance](WINDOWS.md) and [testing](TESTING.md). Existing
[CI](../.github/workflows/ci.yml) covers Windows CPU/launcher contracts and Linux
CPU/repository checks. Neither establishes Windows GPU correctness.

For API changes, build `openai_chat_test` and `http_server_test` through
`tools/windows/build.ps1` before running their exact CTest selection. Include
Qwen template checks for rendering changes. GUI changes use the existing GUI
Python tests and `node tests/tools/gui_dom_test.mjs`; setup changes use the
Windows fixture tests. Shared code also needs Linux CI. Use the pinned formatter
for C++ changes.

Inference changes require relevant matched-token logit and perplexity comparisons
against a recorded baseline, with identical model artifacts, corpus, compiler
and runtime settings. Require exact full-logit equality when arithmetic should
be unchanged; declare numerical limits for intentional arithmetic changes.
Add affected state, sampling, rollback and vision checks. Preserve old cache
directories for rollback; reverting source alone does not restore cache data.
Record commands, build identities, results, skipped checks and remaining gaps.

Promote only after affected checks pass and included features are explicitly
reviewed. Record the upstream SHA, imported/adapted/deferred fixes, retained fork
behavior and validation in the PR. Merge the PR into `windows-port` using a
regular merge when it contains an official checkpoint merge: squashing/rebasing
would lose that shared ancestry. Tag qualified milestones for rollback. Review
new official fixes regularly to keep batches manageable.

Longer term, contribute general correctness fixes and small platform helpers
upstream. This reduces the maintained diff. Keep personal UI and feature changes
focused and separate from those fixes.
