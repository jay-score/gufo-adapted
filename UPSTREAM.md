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

No entries have been recorded yet. Earlier imports have not been backfilled;
absence from this record does not mean an upstream change is absent from the fork.

## Model-specific provenance

- [DeepSeek V4 Flash](src/models/deepseek_v4_flash/UPSTREAM.md): imported engine
  revision, integration boundaries, and update policy.
- [Qwen-Image-2.1](src/models/qwen_image_21/UPSTREAM.md): pinned model and reference
  implementation contracts.
