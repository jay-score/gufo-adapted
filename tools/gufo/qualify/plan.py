"""Select affected models and qualification scopes from a committed diff."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import (
    QualifyConfig,
    QualifyError,
    load_configs,
    ordered_union,
    path_matches,
)
from .util import REVIEW_CONTEXT, platform_name, utc_now

ARTIFACT_TYPE = "gufo-qualify-plan"
SCHEMA_VERSION = 1
CODE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".hip",
    ".py",
}
# Text-server API files belong to the API-contract workflow, not to a model
# arithmetic gate. Keep a changed inference_backend/scheduler visible.
API_ONLY_PREFIXES = (
    "src/cli/serve/openai_chat",
    "src/cli/serve/http_server",
    "src/cli/serve/serve.",
    "src/cli/serve/generation_metrics",
    "src/cli/serve/messages",
    "src/cli/serve/responses",
)


def _run_review_context(
    root: Path,
    base: str | None,
    head: str,
    review_context: Path,
) -> dict[str, Any]:
    if not review_context.is_file():
        raise QualifyError(f"missing review-context tool: {review_context}")
    argv = [
        sys.executable,
        str(review_context),
        "--repo",
        str(root),
        "--head",
        head,
        "--format",
        "json",
        "--platform",
        platform_name(),
    ]
    if base:
        argv.extend(["--base", base])
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise QualifyError(f"review-context failed:\n{detail}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise QualifyError(f"review-context returned invalid JSON: {error}") from error


def _requested_models(
    configs: dict[str, QualifyConfig], models: Sequence[str]
) -> list[QualifyConfig]:
    if not models:
        return [configs[key] for key in sorted(configs)]
    missing = sorted(set(models) - set(configs))
    if missing:
        known = ", ".join(sorted(configs)) or "none"
        raise QualifyError(
            f"no qualify.json for {', '.join(missing)}; declared models: {known}"
        )
    return [configs[key] for key in models]


def _selection(
    config: QualifyConfig,
    changed: Sequence[str],
    forced_scopes: Sequence[str],
    explicit: bool,
) -> dict[str, Any] | None:
    matched: dict[str, list[str]] = {}
    requires: list[list[str]] = []
    trigger_rows: list[dict[str, Any]] = []
    for trigger in config.triggers:
        files = [
            path
            for path in changed
            if any(path_matches(path, prefix) for prefix in trigger.paths)
        ]
        if files:
            matched.setdefault(trigger.scope, []).extend(files)
            requires.append(list(trigger.requires))
            trigger_rows.append(
                {
                    "scope": trigger.scope,
                    "paths": list(trigger.paths),
                    "files": sorted(set(files)),
                    "requires": list(trigger.requires),
                }
            )
    available_scopes = {trigger.scope for trigger in config.triggers}
    for scope in forced_scopes:
        if scope not in available_scopes:
            raise QualifyError(
                f"{config.docs}: unknown scope {scope!r}; available: "
                + ", ".join(sorted(available_scopes))
            )
        if scope in matched:
            continue
        for trigger in config.triggers:
            if trigger.scope == scope:
                matched.setdefault(scope, [])
                requires.append(list(trigger.requires))
                trigger_rows.append(
                    {
                        "scope": trigger.scope,
                        "paths": list(trigger.paths),
                        "files": [],
                        "requires": list(trigger.requires),
                    }
                )
    if not matched:
        if not explicit:
            return None
        raise QualifyError(
            f"{config.docs}: explicitly selected but no trigger matches; "
            "pass --scope to choose requirements"
        )

    order_checks = list(config.checks)
    order_captures = list(config.captures)
    required_checks = ordered_union(
        ([item for item in group if item in config.checks] for group in requires),
        order_checks,
    )
    required_captures = ordered_union(
        ([item for item in group if item in config.captures] for group in requires),
        order_captures,
    )
    baseline_captures = [
        item for item in required_captures if config.captures[item].baseline
    ]
    scopes = sorted(matched)
    warnings = list(config.warnings)
    for scope in scopes:
        if scope in ("arithmetic", "prefill", "speculation") and not required_captures:
            warnings.append(
                f"scope {scope!r} requires no baseline capture; confirm this model "
                "has no two-binary logit harness"
            )
        state_cover = {"state", "model", "serving"}.intersection(required_checks)
        if scope in ("arithmetic", "prefill") and not state_cover:
            warnings.append(
                f"scope {scope!r} does not require state checks; confirm cache and "
                "snapshot behavior is covered elsewhere"
            )
    return {
        "model": config.docs,
        "modelId": config.model_id,
        "source": config.source,
        "category": config.category,
        "config": config.config_rel,
        "scopes": scopes,
        "triggers": trigger_rows,
        "checks": required_checks,
        "captures": required_captures,
        "baselineCaptures": baseline_captures,
        "profiles": {name: list(items) for name, items in config.profiles.items()},
        "itemDescriptions": {
            item: (
                config.checks[item].description
                if item in config.checks
                else config.captures[item].description
            )
            for item in [*required_checks, *required_captures]
        },
        "warnings": warnings,
    }


def _code_files(packet: dict[str, Any]) -> list[str]:
    """Production inference paths that a qualification trigger should cover."""
    result: list[str] = []
    for item in packet.get("files", []):
        for path in (item.get("path"), item.get("old_path")):
            if not path or not path.startswith("src/") or Path(path).suffix not in CODE_SUFFIXES:
                continue
            if path.startswith(API_ONLY_PREFIXES):
                continue
            result.append(path)
    return sorted(set(result))


def build_plan(
    root: Path,
    *,
    base: str | None,
    head: str,
    models: Sequence[str] = (),
    scopes: Sequence[str] = (),
    review_context: Path = REVIEW_CONTEXT,
) -> dict[str, Any]:
    root = root.resolve()
    configs = load_configs(root)
    selected = _requested_models(configs, models)
    packet = _run_review_context(root, base, head, review_context)
    changed = sorted({
        path for item in packet.get("files", [])
        for path in (item.get("path"), item.get("old_path")) if path
    })

    entries: list[dict[str, Any]] = []
    for config in selected:
        explicit = config.docs in models
        entry = _selection(config, changed, scopes, explicit)
        if entry is not None:
            entries.append(entry)

    matched_files: set[str] = set()
    for entry in entries:
        for trigger in entry["triggers"]:
            matched_files.update(trigger["files"])
    uncovered = [path for path in _code_files(packet) if path not in matched_files]

    warnings: list[str] = []
    if packet.get("dirty"):
        warnings.append(
            "working tree is dirty; the packet describes committed changes only"
        )
    if packet.get("untracked"):
        warnings.append(
            f"{len(packet['untracked'])} untracked path(s) are not part of the plan"
        )
    if not entries:
        warnings.append("no qualification config matches the changed paths")
    if uncovered:
        warnings.append(
            "changed code paths without a qualification trigger: "
            + ", ".join(uncovered[:12])
            + (" ..." if len(uncovered) > 12 else "")
        )

    return {
        "artifactType": ARTIFACT_TYPE,
        "schemaVersion": SCHEMA_VERSION,
        "generated": utc_now(),
        "platform": platform_name(),
        "baseRef": packet.get("base_ref"),
        "baseSha": packet.get("base_sha"),
        "headRef": packet.get("head_ref", head),
        "headSha": packet.get("head_sha"),
        "mergeBase": packet.get("merge_base"),
        "dirty": bool(packet.get("dirty")),
        "untracked": list(packet.get("untracked", [])),
        "intent": packet.get("intent", ""),
        "commitSubjects": [
            commit.get("subject", "") for commit in packet.get("commits", [])
        ],
        "models": entries,
        "uncovered": uncovered,
        "warnings": warnings,
    }


def plan_commands(entry: dict[str, Any]) -> list[str]:
    """Human-facing next steps for one model entry."""
    model = entry["model"]
    checks = ", ".join(entry["checks"]) or "none"
    captures = ", ".join(entry["captures"]) or "none"
    baseline = ", ".join(entry["baselineCaptures"]) or "none"
    return [
        f"# {model}: checks [{checks}]; captures [{captures}]; baseline [{baseline}]",
        "python tools/models/qualify.py record --model "
        f"{model} --side baseline --plan <plan.json> --gufo <baseline-gufo> "
        "--file <role>=<path> --tool <name>=<path> --repo <baseline-worktree> "
        "--out <run>/baseline",
        "python tools/models/qualify.py record --model "
        f"{model} --side candidate --plan <plan.json> --gufo <candidate-gufo> "
        "--file <role>=<path> --tool <name>=<path> --repo <candidate-worktree> "
        "--out <run>/candidate",
        "python tools/models/qualify.py compare --model "
        f"{model} --baseline <run>/baseline --candidate <run>/candidate "
        "--out <run>",
    ]


def render_markdown(plan: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Qualification plan")
    lines.append("")
    lines.append(f"Generated: {plan['generated']} ({plan['platform']})")
    lines.append("")
    lines.append(f"- Base: `{plan.get('baseRef')}` (`{str(plan.get('baseSha'))[:12]}`)")
    lines.append(f"- Head: `{plan.get('headRef')}` (`{str(plan.get('headSha'))[:12]}`)")
    lines.append(f"- Merge base: `{str(plan.get('mergeBase'))[:12]}`")
    lines.append(f"- Working tree: {'dirty' if plan['dirty'] else 'clean'}")
    if plan.get("commitSubjects"):
        lines.append("- Commits:")
        for subject in plan["commitSubjects"][:10]:
            lines.append(f"  - {subject}")
        if len(plan["commitSubjects"]) > 10:
            lines.append(f"  - ... and {len(plan['commitSubjects']) - 10} more")
    lines.append("")

    lines.append("## Affected models")
    lines.append("")
    if plan["models"]:
        lines.append("| Model | Scopes | Checks | Captures | Baseline captures |")
        lines.append("| --- | --- | --- | --- | --- |")
        for entry in plan["models"]:
            lines.append(
                "| {model} | {scopes} | {checks} | {captures} | {baseline} |".format(
                    model=entry["model"],
                    scopes=", ".join(f"`{scope}`" for scope in entry["scopes"]),
                    checks=", ".join(f"`{item}`" for item in entry["checks"]) or "-",
                    captures=", ".join(f"`{item}`" for item in entry["captures"]) or "-",
                    baseline=", ".join(
                        f"`{item}`" for item in entry["baselineCaptures"]
                    )
                    or "-",
                )
            )
    else:
        lines.append("_No qualification config matches the changed paths._")
    lines.append("")

    for entry in plan["models"]:
        lines.append(f"## {entry['model']}")
        lines.append("")
        lines.append(entry.get("modelId", ""))
        lines.append("")
        lines.append(f"Config: `{entry['config']}`")
        lines.append("")
        if entry.get("profiles"):
            lines.append(
                "Profiles: "
                + ", ".join(
                    f"`{name}` ({', '.join(items)})"
                    for name, items in entry["profiles"].items()
                )
            )
            lines.append("")
        descriptions = entry.get("itemDescriptions", {})
        if descriptions:
            lines.append("### Required items")
            lines.append("")
            lines.append("| Item | Kind | Description |")
            lines.append("| --- | --- | --- |")
            for item, description in descriptions.items():
                kind = "check" if item in entry["checks"] else "capture"
                lines.append(f"| `{item}` | {kind} | {description} |")
            lines.append("")
        lines.append("### Triggers")
        lines.append("")
        lines.append("| Scope | Changed paths | Requires |")
        lines.append("| --- | --- | --- |")
        for trigger in entry["triggers"]:
            files = ", ".join(f"`{path}`" for path in trigger["files"][:6]) or "(explicit)"
            if len(trigger["files"]) > 6:
                files += f" (+{len(trigger['files']) - 6})"
            lines.append(
                f"| `{trigger['scope']}` | {files} | "
                + ", ".join(f"`{item}`" for item in trigger["requires"])
                + " |"
            )
        lines.append("")
        lines.append("### Next commands")
        lines.append("")
        lines.append("```text")
        lines.extend(plan_commands(entry))
        lines.append("```")
        lines.append("")
        for warning in entry["warnings"]:
            lines.append(f"- Warning: {warning}")
        if entry["warnings"]:
            lines.append("")

    if plan["uncovered"]:
        lines.append("## Uncovered changed code")
        lines.append("")
        lines.append(
            "These changed code paths match no qualify.json trigger. Confirm the "
            "model list or add the shared path to each affected declaration."
        )
        lines.append("")
        for path in plan["uncovered"]:
            lines.append(f"- `{path}`")
        lines.append("")

    if plan["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for warning in plan["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Not verified by this tool")
    lines.append("")
    lines.append(
        "The plan only selects declared work. It does not build Gufo, load a "
        "model, run a check, compare logits or prove that the declaration covers "
        "the change. `record` and `compare` must produce the evidence."
    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"
