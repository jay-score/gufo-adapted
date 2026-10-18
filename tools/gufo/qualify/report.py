"""Render a qualification result as a concise Markdown evidence report."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .util import redact_command


def _status_text(status: str) -> str:
    return {
        "PASS": "PASS - required checks and comparisons passed.",
        "FAIL": "FAIL - at least one required check, comparison or limit failed.",
        "UNVERIFIED": "UNVERIFIED - no failure was observed, but a required identity or coverage claim is missing.",
        "CONTROL": "CONTROL - the same binary ran on both sides; this validates the harness, not a change.",
    }.get(status, status)


def _sha(value: Any) -> str:
    return "-" if not value else str(value)[:12]


def _record_label(value: Any) -> str:
    if not value:
        return "-"
    path = Path(str(value))
    return f"{path.parent.name}/{path.name}"


def _artifacts(result: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for side in ("baseline", "candidate"):
        record = result.get(side, {})
        binary = record.get("binary", {})
        if binary.get("path"):
            paths.append(Path(binary["path"]))
    for identity in result.get("identity", {}).get("files", []):
        for side in ("baseline", "candidate"):
            item = identity.get(side)
            if isinstance(item, dict) and item.get("path"):
                paths.append(Path(item["path"]))
    return paths


def _commands(
    commands: Iterable[dict[str, Any]], root: Path, artifacts: list[Path]
) -> list[str]:
    lines: list[str] = []
    for command in commands:
        argv = command.get("argv")
        if not argv:
            continue
        lines.append(" ".join(redact_command(argv, root, artifacts)))
    return lines


def render_markdown(result: dict[str, Any], root: Path) -> str:
    root = root.resolve()
    artifacts = _artifacts(result)
    status = result.get("status", "FAIL")
    lines: list[str] = []
    lines.append(f"# Qualification: {result.get('model', {}).get('id', '')}")
    lines.append("")
    lines.append(f"**Status: {status}.** {_status_text(status)}")
    lines.append("")
    lines.append(f"Generated: {result.get('generated')}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| | Baseline | Candidate |")
    lines.append("| --- | --- | --- |")
    baseline = result.get("baseline", {})
    candidate = result.get("candidate", {})
    lines.append(
        f"| Source revision | `{_sha(baseline.get('source', {}).get('revision'))}`"
        f"{' (dirty)' if baseline.get('source', {}).get('dirty') else ''} | "
        f"`{_sha(candidate.get('source', {}).get('revision'))}`"
        f"{' (dirty)' if candidate.get('source', {}).get('dirty') else ''} |"
    )
    lines.append(
        f"| Binary SHA-256 | `{_sha(baseline.get('binary', {}).get('sha256'))}` | "
        f"`{_sha(candidate.get('binary', {}).get('sha256'))}` |"
    )
    lines.append(
        f"| Binary version | {baseline.get('binary', {}).get('version') or '-'} | "
        f"{candidate.get('binary', {}).get('version') or '-'} |"
    )
    fingerprint = result.get("identity", {}).get("fingerprint", {})
    lines.append(
        f"| Fingerprint | `{_sha(fingerprint.get('baseline'))}` | "
        f"`{_sha(fingerprint.get('candidate'))}` |"
    )
    lines.append(
        f"| Platform | {baseline.get('platform') or '-'} | "
        f"{candidate.get('platform') or '-'} |"
    )
    lines.append("")

    plan = result.get("plan")
    if plan:
        lines.append("## Scope")
        lines.append("")
        lines.append(
            f"- Plan: base `{_sha(plan.get('baseSha'))}`, head "
            f"`{_sha(plan.get('headSha'))}`, merge base "
            f"`{_sha(plan.get('mergeBase'))}`"
        )
        lines.append(
            "- Scopes: "
            + (", ".join(f"`{scope}`" for scope in plan.get("scopes", [])) or "-")
        )
        lines.append("")

    lines.append("## Coverage")
    lines.append("")
    coverage = result.get("coverage", {})
    lines.append(
        "- Required checks: "
        + (", ".join(f"`{item}`" for item in coverage.get("requiredChecks", [])) or "none")
    )
    lines.append(
        "- Required captures: "
        + (", ".join(f"`{item}`" for item in coverage.get("requiredCaptures", [])) or "none")
    )
    lines.append(
        "- Baseline captures: "
        + (", ".join(f"`{item}`" for item in coverage.get("baselineCaptures", [])) or "none")
    )
    lines.append("")

    lines.append("## Identity")
    lines.append("")
    identity_rows = [
        item
        for item in result.get("identity", {}).get("files", [])
        if item.get("kind") in ("file", "corpus")
    ]
    if identity_rows:
        lines.append("| Kind | Role | Artifact | Bytes | SHA-256 | Match |")
        lines.append("| --- | --- | --- | ---: | --- | --- |")
        for item in identity_rows:
            baseline_item = item.get("baseline") or {}
            candidate_item = item.get("candidate") or {}
            value = baseline_item or candidate_item
            lines.append(
                f"| {item.get('kind')} | `{item.get('role')}` | "
                f"`{item.get('name')}` | {value.get('bytes', '-')} | "
                f"`{_sha(value.get('sha256'))}` | "
                f"{'yes' if item.get('match') else 'NO'} |"
            )
    else:
        lines.append("_No model-file or corpus identity was recorded._")
    lines.append("")

    lines.append("## Checks")
    lines.append("")
    checks = result.get("checks", [])
    if checks:
        lines.append("| Check | Status | Command |")
        lines.append("| --- | --- | --- |")
        for item in checks:
            command = " ; ".join(_commands(item.get("commands", []), root, artifacts))
            lines.append(
                f"| `{item.get('id')}` | {item.get('status')} | "
                f"`{command}` |"
            )
    else:
        lines.append("_No checks were required._")
    lines.append("")

    lines.append("## Captures and comparisons")
    lines.append("")
    captures = result.get("captures", [])
    if captures:
        lines.append("| Capture | Status | Metrics |")
        lines.append("| --- | --- | --- |")
        for item in captures:
            metrics = ", ".join(
                f"{name}={value}" for name, value in sorted(item.get("metrics", {}).items())
            )
            lines.append(
                f"| `{item.get('id')}` | {item.get('status')} | {metrics or '-'} |"
            )
    else:
        lines.append("_No captures were required._")
    lines.append("")
    comparisons = result.get("comparisons", [])
    if comparisons:
        lines.append("| Capture | Output | Comparator | Result | Baseline | Candidate |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for item in comparisons:
            baseline_item = item.get("baseline") or {}
            candidate_item = item.get("candidate") or {}
            lines.append(
                f"| `{item.get('capture')}` | `{item.get('output')}` | "
                f"`{item.get('comparator')}` | **{item.get('result')}** | "
                f"`{_sha(baseline_item.get('sha256'))}` | "
                f"`{_sha(candidate_item.get('sha256'))}` |"
            )
        details = [
            item
            for item in comparisons
            if item.get("result") not in ("match", "not-compared")
        ]
        for item in details:
            if item.get("detail"):
                lines.append("")
                lines.append(
                    f"> `{item.get('capture')}` / `{item.get('output')}`: "
                    f"{item.get('detail')}"
                )
    lines.append("")

    limits = result.get("limits", [])
    if limits:
        lines.append("## Declared limits")
        lines.append("")
        lines.append("| Capture | Metric | Bound | Baseline | Candidate | Result | Source |")
        lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
        for item in limits:
            bound = (
                f"max {item.get('max')}"
                if item.get("max") is not None
                else f"min {item.get('min')}"
            )
            lines.append(
                f"| `{item.get('capture')}` | `{item.get('metric')}` | {bound} | "
                f"{item.get('baseline')} | {item.get('candidate')} | "
                f"{'PASS' if item.get('pass') else 'FAIL'} | {item.get('source')} |"
            )
        lines.append("")

    lines.append("## Not verified")
    lines.append("")
    unverified = result.get("unverified", [])
    if unverified:
        for item in unverified:
            lines.append(f"- {item}")
    else:
        lines.append("- Nothing declared unverified.")
    lines.append("")

    if result.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for item in result["warnings"]:
            lines.append(f"- {item}")
        lines.append("")

    if result.get("errors"):
        lines.append("## Errors")
        lines.append("")
        for item in result["errors"]:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append(f"- Config: `{result.get('config', {}).get('path')}` "
                 f"(`{_sha(result.get('config', {}).get('sha256'))}`)")
    lines.append(
        f"- Baseline record: `{_record_label(baseline.get('record'))}`"
    )
    lines.append(
        f"- Candidate record: `{_record_label(candidate.get('record'))}`"
    )
    if plan:
        lines.append(
            f"- Plan: base `{_sha(plan.get('baseSha'))}`, head "
            f"`{_sha(plan.get('headSha'))}`"
        )
    lines.append("")
    command_lines: list[str] = []
    for item in (result.get("checks") or []) + (result.get("captures") or []):
        command_lines.extend(_commands(item.get("commands", []), root, artifacts))
    if command_lines:
        lines.append("```text")
        lines.extend(command_lines)
        lines.append("```")
        lines.append("")

    lines.append("## Limits of this report")
    lines.append("")
    lines.append(
        "The record JSON retains the exact commands and paths. This report "
        "reduces private absolute paths to artifact names."
    )
    lines.append("")
    lines.append(
        "A PASS requires exact equality for unchanged arithmetic or a declared "
        "model-owned limit for an intentional change. Matching two Gufo builds "
        "does not establish original-model accuracy, and a skipped check is not "
        "a pass. Keep raw logit dumps outside Git; this report retains command, "
        "artifact and identity evidence only."
    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"
