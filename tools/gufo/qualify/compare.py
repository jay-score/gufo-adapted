"""Compare baseline and candidate records and retain the qualification verdict."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .config import QualifyConfig, QualifyError, config_for
from .util import (
    atomic_json,
    hash_file,
    load_json,
    run_logit_comparator,
    utc_now,
)

ARTIFACT_TYPE = "gufo-qualify-result"
SCHEMA_VERSION = 1


class CompareError(QualifyError):
    """A user-facing compare error."""


def _record_path(path: Path) -> Path:
    if path.is_dir():
        return path / "record.json"
    return path


def load_record(path: Path) -> dict[str, Any]:
    record = load_json(_record_path(path))
    if record.get("artifactType") != "gufo-qualify-record":
        raise CompareError(f"{_record_path(path)} is not a qualification record")
    return record


def _entry(plan: dict[str, Any], model: str) -> dict[str, Any]:
    for entry in plan.get("models", []):
        if entry.get("model") == model:
            return entry
    raise CompareError(f"{model} is not in the qualification plan")


def _identity_by_name(entries: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {entry["name"]: entry for entry in entries}


def _compare_identity(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_corpora: set[str],
    errors: list[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_files = baseline.get("files", {})
    candidate_files = candidate.get("files", {})
    roles = sorted(set(baseline_files) | set(candidate_files))
    for role in roles:
        left = _identity_by_name(baseline_files.get(role, []))
        right = _identity_by_name(candidate_files.get(role, []))
        names = sorted(set(left) | set(right))
        if not names:
            continue
        if set(left) != set(right):
            errors.append(
                f"file role {role!r} differs between baseline and candidate: "
                f"{', '.join(sorted(left)) or 'missing'} vs "
                f"{', '.join(sorted(right)) or 'missing'}"
            )
            for name in names:
                rows.append(
                    {
                        "kind": "file",
                        "role": role,
                        "name": name,
                        "baseline": left.get(name),
                        "candidate": right.get(name),
                        "match": False,
                    }
                )
            continue
        for name in names:
            item = left[name]
            other = right[name]
            match = (
                item.get("type") == other.get("type")
                and item.get("bytes") == other.get("bytes")
                and item.get("sha256") == other.get("sha256")
            )
            if not match:
                errors.append(
                    f"file role {role!r}: {name!r} differs between baseline and candidate"
                )
            rows.append(
                {
                    "kind": "file",
                    "role": role,
                    "name": name,
                    "baseline": item,
                    "candidate": other,
                    "match": match,
                }
            )
    baseline_corpus = baseline.get("corpus", {})
    candidate_corpus = candidate.get("corpus", {})
    for name in sorted(set(baseline_corpus) | set(candidate_corpus)):
        item = baseline_corpus.get(name)
        other = candidate_corpus.get(name)
        if item is None or other is None:
            if name in baseline_corpora:
                errors.append(
                    f"corpus {name!r} is used by a baseline capture but is "
                    "missing from one record"
                )
            else:
                warnings.append(
                    f"corpus {name!r} is used by a candidate-only capture; "
                    "its baseline identity was not compared"
                )
            rows.append(
                {
                    "kind": "corpus",
                    "role": "corpus",
                    "name": name,
                    "baseline": item,
                    "candidate": other,
                    "match": False,
                }
            )
            continue
        match = (
            item.get("bytes") == other.get("bytes")
            and item.get("sha256") == other.get("sha256")
        )
        if not match:
            errors.append(f"corpus {name!r} differs between baseline and candidate")
        rows.append(
            {
                "kind": "corpus",
                "role": "corpus",
                "name": name,
                "baseline": item,
                "candidate": other,
                "match": match,
            }
        )
    return rows


def _revision_matches(left: str | None, right: str | None) -> bool:
    """Accept full and abbreviated Git revisions for the same commit."""
    if not left or not right:
        return False
    return left == right or left.startswith(right) or right.startswith(left)


def _fingerprint_id(record: dict[str, Any]) -> str | None:
    fingerprint = record.get("binary", {}).get("fingerprint")
    if isinstance(fingerprint, dict):
        value = fingerprint.get("fingerprintId")
        if isinstance(value, str):
            return value
    return None


def _required_lists(
    candidate: dict[str, Any],
    plan: dict[str, Any] | None,
    model: str,
) -> tuple[list[str], list[str], list[str], dict[str, Any] | None]:
    plan_entry = _entry(plan, model) if plan is not None else None
    stored = candidate.get("plan")
    if plan_entry is not None and stored is not None:
        for key in ("checks", "captures", "baselineCaptures"):
            if list(stored.get(key, [])) != list(plan_entry.get(key, [])):
                raise CompareError(
                    f"candidate was recorded against a different plan ({key})"
                )
    if plan_entry is not None:
        return (
            list(plan_entry.get("checks", [])),
            list(plan_entry.get("captures", [])),
            list(plan_entry.get("baselineCaptures", [])),
            plan_entry,
        )
    if stored is not None:
        return (
            list(stored.get("checks", [])),
            list(stored.get("captures", [])),
            list(stored.get("baselineCaptures", [])),
            stored,
        )
    return [], [], [], None


def _compare_outputs(
    config: QualifyConfig,
    capture_id: str,
    baseline_item: dict[str, Any],
    candidate_item: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    capture = config.captures[capture_id]
    comparator = capture.compare
    left = {
        output["logical"]: output for output in baseline_item.get("outputs", [])
    }
    right = {
        output["logical"]: output for output in candidate_item.get("outputs", [])
    }
    names = sorted(set(left) | set(right))
    if not names:
        errors.append(f"capture {capture_id!r}: no declared outputs were recorded")
        return []
    rows: list[dict[str, Any]] = []
    for name in names:
        item = left.get(name)
        other = right.get(name)
        if item is None or other is None:
            errors.append(
                f"capture {capture_id!r}: output {name!r} is missing on one side"
            )
            rows.append(
                {
                    "capture": capture_id,
                    "output": name,
                    "comparator": comparator,
                    "result": "missing",
                    "baseline": item,
                    "candidate": other,
                }
            )
            continue
        if comparator == "none":
            warnings.append(
                f"capture {capture_id!r} has no comparator; {name!r} was not compared"
            )
            rows.append(
                {
                    "capture": capture_id,
                    "output": name,
                    "comparator": comparator,
                    "result": "not-compared",
                    "baseline": item,
                    "candidate": other,
                }
            )
            continue
        if comparator == "file-sha256":
            match = (
                item.get("bytes") == other.get("bytes")
                and item.get("sha256") == other.get("sha256")
            )
            result = "match" if match else "differ"
            if not match:
                errors.append(
                    f"capture {capture_id!r}: {name!r} differs between baseline and candidate"
                )
            rows.append(
                {
                    "capture": capture_id,
                    "output": name,
                    "comparator": comparator,
                    "result": result,
                    "baseline": item,
                    "candidate": other,
                }
            )
            continue
        if comparator == "logit-sha256":
            outcome = run_logit_comparator(
                Path(item["path"]), Path(other["path"])
            )
            if outcome["result"] != "match":
                errors.append(
                    f"capture {capture_id!r}: {name!r} {outcome['result']}: "
                    f"{outcome['detail']}"
                )
            rows.append(
                {
                    "capture": capture_id,
                    "output": name,
                    "comparator": comparator,
                    "result": outcome["result"],
                    "detail": outcome["detail"],
                    "baseline": item,
                    "candidate": other,
                }
            )
            continue
        raise CompareError(f"unknown comparator {comparator!r}")
    return rows


def compare(
    root: Path,
    *,
    model: str,
    baseline_path: Path,
    candidate_path: Path,
    out: Path,
    plan_path: Path | None,
    control: bool,
    allow_fingerprint_mismatch: bool,
    allow_source_mismatch: bool,
    allow_baseline_mismatch: bool,
) -> dict[str, Any]:
    root = root.resolve()
    config = config_for(root, model)
    baseline = load_record(baseline_path)
    candidate = load_record(candidate_path)
    if baseline.get("side") != "baseline":
        raise CompareError("the baseline record was not written with --side baseline")
    if candidate.get("side") != "candidate":
        raise CompareError("the candidate record was not written with --side candidate")
    if baseline.get("model", {}).get("docs") != model:
        raise CompareError(
            f"baseline record is for {baseline.get('model', {}).get('docs')!r}, not {model!r}"
        )
    if candidate.get("model", {}).get("docs") != model:
        raise CompareError(
            f"candidate record is for {candidate.get('model', {}).get('docs')!r}, not {model!r}"
        )

    plan = load_json(plan_path) if plan_path is not None else None
    if plan is not None and plan.get("artifactType") != "gufo-qualify-plan":
        raise CompareError(f"{plan_path} is not a qualification plan")

    errors: list[str] = []
    warnings: list[str] = []
    unverified: list[str] = []

    current_config = hash_file(config.path)
    for label, record in (("baseline", baseline), ("candidate", candidate)):
        declared = record.get("config", {})
        if declared.get("sha256") != current_config["sha256"]:
            errors.append(
                f"{label} record used a different qualify.json "
                f"({declared.get('sha256')} != {current_config['sha256']})"
            )

    required_checks, required_captures, baseline_captures, plan_entry = _required_lists(
        candidate, plan, model
    )
    if plan is None and candidate.get("plan") is None:
        selected = candidate.get("selected", {})
        required_checks = list(selected.get("checks", []))
        required_captures = list(selected.get("captures", []))
        baseline_captures = [
            item_id
            for item_id in required_captures
            if item_id in config.captures and config.captures[item_id].baseline
        ]
        unverified.append(
            "no qualification plan was recorded; coverage is limited to the "
            "recorded selection"
        )

    baseline_corpora: set[str] = set()
    for item_id in baseline_captures:
        item = baseline.get("items", {}).get(f"capture:{item_id}")
        if item is not None:
            baseline_corpora.update(item.get("corpora", []))
    identity = _compare_identity(
        baseline, candidate, baseline_corpora, errors, warnings
    )

    baseline_fp = _fingerprint_id(baseline)
    candidate_fp = _fingerprint_id(candidate)
    if baseline_fp is None or candidate_fp is None:
        unverified.append(
            "machine fingerprint is missing on at least one side; numerical "
            "agreement across different drivers or hardware is not established"
        )
    elif baseline_fp != candidate_fp:
        message = (
            f"machine fingerprints differ ({baseline_fp} != {candidate_fp}); "
            "compare on one host/toolchain or state why the difference is irrelevant"
        )
        if allow_fingerprint_mismatch:
            unverified.append(message + " (explicitly allowed)")
        else:
            errors.append(message)

    if baseline.get("binary", {}).get("sha256") == candidate.get("binary", {}).get("sha256"):
        if control:
            warnings.append(
                "baseline and candidate are the same binary; this is a harness "
                "control, not evidence that a change is qualified"
            )
        else:
            errors.append(
                "baseline and candidate binaries are identical; a qualification "
                "needs two builds (use --control only for a harness self-check)"
            )
    elif control:
        errors.append("--control requires the same binary on both sides")

    if plan is not None:
        plan_head = plan.get("headSha")
        candidate_revision = candidate.get("source", {}).get("revision")
        if plan_head and not _revision_matches(candidate_revision, plan_head):
            message = (
                f"candidate revision {candidate_revision} "
                f"is not the planned head {plan_head}"
            )
            if allow_source_mismatch:
                unverified.append(message + " (explicitly allowed)")
            else:
                errors.append(message)
        plan_base = plan.get("baseSha")
        merge_base = plan.get("mergeBase")
        baseline_revision = baseline.get("source", {}).get("revision")
        if plan_base and not (
            _revision_matches(baseline_revision, plan_base)
            or _revision_matches(baseline_revision, merge_base)
        ):
            message = (
                f"baseline revision {baseline_revision} is neither the planned "
                f"base {plan_base} nor its merge base {merge_base}"
            )
            if allow_baseline_mismatch:
                unverified.append(message + " (explicitly allowed)")
            else:
                errors.append(message)
        elif (
            plan_base
            and _revision_matches(baseline_revision, merge_base)
            and not _revision_matches(merge_base, plan_base)
        ):
            warnings.append(
                "baseline was built at the merge base, not the planned base; "
                "confirm the control is the intended one"
            )

    checks: list[dict[str, Any]] = []
    for item_id in required_checks:
        item = candidate.get("items", {}).get(f"check:{item_id}")
        if item is None:
            errors.append(f"required check {item_id!r} is missing from the candidate record")
            checks.append({"id": item_id, "status": "missing"})
            continue
        status = item.get("status")
        if status != "pass":
            errors.append(
                f"required check {item_id!r} did not pass ({status}"
                + (f": {item.get('reason')}" if item.get("reason") else "")
                + ")"
            )
        checks.append(
            {
                "id": item_id,
                "description": item.get("description"),
                "status": status,
                "reason": item.get("reason"),
                "commands": item.get("commands", []),
            }
        )

    captures: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for item_id in required_captures:
        item = candidate.get("items", {}).get(f"capture:{item_id}")
        if item is None:
            errors.append(f"required capture {item_id!r} is missing from the candidate record")
            captures.append({"id": item_id, "status": "missing"})
            continue
        status = item.get("status")
        if status != "pass":
            errors.append(
                f"required capture {item_id!r} did not pass ({status}"
                + (f": {item.get('reason')}" if item.get("reason") else "")
                + ")"
            )
        captures.append(
            {
                "id": item_id,
                "description": item.get("description"),
                "status": status,
                "reason": item.get("reason"),
                "commands": item.get("commands", []),
                "metrics": item.get("metrics", {}),
                "tools": item.get("tools", []),
            }
        )
        if item_id in baseline_captures:
            baseline_item = baseline.get("items", {}).get(f"capture:{item_id}")
            if baseline_item is None:
                errors.append(
                    f"baseline capture {item_id!r} is missing from the baseline record"
                )
                continue
            if baseline_item.get("status") != "pass":
                errors.append(
                    f"baseline capture {item_id!r} did not pass "
                    f"({baseline_item.get('status')})"
                )
                continue
            if status != "pass":
                continue
            for tool in baseline_item.get("tools", []):
                if tool not in baseline.get("tools", {}):
                    errors.append(
                        f"baseline capture {item_id!r}: tool {tool!r} was not recorded"
                    )
            for tool in item.get("tools", []):
                if tool not in candidate.get("tools", {}):
                    errors.append(
                        f"candidate capture {item_id!r}: tool {tool!r} was not recorded"
                    )
            comparisons.extend(
                _compare_outputs(
                    config,
                    item_id,
                    baseline_item,
                    item,
                    errors,
                    warnings,
                )
            )

    limits: list[dict[str, Any]] = []
    for limit in config.limits:
        if limit.capture not in required_captures:
            continue
        item = candidate.get("items", {}).get(f"capture:{limit.capture}")
        if item is None or item.get("status") != "pass":
            # Coverage already reports the missing or failed capture.
            continue
        value = item.get("metrics", {}).get(limit.metric)
        if value is None:
            errors.append(
                f"limit for {limit.capture!r}.{limit.metric} cannot be evaluated: "
                "the command did not report that metric"
            )
            limits.append(
                {
                    "capture": limit.capture,
                    "metric": limit.metric,
                    "max": limit.maximum,
                    "min": limit.minimum,
                    "source": limit.source,
                    "candidate": None,
                    "baseline": None,
                    "pass": False,
                }
            )
            continue
        baseline_item = baseline.get("items", {}).get(f"capture:{limit.capture}")
        baseline_value = (
            baseline_item.get("metrics", {}).get(limit.metric)
            if baseline_item
            else None
        )
        passed = (
            value <= limit.maximum
            if limit.maximum is not None
            else value >= limit.minimum  # type: ignore[operator]
        )
        if not passed:
            errors.append(
                f"capture {limit.capture!r}.{limit.metric}={value} violates "
                f"{'max' if limit.maximum is not None else 'min'} "
                f"{limit.maximum if limit.maximum is not None else limit.minimum} "
                f"({limit.source})"
            )
        limits.append(
            {
                "capture": limit.capture,
                "metric": limit.metric,
                "max": limit.maximum,
                "min": limit.minimum,
                "source": limit.source,
                "candidate": value,
                "baseline": baseline_value,
                "pass": passed,
            }
        )

    if errors:
        status = "FAIL"
    elif control:
        status = "CONTROL"
    elif unverified:
        status = "UNVERIFIED"
    else:
        status = "PASS"

    result: dict[str, Any] = {
        "artifactType": ARTIFACT_TYPE,
        "schemaVersion": SCHEMA_VERSION,
        "generated": utc_now(),
        "status": status,
        "control": control,
        "model": {
            "id": config.model_id,
            "docs": config.docs,
            "source": config.source,
            "category": config.category,
        },
        "config": {
            "path": config.config_rel,
            "sha256": current_config["sha256"],
        },
        "plan": None
        if plan_entry is None
        else {
            "baseSha": (plan or candidate.get("plan") or {}).get("baseSha"),
            "headSha": (plan or candidate.get("plan") or {}).get("headSha"),
            "mergeBase": (plan or candidate.get("plan") or {}).get("mergeBase"),
            "checks": required_checks,
            "captures": required_captures,
            "baselineCaptures": baseline_captures,
            "scopes": list(plan_entry.get("scopes", [])),
        },
        "baseline": {
            "record": str(_record_path(baseline_path).resolve()),
            "source": baseline.get("source"),
            "binary": baseline.get("binary"),
            "platform": baseline.get("platform"),
        },
        "candidate": {
            "record": str(_record_path(candidate_path).resolve()),
            "source": candidate.get("source"),
            "binary": candidate.get("binary"),
            "platform": candidate.get("platform"),
        },
        "identity": {
            "files": identity,
            "fingerprint": {
                "baseline": baseline_fp,
                "candidate": candidate_fp,
                "match": baseline_fp == candidate_fp and baseline_fp is not None,
            },
        },
        "coverage": {
            "requiredChecks": required_checks,
            "requiredCaptures": required_captures,
            "baselineCaptures": baseline_captures,
        },
        "checks": checks,
        "captures": captures,
        "comparisons": comparisons,
        "limits": limits,
        "errors": errors,
        "warnings": warnings,
        "unverified": unverified,
    }
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "qualification.json"
    atomic_json(result_path, result)
    from .report import render_markdown

    (out / "qualification.md").write_text(
        render_markdown(result, root), encoding="utf-8", newline="\n"
    )
    print(f"status: {status}")
    for message in errors:
        print(f"error: {message}")
    for message in unverified:
        print(f"unverified: {message}")
    for message in warnings:
        print(f"warning: {message}")
    print(f"result: {result_path}")
    result["resultPath"] = str(result_path)
    return result
