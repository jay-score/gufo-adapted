"""Run one side of a qualification and retain provenance plus command evidence."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import (
    Capture,
    Check,
    Command,
    QualifyConfig,
    QualifyError,
    config_for,
    ordered_union,
)
from .util import (
    atomic_json,
    display_path,
    executable_suffix,
    git_identity,
    hash_file,
    hash_path,
    parse_metrics,
    platform_name,
    utc_now,
)

ARTIFACT_TYPE = "gufo-qualify-record"
SCHEMA_VERSION = 1
SKIP_RETURN_CODE = 77

_TOKEN_RE = re.compile(r"\{([^{}]+)\}")
_SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$")


class RecordError(QualifyError):
    """A user-facing record error."""


@dataclass
class RecordContext:
    root: Path
    out: Path
    gufo: Path
    repo: Path
    config: QualifyConfig
    files: dict[str, list[Path]]
    tools: dict[str, Path]
    corpus: dict[str, Path]
    env: dict[str, str]
    timeout: float


@dataclass
class ItemResult:
    kind: str
    id: str
    status: str
    description: str
    commands: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    toolsUsed: list[str] = field(default_factory=list)
    corporaUsed: list[str] = field(default_factory=list)
    reason: str | None = None


def _expand_shards(paths: Iterable[Path]) -> list[Path]:
    """Expand a first GGUF shard to every sibling declared by its name."""
    expanded: list[Path] = []
    for path in paths:
        match = _SHARD_RE.match(path.name)
        if not match:
            expanded.append(path)
            continue
        count = int(match.group("count"))
        prefix = match.group("prefix")
        for index in range(1, count + 1):
            shard = path.parent / f"{prefix}-{index:05d}-of-{count:05d}.gguf"
            if not shard.is_file():
                raise RecordError(
                    f"sharded GGUF is incomplete: missing {shard.name} "
                    f"(from {path.name})"
                )
            expanded.append(shard)
    result: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def parse_file_args(values: Sequence[str]) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {}
    for value in values:
        role, separator, raw = value.partition("=")
        if not separator or not role or not raw:
            raise RecordError(f"--file expects ROLE=PATH, got {value!r}")
        path = Path(raw).expanduser()
        if not path.exists():
            raise RecordError(f"--file {role}={raw}: path does not exist")
        files.setdefault(role, []).append(path)
    return {role: _expand_shards(paths) for role, paths in files.items()}


def parse_tool_args(values: Sequence[str]) -> dict[str, Path]:
    tools: dict[str, Path] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name or not raw:
            raise RecordError(f"--tool expects NAME=PATH, got {value!r}")
        path = Path(raw).expanduser()
        if not path.exists():
            raise RecordError(f"--tool {name}={raw}: path does not exist")
        tools[name] = path.resolve()
    return tools


def parse_env_args(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name:
            raise RecordError(f"--env expects NAME=VALUE, got {value!r}")
        result[name] = raw
    return result


def _parse_ids(raw: Sequence[str] | None, available: Sequence[str]) -> list[str] | None:
    if not raw:
        return None
    requested: set[str] = set()
    for value in raw:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if item == "all":
                requested.update(available)
            else:
                requested.add(item)
    unknown = sorted(requested - set(available))
    if unknown:
        raise RecordError(
            f"unknown item(s): {', '.join(unknown)}; available: {', '.join(available)}"
        )
    return [item for item in available if item in requested]


def _find_plan_entry(plan: dict[str, Any], model: str) -> dict[str, Any]:
    for entry in plan.get("models", []):
        if entry.get("model") == model:
            return entry
    raise RecordError(f"{model} is not in the qualification plan")


def _used_placeholders(
    commands: Iterable[Command],
) -> tuple[set[str], set[str], set[str]]:
    files: set[str] = set()
    corpus: set[str] = set()
    tools: set[str] = set()
    for command in commands:
        for part in command.argv:
            for raw in _TOKEN_RE.findall(part):
                kind, _, name = raw.partition(":")
                if kind == "file" or kind == "files":
                    files.add(name)
                elif kind == "corpus":
                    corpus.add(name)
                elif kind == "tool":
                    tools.add(name)
        for value in command.env.values():
            for raw in _TOKEN_RE.findall(value):
                kind, _, name = raw.partition(":")
                if kind == "file" or kind == "files":
                    files.add(name)
                elif kind == "corpus":
                    corpus.add(name)
                elif kind == "tool":
                    tools.add(name)
    return files, corpus, tools


def _tools_used(commands: Iterable[Command]) -> list[str]:
    return sorted(_used_placeholders(commands)[2])


def _corpora_used(commands: Iterable[Command]) -> list[str]:
    return sorted(_used_placeholders(commands)[1])


def _find_test_executable(name: str, ctx: RecordContext) -> Path:
    """Locate a gpu-test executable without hard-coding the platform build layout."""
    suffix = executable_suffix()
    candidates: list[Path] = []
    for root in (ctx.repo, ctx.root):
        build = root / "build" / "gpu-test"
        if not build.is_dir():
            continue
        direct = build / f"{name}{suffix}"
        if direct.is_file():
            candidates.append(direct)
        candidates.extend(build.rglob(f"{name}{suffix}"))
    unique = sorted(
        {path.resolve() for path in candidates},
        key=lambda path: (len(path.parts), str(path)),
    )
    if not unique:
        raise RecordError(
            f"cannot find gpu-test executable {name!r} under "
            f"{ctx.repo / 'build' / 'gpu-test'} or "
            f"{ctx.root / 'build' / 'gpu-test'}"
        )
    return unique[0]


def _resolve_text(
    text: str,
    ctx: RecordContext,
    item_out: Path,
) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        if raw == "gufo":
            return str(ctx.gufo)
        if raw == "python":
            return sys.executable
        if raw == "repo":
            return str(ctx.repo)
        if raw == "out":
            return str(item_out)
        if raw == "exe":
            return executable_suffix()
        if raw == "docs_model":
            return ctx.config.docs
        if raw == "source":
            return ctx.config.source
        if raw == "config":
            return str(ctx.config.path)
        kind, separator, name = raw.partition(":")
        if not separator:
            raise RecordError(f"unknown placeholder {{{raw}}}")
        if kind == "file":
            paths = ctx.files.get(name)
            if not paths:
                raise RecordError(f"missing --file {name}=PATH for {{{raw}}}")
            return str(paths[0])
        if kind == "corpus":
            path = ctx.corpus.get(name)
            if path is None:
                raise RecordError(f"undeclared corpus {name!r}")
            return str(path)
        if kind == "tool":
            path = ctx.tools.get(name)
            if path is None:
                raise RecordError(f"missing --tool {name}=PATH for {{{raw}}}")
            return str(path)
        if kind == "test":
            return str(_find_test_executable(name, ctx))
        if kind == "env":
            value = ctx.env.get(name)
            if value is None:
                raise RecordError(f"environment variable {name!r} is not set")
            return value
        raise RecordError(f"unknown placeholder {{{raw}}}")

    return _TOKEN_RE.sub(replace, text)


def _resolve_argv(argv: Sequence[str], ctx: RecordContext, item_out: Path) -> list[str]:
    resolved: list[str] = []
    for part in argv:
        whole = re.fullmatch(r"\{files:([^{}]+)\}", part)
        if whole:
            paths = ctx.files.get(whole.group(1))
            if not paths:
                raise RecordError(f"missing --file {whole.group(1)}=PATH")
            resolved.extend(str(path) for path in paths)
            continue
        resolved.append(_resolve_text(part, ctx, item_out))
    return resolved


def _log_path(out: Path, kind: str, item_id: str, index: int, suffix: str) -> Path:
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs / f"{kind}-{item_id}-{index}.{suffix}"


def _run_command(
    command: Command,
    ctx: RecordContext,
    *,
    kind: str,
    item_id: str,
    index: int,
    item_out: Path,
) -> dict[str, Any]:
    if platform_name() not in command.platforms:
        return {
            "argv": list(command.argv),
            "status": "not-applicable",
            "reason": f"declared only for {', '.join(command.platforms)}",
        }
    try:
        argv = _resolve_argv(command.argv, ctx, item_out)
        declared_env = {
            key: _resolve_text(value, ctx, item_out)
            for key, value in command.env.items()
        }
        env = os.environ.copy()
        env.update(declared_env)
        cwd = (ctx.root / _resolve_text(command.cwd, ctx, item_out)).resolve()
    except RecordError as error:
        return {"argv": list(command.argv), "status": "error", "reason": str(error)}
    if cwd.is_dir() is False:
        return {
            "argv": list(command.argv),
            "status": "error",
            "reason": f"working directory does not exist: {cwd}",
        }
    timeout = command.timeout or ctx.timeout
    started = utc_now()
    monotonic_start = time.monotonic()
    print(f"  $ {' '.join(argv)}", flush=True)
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        _log_path(ctx.out, kind, item_id, index, "stdout.txt").write_text(
            stdout, encoding="utf-8", newline="\n"
        )
        _log_path(ctx.out, kind, item_id, index, "stderr.txt").write_text(
            stderr, encoding="utf-8", newline="\n"
        )
        return {
            "argv": argv,
            "cwd": str(cwd),
            "env": declared_env,
            "status": "timeout",
            "started": started,
            "finished": utc_now(),
            "durationSeconds": round(time.monotonic() - monotonic_start, 3),
            "timeoutSeconds": timeout,
            "reason": f"command exceeded {timeout:.0f} s",
        }
    except OSError as error:
        return {
            "argv": argv,
            "cwd": str(cwd),
            "env": declared_env,
            "status": "error",
            "started": started,
            "finished": utc_now(),
            "durationSeconds": round(time.monotonic() - monotonic_start, 3),
            "reason": f"cannot execute: {error}",
        }
    stdout_path = _log_path(ctx.out, kind, item_id, index, "stdout.txt")
    stderr_path = _log_path(ctx.out, kind, item_id, index, "stderr.txt")
    stdout_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
    stderr_path.write_text(completed.stderr, encoding="utf-8", newline="\n")
    if completed.returncode == 0:
        status = "pass"
    elif completed.returncode == SKIP_RETURN_CODE:
        status = "skipped"
    else:
        status = "fail"
    return {
        "argv": argv,
        "cwd": str(cwd),
        "env": declared_env,
        "status": status,
        "exitCode": completed.returncode,
        "started": started,
        "finished": utc_now(),
        "durationSeconds": round(time.monotonic() - monotonic_start, 3),
        "stdout": stdout_path.relative_to(ctx.out).as_posix(),
        "stderr": stderr_path.relative_to(ctx.out).as_posix(),
        "stdoutTail": completed.stdout[-4000:],
        "stderrTail": completed.stderr[-4000:],
        "metrics": parse_metrics(completed.stdout + "\n" + completed.stderr),
    }


def _run_item(
    item: Check | Capture,
    kind: str,
    ctx: RecordContext,
) -> ItemResult:
    result = ItemResult(kind, item.id, "pass", item.description)
    item_out = ctx.out / ("captures" if kind == "capture" else "checks") / item.id
    item_out.mkdir(parents=True, exist_ok=True)
    print(f"== {kind} {item.id}: {item.description}", flush=True)
    result.toolsUsed = _tools_used(item.commands)
    result.corporaUsed = _corpora_used(item.commands)
    for index, command in enumerate(item.commands):
        record = _run_command(
            command,
            ctx,
            kind=kind,
            item_id=item.id,
            index=index,
            item_out=item_out,
        )
        result.commands.append(record)
        result.metrics.update(record.get("metrics", {}))
        status = record["status"]
        if status == "pass":
            continue
        if status == "not-applicable" and command.optional:
            print(
                f"  -> not-applicable (optional): {record.get('reason')}",
                flush=True,
            )
            continue
        result.status = status
        result.reason = record.get("reason")
        print(f"  -> {status}" + (f": {result.reason}" if result.reason else ""), flush=True)
        return result

    if isinstance(item, Capture):
        for pattern in item.outputs:
            try:
                resolved = Path(_resolve_text(pattern, ctx, item_out))
            except RecordError as error:
                result.status = "error"
                result.reason = str(error)
                return result
            if not resolved.exists():
                result.status = "fail"
                result.reason = f"declared output was not produced: {resolved.name}"
                return result
            identity = hash_path(resolved)
            try:
                logical = resolved.relative_to(item_out).as_posix()
            except ValueError:
                logical = resolved.name
            identity["logical"] = logical
            identity["path"] = str(resolved)
            result.outputs.append(identity)
    print(
        f"  -> {result.status}"
        + (f" ({len(result.outputs)} output(s))" if result.outputs else ""),
        flush=True,
    )
    return result


def _item_record(result: ItemResult) -> dict[str, Any]:
    return {
        "kind": result.kind,
        "id": result.id,
        "description": result.description,
        "status": result.status,
        "reason": result.reason,
        "tools": result.toolsUsed,
        "corpora": result.corporaUsed,
        "commands": result.commands,
        "outputs": result.outputs,
        "metrics": result.metrics,
    }


def record(
    root: Path,
    *,
    model: str,
    side: str,
    gufo: Path,
    out: Path,
    plan_path: Path | None,
    captures: Sequence[str] | None,
    checks: Sequence[str] | None,
    profiles: Sequence[str] | None,
    files: dict[str, list[Path]],
    tools: dict[str, Path],
    env: dict[str, str],
    repo: Path,
    fingerprint_path: Path | None,
    timeout: float,
    revision: str | None,
    source_dirty: bool | None,
) -> Path:
    if side not in ("baseline", "candidate"):
        raise RecordError("--side must be baseline or candidate")
    root = root.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = config_for(root, model)

    plan_entry: dict[str, Any] | None = None
    plan_data: dict[str, Any] | None = None
    required_checks: list[str] = []
    required_captures: list[str] = []
    baseline_captures: list[str] = []
    if plan_path is not None:
        try:
            plan_data = json.loads(plan_path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError as error:
            raise RecordError(f"missing plan: {plan_path}") from error
        except json.JSONDecodeError as error:
            raise RecordError(f"invalid plan JSON in {plan_path}: {error}") from error
        plan_entry = _find_plan_entry(plan_data, model)
        required_checks = list(plan_entry.get("checks", []))
        required_captures = list(plan_entry.get("captures", []))
        baseline_captures = list(plan_entry.get("baselineCaptures", []))

    profile_checks: list[str] = []
    profile_captures: list[str] = []
    for name in profiles or []:
        if name not in config.profiles:
            raise RecordError(
                f"unknown profile {name!r}; available: "
                + ", ".join(config.profiles)
            )
        for item in config.profiles[name]:
            if item in config.checks:
                profile_checks.append(item)
            else:
                profile_captures.append(item)
    explicit_checks = _parse_ids(checks, list(config.checks))
    explicit_captures = _parse_ids(captures, list(config.captures))
    explicit = bool(profiles or checks or captures)
    if explicit:
        selected_checks = ordered_union(
            [profile_checks, explicit_checks or []], config.checks
        )
        selected_captures = ordered_union(
            [profile_captures, explicit_captures or []], config.captures
        )
        if side == "baseline" and profile_checks and not checks:
            print(
                "baseline side ignores profile checks; recording captures only",
                file=sys.stderr,
            )
            selected_checks = []
    else:
        selected_checks = [] if side == "baseline" else list(required_checks)
        selected_captures = list(
            baseline_captures if side == "baseline" else required_captures
        )
    if side == "baseline":
        if selected_checks:
            raise RecordError(
                "the baseline side records captures only; checks run on the candidate"
            )
        if selected_captures:
            invalid = [
                item
                for item in selected_captures
                if item not in config.captures or not config.captures[item].baseline
            ]
            if invalid:
                raise RecordError(
                    "baseline side requested non-baseline captures: " + ", ".join(invalid)
                )
    if not selected_checks and not selected_captures:
        if side == "baseline" and plan_data is not None and not required_captures:
            print(
                "no baseline captures are declared for this plan; recording "
                "identity only",
                file=sys.stderr,
            )
        else:
            raise RecordError(
                "nothing selected; pass --plan, --check or --capture"
            )
    unknown_checks = [item for item in selected_checks if item not in config.checks]
    unknown_captures = [item for item in selected_captures if item not in config.captures]
    if unknown_checks or unknown_captures:
        raise RecordError(
            "unknown selection: "
            + ", ".join(unknown_checks + unknown_captures)
        )
    selected_checks = ordered_union([selected_checks], config.checks)
    selected_captures = ordered_union([selected_captures], config.captures)

    commands: list[Command] = []
    for item in selected_checks:
        commands.extend(config.checks[item].commands)
    for item in selected_captures:
        commands.extend(config.captures[item].commands)
    used_files, used_corpus, used_tools = _used_placeholders(commands)
    missing_files = sorted(used_files - set(files))
    if missing_files:
        raise RecordError(
            "missing --file role(s): " + ", ".join(missing_files)
        )
    missing_tools = sorted(used_tools - set(tools))
    if missing_tools:
        raise RecordError("missing --tool name(s): " + ", ".join(missing_tools))
    extra_files = sorted(set(files) - set(config.files))
    if extra_files:
        raise RecordError(
            "unknown file role(s): " + ", ".join(extra_files)
        )
    extra_tools = sorted(set(tools) - set(config.tools))
    if extra_tools:
        raise RecordError(
            "unknown tool name(s): " + ", ".join(extra_tools)
        )

    if not gufo.is_file():
        raise RecordError(f"--gufo does not exist: {gufo}")
    repo = repo.resolve()
    if revision is None:
        source = git_identity(repo)
    else:
        source = {
            "revision": revision,
            "dirty": bool(source_dirty),
            "worktree": repo.name,
        }
    source["repo"] = display_path(repo, root)
    if source_dirty is not None:
        source["dirty"] = source_dirty

    print(f"hashing {side} binary {gufo}", file=sys.stderr, flush=True)
    binary = hash_file(gufo)
    try:
        version = subprocess.run(
            [str(gufo), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        binary["version"] = (version.stdout or version.stderr).strip() or None
        binary["versionExitCode"] = version.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        binary["version"] = None
        binary["versionError"] = str(error)
    fingerprint: dict[str, Any] | None = None
    fingerprint_error: str | None = None
    if fingerprint_path is not None:
        try:
            fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as error:
            fingerprint_error = f"cannot read --fingerprint: {error}"
    else:
        try:
            completed = subprocess.run(
                [str(gufo), "diagnose", "--fingerprint", "--json"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            if completed.returncode == 0:
                fingerprint = json.loads(completed.stdout)
            else:
                fingerprint_error = (
                    completed.stderr or completed.stdout
                ).strip() or f"exit {completed.returncode}"
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            fingerprint_error = str(error)
    binary["fingerprint"] = fingerprint
    binary["fingerprintError"] = fingerprint_error

    def identity_with_path(identity: dict[str, Any], path: Path) -> dict[str, Any]:
        return {**identity, "path": str(path.resolve())}

    tool_identities = {
        name: identity_with_path(hash_path(path), path)
        for name, path in sorted(tools.items())
    }
    file_identities = {
        role: [
            identity_with_path(hash_path(path), path)
            for path in sorted(paths)
        ]
        for role, paths in sorted(files.items())
    }
    corpus_identities = {}
    for name in sorted(used_corpus):
        path = config.corpus_path(name)
        corpus_identities[name] = identity_with_path(hash_path(path), path)
    config_identity = hash_file(config.path)

    record_data: dict[str, Any] = {
        "artifactType": ARTIFACT_TYPE,
        "schemaVersion": SCHEMA_VERSION,
        "side": side,
        "recorded": utc_now(),
        "platform": platform_name(),
        "model": {
            "id": config.model_id,
            "docs": config.docs,
            "source": config.source,
            "category": config.category,
        },
        "config": {
            "path": config.config_rel,
            "bytes": config_identity["bytes"],
            "sha256": config_identity["sha256"],
        },
        "source": source,
        "binary": binary,
        "tools": tool_identities,
        "files": file_identities,
        "corpus": corpus_identities,
        "env": dict(sorted(env.items())),
        "plan": None
        if plan_data is None
        else {
            "baseSha": plan_data.get("baseSha"),
            "headSha": plan_data.get("headSha"),
            "mergeBase": plan_data.get("mergeBase"),
            "checks": required_checks,
            "captures": required_captures,
            "baselineCaptures": baseline_captures,
            "scopes": list(plan_entry.get("scopes", [])) if plan_entry else [],
        },
        "selected": {
            "checks": selected_checks,
            "captures": selected_captures,
        },
        "items": {},
    }
    record_path = out / "record.json"
    atomic_json(record_path, record_data)
    context = _context(
        root,
        out,
        gufo,
        repo,
        config,
        files,
        tools,
        env,
        timeout,
        used_corpus,
    )
    for item_id in selected_checks:
        result = _run_item(config.checks[item_id], "check", context)
        record_data["items"][f"check:{item_id}"] = _item_record(result)
        atomic_json(record_path, record_data)
    for item_id in selected_captures:
        result = _run_item(config.captures[item_id], "capture", context)
        record_data["items"][f"capture:{item_id}"] = _item_record(result)
        atomic_json(record_path, record_data)

    failures = [
        (item_id, item)
        for item_id, item in record_data["items"].items()
        if item["status"] != "pass"
    ]
    print(f"record: {record_path}", flush=True)
    if failures:
        detail = "; ".join(
            f"{item_id} {item['status']}"
            + (f" ({item.get('reason')})" if item.get("reason") else "")
            for item_id, item in failures
        )
        print(f"failed or unverified items: {detail}", file=sys.stderr)
        raise RecordError(
            f"{len(failures)} item(s) did not pass: {detail}; see {record_path}"
        )
    return record_path


def _context(
    root: Path,
    out: Path,
    gufo: Path,
    repo: Path,
    config: QualifyConfig,
    files: dict[str, list[Path]],
    tools: dict[str, Path],
    env: dict[str, str],
    timeout: float,
    used_corpus: set[str],
) -> RecordContext:
    return RecordContext(
        root=root,
        out=out,
        gufo=gufo.resolve(),
        repo=repo,
        config=config,
        files=files,
        tools=tools,
        corpus={
            name: config.corpus_path(name)
            for name in used_corpus
        },
        env={**os.environ, **env},
        timeout=timeout,
    )
