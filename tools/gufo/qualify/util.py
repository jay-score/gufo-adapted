"""Shared helpers for qualification records and reports."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
REVIEW_CONTEXT = PACKAGE_ROOT / "tools" / "ci" / "review-context.py"
LOGIT_COMPARATOR = PACKAGE_ROOT / "tools" / "bench" / "logit-eval.py"

METRIC_RE = re.compile(
    r"(?<![\w.-])([A-Za-z_][A-Za-z0-9_.-]*)="
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class QualifyIOError(RuntimeError):
    """A user-facing qualification I/O error."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def platform_name() -> str:
    return "windows" if os.name == "nt" else "linux"


def executable_suffix() -> str:
    return ".exe" if os.name == "nt" else ""


def parse_metrics(text: str) -> dict[str, float]:
    """Collect the last numeric ``name=value`` occurrence in command output."""
    metrics: dict[str, float] = {}
    for match in METRIC_RE.finditer(text):
        try:
            metrics[match.group(1)] = float(match.group(2))
        except ValueError:
            continue
    return metrics


def parse_sha256(text: str) -> str | None:
    match = SHA256_RE.search(text)
    return match.group(0) if match else None


def hash_file(path: Path) -> dict[str, Any]:
    """Hash one file; progress stays on stderr for long model artifacts."""
    path = path.resolve()
    size = path.stat().st_size
    if size >= (64 << 20):
        print(
            f"hashing {path.name} ({size / (1 << 30):.2f} GiB) ...",
            file=sys.stderr,
            flush=True,
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return {
        "name": path.name,
        "type": "file",
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def hash_path(path: Path) -> dict[str, Any]:
    """Hash a file, or a directory tree as sorted relative paths and file hashes."""
    path = path.resolve()
    if path.is_file():
        return hash_file(path)
    if not path.is_dir():
        raise QualifyIOError(f"artifact does not exist: {path}")
    entries: list[dict[str, Any]] = []
    total = 0
    for item in sorted(child for child in path.rglob("*") if child.is_file()):
        identity = hash_file(item)
        entries.append(
            {
                "path": item.relative_to(path).as_posix(),
                "bytes": identity["bytes"],
                "sha256": identity["sha256"],
            }
        )
        total += identity["bytes"]
    tree = hashlib.sha256()
    for entry in entries:
        tree.update(
            f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode(
                "utf-8"
            )
        )
    return {
        "name": path.name,
        "type": "directory",
        "bytes": total,
        "sha256": tree.hexdigest(),
        "entries": entries,
    }


def sha_file(path: Path) -> str:
    """Return only the SHA-256 digest of one file."""
    return str(hash_file(path)["sha256"])


def git_identity(repo: Path) -> dict[str, Any]:
    """Capture the revision and dirty state of the checkout that built a binary."""
    try:
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise QualifyIOError(f"cannot read git identity from {repo}: {error}") from error
    return {
        "revision": revision,
        "dirty": bool(status.strip()),
        "worktree": repo.name,
    }


def run_logit_comparator(left: Path, right: Path) -> dict[str, Any]:
    """Run the canonical raw-logit row comparator."""
    if not LOGIT_COMPARATOR.is_file():
        raise QualifyIOError(f"missing logit comparator: {LOGIT_COMPARATOR}")
    completed = subprocess.run(
        [sys.executable, str(LOGIT_COMPARATOR), str(left), str(right)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode == 0:
        result = "match"
    elif completed.returncode == 1:
        result = "differ"
    else:
        result = "invalid"
    output = (completed.stdout + completed.stderr).strip()
    return {
        "result": result,
        "exitCode": completed.returncode,
        "detail": output,
    }


def display_path(path: str | Path | None, root: Path | None) -> str | None:
    """Prefer a repository-relative path; otherwise keep only the file name."""
    if path is None:
        return None
    resolved = Path(path)
    if root is not None:
        try:
            return resolved.resolve().relative_to(root.resolve()).as_posix()
        except (ValueError, OSError):
            pass
    return resolved.name


def redact_argument(value: str, root: Path, artifacts: Iterable[Path]) -> str:
    """Keep commands readable without leaking private absolute paths."""
    for artifact in artifacts:
        raw = str(artifact)
        if value == raw:
            return artifact.name
        if raw in value:
            value = value.replace(raw, artifact.name)
    try:
        relative = Path(value).resolve().relative_to(root.resolve()).as_posix()
        if not Path(value).is_absolute():
            return value
        return relative
    except (ValueError, OSError):
        return Path(value).name if Path(value).is_absolute() else value


def redact_command(
    argv: Iterable[str], root: Path, artifacts: Iterable[Path]
) -> list[str]:
    return [redact_argument(part, root, artifacts) for part in argv]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Write JSON atomically (shared behavior with the benchmark driver)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as error:
        raise QualifyIOError(f"missing file: {path}") from error
    except json.JSONDecodeError as error:
        raise QualifyIOError(f"invalid JSON in {path}: {error}") from error
