#!/usr/bin/env python3
"""Build a read-only review context packet for a Gufo change.

The packet classifies a committed diff by ownership area, recommends the checks
required by docs/TESTING.md and AGENTS.md, and flags hazards that need human or
model judgment. It never edits the worktree.

Typical use:

    python tools/ci/review-context.py --base origin/windows-port
    python tools/ci/review-context.py --base HEAD~3 --intent "fix cache replay"
    python tools/ci/review-context.py --base main --format json --output review.json

The output is advisory. The review skill is responsible for reading the changed
code, verifying claims and reporting what it could not verify.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

# Source model directory -> documentation directory. New models are resolved by
# looking for a matching directory under docs/models.
MODEL_DOC_DIRS = {
    "qwen": "qwen3.8-27b",
    "qwen38_flash_next": "qwen3.8-flash-next",
    "deepseek_v4_flash": "deepseek-v4-flash",
    "qwen3_asr": "qwen3-asr",
    "qwen3_tts": "qwen3-tts",
    "qwen_image_21": "qwen-image-2.1",
    "minimax_h3": "minimax-h3",
}

CODE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
                 ".hip", ".m", ".mm"}

# Heuristic areas. A path can belong to more than one area.
ARITHMETIC_RE = re.compile(
    r"(^|/)(kernels?|hip|quant|dequant|gemm|mmq|mmvq|wmma|blaslt|attention|"
    r"softmax|rope|norm|silu|swiglu|sampling|speculative|verifier|forward|"
    r"executor|engine|logits?)(/|\.|$)",
    re.IGNORECASE,
)
LOADER_TOKENIZER_RE = re.compile(
    r"^src/(core/(gguf|tokenizer)|models/[^/]+/"
    r"(config|weights|tokenizer|chat_template|model_loader))",
    re.IGNORECASE,
)
STATE_RE = re.compile(
    r"(snapshot|continuation|cache|kv[_-]?cache|rollback|checkpoint|session|"
    r"recurrent|draft)",
    re.IGNORECASE,
)
API_RE = re.compile(
    r"^src/cli/serve/(openai_chat|http_server|serve|inference_backend|"
    r"text_generation_scheduler|responses|messages)"
)
WINDOWS_PATH_RE = re.compile(
    r"^(compat/win32/|tools/windows/|src/core/platform/)|\.(bat|ps1)$"
)
BUILD_PATH_RE = re.compile(
    r"^(CMakeLists\.txt|CMakePresets\.json|cmake/|vcpkg\.json|vcpkg\.jsonc|"
    r"flake\.nix|flake\.lock|\.devops/)"
)
PERF_TOOL_RE = re.compile(r"^tools/(bench|prof)/")


class ReviewError(RuntimeError):
    """A user-facing review-context error."""


@dataclass
class ChangedFile:
    path: str
    status: str
    added: int
    deleted: int
    binary: bool
    areas: list[str] = field(default_factory=list)
    old_path: str | None = None


@dataclass
class Commit:
    sha: str
    subject: str
    body: str


@dataclass
class Check:
    name: str
    priority: str
    command: str
    reason: str


@dataclass
class Flag:
    severity: str
    title: str
    detail: str


@dataclass
class CheapCheck:
    name: str
    command: str
    status: str
    output: str = ""


@dataclass
class Packet:
    repo: str
    branch: str
    base_ref: str
    base_sha: str
    head_ref: str
    head_sha: str
    merge_base: str
    generated: str
    platform: str
    intent: str
    dirty: bool
    untracked: list[str]
    commits: list[Commit]
    files: list[ChangedFile]
    areas: dict[str, int]
    checks: list[Check]
    flags: list[Flag]
    cheap_checks: list[CheapCheck]
    warnings: list[str]


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in repo and capture text output."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as error:
        raise ReviewError("git is not available on PATH") from error


def git_out(repo: Path, *args: str) -> str:
    result = git(repo, *args)
    if result.returncode != 0:
        raise ReviewError(
            f"git {' '.join(args)} failed:\n{result.stderr.strip()}"
        )
    return result.stdout


def git_ok(repo: Path, *args: str) -> bool:
    return git(repo, *args).returncode == 0


def resolve_commit(repo: Path, ref: str) -> str:
    result = git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if result.returncode != 0:
        raise ReviewError(f"cannot resolve ref '{ref}' to a commit")
    return result.stdout.strip()


def default_base(repo: Path) -> str:
    for candidate in ("origin/windows-port", "windows-port",
                      "origin/main", "main", "HEAD~1"):
        if git_ok(repo, "rev-parse", "--verify", f"{candidate}^{{commit}}"):
            return candidate
    raise ReviewError(
        "cannot infer a base ref; pass --base (for example origin/windows-port)"
    )


def current_branch(repo: Path) -> str:
    result = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    return result.stdout.strip() or "HEAD"


def merge_base(repo: Path, base: str, head: str,
               warnings: list[str]) -> str:
    result = git(repo, "merge-base", base, head)
    if result.returncode != 0 or not result.stdout.strip():
        warnings.append(
            "merge-base failed; comparing the base commit directly. "
            "History may be shallow or unrelated."
        )
        return base
    return result.stdout.strip()


def parse_name_status(raw: str) -> dict[str, tuple[str, str | None]]:
    """Parse `git diff --name-status -z` into path -> (status, old path)."""
    statuses: dict[str, tuple[str, str | None]] = {}
    parts = raw.split("\0")
    index = 0
    while index < len(parts):
        status = parts[index].strip()
        index += 1
        if not status:
            continue
        if status[0] in "RC":
            if index + 1 >= len(parts):
                break
            old_path = parts[index]
            index += 1
            new_path = parts[index]
            index += 1
            statuses[new_path] = (status, old_path)
        else:
            if index >= len(parts):
                break
            path = parts[index]
            index += 1
            statuses[path] = (status, None)
    return statuses


def parse_numstat(raw: str) -> dict[str, tuple[int, int, bool]]:
    """Parse `git diff --numstat -z` into path -> (added, deleted, binary)."""
    stats: dict[str, tuple[int, int, bool]] = {}
    parts = iter(raw.split("\0"))
    for entry in parts:
        columns = entry.split("\t", 2)
        if len(columns) < 3:
            continue
        added, deleted = columns[0], columns[1]
        path = columns[2]
        if not path:
            next(parts)  # old path
            path = next(parts)
        binary = added == "-" and deleted == "-"
        try:
            counts = (
                0 if binary else int(added),
                0 if binary else int(deleted),
                binary,
            )
        except ValueError:
            continue
        stats[path] = counts
    return stats


def classify_path(path: str) -> list[str]:
    areas: list[str] = []

    model_match = re.match(r"^src/models/([^/]+)/", path)
    if model_match:
        model = model_match.group(1)
        areas.append(f"model:{model}")
        if ("/kernels/" in path or "/hip/" in path
                or path.endswith((".hip", ".hip.cpp"))):
            areas.append(f"model-kernels:{model}")

    test_model_match = re.match(r"^tests/models/([^/]+)/", path)
    if test_model_match:
        areas.append(f"model-tests:{test_model_match.group(1)}")

    if path.startswith("src/core/quant/"):
        areas.append("core-quant")
    elif path.startswith("src/core/"):
        areas.append("core")

    if path.startswith("src/cli/serve/"):
        areas.append("server")
    elif path.startswith("src/cli/"):
        areas.append("cli")

    if path.startswith(("src/eval/", "src/testing/")):
        areas.append("eval")

    if WINDOWS_PATH_RE.search(path):
        areas.append("windows")

    if path.startswith("tools/gui/"):
        areas.append("gui")

    if path.startswith(("tools/bench/", "tools/prof/", "tools/models/",
                        "tools/serving/", "tools/ds4/", "tools/qwen27b/",
                        "tools/qwen-flash/", "tools/h3/", "tools/audio/")):
        areas.append("tools")

    if path.startswith(("tools/ci/", ".github/workflows/")):
        areas.append("ci")

    if (path.startswith("docs/")
            or (path.endswith(".md") and "/" not in path)
            or path in {"README.md", "AGENTS.md", "CLAUDE.md"}):
        areas.append("docs")

    if BUILD_PATH_RE.search(path):
        areas.append("build")

    if (path.startswith("licenses/") or path == "THIRD_PARTY_NOTICES.md"
            or path == "NOTICE"):
        areas.append("licenses")

    if path.startswith(".agents/skills/"):
        areas.append("skills")

    if path.startswith("tests/"):
        areas.append("tests")

    return sorted(set(areas)) or ["other"]


def collect_files(repo: Path, base: str, head: str) -> list[ChangedFile]:
    statuses = parse_name_status(
        git_out(repo, "diff", "--name-status", "-z", "--find-renames",
                base, head)
    )
    stats = parse_numstat(
        git_out(repo, "diff", "--numstat", "-z", "--find-renames", base, head)
    )
    files: list[ChangedFile] = []
    for path, (status, old_path) in statuses.items():
        added, deleted, binary = stats.get(path, (0, 0, False))
        files.append(ChangedFile(
            path=path,
            status=status,
            added=added,
            deleted=deleted,
            binary=binary,
            areas=sorted(set(classify_path(path)) | (
                set(classify_path(old_path)) if old_path else set()
            )),
            old_path=old_path,
        ))
    return sorted(files, key=lambda item: item.path)


def collect_commits(repo: Path, base: str, head: str) -> list[Commit]:
    raw = git_out(repo, "log", "--format=%H%x1f%s%x1f%b%x1e",
                  f"{base}..{head}")
    commits: list[Commit] = []
    for record in raw.split("\x1e"):
        record = record.strip("\n")
        if not record:
            continue
        parts = record.split("\x1f", 2)
        if len(parts) != 3:
            continue
        commits.append(Commit(
            sha=parts[0].strip(),
            subject=parts[1].strip(),
            body=parts[2].strip(),
        ))
    return commits


def changed_files_matching(files: Iterable[ChangedFile],
                           pattern: re.Pattern[str]) -> list[str]:
    return [path for path in changed_paths(files) if pattern.search(path)]


def changed_paths(files: Iterable[ChangedFile]) -> list[str]:
    """Include the source of a rename when selecting affected ownership."""
    return sorted({
        path for item in files for path in (item.path, item.old_path) if path
    })


def total_changed_lines(repo: Path, base: str, head: str,
                        ignore_whitespace: bool) -> int:
    args = ["diff", "--numstat", "-z", "--find-renames"]
    if ignore_whitespace:
        args.append("-w")
    args.extend([base, head])
    return sum(added + deleted for added, deleted, _ in
               parse_numstat(git_out(repo, *args)).values())


def file_size_in_head(repo: Path, head: str, path: str) -> int:
    result = git(repo, "cat-file", "-s", f"{head}:{path}")
    if result.returncode != 0:
        return 0
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def doc_dir_for(repo: Path, model: str) -> str | None:
    if model in MODEL_DOC_DIRS:
        return MODEL_DOC_DIRS[model]
    normalized = model.replace("_", "-").lower()
    docs_root = repo / "docs" / "models"
    if not docs_root.is_dir():
        return None
    for child in docs_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if name == normalized or name.replace(".", "-") == normalized:
            return child.name
    return None


def shared_model_includes(repo: Path, base: str, head: str,
                          files: Sequence[ChangedFile]) -> list[str]:
    """Find a changed model file that includes another model package."""
    findings: list[str] = []
    model_files = [
        item.path for item in files
        if item.path.startswith("src/models/") and item.status != "D"
    ]
    for path in model_files[:40]:
        model_match = re.match(r"^src/models/([^/]+)/", path)
        if not model_match:
            continue
        model = model_match.group(1)
        diff = git_out(repo, "diff", "--unified=0", "--find-renames",
                       base, head, "--", path)
        for line in diff.splitlines():
            if not line.startswith("+"):
                continue
            match = re.search(r'#include\s+"src/models/([^/"]+)/', line)
            if match and match.group(1) != model:
                findings.append(
                    f"{path} includes src/models/{match.group(1)}/; "
                    "model code stays with its model"
                )
    return findings


def add_check(checks: list[Check], name: str, priority: str,
              command: str, reason: str) -> None:
    for existing in checks:
        if existing.name == name:
            if reason not in existing.reason:
                existing.reason += f" {reason}"
            if priority == "required":
                existing.priority = "required"
            return
    checks.append(Check(name, priority, command, reason))


def ctest_pattern(targets: Iterable[str]) -> str:
    return "|".join(sorted({target for target in targets if target}))


def recommend_checks(files: Sequence[ChangedFile], platform_name: str) -> list[Check]:
    paths = changed_paths(files)
    code_paths = [path for path in paths if Path(path).suffix in CODE_SUFFIXES]
    any_code = bool(code_paths)
    any_docs = any(path.endswith(".md") for path in paths)
    any_model = any(path.startswith("src/models/") for path in paths)
    any_model_kernels = any(
        path.startswith("src/models/")
        and ("/kernels/" in path or "/hip/" in path
             or path.endswith((".hip", ".hip.cpp")))
        for path in paths
    )
    any_model_execution = any(
        path.startswith("src/models/")
        and re.search(r"(engine|executor|forward|runtime|kernels?|hip|"
                      r"attention|model_loader|weights)", path,
                      re.IGNORECASE)
        for path in paths
    )
    any_core = any(path.startswith("src/core/") for path in paths)
    any_server = any(path.startswith("src/cli/serve/") for path in paths)
    any_cli = any(path.startswith("src/cli/") for path in paths)
    any_windows = any(WINDOWS_PATH_RE.search(path) for path in paths)
    any_build = any(BUILD_PATH_RE.search(path) for path in paths)
    any_license = any(path.startswith("licenses/")
                      or path == "THIRD_PARTY_NOTICES.md" for path in paths)
    any_gui = any(path.startswith("tools/gui/") for path in paths)
    any_skills = any(path.startswith(".agents/skills/") for path in paths)
    any_tests = any(path.startswith("tests/") for path in paths)
    any_perf_tools = any(PERF_TOOL_RE.search(path) for path in paths)
    any_state = any(STATE_RE.search(path) for path in paths)
    any_api = any(API_RE.search(path) for path in paths)
    any_arith = any(
        ARITHMETIC_RE.search(path) or LOADER_TOKENIZER_RE.search(path)
        for path in paths
    )
    any_vision = any(
        "vision" in path or "mmproj" in path or "projector" in path
        for path in paths
    )

    checks: list[Check] = []

    if any_code:
        add_check(
            checks, "format", "required",
            "python tools/ci/check-format.py",
            "C++ formatting gate; use CI's clang-format on PATH.",
        )

    if any_docs or any_model or any_cli or any_windows:
        add_check(
            checks, "docs", "required",
            "python tools/ci/check-docs.py",
            "Markdown links, anchors, required files and JSON blocks.",
        )

    if any_build or any_license:
        add_check(
            checks, "dependencies", "required",
            "python tools/ci/check-dependencies.py",
            "Dependency and license inventory changed.",
        )

    if any_core or any_server or any_cli or any_model or any_build or any_windows:
        add_check(
            checks, "linux-ci", "required",
            "nix build .#checks.x86_64-linux.pr  "
            "# or: cmake --build --preset pr",
            "Hosted CPU/repository selection for shared code. Linux CI is "
            "still required; Windows-only validation is not a substitute.",
        )

    cpu_targets: set[str] = set()
    cpu_rules = [
        (r"^src/core/quant/", ["ggml_dequant_test",
                               "ggml_block_layout_parity_test",
                               "ggml_gemm_parity_test"]),
        (r"^src/core/gguf", ["gguf_reader_test", "gguf_identity_test"]),
        (r"^src/core/sampling", ["logit_sampler_test"]),
        (r"^src/core/speculative", ["speculative_verification_test"]),
        (r"^src/cli/serve/openai_chat", ["openai_chat_test",
                                         "http_server_test"]),
        (r"^src/cli/serve/http_server", ["http_server_test",
                                         "openai_chat_test"]),
        (r"^src/cli/serve/text_generation_scheduler",
         ["text_generation_scheduler_test", "http_server_test"]),
        (r"^src/cli/serve/continuation_",
         ["continuation_disk_store_test", "text_model_runner_test"]),
        (r"^src/cli/serve/inference_backend",
         ["text_model_runner_test", "openai_chat_test",
          "http_server_test"]),
        (r"^src/cli/serve/", ["http_server_test"]),
        (r"^src/cli/bench", ["bench_cli_test"]),
        (r"^src/cli/prompt", ["prompt_cli_test"]),
        (r"^src/cli/", ["gufo_help"]),
        (r"^src/models/qwen/", ["qwen_tokenizer_test",
                                "qwen_chat_template_test"]),
        (r"^src/models/qwen38_flash_next/",
         ["qwen38_flash_next_config_test",
          "qwen38_flash_next_mtp_sampling_test"]),
        (r"^src/models/deepseek_v4_flash/",
         ["ds4_sampling_test", "ds4_chat_template_test", "ds4_cli_test"]),
        (r"^src/models/qwen3_asr/",
         ["qwen3_asr_config_test", "qwen3_asr_audio_api_test"]),
        (r"^src/models/qwen3_tts/",
         ["qwen3_tts_config_test", "qwen3_tts_audio_api_test"]),
        (r"^src/models/qwen_image_21/", ["qwen_image_21_test"]),
        (r"^src/models/minimax_h3/",
         ["video_api_test", "minimax_h3_sampling_test",
          "minimax_h3_runtime_test"]),
        (r"^tools/ci/check-kernel-resources",
         ["kernel_resources_parser_test"]),
    ]
    for pattern, targets in cpu_rules:
        if any(re.search(pattern, path) for path in paths):
            cpu_targets.update(targets)

    if cpu_targets:
        test_names = {
            "qwen38_flash_next_config_test": "qwen38_flash_next.config",
            "qwen38_flash_next_mtp_sampling_test": "qwen38_flash_next.mtp_sampling",
            "ds4_sampling_test": "ds4.sampling",
            "ds4_chat_template_test": "ds4.template",
            "ds4_cli_test": "ds4.cli",
        }
        pattern = ctest_pattern(
            re.escape(test_names.get(target, target)) for target in cpu_targets
        )
        build_targets = {"gufo" if target == "gufo_help" else target
                         for target in cpu_targets}
        if platform_name == "windows":
            command = (
                "powershell -ExecutionPolicy Bypass -File "
                "tools\\windows\\build.ps1 -Preset gpu-test -Target "
                "<target> -Jobs 4\n"
                "ctest --test-dir build/gpu-test -R "
                f"'^({pattern})$' --output-on-failure --no-tests=error"
            )
            reason = (
                "Build each target below, then run the focused CTest "
                "selection."
            )
            target_list = ", ".join(sorted(build_targets))
            command += f"\n# targets: {target_list}"
        else:
            targets = " ".join(sorted(build_targets))
            command = (
                f"nix develop -c cmake --build --preset gpu-test "
                f"--target {targets}\n"
                f"nix develop -c ctest --preset gpu-full -R "
                f"'^({pattern})$' --output-on-failure --no-tests=error"
            )
            reason = "Build and run the focused CPU/contract targets."
        add_check(checks, "cpu-targets", "required", command, reason)

    if (any_model_kernels or any_model_execution
            or any(re.search(r"\.hip($|\.)", path) for path in paths)):
        affected = sorted({
            item.split("/")[2] for item in paths
            if item.startswith("src/models/") and item.count("/") >= 2
        })
        add_check(
            checks, "gpu", "required",
            "Build the affected GPU target and run its focused CTest; "
            "a missing-model or missing-device skip is not a pass.",
            "HIP kernels or model execution changed"
            + (f" ({', '.join(affected)})" if affected else "") + ".",
        )
        add_check(
            checks, "kernel-resources", "conditional",
            "ctest --test-dir build/gpu-test -R '^kernel_resources_test$' "
            "--output-on-failure --no-tests=error",
            "Check compiled gfx1151 scratch/spill budgets after kernel "
            "changes.",
        )

    if any_arith:
        add_check(
            checks, "numerical", "required",
            "Run the affected model's matched-token logits/PPL and state "
            "checks; see docs/models/<model>/QUALITY.md and "
            "docs/TESTING.md#matched-token-and-layer-comparisons. Record "
            "baseline/candidate revisions, binary hashes, model/sidecar "
            "identities, corpus hash, settings and exact commands.",
            "Arithmetic, loader or tokenizer paths changed. Require exact full-logit "
            "equality for unchanged arithmetic, or declared limits for an "
            "intentional numerical change.",
        )

    if any_state:
        add_check(
            checks, "state", "required",
            "Run snapshot/restore, cache replay/edits, rollback, seeded "
            "replay, EOS and cancellation checks; see docs/TESTING.md.",
            "State, cache, snapshot or rollback paths changed.",
        )

    if any_api:
        add_check(
            checks, "api", "required",
            "ctest --test-dir build/gpu-test -R "
            "'^(openai_chat_test|http_server_test)$' "
            "--output-on-failure --no-tests=error",
            "Cover buffered and streaming Chat Completions and Responses, "
            "reasoning/tool-call parsing, history replay and image order.",
        )

    if any_vision or any_api or any_server:
        add_check(
            checks, "capability", "required",
            "Verify GET /v1/models and GET /props reflect actual loaded "
            "capabilities; test direct attachments, images in tool results, "
            "history replay and streaming.",
            "Server or vision capability reporting may be affected.",
        )

    if any_windows or any_core or any_server or any_model or any_build:
        add_check(
            checks, "windows", "required",
            "powershell -ExecutionPolicy Bypass -File "
            "tools\\windows\\check.ps1\n"
            "powershell -ExecutionPolicy Bypass -File "
            "tools\\windows\\build.ps1 -Preset gpu-test -Target "
            "<affected-target> -Jobs 4",
            "Windows is the primary fork target; shared changes need a "
            "Windows build and focused run.",
        )

    if any_model:
        add_check(
            checks, "model-docs", "required",
            "python tools/ci/check-docs.py; update "
            "docs/models/<model>/{README,BENCHMARKS,QUALITY,EXPERIMENTS}.md, "
            "docs/models/README.md and the checker model list as applicable.",
            "Model code changed; keep model-owned documentation and the "
            "checker in sync.",
        )

    if any_gui:
        add_check(
            checks, "gui-tests", "required",
            "python -m unittest discover -s tests/tools "
            "-p \"gui_*_test.py\"",
            "GUI config, presets, process or DOM behavior changed.",
        )

    if any_windows:
        add_check(
            checks, "windows-scripts", "required",
            "python tests/tools/windows_launcher_test.py\n"
            "python tests/tools/windows_setup_test.py\n"
            "python tests/tools/windows_update_test.py\n"
            "python tests/tools/windows_vcpkg_path_test.py\n"
            "python tests/tools/windows_entrypoints_test.py",
            "Windows launcher, setup, update or entrypoint scripts changed.",
        )

    if any_model or any_vision or any_server:
        add_check(
            checks, "real-model", "conditional",
            "Run a bounded real-model smoke with the documented artifacts; "
            "a missing-model skip is not a pass.",
            "Loader, inference or vision behavior may need end-to-end "
            "validation.",
        )

    if any_perf_tools or any_model_kernels:
        add_check(
            checks, "benchmark", "conditional",
            "python tools/bench/model-bench.py --model <model> ... run "
            "--target gufo --table <id>; render with --check",
            "Performance claims or benchmark tooling changed. Keep workload "
            "identity, artifacts and provenance.",
        )

    if any_skills:
        add_check(
            checks, "skills", "conditional",
            "python tools/ci/check-docs.py",
            "Verify every skill path and referenced tool still exists; "
            "update .agents/skills/README.md when the catalog changes.",
        )

    if not any_tests and any_code:
        add_check(
            checks, "test-change", "conditional",
            "Add or identify the focused regression test; no test file "
            "changed in this range.",
            "Code changed without a matching test change.",
        )

    return checks


def detect_flags(repo: Path, base: str, head: str,
                 files: Sequence[ChangedFile]) -> list[Flag]:
    flags: list[Flag] = []
    paths = changed_paths(files)
    added_paths = [item.path for item in files if item.status.startswith("A")]

    def add_flag(severity: str, title: str, detail: str) -> None:
        flags.append(Flag(severity, title, detail))

    def summarize(items: Iterable[str], limit: int = 8) -> str:
        unique = sorted(set(items))
        shown = unique[:limit]
        suffix = "" if len(unique) <= limit else f" (+{len(unique) - limit} more)"
        return ", ".join(shown) + suffix

    arithmetic = changed_files_matching(files, ARITHMETIC_RE)
    if arithmetic:
        add_flag(
            "major", "Arithmetic-sensitive paths changed",
            "Require matched-token logits/PPL and state checks before "
            "accepting the change. Files: " + summarize(arithmetic),
        )

    state = changed_files_matching(files, STATE_RE)
    if state:
        add_flag(
            "major", "State, cache or snapshot paths changed",
            "Verify snapshot payload version, cache identity, rollback, "
            "seeded replay and cancellation. Files: " + summarize(state),
        )

    api = changed_files_matching(files, API_RE)
    if api:
        add_flag(
            "major", "API or scheduler contract changed",
            "Cover buffered and streaming Chat Completions/Responses, tool "
            "calls, reasoning, history replay and capability reporting. "
            "Files: " + summarize(api),
        )

    windows = [path for path in paths if WINDOWS_PATH_RE.search(path)]
    if windows:
        add_flag(
            "major", "Windows-specific paths changed",
            "Check POSIX shims, unified-memory placement, DLL staging, GUI "
            "settings and Linux preservation. Files: " + summarize(windows),
        )

    build = [path for path in paths if BUILD_PATH_RE.search(path)]
    if build:
        add_flag(
            "major", "Build, dependency or toolchain paths changed",
            "Validate pins, Linux CI, Windows configure/build and kernel "
            "resource budgets. Files: " + summarize(build),
        )

    licenses = [path for path in paths if path.startswith("licenses/")
                or path in {"THIRD_PARTY_NOTICES.md", "NOTICE"}]
    if licenses:
        add_flag(
            "major", "License or third-party inventory changed",
            "Update THIRD_PARTY_NOTICES.md and run "
            "check-dependencies.py. Files: " + summarize(licenses),
        )

    model_paths = [path for path in paths if path.startswith("src/models/")]
    if model_paths:
        models = sorted({path.split("/")[2] for path in model_paths})
        for model in models:
            doc_dir = doc_dir_for(repo, model)
            new_model = any(path.startswith(f"src/models/{model}/")
                            for path in added_paths)
            if doc_dir is None and new_model:
                add_flag(
                    "major", f"New model package: {model}",
                    "Add the four docs under docs/models/, a docs/models/"
                    "README.md row and a tools/ci/check-docs.py model-list "
                    "entry. Wire general.architecture in "
                    "inference_backend.cpp, bench.cpp and prompt.cpp; add "
                    "tests and dispatch coverage.",
                )
            elif doc_dir is not None:
                doc_changes = [path for path in paths
                               if path.startswith(f"docs/models/{doc_dir}/")]
                if not doc_changes:
                    add_flag(
                        "major", f"Model code changed without {doc_dir} docs",
                        "Update README/QUALITY/BENCHMARKS/EXPERIMENTS or "
                        "record why no doc change is needed.",
                    )

    model_test_changes = {
        path.split("/")[2] for path in paths if path.startswith("tests/models/")
    }
    for model in sorted({path.split("/")[2] for path in model_paths}):
        if model not in model_test_changes:
            add_flag(
                "minor", f"No test changes for model {model}",
                "Confirm existing tests cover the change, or add a focused "
                "regression test.",
            )

    diff_text = ""
    if files:
        result = git(repo, "diff", "--find-renames", "--unified=0",
                     base, head)
        if result.returncode == 0:
            diff_text = result.stdout
    if state and "kSnapshotPayloadVersion" not in diff_text:
        add_flag(
            "major", "State files changed without a payload-version change",
            "Determine whether kSnapshotPayloadVersion or an equivalent "
            "cache/snapshot identity must be bumped. If not, record why.",
        )

    for finding in shared_model_includes(repo, base, head, files):
        add_flag("major", "Shared kernel/code across model packages", finding)

    if "src/cli/serve/serve.cpp" in paths:
        add_flag(
            "minor", "Server flags changed",
            "Keep PrintServeHelp and option parsing in sync; one canonical "
            "long option per behavior; no aliases. Update docs/CLI.md or "
            "docs/SERVER.md when behavior changes.",
        )

    if "src/cli/serve/inference_backend.cpp" in paths or any(
        "vision" in path or "mmproj" in path for path in paths
    ):
        add_flag(
            "major", "Capability reporting may have drifted",
            "GET /v1/models and GET /props must reflect actual loaded image "
            "support. Cover direct attachments, images in tool results, "
            "history replay and streaming.",
        )

    normal_lines = total_changed_lines(repo, base, head, False)
    ignored_lines = total_changed_lines(repo, base, head, True)
    if normal_lines >= 40 and ignored_lines * 2 <= normal_lines:
        detail = (
            f"{normal_lines} changed lines, {ignored_lines} after ignoring "
            "whitespace. Separate formatting churn from the functional diff."
        )
        severity = "major" if ignored_lines == 0 else "minor"
        add_flag(severity, "Formatting or whitespace churn", detail)

    large_files = []
    for item in files:
        if item.status.startswith("D") or item.binary:
            continue
        size = file_size_in_head(repo, head, item.path)
        if size >= 1_000_000:
            large_files.append(f"{item.path} ({size / 1_000_000:.1f} MB)")
    if large_files:
        add_flag(
            "minor", "Large files in the diff",
            "Confirm the artifact belongs in Git; retained evidence should "
            "stay small: " + summarize(large_files, 5),
        )

    binary_files = [item.path for item in files if item.binary]
    if binary_files:
        add_flag(
            "note", "Binary files changed",
            "Review provenance and size: " + summarize(binary_files, 5),
        )

    status = git(repo, "status", "--porcelain=v1")
    if status.returncode == 0:
        lines = [line for line in status.stdout.splitlines() if line]
        untracked = [line[3:] for line in lines if line.startswith("??")]
        dirty = any(not line.startswith("??") for line in lines)
        if dirty:
            add_flag(
                "note", "Working tree has uncommitted changes",
                "The packet describes the committed range. Uncommitted edits "
                "can change build/test results.",
            )
        if untracked:
            add_flag(
                "note", "Untracked files present",
                "Nix flakes only see tracked files; stage task-owned paths "
                "before a Nix build. Untracked: " + summarize(untracked, 8),
            )

    if not files:
        add_flag(
            "note", "Empty diff",
            f"No committed changes between {base} and {head}.",
        )

    return flags


def run_cheap_checks(repo: Path, files: Sequence[ChangedFile]) -> list[CheapCheck]:
    paths = changed_paths(files)
    any_code = any(Path(path).suffix in CODE_SUFFIXES for path in paths)
    any_docs = any(path.endswith(".md") for path in paths)
    any_model = any(path.startswith("src/models/") for path in paths)
    any_build = any(BUILD_PATH_RE.search(path) for path in paths)
    any_license = any(path.startswith("licenses/")
                      or path == "THIRD_PARTY_NOTICES.md" for path in paths)

    requested: list[tuple[str, str]] = []
    if any_code:
        requested.append(("format", "python tools/ci/check-format.py"))
    if any_docs or any_model:
        requested.append(("docs", "python tools/ci/check-docs.py"))
    if any_build or any_license:
        requested.append(("dependencies", "python tools/ci/check-dependencies.py"))

    results: list[CheapCheck] = []
    for name, command in requested:
        if name == "format" and shutil.which("clang-format") is None:
            results.append(CheapCheck(
                name, command, "unavailable",
                "clang-format is not on PATH; install CI's clang-tools or "
                "rerun without --run-checks.",
            ))
            continue
        argv = [sys.executable, *command.split()[1:]]
        try:
            result = subprocess.run(
                argv,
                cwd=repo,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
            )
        except FileNotFoundError:
            results.append(CheapCheck(name, command, "unavailable",
                                      "Python interpreter not found"))
            continue
        except subprocess.TimeoutExpired:
            results.append(CheapCheck(name, command, "timeout",
                                      "check exceeded 180 seconds"))
            continue
        output = (result.stdout + result.stderr).strip()
        if len(output) > 4000:
            output = output[-4000:]
        results.append(CheapCheck(
            name=name,
            command=command,
            status="pass" if result.returncode == 0 else "fail",
            output=output,
        ))
    return results


def build_packet(args: argparse.Namespace) -> Packet:
    repo = (Path(args.repo).resolve() if args.repo else REPO_ROOT)
    if not (repo / ".git").exists() and not git_ok(repo, "rev-parse",
                                                   "--git-dir"):
        raise ReviewError(f"not a git repository: {repo}")

    warnings: list[str] = []
    base_ref = args.base or default_base(repo)
    base_sha = resolve_commit(repo, base_ref)
    head_ref = args.head
    head_sha = resolve_commit(repo, head_ref)
    if args.run_checks and resolve_commit(repo, "HEAD") != head_sha:
        raise ReviewError(
            "--run-checks requires the checkout HEAD to match --head; "
            "use a checkout at the reviewed revision or omit --run-checks"
        )
    merge_base_sha = merge_base(repo, base_sha, head_sha, warnings)

    files = collect_files(repo, merge_base_sha, head_sha)
    commits = collect_commits(repo, merge_base_sha, head_sha)

    if args.intent:
        intent = args.intent.strip()
    elif args.intent_file:
        intent = Path(args.intent_file).read_text(encoding="utf-8").strip()
    else:
        subjects = [commit.subject for commit in commits[:10]]
        if len(commits) > 10:
            subjects.append(f"... and {len(commits) - 10} more commits")
        intent = "\n".join(subjects)

    status = git(repo, "status", "--porcelain=v1")
    lines = status.stdout.splitlines() if status.returncode == 0 else []
    untracked = [line[3:] for line in lines if line.startswith("??")]
    dirty = any(not line.startswith("??") for line in lines)

    platform_name = args.platform
    if platform_name == "auto":
        platform_name = "windows" if os.name == "nt" else "linux"

    areas: dict[str, int] = {}
    for item in files:
        for area in item.areas:
            areas[area] = areas.get(area, 0) + 1

    checks = recommend_checks(files, platform_name)
    flags = detect_flags(repo, merge_base_sha, head_sha, files)
    cheap_checks = run_cheap_checks(repo, files) if args.run_checks else []

    if not commits and not args.intent and not args.intent_file:
        warnings.append("No commits and no --intent provided; intent unknown.")

    return Packet(
        repo=str(repo),
        branch=current_branch(repo),
        base_ref=base_ref,
        base_sha=base_sha,
        head_ref=head_ref,
        head_sha=head_sha,
        merge_base=merge_base_sha,
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        platform=platform_name,
        intent=intent,
        dirty=dirty,
        untracked=untracked,
        commits=commits,
        files=files,
        areas=areas,
        checks=checks,
        flags=flags,
        cheap_checks=cheap_checks,
        warnings=warnings,
    )


def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def render_markdown(packet: Packet, max_files: int) -> str:
    lines: list[str] = []
    lines.append(f"# Review context: {packet.branch}")
    lines.append("")
    lines.append(f"Generated: {packet.generated}")
    lines.append("")
    lines.append(f"- Base: `{packet.base_ref}` (`{packet.base_sha[:12]}`)")
    lines.append(f"- Head: `{packet.head_ref}` (`{packet.head_sha[:12]}`)")
    lines.append(f"- Merge base: `{packet.merge_base[:12]}`")
    lines.append(f"- Platform: {packet.platform}")
    lines.append(f"- Working tree: {'dirty' if packet.dirty else 'clean'}")
    lines.append(f"- Commits: {len(packet.commits)}")
    lines.append("")

    lines.append("## Intent")
    lines.append("")
    if packet.intent:
        lines.append(packet.intent)
    else:
        lines.append("_No intent supplied. Infer it from the diff and mark it "
                     "unconfirmed._")
    lines.append("")

    lines.append("## Areas")
    lines.append("")
    if packet.areas:
        lines.append("| Area | Files |")
        lines.append("| --- | ---: |")
        for area, count in sorted(packet.areas.items()):
            lines.append(f"| `{md_escape(area)}` | {count} |")
    else:
        lines.append("_No files changed._")
    lines.append("")

    lines.append("## Changed files")
    lines.append("")
    if packet.files:
        lines.append("| Status | File | + | - | Areas |")
        lines.append("| --- | --- | ---: | ---: | --- |")
        for item in packet.files[:max_files]:
            delta_add = "-" if item.binary else str(item.added)
            delta_del = "-" if item.binary else str(item.deleted)
            areas = ", ".join(f"`{md_escape(area)}`" for area in item.areas)
            path = (f"{item.old_path} -> {item.path}" if item.old_path else item.path)
            lines.append(
                f"| {md_escape(item.status)} | {md_escape(path)} | "
                f"{delta_add} | {delta_del} | {areas} |"
            )
        if len(packet.files) > max_files:
            lines.append(
                f"| ... | _{len(packet.files) - max_files} more files_ | | | |"
            )
    else:
        lines.append("_No files changed._")
    lines.append("")

    lines.append("## Recommended checks")
    lines.append("")
    required = [check for check in packet.checks if check.priority == "required"]
    conditional = [check for check in packet.checks
                   if check.priority != "required"]
    if required:
        lines.append("### Required")
        lines.append("")
        for check in required:
            lines.append(f"- **{check.name}**: {check.reason}")
            lines.append("")
            lines.append("  ```text")
            for command_line in check.command.splitlines():
                lines.append(f"  {command_line}")
            lines.append("  ```")
        lines.append("")
    if conditional:
        lines.append("### Conditional / likely")
        lines.append("")
        for check in conditional:
            lines.append(f"- **{check.name}**: {check.reason}")
            lines.append("")
            lines.append("  ```text")
            for command_line in check.command.splitlines():
                lines.append(f"  {command_line}")
            lines.append("  ```")
        lines.append("")
    if not packet.checks:
        lines.append("_No checks recommended._")
        lines.append("")

    lines.append("## Risk flags")
    lines.append("")
    if packet.flags:
        lines.append("| Severity | Flag | Detail |")
        lines.append("| --- | --- | --- |")
        for flag in packet.flags:
            lines.append(
                f"| {flag.severity} | {md_escape(flag.title)} | "
                f"{md_escape(flag.detail)} |"
            )
    else:
        lines.append("_No flags._")
    lines.append("")

    if packet.cheap_checks:
        lines.append("## Cheap checks run")
        lines.append("")
        lines.append("| Check | Status | Command |")
        lines.append("| --- | --- | --- |")
        for check in packet.cheap_checks:
            lines.append(
                f"| {check.name} | {check.status} | "
                f"`{md_escape(check.command)}` |"
            )
        lines.append("")
        for check in packet.cheap_checks:
            if check.output:
                lines.append(f"### {check.name} output")
                lines.append("")
                lines.append("```text")
                lines.append(check.output)
                lines.append("```")
                lines.append("")

    lines.append("## Review checklist")
    lines.append("")
    for item in (
        "Intent is stated and the diff matches it.",
        "Correctness: arithmetic boundaries, tolerances and independent "
        "reference are identified.",
        "State: snapshot/cache identity, rollback, seeded replay and "
        "cancellation are covered.",
        "API: buffered/streaming behavior, tool calls, reasoning and history "
        "replay are covered.",
        "Concurrency: ownership, lifetimes, cancellation and scratch reuse "
        "are safe.",
        "Platform: Windows behavior is intentional and Linux is preserved.",
        "Build/dependencies/licenses are recorded and validated.",
        "Required tests were actually run, with revisions and artifacts.",
        "Docs and model cards match the new behavior.",
        "Performance claims use matched workload identity and retained "
        "numbers.",
        "Commits are focused and contain no unrelated formatting churn.",
    ):
        lines.append(f"- [ ] {item}")
    lines.append("")

    lines.append("## Not verified by this tool")
    lines.append("")
    lines.append("This packet is read-only context. It does not build Gufo, "
                 "run GPU tests, load models, verify numerical parity or "
                 "prove that a claim in the diff is correct. Record what the "
                 "review actually ran.")
    lines.append("")

    if packet.untracked:
        lines.append("## Untracked files")
        lines.append("")
        for path in packet.untracked[:50]:
            lines.append(f"- `{path}`")
        if len(packet.untracked) > 50:
            lines.append(f"- _... and {len(packet.untracked) - 50} more_")
        lines.append("")

    if packet.warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in packet.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a read-only review context packet for a Gufo change."
    )
    parser.add_argument("--base", help="base ref or commit (default: the "
                        "first of origin/windows-port, windows-port, "
                        "origin/main, main, HEAD~1)")
    parser.add_argument("--head", default="HEAD", help="head ref (default: HEAD)")
    parser.add_argument("--repo", help="repository root (default: this checkout)")
    intent = parser.add_mutually_exclusive_group()
    intent.add_argument("--intent", help="stated intent/PR description")
    intent.add_argument("--intent-file", help="file containing the intent")
    parser.add_argument("--format", choices=("markdown", "json"),
                        default="markdown")
    parser.add_argument("--output", help="write the packet to this path")
    parser.add_argument("--max-files", type=int, default=200,
                        help="maximum files to list (default: 200)")
    parser.add_argument("--platform", choices=("auto", "windows", "linux"),
                        default="auto", help="command style (default: auto)")
    parser.add_argument("--run-checks", action="store_true",
                        help="run check-format/check-docs/check-dependencies "
                             "when applicable")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        packet = build_packet(args)
    except ReviewError as error:
        print(f"review-context: {error}", file=sys.stderr)
        return 2

    if args.format == "json":
        rendered = json.dumps(asdict(packet), indent=2) + "\n"
    else:
        rendered = render_markdown(packet, args.max_files)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"wrote {output}", file=sys.stderr)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
