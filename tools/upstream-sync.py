#!/usr/bin/env python3
"""Review official upstream history and prepare an isolated sync worktree."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys


OFFICIAL_URL = "https://github.com/gufo-org/gufo.git"


def git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.rstrip("\n")


def revision(root, value):
    return git(root, "rev-parse", "--verify", "--end-of-options", value + "^{commit}")


def review(root, base, tip):
    common = git(root, "merge-base", base, tip)
    counts = git(root, "rev-list", "--left-right", "--count", base + "..." + tip).split()
    equivalent = {
        line.split()[1] for line in git(root, "cherry", base, tip).splitlines()
        if line.startswith("- ")
    }
    # Attribution is evidence of an import, not proof the whole patch is present.
    attributions = {}
    records = git(root, "log", "--format=%H%x00%B%x00", common + ".." + base).split("\0")
    for index in range(0, len(records) - 1, 2):
        local_sha, message = records[index].strip(), records[index + 1]
        for source_sha in re.findall(r"\b[0-9a-f]{40}\b", message):
            attributions.setdefault(source_sha, []).append(local_sha)
    incoming = []
    for line in git(root, "log", "--reverse", "--format=%H%x09%s", base + ".." + tip).splitlines():
        sha, subject = line.split("\t", 1)
        incoming.append({
            "sha": sha, "subject": subject,
            "patch_equivalent": sha in equivalent,
            "recorded_references": attributions.get(sha, []),
        })
    fork_paths = set(git(root, "diff", "--name-only", "-z", common, base).split("\0")) - {""}
    upstream_paths = set(git(root, "diff", "--name-only", "-z", common, tip).split("\0")) - {""}
    return {
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "official_url": OFFICIAL_URL,
        "base_sha": base, "official_sha": tip, "merge_base": common,
        "fork_only_commits": int(counts[0]), "upstream_only_commits": int(counts[1]),
        "incoming": incoming,
        "overlapping_paths": sorted(fork_paths & upstream_paths),
        "checks": [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="windows-port", help="fork branch or commit (default: windows-port)")
    parser.add_argument("--tip", default="official/main", help="official checkpoint to review (default: official/main)")
    parser.add_argument("--offline", action="store_true", help="use cached official/main without fetching")
    parser.add_argument("--prepare", type=Path, help="create a new worktree at this unused path; do not merge")
    parser.add_argument("--branch", help="new sync branch name; requires --prepare")
    parser.add_argument("--check", action="store_true", help="run documentation and dependency checks in the selected tree")
    args = parser.parse_args()
    if args.branch and not args.prepare:
        parser.error("--branch requires --prepare")
    root = Path(git(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel"))
    if git(root, "rev-parse", "--is-shallow-repository") == "true":
        raise RuntimeError("Full history is required; deepen this checkout before reviewing upstream.")
    remote = git(root, "remote", "get-url", "official")
    if remote not in (OFFICIAL_URL, OFFICIAL_URL.removesuffix(".git"), "git@github.com:gufo-org/gufo.git"):
        raise RuntimeError("The official remote must identify gufo-org/gufo; existing remotes were not changed.")
    if args.prepare and args.prepare.resolve().exists():
        raise RuntimeError("The sync worktree path already exists; choose an unused path.")
    if not args.offline:
        git(root, "fetch", "--no-tags", "official", "+refs/heads/main:refs/remotes/official/main")
    base, tip = revision(root, args.base), revision(root, args.tip)
    official_main = revision(root, "official/main")
    try:
        git(root, "merge-base", "--is-ancestor", tip, official_main)
    except RuntimeError:
        raise RuntimeError("Selected --tip is not an ancestor of official/main.") from None
    report = review(root, base, tip)
    print("Cached official history (offline)" if args.offline else "Fetched official main")
    print(f"Fork: {base}\nOfficial checkpoint: {tip}\nShared baseline: {report['merge_base']}")
    print(f"History: {report['fork_only_commits']} fork-only, {report['upstream_only_commits']} upstream-only commits")
    print("Incoming commits (oldest first; review fixes before features):")
    for commit in report["incoming"]:
        evidence = " [patch equivalent]" if commit["patch_equivalent"] else ""
        if commit["recorded_references"]:
            evidence += " [referenced locally; inspect scope]"
        print(f"  {commit['sha'][:12]} {commit['subject']}{evidence}")
    print(f"Overlapping paths ({len(report['overlapping_paths'])}; not a conflict prediction):")
    for path in report["overlapping_paths"]:
        print("  " + path)
    selected_tree = root
    if args.prepare:
        branch = args.branch or f"sync/{datetime.now(timezone.utc):%Y-%m-%d}-{tip[:8]}"
        git(root, "check-ref-format", "--branch", branch)
        selected_tree = args.prepare.resolve()
        git(root, "worktree", "add", "-b", branch, str(selected_tree), base)
        report["worktree"] = str(selected_tree)
        report["branch"] = branch
        print(f"Prepared {branch} at {selected_tree}; no upstream changes applied.")
        print("Start with reviewed fixes: git cherry-pick -x <official-fix-SHA>")
        print(f"For a fully reviewed checkpoint: git merge --no-ff --no-commit {tip}")
    report_path = selected_tree / "build/upstream-sync" / f"{base[:12]}-{tip[:12]}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    print(f"Report: {report_path}")
    if args.check:
        print(f"Checking {selected_tree}; these checks do not validate an upstream integration.")
        report["checked_tree_sha"] = revision(selected_tree, "HEAD")
        report["checked_tree_status"] = git(selected_tree, "status", "--porcelain")
        save()
        sys.stdout.flush()
        for script in ("tools/ci/check-docs.py", "tools/ci/check-dependencies.py"):
            result = subprocess.run([sys.executable, script], cwd=selected_tree)
            report["checks"].append({"script": script, "exit_code": result.returncode})
            save()
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as error:
        print(f"upstream-sync: {error}", file=sys.stderr)
        sys.exit(1)
