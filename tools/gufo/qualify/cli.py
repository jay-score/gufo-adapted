"""Command line for the per-model inference qualification workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .compare import CompareError, compare
from .config import QualifyError
from .plan import build_plan, render_markdown as render_plan
from .record import (
    RecordError,
    parse_env_args,
    parse_file_args,
    parse_tool_args,
    record,
)
from .report import render_markdown as render_report
from .util import PACKAGE_ROOT, REVIEW_CONTEXT, QualifyIOError, load_json


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=PACKAGE_ROOT,
        help="repository root (default: this checkout)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qualify",
        description=(
            "Select, record and compare per-model inference qualification "
            "evidence."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser(
        "plan",
        help="select affected models and scopes from a committed diff",
    )
    _common(plan)
    plan.add_argument("--base", help="base ref (default: review-context's choice)")
    plan.add_argument("--head", default="HEAD", help="head ref (default: HEAD)")
    plan.add_argument(
        "--model",
        action="append",
        default=[],
        help="include this docs model even if no trigger matches (repeatable)",
    )
    plan.add_argument(
        "--scope",
        action="append",
        default=[],
        help="force a scope's requirements (repeatable)",
    )
    plan.add_argument(
        "--format", choices=("markdown", "json"), default="markdown"
    )
    plan.add_argument(
        "--output",
        type=Path,
        help="write the plan to this path (relative paths are repository-relative)",
    )
    plan.add_argument(
        "--review-context",
        type=Path,
        default=REVIEW_CONTEXT,
        help="review-context.py used to collect the diff",
    )

    record_parser = sub.add_parser(
        "record", help="run one side's captures and checks"
    )
    _common(record_parser)
    record_parser.add_argument("--model", required=True, help="docs model directory")
    record_parser.add_argument(
        "--side", required=True, choices=("baseline", "candidate")
    )
    record_parser.add_argument("--gufo", type=Path, required=True, help="Gufo binary")
    record_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="record directory (relative paths are repository-relative)",
    )
    record_parser.add_argument("--plan", type=Path, help="plan.json from `plan`")
    record_parser.add_argument(
        "--capture", action="append", default=[], help="capture id, comma-separated, or all"
    )
    record_parser.add_argument(
        "--check", action="append", default=[], help="check id, comma-separated, or all"
    )
    record_parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="declared profile name (for example fast or full; repeatable)",
    )
    record_parser.add_argument(
        "--file", action="append", default=[], metavar="ROLE=PATH",
        help="model artifact for a declared file role (repeatable)",
    )
    record_parser.add_argument(
        "--tool", action="append", default=[], metavar="NAME=PATH",
        help="test/tool executable for a declared tool role (repeatable)",
    )
    record_parser.add_argument(
        "--env", action="append", default=[], metavar="NAME=VALUE",
        help="environment override for every command (repeatable)",
    )
    record_parser.add_argument(
        "--repo",
        type=Path,
        help="checkout that built --gufo and resolves {repo} (default: --root)",
    )
    record_parser.add_argument(
        "--fingerprint", type=Path, help="JSON from `gufo diagnose --fingerprint --json`"
    )
    record_parser.add_argument(
        "--timeout", type=float, default=7200.0, help="per-command timeout in seconds"
    )
    record_parser.add_argument("--revision", help="override the source revision")
    dirty = record_parser.add_mutually_exclusive_group()
    dirty.add_argument("--dirty", action="store_true", help="mark the source tree dirty")
    dirty.add_argument("--clean", action="store_true", help="mark the source tree clean")

    compare_parser = sub.add_parser(
        "compare", help="compare baseline and candidate records"
    )
    _common(compare_parser)
    compare_parser.add_argument("--model", required=True, help="docs model directory")
    compare_parser.add_argument(
        "--baseline", type=Path, required=True, help="baseline record directory or JSON"
    )
    compare_parser.add_argument(
        "--candidate", type=Path, required=True, help="candidate record directory or JSON"
    )
    compare_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="result directory (relative paths are repository-relative)",
    )
    compare_parser.add_argument("--plan", type=Path, help="plan.json from `plan`")
    compare_parser.add_argument(
        "--control",
        action="store_true",
        help="allow the same binary on both sides as a harness self-check",
    )
    compare_parser.add_argument(
        "--allow-fingerprint-mismatch", action="store_true"
    )
    compare_parser.add_argument("--allow-source-mismatch", action="store_true")
    compare_parser.add_argument("--allow-baseline-mismatch", action="store_true")

    report = sub.add_parser("report", help="re-render a qualification result")
    _common(report)
    report.add_argument(
        "--artifact", type=Path, required=True, help="qualification.json or result directory"
    )
    report.add_argument(
        "--output",
        type=Path,
        help="write Markdown here instead of stdout (relative paths are repository-relative)",
    )

    return parser


def _under_root(path: Path | None, root: Path) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return root / path


def _write_or_print(text: str, output: Path | None, root: Path) -> None:
    if output is None:
        sys.stdout.write(text)
        return
    target = output if output.is_absolute() else (root / output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {target}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    try:
        if args.command == "plan":
            plan = build_plan(
                root,
                base=args.base,
                head=args.head,
                models=args.model,
                scopes=args.scope,
                review_context=args.review_context,
            )
            if args.format == "json":
                rendered = json.dumps(plan, indent=2) + "\n"
            else:
                rendered = render_plan(plan)
            _write_or_print(rendered, args.output, root)
            return 0
        if args.command == "record":
            dirty = True if args.dirty else (False if args.clean else None)
            record(
                root,
                model=args.model,
                side=args.side,
                gufo=args.gufo if args.gufo.is_absolute() else (root / args.gufo),
                out=args.out if args.out.is_absolute() else (root / args.out),
                plan_path=_under_root(args.plan, root),
                captures=args.capture,
                checks=args.check,
                profiles=args.profile,
                files=parse_file_args(args.file),
                tools=parse_tool_args(args.tool),
                env=parse_env_args(args.env),
                repo=_under_root(args.repo, root) or root,
                fingerprint_path=_under_root(args.fingerprint, root),
                timeout=args.timeout,
                revision=args.revision,
                source_dirty=dirty,
            )
            return 0
        if args.command == "compare":
            result = compare(
                root,
                model=args.model,
                baseline_path=_under_root(args.baseline, root),
                candidate_path=_under_root(args.candidate, root),
                out=_under_root(args.out, root),
                plan_path=_under_root(args.plan, root),
                control=args.control,
                allow_fingerprint_mismatch=args.allow_fingerprint_mismatch,
                allow_source_mismatch=args.allow_source_mismatch,
                allow_baseline_mismatch=args.allow_baseline_mismatch,
            )
            return 0 if result.get("status") in ("PASS", "CONTROL") else 1
        if args.command == "report":
            artifact = _under_root(args.artifact, root)
            assert artifact is not None
            if artifact.is_dir():
                artifact = artifact / "qualification.json"
            result = load_json(artifact)
            if result.get("artifactType") != "gufo-qualify-result":
                raise CompareError(f"{artifact} is not a qualification result")
            _write_or_print(render_report(result, root), args.output, root)
            return 0
    except (QualifyError, QualifyIOError, RecordError, CompareError, OSError) as error:
        print(f"qualify: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
