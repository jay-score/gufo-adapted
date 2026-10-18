"""Focused checks for tools/models/qualify.py; no model or GPU required."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOL = Path(__file__).resolve().parents[2] / "tools/models/qualify.py"
sys.path.insert(0, str(TOOL.parents[1]))

from gufo.qualify.config import Check, Command, config_for
from gufo.qualify.record import (
    RecordError, _context, _find_test_executable, _run_command, _run_item,
)

MAKE_LOGITS = """\
import hashlib
import os
import sys

corpus, output, _model = sys.argv[1:4]
mode = os.environ.get("FIXTURE_MODE", "ok")
if mode == "missing":
    sys.exit(0)
data = open(corpus, "rb").read()
lines = ["GFLE-SHA256 1 4 2"]
for row in (1, 2):
    salt = b"x" if mode == "mismatch" else b""
    digest = hashlib.sha256(data + bytes([row]) + salt).hexdigest()
    lines.append(f"{row} {row % 4} {digest}")
open(output, "w", encoding="utf-8", newline="\\n").write("\\n".join(lines) + "\\n")
if mode == "limit":
    print("mean_kl=0.005")
else:
    print("mean_kl=0.0005")
print("rows=2")
"""

QUALIFY = """\
{
  "schema": "gufo-model-qualify/1",
  "model": {
    "id": "Example",
    "docs": "example",
    "source": "example",
    "category": "llm"
  },
  "files": {"gguf": "fixture model"},
  "corpus": {"logits": "artifacts/corpus.txt"},
  "tools": {},
  "triggers": [
    {
      "scope": "arithmetic",
      "paths": ["src/models/example/", "src/core/quant/"],
      "requires": ["build", "fast", "logits"]
    },
    {
      "scope": "state",
      "paths": ["src/cli/serve/"],
      "requires": ["build", "fast"]
    }
  ],
  "profiles": {
    "fast": ["build", "fast"],
    "full": ["build", "fast", "logits"],
    "logits": ["logits"]
  },
  "checks": {
    "build": {
      "description": "fixture build",
      "commands": [{"argv": ["{python}", "-c", "print('build=1')"]}]
    },
    "fast": {
      "description": "fixture fast suite",
      "commands": [
        {"argv": ["{python}", "-c", "print('fast=1')"]},
        {
          "argv": ["{python}", "-c", "print('linux-extra=1')"],
          "platforms": ["linux"],
          "optional": true
        }
      ]
    }
  },
  "captures": {
    "logits": {
      "description": "fixture raw-logit rows",
      "baseline": true,
      "commands": [
        {
          "argv": [
            "{python}",
            "{repo}/tools/make_logits.py",
            "{corpus:logits}",
            "{out}/result.bin.sha256",
            "{file:gguf}"
          ],
          "env": {"FIXTURE_MODE": "{env:FIXTURE_MODE}"}
        }
      ],
      "outputs": ["{out}/result.bin.sha256"],
      "compare": "logit-sha256"
    }
  },
  "limits": [
    {
      "capture": "logits",
      "metric": "mean_kl",
      "max": 0.002,
      "source": "fixture limit"
    }
  ]
}
"""

OTHER = """\
{
  "schema": "gufo-model-qualify/1",
  "model": {
    "id": "Other",
    "docs": "other",
    "source": "other",
    "category": "llm"
  },
  "files": {"gguf": "fixture model"},
  "corpus": {},
  "tools": {},
  "triggers": [
    {
      "scope": "arithmetic",
      "paths": ["src/models/other/", "src/core/quant/"],
      "requires": ["fast"]
    }
  ],
  "checks": {
    "fast": {
      "description": "fixture fast suite",
      "commands": [{"argv": ["{python}", "-c", "print('fast=1')"]}]
    }
  },
  "captures": {},
  "limits": []
}
"""


class QualifyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="gufo-qualify-")
        self.root = Path(self.temporary.name)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "qualify-test@example.com")
        self.git("config", "user.name", "Qualify Test")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.autocrlf", "false")
        self.write("README.md", "# Fixture\n")
        self.write("tools/make_logits.py", MAKE_LOGITS)
        self.write("docs/models/example/artifacts/qualify.json", QUALIFY)
        self.write(
            "docs/models/example/artifacts/corpus.txt",
            "A fixed corpus for the fixture logit rows.\n",
        )
        self.write("docs/models/other/artifacts/qualify.json", OTHER)
        self.write("src/models/example/config.cpp", "int config = 1;\n")
        self.write("src/models/other/config.cpp", "int config = 1;\n")
        self.write("src/core/quant/dequant.cpp", "int dequant = 1;\n")
        self.write("src/cli/serve/continuation_cache.cpp", "int cache = 1;\n")
        self.write("model.gguf", "fake model\n")
        self.write("model-other.gguf", "different fake model\n")
        self.write("gufo-baseline.bin", "baseline binary\n")
        self.write("gufo-candidate.bin", "candidate binary\n")
        self.write(
            "fingerprint.json",
            json.dumps({"schemaVersion": "1.0.0", "fingerprintId": "fixture-machine"}),
        )
        self.commit("chore: fixture base")
        self.base_sha = self.git("rev-parse", "HEAD").stdout.strip()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    def commit(self, message: str) -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD").stdout.strip()

    def run_tool(self, *args: str) -> subprocess.CompletedProcess[str]:
        command, *rest = args
        return subprocess.run(
            [
                sys.executable,
                str(TOOL),
                command,
                "--root",
                str(self.root),
                *rest,
            ],
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def plan(self, base: str = "HEAD~1", head: str = "HEAD") -> dict:
        result = self.run_tool(
            "plan", "--base", base, "--head", head, "--format", "json"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def change(self, path: str, text: str, message: str) -> str:
        self.write(path, text)
        return self.commit(message)

    def record(
        self,
        side: str,
        gufo: str,
        out: str,
        *,
        plan: Path | None = None,
        check: list[str] | None = None,
        capture: list[str] | None = None,
        profile: list[str] | None = None,
        file: str = "gguf=model.gguf",
        fingerprint: str = "fingerprint.json",
        mode: str = "ok",
        revision: str | None = None,
        extra: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        args = [
            "record",
            "--model",
            "example",
            "--side",
            side,
            "--gufo",
            gufo,
            "--out",
            out,
            "--file",
            file,
            "--fingerprint",
            fingerprint,
            "--env",
            f"FIXTURE_MODE={mode}",
        ]
        if plan is not None:
            args.extend(["--plan", str(plan)])
        for item in check or []:
            args.extend(["--check", item])
        for item in capture or []:
            args.extend(["--capture", item])
        for item in profile or []:
            args.extend(["--profile", item])
        if revision is not None:
            args.extend(["--revision", revision, "--clean"])
        if extra:
            args.extend(extra)
        return self.run_tool(*args)

    def planned_run(self) -> tuple[Path, Path, str]:
        """Create a core-quant diff, plan it, and record both sides."""
        self.change(
            "src/core/quant/dequant.cpp",
            "int dequant = 2;\n",
            "fix: change shared quantization",
        )
        head_sha = self.git("rev-parse", "HEAD").stdout.strip()
        plan_path = self.root / "plan.json"
        result = self.run_tool(
            "plan",
            "--base",
            self.base_sha,
            "--head",
            head_sha,
            "--format",
            "json",
            "--output",
            str(plan_path),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        baseline = self.root / "run/baseline"
        candidate = self.root / "run/candidate"
        recorded = self.record(
            "baseline",
            "gufo-baseline.bin",
            str(baseline),
            plan=plan_path,
            revision=self.base_sha,
        )
        self.assertEqual(recorded.returncode, 0, recorded.stderr)
        recorded = self.record(
            "candidate",
            "gufo-candidate.bin",
            str(candidate),
            plan=plan_path,
            revision=head_sha,
        )
        self.assertEqual(recorded.returncode, 0, recorded.stderr)
        return plan_path, self.root / "run", head_sha

    def test_plan_selects_shared_and_model_paths(self) -> None:
        self.change(
            "src/core/quant/dequant.cpp",
            "int dequant = 2;\n",
            "fix: change shared quantization",
        )
        packet = self.plan()
        models = {entry["model"] for entry in packet["models"]}
        self.assertEqual(models, {"example", "other"})
        self.assertEqual(packet["uncovered"], [])

        self.change(
            "src/models/example/config.cpp",
            "int config = 2;\n",
            "fix: change one model",
        )
        packet = self.plan()
        self.assertEqual([entry["model"] for entry in packet["models"]], ["example"])
        self.assertIn("arithmetic", packet["models"][0]["scopes"])
        self.assertIn("logits", packet["models"][0]["captures"])

    def test_invalid_declaration_fails_loudly(self) -> None:
        bad = {
            "schema": "gufo-model-qualify/1",
            "model": {
                "id": "Bad",
                "docs": "bad",
                "source": "bad",
                "category": "llm",
            },
            "files": {"gguf": "fixture"},
            "corpus": {},
            "tools": {},
            "triggers": [
                {
                    "scope": "arithmetic",
                    "paths": ["src/models/bad/"],
                    "requires": ["missing"],
                }
            ],
            "checks": {
                "fast": {
                    "description": "fixture",
                    "commands": [{"argv": ["{python}", "-c", "print('ok')"]}],
                }
            },
            "captures": {},
            "limits": [],
        }
        self.write(
            "docs/models/bad/artifacts/qualify.json",
            json.dumps(bad, indent=2) + "\n",
        )
        self.commit("chore: add invalid qualification declaration")
        result = self.run_tool("plan", "--base", "HEAD~1", "--head", "HEAD")
        self.assertEqual(result.returncode, 1)
        self.assertIn("unknown check or capture", result.stderr)

    def test_plan_reports_uncovered_code(self) -> None:
        self.change(
            "src/core/unknown/thing.cpp",
            "int thing = 1;\n",
            "feat: add unowned code",
        )
        packet = self.plan()
        self.assertIn("src/core/unknown/thing.cpp", packet["uncovered"])

    def test_plan_covers_both_sides_of_renames(self) -> None:
        self.git("mv", "src/core/quant/dequant.cpp", "tools/moved.cpp")
        self.commit("refactor: move shared quantization")
        packet = self.plan()
        self.assertEqual({entry["model"] for entry in packet["models"]}, {"example", "other"})
        self.assertIn("src/core/quant/dequant.cpp", packet["models"][0]["triggers"][0]["files"])

        self.write("src/core/unknown/thing.cpp", "int thing = 1;\n")
        self.commit("chore: add uncovered fixture")
        self.git("mv", "src/core/unknown/thing.cpp", "tools/thing.cpp")
        self.commit("refactor: move uncovered code")
        self.assertIn("src/core/unknown/thing.cpp", self.plan()["uncovered"])

    def test_matching_records_pass_and_report(self) -> None:
        plan_path, run, _ = self.planned_run()
        result = self.run_tool(
            "compare",
            "--model",
            "example",
            "--baseline",
            str(run / "baseline"),
            "--candidate",
            str(run / "candidate"),
            "--plan",
            str(plan_path),
            "--out",
            str(run),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        qualification = json.loads((run / "qualification.json").read_text())
        self.assertEqual(qualification["status"], "PASS")
        self.assertTrue((run / "qualification.md").is_file())
        report = (run / "qualification.md").read_text()
        for heading in (
            "# Qualification:",
            "## Summary",
            "## Coverage",
            "## Identity",
            "## Checks",
            "## Captures and comparisons",
            "## Reproduction",
            "## Not verified",
        ):
            self.assertIn(heading, report)

        rendered = self.run_tool(
            "report", "--artifact", str(run / "qualification.json")
        )
        self.assertEqual(rendered.returncode, 0, rendered.stderr)
        self.assertIn("**Status: PASS.**", rendered.stdout)

    def test_differing_logits_fail(self) -> None:
        plan_path, run, head_sha = self.planned_run()
        candidate = self.root / "run/mismatch"
        recorded = self.record(
            "candidate",
            "gufo-candidate.bin",
            str(candidate),
            plan=plan_path,
            mode="mismatch",
            revision=head_sha,
        )
        self.assertEqual(recorded.returncode, 0, recorded.stderr)
        result = self.run_tool(
            "compare",
            "--model",
            "example",
            "--baseline",
            str(run / "baseline"),
            "--candidate",
            str(candidate),
            "--plan",
            str(plan_path),
            "--out",
            str(self.root / "run/mismatch-result"),
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("differ", result.stdout.lower())
        qualification = json.loads(
            (self.root / "run/mismatch-result/qualification.json").read_text()
        )
        self.assertEqual(qualification["status"], "FAIL")

    def test_declared_limit_fails(self) -> None:
        plan_path, run, head_sha = self.planned_run()
        candidate = self.root / "run/limit"
        recorded = self.record(
            "candidate",
            "gufo-candidate.bin",
            str(candidate),
            plan=plan_path,
            mode="limit",
            revision=head_sha,
        )
        self.assertEqual(recorded.returncode, 0, recorded.stderr)
        result = self.run_tool(
            "compare",
            "--model",
            "example",
            "--baseline",
            str(run / "baseline"),
            "--candidate",
            str(candidate),
            "--plan",
            str(plan_path),
            "--out",
            str(self.root / "run/limit-result"),
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("mean_kl", result.stdout)

    def test_missing_required_check_fails(self) -> None:
        plan_path, run, head_sha = self.planned_run()
        candidate = self.root / "run/no-build"
        recorded = self.record(
            "candidate",
            "gufo-candidate.bin",
            str(candidate),
            plan=plan_path,
            check=["fast"],
            revision=head_sha,
        )
        self.assertEqual(recorded.returncode, 0, recorded.stderr)
        record = json.loads((candidate / "record.json").read_text())
        self.assertEqual(sorted(record["items"]), ["check:fast"])
        result = self.run_tool(
            "compare",
            "--model",
            "example",
            "--baseline",
            str(run / "baseline"),
            "--candidate",
            str(candidate),
            "--plan",
            str(plan_path),
            "--out",
            str(self.root / "run/no-build-result"),
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("required check 'build'", result.stdout)

    def test_model_identity_mismatch_fails(self) -> None:
        plan_path, run, head_sha = self.planned_run()
        candidate = self.root / "run/other-model"
        recorded = self.record(
            "candidate",
            "gufo-candidate.bin",
            str(candidate),
            plan=plan_path,
            file="gguf=model-other.gguf",
            revision=head_sha,
        )
        self.assertEqual(recorded.returncode, 0, recorded.stderr)
        result = self.run_tool(
            "compare",
            "--model",
            "example",
            "--baseline",
            str(run / "baseline"),
            "--candidate",
            str(candidate),
            "--plan",
            str(plan_path),
            "--out",
            str(self.root / "run/other-model-result"),
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("differs between baseline and candidate", result.stdout)

    def test_identical_binary_needs_control(self) -> None:
        plan_path, run, head_sha = self.planned_run()
        candidate = self.root / "run/same-binary"
        recorded = self.record(
            "candidate",
            "gufo-baseline.bin",
            str(candidate),
            plan=plan_path,
            revision=head_sha,
        )
        self.assertEqual(recorded.returncode, 0, recorded.stderr)
        result = self.run_tool(
            "compare",
            "--model",
            "example",
            "--baseline",
            str(run / "baseline"),
            "--candidate",
            str(candidate),
            "--plan",
            str(plan_path),
            "--out",
            str(self.root / "run/same-result"),
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("identical", result.stdout)

        control = self.run_tool(
            "compare",
            "--model",
            "example",
            "--baseline",
            str(run / "baseline"),
            "--candidate",
            str(candidate),
            "--plan",
            str(plan_path),
            "--control",
            "--out",
            str(self.root / "run/control-result"),
        )
        self.assertEqual(control.returncode, 0, control.stderr)
        qualification = json.loads(
            (self.root / "run/control-result/qualification.json").read_text()
        )
        self.assertEqual(qualification["status"], "CONTROL")

    def test_identity_only_baseline_is_allowed(self) -> None:
        plan = {
            "artifactType": "gufo-qualify-plan",
            "schemaVersion": 1,
            "models": [
                {
                    "model": "example",
                    "scopes": ["state"],
                    "checks": ["fast"],
                    "captures": [],
                    "baselineCaptures": [],
                }
            ],
            "baseSha": None,
            "headSha": None,
            "mergeBase": None,
        }
        plan_path = self.root / "empty-plan.json"
        plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")

        baseline = self.root / "run/identity-baseline"
        result = self.record(
            "baseline",
            "gufo-baseline.bin",
            str(baseline),
            plan=plan_path,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads((baseline / "record.json").read_text())
        self.assertEqual(record["items"], {})

        candidate = self.root / "run/identity-candidate"
        result = self.record(
            "candidate",
            "gufo-candidate.bin",
            str(candidate),
            plan=plan_path,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        result = self.run_tool(
            "compare",
            "--model",
            "example",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--plan",
            str(plan_path),
            "--out",
            str(self.root / "run/identity-result"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = json.loads(
            (self.root / "run/identity-result/qualification.json").read_text()
        )
        self.assertEqual(result["status"], "PASS")

    def test_profile_expands_declared_suite(self) -> None:
        result = self.record(
            "candidate",
            "gufo-candidate.bin",
            str(self.root / "run/profile"),
            profile=["fast"],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads((self.root / "run/profile/record.json").read_text())
        self.assertEqual(record["selected"]["checks"], ["build", "fast"])
        self.assertEqual(record["selected"]["captures"], [])
        self.assertEqual(
            sorted(record["items"]), ["check:build", "check:fast"]
        )

        result = self.record(
            "baseline",
            "gufo-baseline.bin",
            str(self.root / "run/profile-baseline"),
            profile=["full"],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(
            (self.root / "run/profile-baseline/record.json").read_text()
        )
        self.assertEqual(record["selected"]["checks"], [])
        self.assertEqual(record["selected"]["captures"], ["logits"])

    def test_baseline_check_is_rejected(self) -> None:
        result = self.record(
            "baseline",
            "gufo-baseline.bin",
            str(self.root / "run/bad-baseline"),
            check=["fast"],
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("baseline side records captures only", result.stderr)

    def test_missing_capture_output_fails(self) -> None:
        plan_path, run, head_sha = self.planned_run()
        candidate = self.root / "run/no-output"
        recorded = self.record(
            "candidate",
            "gufo-candidate.bin",
            str(candidate),
            plan=plan_path,
            mode="missing",
            revision=head_sha,
        )
        self.assertEqual(recorded.returncode, 1)
        record = json.loads((candidate / "record.json").read_text())
        self.assertIn(
            "declared output was not produced",
            record["items"]["capture:logits"]["reason"],
        )

    def test_capture_rerun_does_not_reuse_previous_outputs(self) -> None:
        plan_path, run, head_sha = self.planned_run()
        candidate = run / "candidate"
        previous = json.loads((candidate / "record.json").read_text())
        old_output = Path(previous["items"]["capture:logits"]["outputs"][0]["path"])
        recorded = self.record(
            "candidate", "gufo-candidate.bin", str(candidate),
            plan=plan_path, mode="missing", revision=head_sha,
        )
        self.assertEqual(recorded.returncode, 1, recorded.stdout)
        current = json.loads((candidate / "record.json").read_text())
        self.assertEqual(current["items"]["capture:logits"]["status"], "fail")
        self.assertEqual(current["items"]["capture:logits"]["outputs"], [])
        self.assertTrue(old_output.is_file())

    def test_embedded_plan_enforces_both_revisions(self) -> None:
        plan_path, run, head_sha = self.planned_run()
        args = (
            "compare", "--model", "example",
            "--baseline", str(run / "baseline"),
            "--candidate", str(run / "candidate"), "--out", str(run),
        )
        matching = self.run_tool(*args)
        self.assertEqual(matching.returncode, 0, matching.stdout + matching.stderr)
        for side, wrong, correct, expected in (
            ("candidate", self.base_sha, head_sha, "not the planned head"),
            ("baseline", head_sha, self.base_sha, "neither the planned base"),
        ):
            with self.subTest(side=side):
                recorded = self.record(
                    side, f"gufo-{side}.bin", str(run / side),
                    plan=plan_path, revision=wrong,
                )
                self.assertEqual(recorded.returncode, 0, recorded.stderr)
                result = self.run_tool(*args)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(expected, result.stdout)
                self.assertEqual(self.record(
                    side, f"gufo-{side}.bin", str(run / side),
                    plan=plan_path, revision=correct,
                ).returncode, 0)


class RecordCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="gufo-qualify-command-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = config_for(TOOL.parents[2], "qwen3.8-flash-next")
        self.ctx = _context(
            self.root, self.root, Path(sys.executable), self.root, self.config,
            {}, {}, {"QUALIFY_TEST_OVERRIDE": "override"}, 30, set(),
        )

    def run_command(self, command: Command) -> dict:
        with contextlib.redirect_stdout(io.StringIO()):
            return _run_command(
                command, self.ctx, kind="check", item_id="fixture",
                index=0, item_out=self.root,
            )

    def test_commands_run_in_selected_source_checkout(self) -> None:
        self.ctx.repo = self.root / "source"
        for checkout in (self.root, self.ctx.repo):
            tools = checkout / "tools"
            tools.mkdir(parents=True)
            (tools / "probe.py").write_text(
                "from pathlib import Path; Path('built.txt').write_text('source')"
                if checkout == self.ctx.repo else "raise SystemExit(1)"
            )
        result = self.run_command(Command((sys.executable, "tools/probe.py")))
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(Path(result["cwd"]), self.ctx.repo)
        self.assertTrue((self.ctx.repo / "built.txt").is_file())
        self.assertFalse((self.root / "built.txt").exists())

        (self.ctx.repo / "checks").mkdir()
        result = self.run_command(Command((
            sys.executable, "-c", "from pathlib import Path; Path('checked.txt').touch()",
        ), cwd="checks"))
        self.assertEqual(result["status"], "pass", result)
        self.assertTrue((self.ctx.repo / "checks/checked.txt").is_file())

    def test_test_executable_cannot_fall_back_to_driver_checkout(self) -> None:
        self.ctx.repo = self.root / "nested/source"
        suffix = ".exe" if sys.platform == "win32" else ""
        driver = self.root / "build/gpu-test" / f"fixture{suffix}"
        selected = self.ctx.repo / "build/gpu-test/tests" / f"fixture{suffix}"
        for executable in (driver, selected):
            executable.parent.mkdir(parents=True)
            executable.write_text("fixture")
        self.assertEqual(_find_test_executable("fixture", self.ctx), selected)
        selected.unlink()
        with self.assertRaises(RecordError):
            _find_test_executable("fixture", self.ctx)

    def test_model_build_commands_select_each_platform(self) -> None:
        for model in ("qwen3.8-flash-next", "qwen3.8-27b", "deepseek-v4-flash"):
            build = config_for(TOOL.parents[2], model).checks["build"]
            for platform in ("windows", "linux"):
                with self.subTest(model=model, platform=platform), patch(
                    "gufo.qualify.record.platform_name", return_value=platform
                ), patch(
                    "gufo.qualify.record.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, "", ""),
                ) as execute, contextlib.redirect_stdout(io.StringIO()):
                    result = _run_item(build, "check", self.ctx)
                    self.assertEqual(result.status, "pass", result.reason)
                    self.assertEqual(execute.call_count, sum(
                        platform in command.platforms for command in build.commands
                    ))

    def test_item_with_no_required_platform_command_does_not_pass(self) -> None:
        item = Check("foreign", "fixture", (
            Command(("not-executed",), platforms=("linux",)),
            Command(("not-executed",), platforms=("windows",), optional=True),
        ))
        with patch("gufo.qualify.record.platform_name", return_value="windows"), patch(
            "gufo.qualify.record.subprocess.run"
        ) as execute:
            result = _run_item(item, "check", self.ctx)
        self.assertEqual(result.status, "not-applicable")
        execute.assert_not_called()

    def test_environment_overrides_reach_children_and_command_env_wins(self) -> None:
        command = Command((
            sys.executable, "-c",
            "import os; print(os.environ['QUALIFY_TEST_OVERRIDE'])",
        ))
        result = self.run_command(command)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["stdoutTail"].strip(), "override")
        result = self.run_command(Command(
            command.argv, env={"QUALIFY_TEST_OVERRIDE": "command"}
        ))
        self.assertEqual(result["stdoutTail"].strip(), "command")

    def test_ctest_individual_results_are_required(self) -> None:
        python = json.dumps(sys.executable.replace("\\", "/"))
        for mode in ("pass", "skip", "mixed", "disabled", "empty"):
            with self.subTest(mode=mode):
                suite = self.root / mode
                suite.mkdir()
                tests = []
                if mode != "empty":
                    exit_code = 77 if mode in ("skip", "mixed") else 0
                    tests.append(
                        f'add_test(fixture {python} "-c" "raise SystemExit({exit_code})")'
                    )
                    tests.append('set_tests_properties(fixture PROPERTIES SKIP_RETURN_CODE 77)')
                if mode == "mixed":
                    tests.append(f'add_test(passed {python} "-c" "raise SystemExit(0)")')
                if mode == "disabled":
                    tests.append('set_tests_properties(fixture PROPERTIES DISABLED TRUE)')
                (suite / "CTestTestfile.cmake").write_text("\n".join(tests) + "\n")
                result = self.run_command(Command(("ctest", "--test-dir", str(suite))))
                self.assertEqual(result["status"], {
                    "pass": "pass", "skip": "skipped", "mixed": "skipped",
                    "disabled": "skipped", "empty": "fail",
                }[mode], result)

    def test_ctest_cannot_reuse_a_previous_result_file(self) -> None:
        logs = self.root / "logs"
        logs.mkdir()
        previous = logs / "check-fixture-0.junit.xml"
        previous.write_text('<testsuite><testcase name="old"/></testsuite>')
        with patch(
            "gufo.qualify.record.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, "", ""),
        ):
            result = self.run_command(Command(("ctest",)))
        self.assertEqual(result["status"], "error")
        self.assertIn("cannot read CTest results", result["reason"])


if __name__ == "__main__":
    unittest.main()
