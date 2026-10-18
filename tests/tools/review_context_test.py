"""Focused checks for tools/ci/review-context.py; no model or GPU required."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[2] / "tools/ci/review-context.py"


class ReviewContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="gufo-review-")
        self.root = Path(self.temporary.name)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "review-test@example.com")
        self.git("config", "user.name", "Review Test")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.autocrlf", "false")

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

    def commit(self, message: str) -> None:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def base_commit(self) -> None:
        self.write("README.md", "# Fixture\n")
        self.write("src/core/quant/ggml_dequant.cpp", "int base = 1;\n")
        self.write("src/cli/serve/openai_chat.cpp", "int base = 1;\n")
        self.write("src/models/qwen38_flash_next/weights.cpp", "int base = 1;\n")
        self.write("src/cli/serve/continuation_cache.cpp", "int base = 1;\n")
        self.write("compat/win32/posix.cpp", "int base = 1;\n")
        self.commit("chore: fixture base")

    def run_tool(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(TOOL), "--repo", str(self.root), *args],
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def packet(self, *args: str, base: str = "HEAD~1",
               head: str = "HEAD") -> dict:
        result = self.run_tool("--base", base, "--head", head,
                               "--format", "json", *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_classifies_areas_and_required_checks(self) -> None:
        self.base_commit()
        self.write("src/core/quant/ggml_dequant.cpp",
                   "// changed\nint base = 2;\n")
        self.write("src/cli/serve/openai_chat.cpp",
                   "// changed\nint base = 2;\n")
        self.write("compat/win32/posix.cpp", "// changed\nint base = 2;\n")
        self.write("docs/TESTING.md", "changed\n")
        self.commit("fix: exercise review context")

        packet = self.packet("--intent", "exercise review context")
        self.assertEqual(packet["intent"], "exercise review context")
        self.assertIn("core-quant", packet["areas"])
        self.assertIn("server", packet["areas"])
        self.assertIn("windows", packet["areas"])
        self.assertIn("docs", packet["areas"])

        names = {check["name"] for check in packet["checks"]}
        self.assertIn("format", names)
        self.assertIn("docs", names)
        self.assertIn("linux-ci", names)
        self.assertIn("cpu-targets", names)
        self.assertIn("numerical", names)
        self.assertIn("api", names)
        self.assertIn("windows", names)

        cpu = next(check for check in packet["checks"]
                   if check["name"] == "cpu-targets")
        self.assertIn("ggml_dequant_test", cpu["command"])
        self.assertIn("openai_chat_test", cpu["command"])

        titles = {flag["title"] for flag in packet["flags"]}
        self.assertIn("Arithmetic-sensitive paths changed", titles)
        self.assertIn("API or scheduler contract changed", titles)
        self.assertIn("Windows-specific paths changed", titles)

    def test_flags_missing_model_docs_and_shared_includes(self) -> None:
        self.base_commit()
        self.write(
            "src/models/qwen38_flash_next/weights.cpp",
            '#include "src/models/qwen/weights.hpp"\nint changed = 1;\n',
        )
        self.commit("feat: change model weights")

        packet = self.packet()
        titles = {flag["title"] for flag in packet["flags"]}
        self.assertIn("Model code changed without qwen3.8-flash-next docs",
                      titles)
        self.assertIn("Shared kernel/code across model packages", titles)
        self.assertIn("gpu", {check["name"] for check in packet["checks"]})

    def test_flags_new_model_package(self) -> None:
        self.base_commit()
        self.write("src/models/new_model/config.cpp", "int config = 1;\n")
        self.commit("feat: add new model package")

        packet = self.packet()
        titles = {flag["title"] for flag in packet["flags"]}
        self.assertIn("New model package: new_model", titles)
        self.assertIn("No test changes for model new_model", titles)

    def test_flags_state_change_without_payload_version(self) -> None:
        self.base_commit()
        self.write("src/cli/serve/continuation_cache.cpp",
                   "int cache = 2;\n")
        self.commit("fix: change cache state")

        packet = self.packet()
        titles = {flag["title"] for flag in packet["flags"]}
        self.assertIn("State files changed without a payload-version change",
                      titles)

    def test_flags_whitespace_churn(self) -> None:
        self.write("src/core/quant/ggml_dequant.cpp",
                   "".join(f"int value{i} = {i};\n" for i in range(60)))
        self.commit("chore: whitespace fixture base")
        self.write("src/core/quant/ggml_dequant.cpp",
                   "".join(f"int  value{i}  =  {i};\n" for i in range(60)))
        self.commit("style: reformat fixture")

        packet = self.packet()
        titles = {flag["title"] for flag in packet["flags"]}
        self.assertIn("Formatting or whitespace churn", titles)

    def test_markdown_output_and_read_only_behavior(self) -> None:
        self.base_commit()
        self.write("src/cli/serve/openai_chat.cpp", "// changed\n")
        self.commit("fix: change chat parsing")

        result = self.run_tool("--base", "HEAD~1", "--head", "HEAD",
                               "--intent", "change chat parsing")
        self.assertEqual(result.returncode, 0, result.stderr)
        for heading in ("# Review context:", "## Intent",
                        "## Recommended checks", "## Risk flags",
                        "## Review checklist",
                        "## Not verified by this tool"):
            self.assertIn(heading, result.stdout)
        self.assertIn("change chat parsing", result.stdout)

        status = self.git("status", "--porcelain=v1").stdout.strip()
        self.assertEqual(status, "", "review-context must not modify the tree")

    def test_empty_diff_and_untracked_file(self) -> None:
        self.base_commit()
        self.write("scratch.txt", "untracked\n")

        packet = self.packet(base="HEAD", head="HEAD")
        self.assertEqual(packet["files"], [])
        titles = {flag["title"] for flag in packet["flags"]}
        self.assertIn("Empty diff", titles)
        self.assertIn("Untracked files present", titles)
        self.assertIn("scratch.txt", packet["untracked"])

    def test_unknown_base_fails_cleanly(self) -> None:
        self.base_commit()
        result = self.run_tool("--base", "does-not-exist", "--head", "HEAD")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot resolve ref", result.stderr)

    def test_run_checks_requires_the_reviewed_head_to_be_checked_out(self) -> None:
        self.write("README.md", "# Base\n")
        self.write("tools/ci/check-docs.py", "print('old checker passed')\n")
        self.commit("chore: base checker")
        base = self.git("rev-parse", "HEAD").stdout.strip()
        self.write("README.md", "# Candidate\n")
        self.write("tools/ci/check-docs.py", "raise SystemExit('candidate check failed')\n")
        self.commit("fix: candidate checker")
        head = self.git("rev-parse", "HEAD").stdout.strip()
        self.git("checkout", "-q", base)
        result = self.run_tool("--base", base, "--head", head, "--run-checks")
        self.assertEqual(result.returncode, 2)
        self.assertIn("checkout HEAD to match --head", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(self.packet(base=base, head=head)["head_sha"], head)
        self.git("checkout", "-q", head)
        packet = self.packet("--run-checks", base=base, head=head)
        self.assertEqual(packet["cheap_checks"][0]["status"], "fail")
        self.assertIn("candidate check failed", packet["cheap_checks"][0]["output"])

    def test_rename_keeps_source_ownership_and_required_checks(self) -> None:
        self.base_commit()
        (self.root / "tools").mkdir()
        old = "src/core/quant/ggml_dequant.cpp"
        new = "tools/moved.cpp"
        self.git("mv", old, new)
        self.commit("refactor: move shared quantization")
        packet = self.packet()
        self.assertEqual(len(packet["files"]), 1)
        renamed = packet["files"][0]
        self.assertEqual((renamed["path"], renamed["old_path"]), (new, old))
        self.assertIn("core-quant", renamed["areas"])
        checks = {check["name"] for check in packet["checks"]}
        self.assertTrue({"numerical", "linux-ci", "windows", "cpu-targets"} <= checks)
        titles = {flag["title"] for flag in packet["flags"]}
        self.assertIn("Arithmetic-sensitive paths changed", titles)
        rendered = self.run_tool("--base", "HEAD~1")
        self.assertIn(f"{old} -> {new}", rendered.stdout)

    def test_numstat_handles_nested_renames_unicode_and_binary_paths(self) -> None:
        text = "".join(f"int value{i} = {i};\n" for i in range(100))
        self.write("src/models/example/old.cpp", text)
        self.commit("chore: rename fixture base")
        self.git("mv", "src/models/example/old.cpp", "src/models/example/new.cpp")
        self.write("src/models/example/new.cpp", text + "int added = 1;\n")
        unicode_path = "src/core/quant/non-ascii-\u00e9.cpp"
        binary_path = "artifacts/binary-\u00e9.bin"
        self.write(unicode_path, "int added = 1;\n")
        self.write(binary_path, "binary\0data")
        self.commit("fix: edit renamed and quoted paths")
        files = {item["path"]: item for item in self.packet()["files"]}
        for path in ("src/models/example/new.cpp", unicode_path):
            self.assertEqual((files[path]["added"], files[path]["deleted"]), (1, 0))
        self.assertTrue(files[binary_path]["binary"])

    def test_loader_and_tokenizer_changes_require_numerical_checks(self) -> None:
        self.base_commit()
        for path in (
            "src/models/qwen38_flash_next/weights.cpp",
            "src/models/qwen/config.cpp", "src/models/qwen/tokenizer.cpp",
            "src/core/gguf/reader.cpp",
        ):
            with self.subTest(path=path):
                self.write(path, "int changed = 2;\n")
                self.commit("fix: update loader or tokenizer")
                checks = {check["name"]: check for check in self.packet()["checks"]}
                self.assertEqual(checks["numerical"]["priority"], "required")

    def test_generated_ctest_filters_select_registered_names(self) -> None:
        self.base_commit()
        for model, test_names, target_names in (
            ("qwen38_flash_next", ("qwen38_flash_next.config", "qwen38_flash_next.mtp_sampling"),
             ("qwen38_flash_next_config_test", "qwen38_flash_next_mtp_sampling_test")),
            ("deepseek_v4_flash", ("ds4.sampling", "ds4.template", "ds4.cli"),
             ("ds4_sampling_test", "ds4_chat_template_test", "ds4_cli_test")),
        ):
            self.write(f"src/models/{model}/weights.cpp", "int changed = 2;\n")
            self.commit("fix: update model weights")
            with tempfile.TemporaryDirectory(prefix="gufo-review-ctest-") as directory:
                suite = Path(directory)
                python = json.dumps(sys.executable.replace("\\", "/"))
                (suite / "CTestTestfile.cmake").write_text("\n".join(
                    f'add_test({name} {python} "-c" "raise SystemExit(0)")'
                    for name in test_names
                ) + "\n")
                for platform in ("linux", "windows"):
                    with self.subTest(model=model, platform=platform):
                        packet = self.packet("--platform", platform)
                        command = next(check["command"] for check in packet["checks"]
                                       if check["name"] == "cpu-targets")
                        for target in target_names:
                            self.assertIn(target, command)
                        self.assertIn("--no-tests=error", command)
                        pattern = re.search(r"-R '([^']+)'", command).group(1)
                        self.assertFalse(re.fullmatch(pattern, test_names[0].replace(".", "X")))
                        result = subprocess.run(
                            ["ctest", "--test-dir", str(suite), "-R", pattern,
                             "--no-tests=error"], capture_output=True, text=True,
                        )
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn(f"{len(test_names)}/{len(test_names)}", result.stdout)


if __name__ == "__main__":
    unittest.main()
