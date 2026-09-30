"""Exercise sync review and worktree safety with disposable Git repositories."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[2] / "tools/upstream-sync.py"


class UpstreamSyncTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gufo-sync-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "repo with spaces"
        self.root.mkdir()
        self.git("init", "-b", "main")
        self.git("config", "user.name", "Sync test")
        self.git("config", "user.email", "sync@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.autocrlf", "false")
        (self.root / "tools").mkdir()
        shutil.copy2(SOURCE, self.root / "tools/upstream-sync.py")
        (self.root / ".gitignore").write_text("build/\n", encoding="utf-8")
        self.commit("shared.txt", "common\n", "initial baseline")
        self.common = self.git("rev-parse", "HEAD")
        self.commit("exact.txt", "fix\n", "fix(server): exact fix")
        self.exact = self.git("rev-parse", "HEAD")
        self.commit("shared.txt", "upstream fix\n", "fix(cache): upstream fix")
        self.adapted = self.git("rev-parse", "HEAD")
        self.commit("feature.txt", "feature\n", "feat(server): optional feature")
        self.tip = self.git("rev-parse", "HEAD")
        self.git("remote", "add", "official", "https://github.com/gufo-org/gufo.git")
        self.git("update-ref", "refs/remotes/official/main", self.tip)
        self.git("checkout", "-b", "windows-port", self.common)
        self.git("cherry-pick", self.exact)
        self.git("commit", "--amend", "-m", "fix(server): imported exact fix")
        self.commit("shared.txt", "adapted Windows fix\n", f"fix(cache): adapted fix (upstream {self.adapted})")
        self.base = self.git("rev-parse", "HEAD")

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.root), *args], check=True,
            capture_output=True, text=True, encoding="utf-8",
        ).stdout.strip()

    def commit(self, path, content, message):
        (self.root / path).write_text(content, encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-m", message)

    def run_sync(self, *args):
        return subprocess.run(
            [sys.executable, str(self.root / "tools/upstream-sync.py"), "--offline", *args],
            cwd=self.directory, capture_output=True, text=True, encoding="utf-8",
        )

    def report(self, tree=None):
        path = next(((tree or self.root) / "build/upstream-sync").glob("*.json"))
        return json.loads(path.read_text(encoding="utf-8"))

    def test_review_recognizes_imports_and_preserves_local_edits(self):
        (self.root / "shared.txt").write_text("unsaved local edit\n", encoding="utf-8")
        (self.root / "untracked.txt").write_text("local\n", encoding="utf-8")
        status = self.git("status", "--porcelain")
        result = self.run_sync()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.report()
        incoming = {commit["sha"]: commit for commit in report["incoming"]}
        self.assertTrue(incoming[self.exact]["patch_equivalent"])
        self.assertEqual(incoming[self.adapted]["recorded_references"], [self.base])
        self.assertFalse(incoming[self.adapted]["patch_equivalent"])
        self.assertEqual(report["overlapping_paths"], ["exact.txt", "shared.txt"])
        self.assertEqual(self.git("status", "--porcelain"), status)
        self.assertEqual(self.git("rev-parse", "HEAD"), self.base)

    def test_prepare_pins_base_without_merging_or_touching_dirty_checkout(self):
        (self.root / "shared.txt").write_text("unsaved\n", encoding="utf-8")
        worktree = self.directory / "sync worktree"
        result = self.run_sync("--prepare", str(worktree), "--branch", "sync/test")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git("rev-parse", "sync/test"), self.base)
        self.assertFalse((worktree / "feature.txt").exists())
        self.assertEqual((self.root / "shared.txt").read_text(), "unsaved\n")
        self.assertEqual(self.report(worktree)["official_sha"], self.tip)
        self.assertIn("git merge --no-ff --no-commit " + self.tip, result.stdout)

    def test_existing_path_and_branch_are_preserved(self):
        occupied = self.directory / "occupied"
        occupied.mkdir()
        sentinel = occupied / "sentinel.txt"
        sentinel.write_text("keep\n")
        result = self.run_sync("--prepare", str(occupied), "--branch", "sync/new")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(), "keep\n")
        self.git("branch", "sync/existing", self.common)
        result = self.run_sync("--prepare", str(self.directory / "unused"), "--branch", "sync/existing")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.git("rev-parse", "sync/existing"), self.common)
        self.assertEqual(self.git("rev-parse", "HEAD"), self.base)

    def test_foreign_tip_or_remote_cannot_be_treated_as_official(self):
        result = self.run_sync("--tip", "windows-port")
        self.assertNotEqual(result.returncode, 0)
        self.git("remote", "set-url", "official", "https://github.com/other/gufo.git")
        result = self.run_sync()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.git("remote", "get-url", "official"), "https://github.com/other/gufo.git")

    def test_check_failure_is_returned_and_recorded(self):
        scripts = self.root / "tools/ci"
        scripts.mkdir()
        (scripts / "check-docs.py").write_text("raise SystemExit(17)\n", encoding="utf-8")
        result = self.run_sync("--check")
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(self.report()["checks"], [{"script": "tools/ci/check-docs.py", "exit_code": 17}])

    def test_shallow_history_is_rejected(self):
        shallow = self.directory / "shallow"
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", "windows-port", self.root.as_uri(), str(shallow)],
            check=True, capture_output=True,
        )
        result = subprocess.run(
            [sys.executable, str(shallow / "tools/upstream-sync.py"), "--offline"],
            capture_output=True, text=True, encoding="utf-8",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Full history is required", result.stderr)
        self.assertFalse((shallow / "build/upstream-sync").exists())


if __name__ == "__main__":
    unittest.main()
