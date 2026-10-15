"""Without -Vcpkg, check.ps1 and build.ps1 (via find-vcpkg.ps1) use setup's recorded vcpkg."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.name == "nt", "Windows scripts require Windows")
class WindowsVcpkgPathTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gufo vcpkg [test] ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        # A copy of the scripts in a fake checkout, so it reads this test's
        # build\windows-setup.json instead of the developer's.
        scripts = self.root / "checkout/tools/windows"
        scripts.mkdir(parents=True)
        for script in ("check.ps1", "find-vcpkg.ps1"):
            shutil.copy(ROOT / "tools/windows" / script, scripts)
        self.check = scripts / "check.ps1"
        self.state = self.root / "checkout/build/windows-setup.json"
        self.state.parent.mkdir()
        self.saved = self.vcpkg("saved vcpkg")
        self.env_root = self.vcpkg("env vcpkg")
        self.shell = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
        self.assertIsNotNone(self.shell)

    def vcpkg(self, name):
        path = self.root / name
        (path / "scripts/buildsystems").mkdir(parents=True)
        (path / "scripts/buildsystems/vcpkg.cmake").touch()
        (path / "vcpkg.exe").touch()
        return path

    def found(self, *arguments, vcpkg_root=None):
        environment = {key: value for key, value in os.environ.items()
                       if key.lower() not in ("psmodulepath", "vcpkg_root")}
        if vcpkg_root:
            environment["VCPKG_ROOT"] = str(vcpkg_root)
        result = subprocess.run(
            [self.shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(self.check),
             *arguments],
            env=environment, capture_output=True, text=True, timeout=120)
        match = re.search(r"^\s+(ok|FAIL)\s+vcpkg\s+(.+?)\s*$", result.stdout, re.MULTILINE)
        self.assertIsNotNone(match, result.stdout + result.stderr)
        return match.group(1), match.group(2)

    def test_saved_checkout_then_vcpkg_root_then_explicit(self):
        self.state.write_text(json.dumps({"Vcpkg": str(self.saved)}), encoding="utf-8")
        self.assertEqual(self.found(vcpkg_root=self.env_root), ("ok", str(self.saved)))
        # An explicit path wins over the recorded one.
        self.assertEqual(self.found("-Vcpkg", str(self.env_root)), ("ok", str(self.env_root)))
        # A recorded checkout that has since moved falls through to VCPKG_ROOT.
        self.state.write_text(json.dumps({"Vcpkg": str(self.root / "moved")}), encoding="utf-8")
        self.assertEqual(self.found(vcpkg_root=self.env_root), ("ok", str(self.env_root)))

    def test_unreadable_setup_state_is_ignored(self):
        self.state.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.found(vcpkg_root=self.env_root), ("ok", str(self.env_root)))


if __name__ == "__main__":
    unittest.main()
