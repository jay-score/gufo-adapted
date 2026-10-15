"""Exercise GUI preparation and build staging with native stubs; no installs or GPU."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.name == "nt", "Windows entrypoints require Windows")
class WindowsEntrypointsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binaries = tempfile.TemporaryDirectory(prefix="gufo script binaries ")
        cls.addClassCleanup(cls.binaries.cleanup)
        cls.stub = Path(cls.binaries.name) / "stub.exe"
        source = cls.stub.with_suffix(".cs")
        source.write_text(r'''using System;
using System.IO;
class Stub {
  static int Main(string[] args) {
    string name = Path.GetFileNameWithoutExtension(Environment.GetCommandLineArgs()[0]);
    File.AppendAllText(Environment.GetEnvironmentVariable("GUFO_SCRIPT_LOG"),
      name + " " + string.Join("|", args) + "\n");
    if (name == "vswhere") {
      Console.WriteLine(Environment.GetEnvironmentVariable("GUFO_SCRIPT_VS"));
    } else if (name == "cmake" && args[0] == "-S") {
      string build = Path.Combine(args[1], "build", "release");
      Directory.CreateDirectory(build);
      File.WriteAllText(Path.Combine(build, "CMakeCache.txt"),
        "OpenMP_libomp_LIBRARY:FILEPATH=" + Environment.GetEnvironmentVariable("GUFO_SCRIPT_OPENMP") + "\n");
    } else if (name == "python") {
      string ready = Environment.GetEnvironmentVariable("GUFO_SCRIPT_READY");
      if (args[0] == "-c" && !File.Exists(ready)) return 1;
      if (args.Length > 1 && args[1] == "pip") {
        if (Environment.GetEnvironmentVariable("GUFO_SCRIPT_PIP_FAIL") == "1") return 23;
        File.WriteAllText(ready, "ready");
      }
    }
    return 0;
  }
}
''')
        compiler = Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
        subprocess.run([str(compiler), "/nologo", f"/out:{cls.stub}", str(source)],
                       check=True, capture_output=True, text=True)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gufo scripts [test] & spaces ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "checkout [test]"
        self.scripts = self.checkout / "tools/windows"
        self.scripts.mkdir(parents=True)
        for name in ("gui.ps1", "build.ps1"):
            shutil.copy2(ROOT / "tools/windows" / name, self.scripts / name)
        self.shells = [path for name in ("powershell.exe", "pwsh.exe")
                       if (path := shutil.which(name))]
        self.assertTrue(self.shells)
        self.environment = {key: value for key, value in os.environ.items()
                            if key.lower() not in ("psmodulepath", "vctoolsredistdir")}
        self.log = self.root / "commands.txt"
        self.ready = self.root / "packages-ready"
        self.environment.update(GUFO_SCRIPT_LOG=str(self.log), GUFO_SCRIPT_READY=str(self.ready))

    def binary(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.stub, path)
        return path

    def run_script(self, shell, name, *arguments):
        return subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                               str(self.scripts / name), *map(str, arguments)], cwd=self.root,
                              env=self.environment, capture_output=True, text=True, timeout=30)

    def prepare_gui(self):
        self.binary(self.checkout / "build/gui-env/Scripts/python.exe")
        requirements = self.checkout / "tools/gui/requirements.txt"
        requirements.parent.mkdir(parents=True)
        requirements.write_bytes((ROOT / "tools/gui/requirements.txt").read_bytes())
        self.stamp = self.checkout / "build/gui-env/gufo-requirements.sha256"
        self.hash = hashlib.sha256(requirements.read_bytes()).hexdigest().upper()
        self.stamp.write_text(self.hash)

    def test_gui_repairs_missing_packages_and_reuses_healthy_environment(self):
        self.prepare_gui()
        for shell in self.shells:
            with self.subTest(shell=shell):
                self.ready.unlink(missing_ok=True)
                self.log.unlink(missing_ok=True)
                result = self.run_script(shell, "gui.ps1", "-PrepareOnly")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                commands = self.log.read_text()
                self.assertEqual(commands.count("-m|pip|install|--force-reinstall|-r|"), 1)
                self.assertIn("-m|ensurepip|--upgrade", commands)
                self.assertEqual(self.stamp.read_text().strip(), self.hash)
                self.log.unlink()
                result = self.run_script(shell, "gui.ps1", "-PrepareOnly")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.log.read_text().splitlines(), ["python -c|import flask, waitress"])

    def test_gui_failed_repair_invalidates_stamp_and_stops_before_launch(self):
        self.prepare_gui()
        self.environment["GUFO_SCRIPT_PIP_FAIL"] = "1"
        for shell in self.shells:
            with self.subTest(shell=shell):
                self.stamp.write_text(self.hash)
                result = self.run_script(shell, "gui.ps1")
                self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
                self.assertFalse(self.stamp.exists())
                self.assertNotIn("server.py", self.log.read_text())

    def test_gui_batch_forwards_options_from_another_directory_and_exits_cleanly(self):
        self.prepare_gui()
        self.ready.touch()
        batch = self.checkout / "launch-gui.bat"
        shutil.copy2(ROOT / "launch-gui.bat", batch)
        config = self.root / "settings [test] & spaces.json"
        command = (f'"{os.environ["COMSPEC"]}" /d /s /c ""{batch}" '
                   f'-Port 8091 -Config "{config}" -NoBrowser"')
        result = subprocess.run(command, cwd=self.root, env=self.environment,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"--port|8091|--config|{config}|--no-browser", self.log.read_text())

    def prepare_build(self, vcvars_exit=0):
        program_files = self.root / "program files [test]"
        self.binary(program_files / "Microsoft Visual Studio/Installer/vswhere.exe")
        vs = self.root / "VS [test] & tools"
        vcvars = vs / "VC/Auxiliary/Build/vcvars64.bat"
        vcvars.parent.mkdir(parents=True)
        vcvars.write_text(f"@exit /b {vcvars_exit}\n")
        self.environment["ProgramFiles(x86)"] = str(program_files)
        self.environment["GUFO_SCRIPT_VS"] = str(vs)
        self.environment["GUFO_SCRIPT_OPENMP"] = (vs / "VC/Tools/MSVC/14.51/lib/x64/libomp.lib").as_posix()
        redist = vs / "VC/Redist/MSVC/14.51"
        crt = redist / "x64/Microsoft.VC145.CRT"
        crt.mkdir(parents=True)
        (crt / "msvcp140.dll").write_bytes(b"crt")
        openmp = redist / "debug_nonredist/x64/Microsoft.VC145.OpenMP.LLVM"
        openmp.mkdir(parents=True)
        (openmp / "libomp140.x86_64.dll").write_bytes(b"openmp")
        self.environment["VCToolsRedistDir"] = str(redist)
        tool_bin = self.root / "tools [test]"
        self.binary(tool_bin / "cmake.exe")
        ninja = self.binary(tool_bin / "ninja.exe")
        self.environment["PATH"] = str(tool_bin) + os.pathsep + self.environment["PATH"]
        self.rock = self.root / "rock [test]"
        (self.rock / ".info").mkdir(parents=True)
        (self.rock / ".info/version").write_text("10.0.0\n")
        (self.rock / "bin/rocblas/nested [test]").mkdir(parents=True)
        (self.rock / "bin/amdhip64_7.dll").write_bytes(b"runtime")
        (self.rock / "bin/rocblas/nested [test]/kernel.bin").write_bytes(b"kernel")
        vcpkg_bin = self.checkout / "build/vcpkg_installed/x64-windows/bin"
        vcpkg_bin.mkdir(parents=True)
        (vcpkg_bin / "dependency.dll").write_bytes(b"dependency")
        self.build = self.checkout / "build/release"
        self.build_arguments = ("-Target", "gufo", "-Rocm", self.rock,
                                "-Vcpkg", self.root / "vcpkg [test]", "-Ninja", ninja)

    def test_build_stops_before_cmake_when_vcvars_fails(self):
        self.prepare_build(vcvars_exit=17)
        for shell in self.shells:
            with self.subTest(shell=shell):
                result = self.run_script(shell, "build.ps1", *self.build_arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Visual Studio environment setup failed (exit 17)",
                              result.stdout + result.stderr)
                self.assertNotIn("cmake ", self.log.read_text())

    def test_build_stages_nested_runtime_files_and_skips_unchanged_files(self):
        self.prepare_build()
        for shell in self.shells:
            with self.subTest(shell=shell):
                result = self.run_script(shell, "build.ps1", *self.build_arguments)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual((self.build / "rocblas/nested [test]/kernel.bin").read_bytes(), b"kernel")
                self.assertEqual((self.build / "dependency.dll").read_bytes(), b"dependency")
                self.assertEqual((self.build / "msvcp140.dll").read_bytes(), b"crt")
                self.assertEqual((self.build / "libomp140.x86_64.dll").read_bytes(), b"openmp")
                # An exclusive lock proves a no-op rebuild does not try to copy this DLL.
                harness = self.root / "locked-build.ps1"
                def quote(value):
                    return "'" + str(value).replace("'", "''") + "'"
                harness.write_text(
                    f"$file = [IO.File]::Open({quote(self.build / 'amdhip64_7.dll')}, "
                    "'Open', 'ReadWrite', 'None')\ntry {\n"
                    f"& {quote(shell)} -NoProfile -ExecutionPolicy Bypass -File "
                    f"{quote(self.scripts / 'build.ps1')} "
                    + " ".join(map(quote, self.build_arguments))
                    + "\nexit $LASTEXITCODE\n} finally { $file.Dispose() }\n",
                    encoding="utf-8-sig",
                )
                result = subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass",
                                         "-File", str(harness)], env=self.environment,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                # Same-size changes with a new timestamp and size-only changes must copy.
                source = self.rock / "bin/amdhip64_7.dll"
                source.write_bytes(b"changed")
                timestamp = (self.build / source.name).stat().st_mtime + 10
                os.utime(source, (timestamp, timestamp))
                nested = self.rock / "bin/rocblas/nested [test]/kernel.bin"
                old_time = nested.stat().st_mtime
                nested.write_bytes(b"larger kernel")
                os.utime(nested, (old_time, old_time))
                result = self.run_script(shell, "build.ps1", *self.build_arguments)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual((self.build / source.name).read_bytes(), b"changed")
                self.assertEqual((self.build / "rocblas/nested [test]/kernel.bin").read_bytes(), b"larger kernel")


if __name__ == "__main__":
    unittest.main()
