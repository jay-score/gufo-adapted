"""Installer safety checks with local Git fixtures; no installs or removals."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
GIT = shutil.which("git")


@unittest.skipUnless(os.name == "nt" and POWERSHELL and GIT, "Windows and Git are required")
class InstallerBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gufo-installer-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.checkout = self.directory / "checkout"
        self.checkout.mkdir()
        (self.checkout / "vcpkg.json").write_text("{}")
        (self.checkout / "setup-windows.bat").write_text("@echo off\n")
        shutil.copy2(ROOT / ".gitignore", self.checkout / ".gitignore")
        for args in (("init",), ("config", "user.name", "Installer test"),
                     ("config", "user.email", "test@example.invalid"), ("add", "."),
                     ("commit", "-m", "fixture"),
                     ("update-ref", "refs/remotes/origin/windows-main", "HEAD")):
            self.git(*args)

    def git(self, *args):
        return subprocess.run([GIT, "-C", str(self.checkout), *args], check=True,
                              capture_output=True, text=True).stdout

    def run_ps(self, body):
        script = self.directory / "test.ps1"
        script.write_text(r'''
. $env:GUFO_SETUP
$env:LOCALAPPDATA = Join-Path $env:GUFO_CASE 'appdata'
function Assert($condition, $message) { if (-not $condition) { throw $message } }
function Get-GufoShortcutPath { return (Join-Path $env:GUFO_CASE 'desktop.lnk') }
function Get-GufoStartMenuShortcutPath { return (Join-Path $env:GUFO_CASE 'setup.lnk') }
function Get-GufoEngineRunning { return $false }
$removed = New-Object System.Collections.Generic.List[string]
function Remove-GufoPath { param([string]$Path) $script:removed.Add($Path) }
$state = New-GufoState -Scheme (New-GufoScheme -Mode None) -InstallDir $env:GUFO_CHECKOUT -Interactive:$false
''' + body, encoding="utf-8-sig")
        result = subprocess.run(
            [POWERSHELL, "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, env={**os.environ, "GUFO_SETUP": str(ROOT / "tools/installer/Gufo-Setup.ps1"),
                             "GUFO_CASE": str(self.directory), "GUFO_CHECKOUT": str(self.checkout)},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ignored_models_are_kept_even_if_source_is_manually_selected(self):
        for name in ("models/personal.gguf", "weights/model.safetensors", "weights/training.ckpt",
                     "weights/pytorch_model-00001.bin", "PERSONAL.GGUF", "models/tokenizer.json"):
            with self.subTest(name=name):
                model = self.checkout / name
                model.parent.mkdir(exist_ok=True)
                model.write_text("model data")
                if name != "PERSONAL.GGUF":
                    self.assertEqual(self.git("status", "--porcelain"), "")
                self.run_ps(r'''
$source = @(Get-GufoUninstallItems $state | Where-Object { $_.Key -eq 'source' })[0]
Assert (-not $source.Checked) 'Model checkout was selected by default'
Assert ($source.Warning -match 'model') 'Missing model warning'
$source.Checked = $true
$result = Invoke-GufoRemove $state @($source)
Assert (-not $result.Ok) 'Model checkout passed the removal guard'
Assert ($removed.Count -eq 0) 'Model checkout reached the deletion helper'
''')
                model.unlink()

    def test_model_guard_is_rechecked_at_removal(self):
        self.run_ps(r'''
$source = @(Get-GufoUninstallItems $state | Where-Object { $_.Key -eq 'source' })[0]
Assert $source.Checked 'Clean published checkout was not selected'
[IO.File]::WriteAllText((Join-Path $state.InstallDir 'download.gguf'), 'new model')
$result = Invoke-GufoRemove $state @($source)
Assert (-not $result.Ok) 'New model was not protected at removal'
Assert ($removed.Count -eq 0) 'New model reached the deletion helper'
''')

    def test_clean_published_checkout_can_be_removed(self):
        self.run_ps(r'''
$source = @(Get-GufoUninstallItems $state | Where-Object { $_.Key -eq 'source' })[0]
Assert $source.Checked 'Clean published checkout was not selected'
$result = Invoke-GufoRemove $state @($source)
Assert $result.Ok 'Clean published checkout failed the removal guard'
Assert ($removed.Count -eq 1) 'Expected one call to the simulated deletion helper'
Assert ($removed[0] -eq $state.InstallDir) 'Wrong folder reached the deletion helper'
''')

    def test_source_is_kept_when_models_cannot_be_checked(self):
        self.run_ps(r'''
function Find-GufoGit { return '' }
$source = @(Get-GufoUninstallItems $state | Where-Object { $_.Key -eq 'source' })[0]
Assert (-not $source.Checked) 'Source was selected without Git'
$source.Checked = $true
Assert (-not (Invoke-GufoRemove $state @($source)).Ok) 'Source was removable without Git'
Assert ($removed.Count -eq 0) 'Unchecked source reached the deletion helper'
''')
        # A source archive has the same source markers, but no Git metadata.
        archive = self.directory / "archive"
        archive.mkdir()
        for name in ("vcpkg.json", "setup-windows.bat"):
            shutil.copy2(self.checkout / name, archive / name)
        self.run_ps(r'''
$state.InstallDir = Join-Path $env:GUFO_CASE 'archive'
$source = @(Get-GufoUninstallItems $state | Where-Object { $_.Key -eq 'source' })[0]
Assert (-not $source.Checked) 'Source archive was selected without metadata'
Assert (-not (Test-GufoRemovable $state $source)) 'Source archive was removable without metadata'
''')

    def test_unpublished_commits_on_any_branch_are_not_selected(self):
        (self.checkout / "local-work.txt").write_text("only copy")
        self.git("add", "local-work.txt")
        self.git("commit", "-m", "local work")
        self.run_ps(r'''
$source = @(Get-GufoUninstallItems $state | Where-Object { $_.Key -eq 'source' })[0]
Assert (-not $source.Checked) 'Unpublished HEAD was selected'
Assert ($source.Warning -match 'unpublished') 'Missing unpublished-work warning'
''')
        self.git("branch", "local-work")
        self.git("checkout", "--detach", "refs/remotes/origin/windows-main")
        self.run_ps(r'''
$source = @(Get-GufoUninstallItems $state | Where-Object { $_.Key -eq 'source' })[0]
Assert (-not $source.Checked) 'Unpublished work on another branch was selected'
''')

    def test_stashes_are_not_selected(self):
        (self.checkout / "vcpkg.json").write_text('{"local":"work"}')
        self.git("stash", "push", "-m", "only copy")
        self.run_ps(r'''
$source = @(Get-GufoUninstallItems $state | Where-Object { $_.Key -eq 'source' })[0]
Assert (-not $source.Checked) 'Checkout containing a stash was selected'
Assert ($source.Warning -match 'stash') 'Missing stash warning'
''')

    def test_unattended_uninstall_skips_protected_source(self):
        (self.checkout / "personal.gguf").write_text("model data")
        self.run_ps(r'''
$state.Machine = @{ Cpu='fixture'; Os='Windows'; Build=26200; RamGiB=64; Problem=$null }
$Action = 'uninstall'
$Yes = $true
Assert ((Invoke-GufoPlain $state) -eq 0) 'Unattended uninstall failed'
Assert ($removed.Count -eq 0) 'Unattended uninstall removed protected source'
''')

    def test_switching_install_folder_keeps_models_in_the_abandoned_clone(self):
        (self.checkout / "personal.gguf").write_text("model data")
        self.run_ps(r'''
$state.AdvancedReturn = 'review'
$state.Plan = @{ root=$state.InstallDir }
$state.PlanKey = Get-GufoPlanKey $state
$state.FetchedDir = $state.InstallDir
$state.InstallDir = Join-Path $env:GUFO_CASE 'different-checkout'
function Start-GufoPrepare { }
Close-GufoAdvanced $state
Assert ($removed.Count -eq 0) 'Switching folders deleted models in the abandoned clone'
''')

    def test_explicit_shortcut_choice_overrides_saved_preference(self):
        self.run_ps(r'''
New-Item -ItemType Directory -Force $state.SettingsDir | Out-Null
$path = Join-Path $state.SettingsDir 'installer.json'
[IO.File]::WriteAllText($path, '{"shortcut":true}')
$explicit = New-GufoState -Shortcut:$false -InstallDir $state.InstallDir
Assert (-not $explicit.Shortcut) 'Explicit NoShortcut was overridden'
[IO.File]::WriteAllText($path, '{"shortcut":false}')
$saved = New-GufoState -InstallDir $state.InstallDir
Assert (-not $saved.Shortcut) 'Saved preference was lost without an explicit choice'
[IO.File]::WriteAllText($path, '{"shortcut":true}')
$NoShortcut = $true
$Plain = $true
$InstallDir = $state.InstallDir
function Get-GufoMachine { return @{} }
function Invoke-GufoPlain {
  param($State)
  Assert (-not $State.Shortcut) 'Installer ignored the NoShortcut flag'
  return 0
}
Assert ((Invoke-GufoInstaller) -eq 0) 'Installer failed'
''')


if __name__ == "__main__":
    unittest.main()
