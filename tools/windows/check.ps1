# Read-only prerequisite check for the stage-one CPU build.
param(
  [string]$Rocm = "C:\TheRock\build",
  [string]$Vcpkg = "C:\vcpkg",
  [string]$Ninja = "ninja"
)
$ErrorActionPreference = "Stop"
if (-not [Environment]::Is64BitOperatingSystem) { throw "Windows x64 is required" }
foreach ($tool in @("cmake", $Ninja, "python", "git")) {
  if (-not (Get-Command $tool -CommandType Application -ErrorAction SilentlyContinue)) {
    throw "$tool is required on PATH (Ninja may be supplied with -Ninja)"
  }
}
$cmakeVersion = [version]((& cmake --version | Select-Object -First 1) -replace "[^0-9.]", "")
if ($cmakeVersion -lt [version]"3.21") { throw "CMake 3.21 or newer is required" }
foreach ($file in @("$Rocm/lib/llvm/bin/clang.exe", "$Rocm/lib/llvm/bin/clang++.exe", "$Vcpkg/vcpkg.exe")) {
  if (-not (Test-Path -LiteralPath $file)) { throw "Missing $file; see docs/WINDOWS.md" }
}
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = if (Test-Path -LiteralPath $vswhere) {
  & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if (-not $vs) { throw "Install Visual Studio Build Tools with Desktop development with C++" }
Write-Host "CPU prerequisites found: CMake $cmakeVersion, TheRock clang, vcpkg, VS C++/SDK, Ninja, Python, Git"
