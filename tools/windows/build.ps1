# Windows build of Gufo against AMD TheRock ROCm (gfx1151).
#
# Prerequisites (paths overridable by parameter; tools\windows\check.ps1
# checks them all):
#   - TheRock 10.0.0 for Windows gfx1151, extracted to C:\TheRock\build:
#     https://stable.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx1151-10.0.0.tar.gz
#     (SHA-256 1293927b06b3b8d4bd7e0265823fb998bc9e0d83c68f33dcfa5d32663b30ce38)
#   - bootstrapped vcpkg (dependencies are pinned in vcpkg.json): -Vcpkg, else
#     the checkout setup-windows.bat recorded, else VCPKG_ROOT, else C:\vcpkg
#   - Visual Studio Build Tools (MSVC STL + Windows SDK; the compiler is TheRock clang)
#   - CMake 3.21+, and ninja on PATH or at C:\tools\ninja\ninja.exe
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1     # release -> build\release\gufo.exe
#   powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset gpu-test -Target <target>
param(
  [string]$Preset = "release",
  [string]$Target = "",
  [string]$Rocm = "C:\TheRock\build",
  [string]$Vcpkg = "",
  [string]$Ninja = "C:\tools\ninja\ninja.exe",
  [int]$Jobs = 0,
  [switch]$Reconfigure,
  [switch]$KeepGoing,
  [string[]]$CMakeArgs = @()
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath "$PSScriptRoot\..\..").Path
$revision = "development"
if ((Test-Path -LiteralPath "$root\.git") -and (Get-Command git -ErrorAction SilentlyContinue)) {
  $revision = & git -C $root describe --always --long --dirty --abbrev=12
  if ($LASTEXITCODE -ne 0) { throw "cannot identify the source revision" }
}
Write-Host "Gufo revision: $revision"

# The port is validated on one TheRock release; a newer clang can round the
# fused kernels differently (see docs\WINDOWS.md).
$validatedRock = "10.0.0"
$rockVersionFile = Join-Path $Rocm ".info\version"
if (-not (Test-Path -LiteralPath $rockVersionFile)) {
  throw "TheRock not found at $Rocm (see docs\WINDOWS.md, or run tools\windows\check.ps1)"
}
$rockVersion = (Get-Content -LiteralPath $rockVersionFile -TotalCount 1).Trim()
if ($rockVersion -ne $validatedRock) {
  Write-Warning "TheRock $rockVersion at $Rocm; this port is validated on $validatedRock"
}

# Ninja: the -Ninja path, else the one on PATH (as tools\windows\check.ps1 accepts).
if (-not (Test-Path -LiteralPath $Ninja)) {
  $onPath = Get-Command ninja -ErrorAction SilentlyContinue
  if (-not $onPath) { throw "ninja not found at $Ninja or on PATH (see tools\windows\check.ps1)" }
  $Ninja = $onPath.Source
}

if (-not $Vcpkg) {
  . "$PSScriptRoot\find-vcpkg.ps1"
  $Vcpkg = Find-Vcpkg $root
}

# Import the MSVC developer environment (headers, libs, link.exe).
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio C++ build tools not found" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$env:PATH = "$(Split-Path $vswhere);$env:PATH"
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
  if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}
if ($LASTEXITCODE -ne 0) { throw "Visual Studio environment setup failed (exit $LASTEXITCODE): $vcvars" }

$env:HIP_PATH = $Rocm.Replace('\', '/')
$env:ROCM_PATH = $Rocm.Replace('\', '/')
$env:HIP_DEVICE_LIB_PATH = "$Rocm/lib/llvm/amdgcn/bitcode".Replace('\', '/')
$env:PATH = "$Rocm\bin;$Rocm\lib\llvm\bin;$(Split-Path $Ninja);$env:PATH"
$clang = "$Rocm\lib\llvm\bin\clang.exe".Replace('\', '/')
$clangxx = "$Rocm\lib\llvm\bin\clang++.exe".Replace('\', '/')

$build = Join-Path $root "build\$Preset"
# Clear cached global-library paths when upgrading a classic-mode build.
if ((Test-Path -LiteralPath "$build\CMakeCache.txt") -and ($Reconfigure -or
    (Select-String -LiteralPath "$build\CMakeCache.txt" -Pattern '^VCPKG_MANIFEST_MODE:BOOL=OFF$' -Quiet))) {
  Remove-Item -LiteralPath "$build\CMakeCache.txt"
}
# Reconfigure so the manifest, tool paths and CMakeArgs take effect on updates.
cmake -S $root --preset $Preset `
  "-DCMAKE_MAKE_PROGRAM=$($Ninja.Replace('\', '/'))" `
  "-DCMAKE_C_COMPILER=$clang" `
  "-DCMAKE_CXX_COMPILER=$clangxx" `
  "-DCMAKE_HIP_COMPILER=$clangxx" `
  "-DCMAKE_TOOLCHAIN_FILE=$($Vcpkg.Replace('\', '/'))/scripts/buildsystems/vcpkg.cmake" `
  "-DVCPKG_TARGET_TRIPLET=x64-windows" `
  "-DVCPKG_MANIFEST_MODE=ON" `
  "-DVCPKG_MANIFEST_DIR=$($root.Replace('\', '/'))" `
  "-DVCPKG_INSTALLED_DIR=$($root.Replace('\', '/'))/build/vcpkg_installed" `
  "-DCMAKE_PREFIX_PATH=$($Rocm.Replace('\', '/'))" `
  "-DCMAKE_LINKER_TYPE=LLD" `
  "-DGUFO_VERSION=$revision" `
  @CMakeArgs
if ($LASTEXITCODE -ne 0) { throw "configure failed" }

$buildArgs = @("--build", $build)
if ($Target) { $buildArgs += @("--target", $Target) }
if ($Jobs -gt 0) { $buildArgs += @("--parallel", $Jobs) }
if ($KeepGoing) { $buildArgs += @("--", "-k", "0") }
cmake @buildArgs
if ($LASTEXITCODE -ne 0) { throw "build failed" }

# Runtime DLLs next to all executables: ROCm (amdhip64, hipblas, hipblaslt, rocblas
# and their kernel libraries) and vcpkg's (icu, curl, ssl, png, jpeg, zlib).
function Copy-RuntimeItem {
  param(
    [Parameter(ValueFromPipeline = $true)][IO.FileSystemInfo]$Source,
    [string]$Destination
  )
  process {
    $target = Join-Path $Destination $Source.Name
    if ($Source.PSIsContainer) {
      New-Item -ItemType Directory -Path $target -Force | Out-Null
      Get-ChildItem -LiteralPath $Source.FullName -Force | Copy-RuntimeItem -Destination $target
    } else {
      $existing = if (Test-Path -LiteralPath $target) { Get-Item -LiteralPath $target -Force } else { $null }
      if (-not $existing -or $existing.Length -ne $Source.Length -or
          $existing.LastWriteTimeUtc -ne $Source.LastWriteTimeUtc) {
        Copy-Item -LiteralPath $Source.FullName -Destination $target -Force
      }
    }
  }
}

$bin = $build
$rocmDlls = @("amdhip64_7.dll", "amd_comgr*.dll", "hipblas.dll", "libhipblaslt.dll", "origami.dll", "rocblas.dll", "rocsolver.dll", "rocsparse.dll", "rocm_kpack.dll", "rocm-openblas*.dll", "hiprtc*.dll")
foreach ($pattern in $rocmDlls) {
  Get-ChildItem -LiteralPath "$Rocm\bin" -Filter $pattern -File -ErrorAction SilentlyContinue | Copy-RuntimeItem -Destination $bin
}
foreach ($dir in @("hipblaslt", "rocblas")) {
  $libraryDir = Join-Path "$Rocm\bin" $dir
  if (Test-Path -LiteralPath $libraryDir) {
    Get-Item -LiteralPath $libraryDir | Copy-RuntimeItem -Destination $bin
  }
}
Get-ChildItem -LiteralPath "$root\build\vcpkg_installed\x64-windows\bin" -Filter '*.dll' -File | Copy-RuntimeItem -Destination $bin

# Source builds must also work outside vcvars64's PATH. These copies are for
# this developer's local use, NOT a redistributable release: VS's LLVM OpenMP
# runtime lives in debug_nonredist. Binary packaging needs a different runtime.
if ($env:VCToolsRedistDir) {
  Get-ChildItem -LiteralPath "$env:VCToolsRedistDir\x64" -Directory -Filter 'Microsoft.VC*.CRT' |
    ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Filter '*.dll' -File } | Copy-RuntimeItem -Destination $bin
}
$openmp = Select-String -LiteralPath "$build\CMakeCache.txt" -Pattern '^OpenMP_libomp_LIBRARY:FILEPATH=(.*)$'
if ($openmp -and $openmp.Matches[0].Groups[1].Value -match '^(.*)/VC/Tools/MSVC/([^/]+)/lib/x64/libomp\.lib$') {
  $redistRoot = Join-Path $Matches[1] 'VC\Redist\MSVC'
  $toolVersion = $Matches[2]
  $redistVersions = @()
  $exactRedist = Join-Path $redistRoot $toolVersion
  if (Test-Path -LiteralPath $exactRedist) {
    $redistVersions += Get-Item -LiteralPath $exactRedist
  }
  $redistVersions += Get-ChildItem -LiteralPath $redistRoot -Directory -Filter '14.*' -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne $toolVersion } |
    Sort-Object { [version]$_.Name } -Descending
  $runtime = $null
  foreach ($redistVersion in $redistVersions) {
    $runtimeRoot = Join-Path $redistVersion.FullName 'debug_nonredist\x64'
    $runtime = Get-ChildItem -LiteralPath $runtimeRoot -Directory -Filter 'Microsoft.VC*.OpenMP.LLVM' -ErrorAction SilentlyContinue |
      ForEach-Object { Get-Item -LiteralPath (Join-Path $_.FullName 'libomp140.x86_64.dll') -ErrorAction SilentlyContinue } |
      Where-Object { $_ } |
      Select-Object -First 1
    if ($runtime) { break }
  }
  if (-not $runtime) {
    $available = ($redistVersions | ForEach-Object Name) -join ', '
    throw "Cannot find a Visual Studio LLVM OpenMP runtime for MSVC $toolVersion or an installed fallback ($available)."
  }
  Write-Host "gufo: staging Visual Studio LLVM OpenMP runtime from $($runtime.FullName)"
  $runtime | Copy-RuntimeItem -Destination $bin
}
Write-Host "gufo: built $build"
