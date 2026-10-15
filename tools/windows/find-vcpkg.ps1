# The vcpkg checkout build.ps1 and check.ps1 use without -Vcpkg: the one
# setup-windows.bat recorded, else VCPKG_ROOT, else C:\vcpkg (the order
# setup.ps1 searches). Returns C:\vcpkg when none is a checkout, so callers
# report it as missing there.
function Find-Vcpkg([string]$Root) {
  $saved = $null
  try {
    $saved = (Get-Content -LiteralPath "$Root\build\windows-setup.json" -Raw -ErrorAction Stop |
      ConvertFrom-Json).Vcpkg
  } catch {}
  $found = @($saved, $env:VCPKG_ROOT, "C:\vcpkg") |
    Where-Object { $_ -and (Test-Path -LiteralPath "$_\scripts\buildsystems\vcpkg.cmake") } |
    Select-Object -First 1
  if ($found) { return $found }
  return "C:\vcpkg"
}
