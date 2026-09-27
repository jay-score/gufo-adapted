# Start the optional local web launcher. The first run creates an isolated venv.
param(
  [int]$Port = 8090,
  [string]$Config = "",
  [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$environment = Join-Path $root "build\gui-env"
$python = Join-Path $environment "Scripts\python.exe"
$requirements = Join-Path $root "tools\gui\requirements.txt"
$stamp = Join-Path $environment "gufo-requirements.sha256"
if (-not (Test-Path $python)) {
  Write-Host "Creating the Gufo launcher environment (Python 3.10+ required)..."
  & python -m venv $environment
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $requirements).Hash
$installedHash = if (Test-Path $stamp) { [string](Get-Content -LiteralPath $stamp -Raw) } else { "" }
if ($installedHash.Trim() -ne $hash) {
  & $python -m pip install -r $requirements
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  Set-Content -LiteralPath $stamp -Value $hash
}
$arguments = @((Join-Path $root "tools\gui\server.py"), "--port", "$Port")
if ($Config) { $arguments += @("--config", $Config) }
if ($NoBrowser) { $arguments += "--no-browser" }
& $python @arguments
exit $LASTEXITCODE
