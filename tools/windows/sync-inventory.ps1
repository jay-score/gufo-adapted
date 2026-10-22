# Inventory an official/main sync and gate it on known Windows regressions.
#
# Emits Markdown: incoming PRs (resolved with gh), the known-regression gate,
# upstream changes to reverted regions, files changed on both sides, diff
# stats in both directions and a sync-log entry skeleton. Run before merging
# with -WindowsBase/-OfficialHead, then again with -Candidate after merging
# and reverting. Exit code 1 means a known regression is not held back.
param(
  [string]$OfficialHead = "official/main",
  [string]$WindowsBase = "HEAD",
  [string]$Candidate = "",
  [string]$Repository = "gufo-org/gufo",
  [string]$Regressions = "",
  [string]$Contracts = "",
  [string]$OutFile = "",
  [switch]$Offline
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot/sync-common.ps1"
$root = (Resolve-Path "$PSScriptRoot/../..").Path
if (-not $Regressions) { $Regressions = Join-Path $PSScriptRoot "known-regressions.json" }
if (-not $Contracts) { $Contracts = Join-Path $root "docs/WINDOWS-COMPATIBILITY.md" }

$official = Resolve-Commit $OfficialHead
$windows = Resolve-Commit $WindowsBase
$candidateSha = if ($Candidate) { Resolve-Commit $Candidate } else { "" }
$ancestor = @(Invoke-Git merge-base $official $windows)[0]
$short = { param($sha) $sha.Substring(0, 8) }
$out = New-Object System.Collections.Generic.List[string]
$failed = $false

# Incoming commits, oldest first. Official main squash-merges PRs, so each PR
# is one commit whose subject ends in "(#N)".
$incoming = @()
foreach ($line in Invoke-Git log --reverse --format="%H%x09%s" "$windows..$official") {
  $sha, $subject = $line -split "`t", 2
  $pr = if ($subject -match "\(#(\d+)\)$") { [int]$Matches[1] } else { 0 }
  $incoming += [pscustomobject]@{ Sha = $sha; Subject = $subject; Pr = $pr; Paths = @(Get-CommitPaths $sha) }
}

$known = @((Read-JsonFile $Regressions).regressions | Where-Object { $_.status -eq "reverted" })
$knownByPr = @{}
foreach ($regression in $known) { $knownByPr[[int]$regression.pr] = $regression }

$out.Add("# Official sync inventory")
$out.Add("")
$out.Add("- Windows base: ``$windows``")
$out.Add("- Official head: ``$official``")
$out.Add("- Previous official ancestor: ``$ancestor``")
if ($candidateSha) { $out.Add("- Candidate: ``$candidateSha``") }
$out.Add("- Incoming commits: $($incoming.Count)")
if (Test-Ancestor $official $windows) {
  $out.Add("")
  $out.Add("Official head is already an ancestor of the Windows base; nothing to merge.")
}

# Known-regression gate.
$out.Add("")
$out.Add("## Known regressions")
foreach ($regression in $known) {
  $out.Add("")
  $out.Add("### #$($regression.pr): $($regression.title)")
  $out.Add("")
  $out.Add("Record: [$($regression.log)]($($regression.log)). $($regression.symptom)")
  $out.Add("")
  $arriving = (Test-CommitExists $regression.mergeSha) -and
    (Test-Ancestor $regression.mergeSha $official) -and -not (Test-Ancestor $regression.mergeSha $windows)
  if ($arriving -and -not $candidateSha) {
    $out.Add("- **FAIL**: this sync imports the regression. Merge, revert it on the candidate, then rerun with -Candidate.")
    $failed = $true
  }
  if ($candidateSha) {
    $present = @(Find-Identifiers $candidateSha @($regression.absentIdentifiers))
    if ($present) {
      $out.Add("- **FAIL**: candidate contains reverted code: $($present -join '; ').")
      $failed = $true
    } else {
      $out.Add("- PASS: none of the $(@($regression.absentIdentifiers).Count) reverted identifiers is present in the candidate.")
    }
    if ($regression.revertSha -and -not $arriving) {
      if ((Test-CommitExists $regression.revertSha) -and (Test-Ancestor $regression.revertSha $candidateSha)) {
        $out.Add("- PASS: revert ``$(& $short $regression.revertSha)`` is an ancestor of the candidate.")
      } else {
        $out.Add("- **FAIL**: revert ``$($regression.revertSha)`` is not an ancestor of the candidate.")
        $failed = $true
      }
    }
    if ($arriving) { $out.Add("- Record the new revert SHA in known-regressions.json and the sync log.") }
  }

  # Did upstream touch the reverted regions or mention the PR? Either may be
  # a fix worth reviewing; neither reapplies the revert automatically.
  $watchSha = if ($regression.revertSha -and (Test-CommitExists $regression.revertSha)) { $regression.revertSha } else { $regression.mergeSha }
  $watch = @{}
  if (Test-CommitExists $watchSha) { foreach ($path in Get-CommitPaths $watchSha) { $watch[$path] = $true } }
  $mentions = @(Invoke-Git log --format=%H --fixed-strings "--grep=#$($regression.pr)" "$windows..$official")
  $touched = 0
  foreach ($commit in $incoming) {
    if ($commit.Sha -eq $regression.mergeSha) { continue }
    $paths = @($commit.Paths | Where-Object { $watch.ContainsKey($_) })
    $mentioned = $mentions -contains $commit.Sha
    if ($paths -or $mentioned) {
      $touched++
      $why = @()
      if ($mentioned) { $why += "mentions #$($regression.pr)" }
      if ($paths) { $why += "touches " + (($paths | ForEach-Object { "``$_``" }) -join ", ") }
      $out.Add("- REVIEW ``$(& $short $commit.Sha)`` $($commit.Subject): $($why -join '; ').")
    }
  }
  if ($touched -eq 0) { $out.Add("- No incoming commit touches the reverted regions or mentions #$($regression.pr).") }
  if (Test-Ancestor $regression.mergeSha $official) {
    $upstream = @(Find-Identifiers $official @($regression.absentIdentifiers) | ForEach-Object { ($_ -split ":")[0] } | Sort-Object -Unique)
    $total = @($regression.absentIdentifiers).Count
    if ($upstream.Count -lt $total) {
      $out.Add("- REVIEW: official head retains $($upstream.Count) of $total reverted identifiers; upstream may have reworked this code.")
    }
  }
  $out.Add("- Reintroduction requires: $(@($regression.reintroductionChecks) -join ' ')")
}

# Incoming PR table, verified against GitHub unless offline.
$rows = @()
$extra = @()
foreach ($commit in $incoming) {
  if (-not $commit.Pr) { $extra += "``$(& $short $commit.Sha)`` $($commit.Subject)"; continue }
  $title = $commit.Subject -replace "\s*\(#\d+\)$", ""
  $url = "https://github.com/$Repository/pull/$($commit.Pr)"
  $note = ""
  if (-not $Offline) {
    $json = & gh pr view $commit.Pr --repo $Repository --json "number,title,url,mergeCommit"
    if ($LASTEXITCODE -ne 0) { throw "gh pr view $($commit.Pr) failed; authenticate gh or pass -Offline" }
    $view = ($json -join "`n") | ConvertFrom-Json
    $title = $view.title
    $url = $view.url
    if ($view.mergeCommit.oid -ne $commit.Sha) { $note = " (GitHub merge commit ``$($view.mergeCommit.oid)`` differs)" }
  } else {
    $note = " (unverified)"
  }
  if ($knownByPr.ContainsKey($commit.Pr)) { $title += "; **reverted/deferred on Windows**" }
  $rows += "| [#$($commit.Pr)]($url) | $title | ``$($commit.Sha)``$note |"
}

$both = @{}
$upstreamFiles = @(Invoke-Git diff --name-only --no-renames $ancestor $official)
foreach ($path in Invoke-Git diff --name-only --no-renames $ancestor $windows) { $both[$path] = $true }
$overlap = @($upstreamFiles | Where-Object { $both.ContainsKey($_) } | Sort-Object)

$out.Add("")
$out.Add("## Files changed on both sides")
$out.Add("")
$out.Add("Review these even when they merge cleanly.")
$out.Add("")
if ($overlap) { foreach ($path in $overlap) { $out.Add("- ``$path``") } } else { $out.Add("- None.") }

$out.Add("")
$out.Add("## Incoming difference")
$out.Add("")
$out.Add('```text')
foreach ($line in Invoke-Git diff --stat=100 $ancestor $official) { $out.Add($line) }
$out.Add('```')
$out.Add("")
$out.Add("## Maintained Windows difference")
$out.Add("")
# Fork-owned additions cannot conflict; show only changed upstream files.
$forkOwned = @(Invoke-Git diff --name-only --no-renames --diff-filter=A $ancestor $windows)
$sharedFiles = @(Invoke-Git diff --name-only --no-renames --diff-filter=a $ancestor $windows)
$out.Add("$($forkOwned.Count) fork-owned files added; changed upstream files:")
$out.Add("")
$out.Add('```text')
if ($sharedFiles) {
  foreach ($line in Invoke-Git diff --stat=100 --no-renames $ancestor $windows '--' @sharedFiles) { $out.Add($line) }
} else {
  $out.Add("(none)")
}
$out.Add('```')

# Sync-log entry skeleton for docs/WINDOWS-UPSTREAM-SYNC.md.
$contractNames = @()
foreach ($line in Get-Content -LiteralPath $Contracts -Encoding UTF8) {
  if ($line -match '^- \[[ xX]\] \*\*(.+?)\.\*\*') { $contractNames += $Matches[1] }
}
$date = (Get-Date).ToString("MMMM d, yyyy", [Globalization.CultureInfo]::InvariantCulture)
$out.Add("")
$out.Add("## Sync log entry skeleton")
$out.Add("")
$out.Add('````markdown')
$out.Add("## $date")
$out.Add("")
$out.Add("Status: **pending**")
$out.Add("")
$out.Add("- Windows base: ``$windows``.")
$out.Add("- Previous official ancestor: ``$(& $short $ancestor)``.")
$out.Add("- Imported official head: ``$official``.")
$out.Add("- Integration: ordinary merge of the recorded official head, merge commit ``TBD``.")
$out.Add("")
$out.Add("| PR | Upstream change | Merge SHA |")
$out.Add("| --- | --- | --- |")
foreach ($row in $rows) { $out.Add($row) }
if ($extra) {
  $out.Add("")
  $out.Add("Additional incoming commits: $($extra -join '; ').")
}
$out.Add("")
$out.Add("### Merge review")
$out.Add("")
$out.Add("Conflicts and resolutions: TBD. Clean merges reviewed: TBD.")
$out.Add("")
$out.Add("### Compatibility and validation")
$out.Add("")
$out.Add("Use validated, reviewed/unchanged, or pending (reason) per contract.")
$out.Add("")
$out.Add("| Contract | Outcome |")
$out.Add("| --- | --- |")
foreach ($contract in $contractNames) { $out.Add("| $contract | pending (TBD) |") }
$out.Add("")
$out.Add("Completed checks: TBD.")
$out.Add("")
$out.Add("### Live checks and limits")
$out.Add("")
$out.Add("TBD.")
$out.Add('````')

if ($OutFile) {
  $target = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutFile)
  New-Item -ItemType Directory -Force (Split-Path -Parent $target) | Out-Null
  [IO.File]::WriteAllLines($target, $out, (New-Object Text.UTF8Encoding $false))
  Write-Host "Wrote $OutFile"
} else {
  $out | ForEach-Object { Write-Output $_ }
}
if ($failed) {
  Write-Host "FAIL: a known Windows regression is not held back; see 'Known regressions'."
  exit 1
}
