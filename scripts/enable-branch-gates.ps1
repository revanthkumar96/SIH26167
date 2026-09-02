# Apply GitHub ruleset so main cannot be updated except via PR + green checks.
# Requires: gh auth login  (scopes: repo, workflow; admin rights on the repo)

$ErrorActionPreference = "Stop"
$env:Path = "$env:ProgramFiles\GitHub CLI;" + $env:Path

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  throw "GitHub CLI is not installed."
}

gh auth status
$remote = git remote get-url origin
if ($remote -notmatch "github.com[:/](?<owner>[^/]+)/(?<repo>[^/.]+)") {
  throw "Could not parse origin remote: $remote"
}
$owner = $Matches.owner
$repo = $Matches.repo

$payloadPath = Join-Path $PSScriptRoot "main-quality-gates.json"
if (-not (Test-Path $payloadPath)) {
  throw "Missing $payloadPath"
}

$rulesets = gh api "repos/$owner/$repo/rulesets" | ConvertFrom-Json
$existing = @($rulesets | Where-Object { $_.name -eq "main-quality-gates" } | Select-Object -ExpandProperty id)

if ($existing.Count -gt 0) {
  $id = $existing[0]
  gh api -X PUT "repos/$owner/$repo/rulesets/$id" --input $payloadPath
  Write-Host "Updated ruleset main-quality-gates ($id) on $owner/$repo"
}
else {
  gh api -X POST "repos/$owner/$repo/rulesets" --input $payloadPath
  Write-Host "Created ruleset main-quality-gates on $owner/$repo"
}

Write-Host "Main is now gated: PRs + required checks (Pre-commit, CI, Gitleaks). Direct pushes and force-pushes are blocked."
