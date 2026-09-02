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

$body = @{
  name        = "main-quality-gates"
  target      = "branch"
  enforcement = "active"
  conditions  = @{
    ref_name = @{
      include = @("refs/heads/main")
      exclude = @()
    }
  }
  rules       = @(
    @{ type = "deletion" }
    @{ type = "non_fast_forward" }
    @{
      type       = "pull_request"
      parameters = @{
        required_approving_review_count     = 0
        dismiss_stale_reviews_on_push       = $true
        require_code_owner_review           = $false
        require_last_push_approval          = $false
        required_review_thread_resolution   = $false
      }
    }
    @{
      type       = "required_status_checks"
      parameters = @{
        strict_required_status_checks_policy = $true
        do_not_enforce_on_create             = $false
        required_status_checks               = @(
          @{ context = "Run pre-commit hooks" }
          @{ context = "Lint and test" }
          @{ context = "Gitleaks" }
        )
      }
    }
  )
} | ConvertTo-Json -Depth 8

$existing = gh api "repos/$owner/$repo/rulesets" --jq ".[] | select(.name==`"main-quality-gates`") | .id" 2>$null
if ($existing) {
  $body | gh api -X PUT "repos/$owner/$repo/rulesets/$existing" --input -
  Write-Host "Updated ruleset main-quality-gates ($existing) on $owner/$repo"
}
else {
  $body | gh api -X POST "repos/$owner/$repo/rulesets" --input -
  Write-Host "Created ruleset main-quality-gates on $owner/$repo"
}

Write-Host "Main is now gated: PRs + required checks (Pre-commit, CI, Gitleaks). Direct pushes and force-pushes are blocked."
