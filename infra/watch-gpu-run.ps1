# Live view of the Stage B run on the rented GPU box.
#
#   powershell -ExecutionPolicy Bypass -File .\infra\watch-gpu-run.ps1
#
# Runs in your own terminal so nothing else stopping can stop it. Ctrl+C to
# quit. Writes live-logs\gpu-run-status.txt as well as printing, so the state is
# readable without the terminal open.
#
# Unlike the data-box watcher this one has a real log to read: the box is
# reachable over SSH, so it reports what the run itself is saying rather than
# inferring progress from network counters.

param(
    [string]$SshHost   = "vastb",
    [string]$Contract  = "50521632",
    [int]$IntervalSec  = 45
)

$out = Join-Path (Split-Path $PSScriptRoot -Parent) "live-logs"
if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }

# The status script lives on the box (infra/box-status.sh, scp'd at launch).
# Sending a quoted shell script over SSH each tick means passing it through
# PowerShell quoting and then bash quoting, and the watcher ends up reporting
# its own syntax errors instead of the run's state.

while ($true) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss")
    $body = ssh -o ConnectTimeout=20 -o BatchMode=yes $SshHost 'bash /workspace/status.sh' 2>&1 |
            Where-Object { $_ -notmatch "Welcome to vast.ai|Have fun!" }

    # Credit is the number that decides whether a retry is affordable, so it is
    # on screen rather than something to go and look up after a failure.
    $credit = ""
    try {
        $credit = python -c @"
import requests
def load(n,p=r'C:/Users/sudik/OneDrive/Desktop/SIH26167/.env'):
    for ln in open(p,encoding='utf-8-sig'):
        if ln.strip().startswith(n): return ln.split('=',1)[1].strip().strip('\"').strip(\"'\")
try:
    h={'Authorization':'Bearer '+load('VAST_API_KEY')}
    r=requests.get('https://console.vast.ai/api/v0/users/current/',headers=h,timeout=30)
    print(f\"{r.json().get('credit',0):.3f}\")
except Exception: print('?')
"@ 2>$null
    } catch { $credit = "?" }

    $lines = @(
        "SatQuery Stage B - live",
        "updated : $stamp UTC",
        "box     : $SshHost (contract $Contract)  RTX 4080S 32GB",
        "credit  : `$$credit remaining",
        ""
    ) + $body

    $text = ($lines -join "`r`n")
    Clear-Host
    Write-Output $text
    $tmp = Join-Path $out "gpu-run-status.txt.tmp"
    Set-Content -Path $tmp -Value $text -Encoding utf8
    Move-Item -Path $tmp -Destination (Join-Path $out "gpu-run-status.txt") -Force

    if ($text -match "STAGE B COMPLETE") { break }
    Start-Sleep -Seconds $IntervalSec
}
