# Live view of the Stage B data box.
#
#   powershell -ExecutionPolicy Bypass -File .\infra\watch-stage-b.ps1
#
# Runs in your own terminal so nothing else stopping can stop it. Ctrl+C to
# quit. Writes live-logs\stage-b-status.txt and prints the same to console.
# The script lives in infra/ because it is a tool; live-logs/ is gitignored
# and holds only what a run emits.
#
# The box is headless -- no SSH, no key, nothing to connect to -- so there is no
# log to tail. Two real signals exist and this shows both:
#
#   the step name    the box writes it to S3 as it goes
#   bytes ingressed  CloudWatch NetworkIn on the instance
#
# The bar tracks network ingress, not job completion, and is labelled that way
# on purpose. Ingress is a genuine measurement; a percentage of "the job" would
# be a number invented to look reassuring. During the prepare and mixture steps
# the box is computing, not downloading, so the bar legitimately stops moving --
# the step name is what is alive then.

param(
    [string]$InstanceId = "i-0dacc08cb77ebce9a",
    [string]$Bucket     = "satquery-869987460914",
    [string]$Region     = "ap-south-1",
    [int]$IntervalSec   = 30
)

$out = Join-Path (Split-Path $PSScriptRoot -Parent) "live-logs"
if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
$prefix = "s3://$Bucket/datasets/prepared/stage-b"

# What the box downloads, in order, and roughly how much. Used only to scale the
# ingress bar; being off by a few GB moves the bar, not the outcome.
$ExpectedGB = 164.0

$Steps = @(
    "installing",
    "pulling the BigEarthNet store from S3",
    "re-preparing the rehearsal slice",
    "pulling benchmark train splits",
    "pulling test splits for the guard",
    "building the Stage B mixture",
    "verifying the written corpus",
    "uploading",
    "done"
)

function Get-Step {
    $s = aws s3 cp "$prefix/_STATUS.txt" - --region $Region 2>$null
    if ($LASTEXITCODE -ne 0) { return "" }
    return ($s | Out-String).Trim()
}

function Get-IngressGB {
    # NetworkIn is a 5-minute Sum in bytes. Summing every datapoint since launch
    # gives cumulative bytes into the instance.
    $start = (Get-Date).ToUniversalTime().AddHours(-6).ToString("yyyy-MM-ddTHH:mm:ssZ")
    $end   = (Get-Date).ToUniversalTime().AddMinutes(5).ToString("yyyy-MM-ddTHH:mm:ssZ")
    $raw = aws cloudwatch get-metric-statistics --region $Region `
        --namespace AWS/EC2 --metric-name NetworkIn `
        --dimensions "Name=InstanceId,Value=$InstanceId" `
        --start-time $start --end-time $end --period 300 --statistics Sum `
        --query "Datapoints[].Sum" --output text 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $raw) { return -1.0 }
    $total = 0.0
    foreach ($v in ($raw -split "\s+")) {
        if ($v -match '^[0-9.eE+]+$') { $total += [double]$v }
    }
    return [math]::Round($total / 1GB, 1)
}

function Get-State {
    $s = aws ec2 describe-instances --region $Region --instance-ids $InstanceId `
        --query "Reservations[].Instances[].State.Name" --output text 2>$null
    if ($LASTEXITCODE -ne 0) { return "unknown" }
    return ($s | Out-String).Trim()
}

function New-Bar {
    param([double]$Fraction, [int]$Width = 42)
    if ($Fraction -lt 0) { $Fraction = 0 }
    if ($Fraction -gt 1) { $Fraction = 1 }
    $filled = [int][math]::Round($Fraction * $Width)
    return "[" + ("#" * $filled) + ("-" * ($Width - $filled)) + "]"
}

$launch = $null
while ($true) {
    $now   = (Get-Date).ToUniversalTime()
    $step  = Get-Step
    $state = Get-State
    $gb    = Get-IngressGB

    if (-not $launch) {
        $lt = aws ec2 describe-instances --region $Region --instance-ids $InstanceId `
              --query "Reservations[].Instances[].LaunchTime" --output text 2>$null
        if ($LASTEXITCODE -eq 0 -and $lt) {
            try { $launch = ([datetime]($lt | Out-String).Trim()).ToUniversalTime() } catch {}
        }
    }
    if ($launch) {
        $mins = [int](($now - $launch).TotalMinutes)
        $elapsed = "{0}h{1:00}m" -f [int]($mins / 60), ($mins % 60)
    } else {
        $elapsed = "?"
    }

    # Which numbered step we are on, for a counter that does not pretend to be a
    # percentage of the whole job.
    $idx = 0
    for ($i = 0; $i -lt $Steps.Count; $i++) {
        if ($step -and $step.StartsWith($Steps[$i].Substring(0, [math]::Min(18, $Steps[$i].Length)))) {
            $idx = $i + 1
        }
    }

    if ($gb -ge 0) {
        $bar = New-Bar ($gb / $ExpectedGB)
        $ingress = "{0} {1,6:N1} / {2:N0} GB in" -f $bar, $gb, $ExpectedGB
    } else {
        $ingress = "(CloudWatch unavailable)"
    }

    $done   = (aws s3 ls "$prefix/_DONE.log"   --region $Region 2>$null)
    $failed = (aws s3 ls "$prefix/_FAILED.log" --region $Region 2>$null)

    $lines = @(
        "SatQuery Stage B data box - live",
        "updated  : $($now.ToString('yyyy-MM-dd HH:mm:ss')) UTC   elapsed $elapsed",
        "instance : $InstanceId  c7i.2xlarge  $state",
        "",
        "step $idx/$($Steps.Count) : $step",
        "ingress  : $ingress",
        ""
    )

    if ($step -and $step -notlike "*pulling*") {
        $lines += "  (computing, not downloading - the bar is expected to sit still here)"
        $lines += ""
    }

    $objs = aws s3 ls "$prefix/" --region $Region 2>$null
    if ($objs) {
        $lines += "outputs appearing at datasets/prepared/stage-b/ :"
        $lines += ($objs | Out-String).TrimEnd()
        $lines += ""
    }

    if ($failed) {
        $lines += "*** FAILED - read the log:"
        $lines += "    aws s3 cp $prefix/_FAILED.log -"
    } elseif ($done) {
        $lines += "*** DONE - read the report before spending GPU hours:"
        $lines += "    aws s3 cp $prefix/mixture-report.txt -"
    }

    $text = ($lines -join "`r`n")
    Clear-Host
    Write-Output $text
    $tmp = Join-Path $out "stage-b-status.txt.tmp"
    Set-Content -Path $tmp -Value $text -Encoding utf8
    Move-Item -Path $tmp -Destination (Join-Path $out "stage-b-status.txt") -Force

    if ($done -or $failed) { break }
    if ($state -eq "terminated") {
        Write-Output ""
        Write-Output "instance terminated with no marker - it died before it could report."
        break
    }
    Start-Sleep -Seconds $IntervalSec
}
