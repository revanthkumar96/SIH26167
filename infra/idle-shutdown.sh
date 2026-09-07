#!/usr/bin/env bash
# Halt the instance when it has been idle long enough to be a forgotten one.
#
# The single likeliest way to lose the $100 grant is an instance left running
# overnight: at on-demand g5.xlarge rates that is ~$24 a night, a quarter of the
# budget, for nothing. Every other cost control here is a warning; this one is
# the structural fix, because it does not depend on anyone remembering.
#
# Idle means both of these, sustained across IDLE_CYCLES consecutive checks:
#   - no GPU utilisation
#   - no active SSH session
#
# Both matter. GPU alone would kill a box someone is actively working on between
# runs; SSH alone would keep a box alive because a forgotten terminal is open.
set -euo pipefail

GPU_THRESHOLD=${GPU_THRESHOLD:-5}      # percent
IDLE_CYCLES=${IDLE_CYCLES:-30}         # with a 60s timer, 30 minutes
STATE=/var/tmp/satquery-idle-count

gpu_busy() {
    local used
    used=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
           | sort -rn | head -1) || return 1
    [ -n "$used" ] && [ "$used" -ge "$GPU_THRESHOLD" ]
}

ssh_active() {
    # `who` lists login sessions; a detached tmux is deliberately not enough to
    # keep the box alive, or a forgotten session would defeat the whole timer.
    [ -n "$(who 2>/dev/null)" ]
}

training_running() {
    # A long data pull or an S3 sync is real work with no GPU load at all.
    pgrep -f 'train_lora.py|train_landcover.py|prepare_bigearthnet.py|aws s3' >/dev/null 2>&1
}

if gpu_busy || ssh_active || training_running; then
    echo 0 > "$STATE"
    exit 0
fi

count=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$count" > "$STATE"
logger -t satquery-idle "idle for $count/$IDLE_CYCLES cycles"

if [ "$count" -ge "$IDLE_CYCLES" ]; then
    logger -t satquery-idle "idle threshold reached; syncing checkpoints and halting"
    # Sync before halting: an unsynced checkpoint on a stopped box is recoverable,
    # but only if someone remembers to start the box again to get it.
    if [ -n "${SATQUERY_S3_CHECKPOINTS:-}" ] && [ -d "$HOME/checkpoints" ]; then
        aws s3 sync "$HOME/checkpoints" "$SATQUERY_S3_CHECKPOINTS" || \
            logger -t satquery-idle "checkpoint sync failed; halting anyway"
    fi
    shutdown -h now "satquery: idle auto-shutdown"
fi
