#!/usr/bin/env bash
# Run at the end of every session, and once more before going to bed.
#
# The habit worth building. Syncs anything unsynced, stops the instance, and
# then *shows what is still running* -- the last step is the point, because a
# stop command that silently failed looks exactly like one that worked.
set -euo pipefail

BUCKET=${SATQUERY_BUCKET:?set SATQUERY_BUCKET}
INSTANCE=${SATQUERY_INSTANCE_ID:-}

if [ -d "$HOME/checkpoints" ]; then
    aws s3 sync "$HOME/checkpoints" "s3://${BUCKET}/checkpoints/"
fi
if [ -d "$HOME/runs" ]; then
    aws s3 sync "$HOME/runs" "s3://${BUCKET}/results/"
fi

if [ -n "$INSTANCE" ]; then
    # Stop, never terminate: the EBS volume holds the 155 GB dataset, and
    # pulling it again is hours.
    aws ec2 stop-instances --instance-ids "$INSTANCE" --output table
fi

echo
echo "instances in this region:"
aws ec2 describe-instances \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]' \
    --output table
