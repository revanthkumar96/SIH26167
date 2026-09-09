#!/usr/bin/env bash
# Stage the BigEarthNet v2.0 image store into S3, using a throwaway CPU box.
#
#   ./infra/load-bigearthnet.sh            # dry run
#   ./infra/load-bigearthnet.sh --apply    # launches a billable instance
#
# The 155 GB store must not travel over a home connection twice. HuggingFace to
# EC2 is free ingress on a fat pipe, and EC2 to S3 in the same region is free
# and fast, so the transfer belongs on an instance in ap-south-1 rather than on
# a laptop in the middle.
#
# The instance is headless: no SSH key, no inbound rule, nothing to connect to.
# It runs the transfer from user-data and terminates itself, so the failure this
# whole project keeps guarding against -- a forgotten running box -- is
# structurally impossible rather than merely watched for.
#
# Cost, honestly: about $0.30 of compute and a couple of hours of wall clock.
# The recurring cost is S3 storage, roughly $3.60/month for 155 GB, which is
# real against a $100 grant and is why --slice exists.
set -euo pipefail

REGION=${AWS_DEFAULT_REGION:-ap-south-1}
BUCKET=${SATQUERY_BUCKET:?set SATQUERY_BUCKET}
PROFILE=${SATQUERY_INSTANCE_PROFILE:-satquery-data-loader}
INSTANCE_TYPE=${SATQUERY_LOADER_TYPE:-c7i.2xlarge}
MODE=${1:-}
SLICE=${SATQUERY_SLICE:-}

if [ "$SLICE" = "1" ]; then
    REPO=hackelle/BigEarthNetV2-Lithuania-Summer-LMDB
    DISK=40
    PREFIX=datasets/bigearthnet-lithuania
    EXPECT="2.5 GB"
else
    REPO=hackelle/BigEarthNetV2-LMDB
    DISK=260
    PREFIX=datasets/bigearthnet
    EXPECT="155 GB"
fi

cat <<PLAN
plan
  region     ${REGION}
  instance   ${INSTANCE_TYPE}, headless, self-terminating
  disk       ${DISK} GB gp3 (deleted with the instance)
  source     ${REPO}  (~${EXPECT})
  target     s3://${BUCKET}/${PREFIX}/
  role       ${PROFILE}  (write access to this bucket only)

  set SATQUERY_SLICE=1 for the 2.5 GB Lithuania slice instead.
PLAN

if [ "$MODE" != "--apply" ]; then
    echo
    echo "dry run. Re-run with --apply to launch."
    exit 0
fi

AMI=$(aws ec2 describe-images --region "$REGION" --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" \
            "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)

USER_DATA=$(cat <<CLOUDINIT
#!/bin/bash
# Everything is logged to S3 as it goes, because there is no way to connect to
# this instance and ask what happened.
exec > >(tee /var/log/satquery-load.log) 2>&1
set -x

fail() {
  aws s3 cp /var/log/satquery-load.log "s3://${BUCKET}/${PREFIX}/_FAILED.log" || true
  shutdown -h now
}
trap fail ERR

apt-get update && apt-get install -y python3-pip awscli
pip3 install --break-system-packages "huggingface_hub[hf_transfer]"

# hf_transfer is a Rust downloader that saturates the link; the default Python
# client leaves most of an EC2 pipe unused on a file this size.
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /data && cd /data

python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="${REPO}", repo_type="dataset", local_dir="/data/store",
    max_workers=8,
)
print("downloaded to", path)
PY

du -sh /data/store
# Multipart with a large part size: a 155 GB object in 8 MB parts is more
# round-trips than throughput.
aws configure set default.s3.multipart_chunksize 256MB
aws configure set default.s3.max_concurrent_requests 20
aws s3 sync /data/store "s3://${BUCKET}/${PREFIX}/" --only-show-errors

aws s3 ls "s3://${BUCKET}/${PREFIX}/" --recursive --summarize | tail -3 > /tmp/manifest.txt
aws s3 cp /tmp/manifest.txt "s3://${BUCKET}/${PREFIX}/_MANIFEST.txt"
aws s3 cp /var/log/satquery-load.log "s3://${BUCKET}/${PREFIX}/_DONE.log"

# The whole point: it goes away by itself.
shutdown -h now
CLOUDINIT
)

INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile "Name=${PROFILE}" \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${DISK},VolumeType=gp3,DeleteOnTermination=true}" \
    --user-data "$USER_DATA" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=satquery-data-loader},{Key=Project,Value=SIH26167}]' \
    --query 'Instances[0].InstanceId' --output text)

cat <<DONE

launched ${INSTANCE_ID}

No SSH, no inbound rule, no key. Watch it from S3 instead:

  aws s3 ls s3://${BUCKET}/${PREFIX}/ --recursive --summarize | tail -3
  aws ec2 describe-instances --region ${REGION} --instance-ids ${INSTANCE_ID} \\
      --query 'Reservations[].Instances[].State.Name' --output text

_DONE.log appears on success, _FAILED.log on error. The instance terminates
itself either way; "terminated" with neither marker means it died before it
could report, and the console output is the place to look.
DONE
