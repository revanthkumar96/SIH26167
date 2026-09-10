#!/usr/bin/env bash
# Repack the prepared corpus into one tarball in S3.
#
#   ./infra/pack-corpus.sh --apply
#
# The corpus is 141,090 objects. Pulling those individually onto a rented GPU is
# thousands of small GETs and easily the slowest part of a timed run -- paid for
# at GPU rates while the card sits idle. One 5 GB object streams instead.
#
# Runs on the same throwaway CPU pattern as the loader: headless, self-
# terminating, reports through S3.
set -euo pipefail
REGION=${AWS_DEFAULT_REGION:-ap-south-1}
BUCKET=${SATQUERY_BUCKET:?set SATQUERY_BUCKET}
PROFILE=${SATQUERY_INSTANCE_PROFILE:-satquery-data-loader}
TYPE=${SATQUERY_PACK_TYPE:-c7i.2xlarge}

echo "plan: ${TYPE} packs s3://${BUCKET}/datasets/prepared/ into prepared-train.tar.gz"
[ "${1:-}" = "--apply" ] || { echo; echo "dry run. --apply to launch."; exit 0; }

SCRATCH=.satquery-tmp; mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT
AMI=$(aws ec2 describe-images --region "$REGION" --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" \
            "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)

cat > "$SCRATCH/bdm.json" <<BDM
[{"DeviceName": "/dev/sda1",
  "Ebs": {"VolumeSize": 60, "VolumeType": "gp3", "DeleteOnTermination": true}}]
BDM

cat > "$SCRATCH/user-data.sh" <<CLOUDINIT
#!/bin/bash
exec > >(tee /var/log/pack.log) 2>&1
set -x
apt-get update && apt-get install -y unzip curl
curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/a.zip
unzip -q /tmp/a.zip -d /tmp && /tmp/aws/install
aws --version || shutdown -h now

DEST="s3://${BUCKET}/datasets"
fail() { aws s3 cp /var/log/pack.log "\${DEST}/_PACK_FAILED.log" || true; shutdown -h now; }
trap fail ERR

mkdir -p /data && cd /data
# Many small objects: parallelism matters far more than part size here.
aws configure set default.s3.max_concurrent_requests 40
for split in train bench; do
  aws s3 sync "\${DEST}/prepared/\${split}/" "/data/\${split}/" --only-show-errors
  tar -czf "/data/prepared-\${split}.tar.gz" -C /data "\${split}"
  aws s3 cp "/data/prepared-\${split}.tar.gz" "\${DEST}/prepared-\${split}.tar.gz"
  ls -la "/data/prepared-\${split}.tar.gz"
done
aws s3 cp /var/log/pack.log "\${DEST}/_PACK_DONE.log"
shutdown -h now
CLOUDINIT

ID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" \
    --iam-instance-profile "Name=${PROFILE}" \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings "file://$SCRATCH/bdm.json" \
    --user-data "file://$SCRATCH/user-data.sh" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=satquery-pack},{Key=Project,Value=SIH26167}]' \
    --query 'Instances[0].InstanceId' --output text)
echo "launched $ID"
echo "  watch: aws s3 ls s3://${BUCKET}/datasets/ | grep -E 'tar.gz|_PACK'"
