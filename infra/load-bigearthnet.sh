#!/usr/bin/env bash
# Stage BigEarthNet into S3 *and* build the adaptation corpus, on one throwaway
# CPU box.
#
#   ./infra/load-bigearthnet.sh            # dry run
#   ./infra/load-bigearthnet.sh --apply    # launches a billable instance
#
# Two jobs on one instance because both need the 155 GB store on local disk, and
# it should land there exactly once. The transfer must not cross a home
# connection twice -- HuggingFace to EC2 is free ingress on a fat pipe and EC2 to
# S3 in-region is free and fast.
#
# Preparation runs here rather than on the GPU box on purpose. Rendering ~40k
# patch pairs and measuring each one is CPU-bound; doing it on a g5.xlarge means
# paying GPU rates for work that never touches the GPU. This box costs about
# $0.09/hr against roughly $0.50, and it takes those hours off the critical path
# once the quota clears.
#
# The instance is headless -- no key pair, no inbound rule, nothing to connect
# to. It runs from user-data and terminates itself, so the forgotten-running-box
# failure is structurally impossible rather than merely watched for. It reports
# by writing to S3, because there is no way to ask it anything.
#
# Ordering matters: the raw store is pushed to S3 *before* preparation starts, so
# a failure in preparation does not cost the download.
set -euo pipefail

REGION=${AWS_DEFAULT_REGION:-ap-south-1}
BUCKET=${SATQUERY_BUCKET:?set SATQUERY_BUCKET}
PROFILE=${SATQUERY_INSTANCE_PROFILE:-satquery-data-loader}
INSTANCE_TYPE=${SATQUERY_LOADER_TYPE:-c7i.2xlarge}
PER_TYPE=${SATQUERY_PER_TYPE:-25000}
SEED=${SATQUERY_SEED:-1234}
MODE=${1:-}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [ "${SATQUERY_SLICE:-}" = "1" ]; then
    HF_REPO=hackelle/BigEarthNetV2-Lithuania-Summer-LMDB
    DISK=60
    PREFIX=datasets/bigearthnet-lithuania
    EXPECT="2.5 GB"
else
    HF_REPO=hackelle/BigEarthNetV2-LMDB
    DISK=300
    PREFIX=datasets/bigearthnet
    EXPECT="155 GB"
fi

cat <<PLAN
plan
  region     ${REGION}
  instance   ${INSTANCE_TYPE}, headless, self-terminating
  disk       ${DISK} GB gp3 (deleted with the instance)
  source     ${HF_REPO}  (~${EXPECT})

  1. download the image store
  2. push it raw to      s3://${BUCKET}/${PREFIX}/
  3. prepare train split (--per-type ${PER_TYPE}, seed ${SEED})
  4. prepare bench split (the cross-modal benchmark, prompts left bare)
  5. push corpora to     s3://${BUCKET}/datasets/prepared/
  6. terminate

  set SATQUERY_SLICE=1 to rehearse on the 2.5 GB Lithuania slice first.
PLAN

if [ "$MODE" != "--apply" ]; then
    echo
    echo "dry run. Re-run with --apply to launch."
    exit 0
fi

# The instance has no git credentials and the repo may be private, so the source
# travels through the bucket it already has a role for. Only what preparation
# needs -- not the data directory, not the venv.
# Scratch lives beside the repo rather than in /tmp. Git Bash hands a POSIX
# /tmp path to a Windows aws.exe that cannot open it; a relative path is
# understood by both.
echo "== staging source =="
SCRATCH=.satquery-tmp
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
BUNDLE="$SCRATCH/satquery-src.tar.gz"
tar -czf "$BUNDLE" -C "$REPO_ROOT" src scripts pyproject.toml README.md
aws s3 cp "$BUNDLE" "s3://${BUCKET}/bootstrap/satquery-src.tar.gz" --only-show-errors
echo "   $(du -h "$BUNDLE" | cut -f1) to s3://${BUCKET}/bootstrap/"

AMI=$(aws ec2 describe-images --region "$REGION" --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" \
            "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)

USER_DATA=$(cat <<CLOUDINIT
#!/bin/bash
exec > >(tee /var/log/satquery-load.log) 2>&1
set -x

STATUS="s3://${BUCKET}/${PREFIX}"
note() { echo "\$1" | aws s3 cp - "\${STATUS}/_STATUS.txt" || true; }
fail() {
  aws s3 cp /var/log/satquery-load.log "\${STATUS}/_FAILED.log" || true
  shutdown -h now
}
trap fail ERR

note "installing"
apt-get update && apt-get install -y python3-pip python3-venv awscli
python3 -m venv /opt/venv
/opt/venv/bin/pip install --quiet "huggingface_hub[hf_transfer]"

mkdir -p /data && cd /data
aws s3 cp "s3://${BUCKET}/bootstrap/satquery-src.tar.gz" /data/src.tar.gz
mkdir -p /data/satquery && tar -xzf /data/src.tar.gz -C /data/satquery
/opt/venv/bin/pip install --quiet -e "/data/satquery[data]"

# --- 1. download -------------------------------------------------------
note "downloading ${HF_REPO}"
# hf_transfer is a Rust downloader; the default client leaves most of an EC2
# pipe unused on a file this size.
export HF_HUB_ENABLE_HF_TRANSFER=1
/opt/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id="${HF_REPO}", repo_type="dataset",
                        local_dir="/data/store", max_workers=8))
PY
du -sh /data/store

# --- 2. raw store to S3, before anything can go wrong downstream -------
note "uploading raw store"
aws configure set default.s3.multipart_chunksize 256MB
aws configure set default.s3.max_concurrent_requests 20
aws s3 sync /data/store "\${STATUS}/" --only-show-errors
aws s3 cp /var/log/satquery-load.log "\${STATUS}/_RAW_DONE.log"

# --- 3/4. prepare the corpora ------------------------------------------
LMDB=\$(find /data/store -maxdepth 2 -name "*.lmdb" -o -maxdepth 2 -name "BENv2*" -type d | head -1)
META=\$(find /data/store -maxdepth 2 -name "metadata.parquet" | head -1)
echo "lmdb=\${LMDB} metadata=\${META}"

note "preparing train split"
/opt/venv/bin/python /data/satquery/scripts/prepare_bigearthnet.py \
    --lmdb "\${LMDB}" --metadata "\${META}" \
    --out /data/prepared/train --split train \
    --per-type ${PER_TYPE} --seed ${SEED} --cache /data/cache

note "preparing bench split"
# The cross-modal benchmark. Preparation refuses to bake evidence into this one;
# an evaluation question carrying its own answer's measurements would score the
# preamble rather than the model.
/opt/venv/bin/python /data/satquery/scripts/prepare_bigearthnet.py \
    --lmdb "\${LMDB}" --metadata "\${META}" \
    --out /data/prepared/bench --split bench \
    --per-type 4000 --seed ${SEED} --cache /data/cache

# --- 5. corpora to S3 ---------------------------------------------------
note "uploading prepared corpora"
aws s3 sync /data/prepared "s3://${BUCKET}/datasets/prepared/" --only-show-errors
for split in train bench; do
  aws s3 cp "/data/prepared/\${split}/manifest.json" \
      "s3://${BUCKET}/datasets/prepared/\${split}/manifest.json" || true
done

aws s3 cp /var/log/satquery-load.log "\${STATUS}/_DONE.log"
note "done"
shutdown -h now
CLOUDINIT
)

# The device name goes in a file rather than on the command line: Git Bash
# rewrites a bare /dev/sda1 argument into a Windows path and the API rejects it.
cat > "$SCRATCH/bdm.json" <<BDM
[{"DeviceName": "/dev/sda1",
  "Ebs": {"VolumeSize": ${DISK}, "VolumeType": "gp3", "DeleteOnTermination": true}}]
BDM
printf %s "$USER_DATA" > "$SCRATCH/user-data.sh"

INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile "Name=${PROFILE}" \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings "file://$SCRATCH/bdm.json" \
    --user-data "file://$SCRATCH/user-data.sh" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=satquery-data-loader},{Key=Project,Value=SIH26167}]' \
    --query 'Instances[0].InstanceId' --output text)

cat <<DONE

launched ${INSTANCE_ID}

No SSH, no inbound rule, no key. Follow it from S3:

  aws s3 cp s3://${BUCKET}/${PREFIX}/_STATUS.txt -            # current step
  aws s3 ls s3://${BUCKET}/datasets/prepared/ --recursive     # corpora appearing
  aws ec2 describe-instances --region ${REGION} --instance-ids ${INSTANCE_ID} \\
      --query 'Reservations[].Instances[].State.Name' --output text

  _RAW_DONE.log   the 155 GB store is safely in S3
  _DONE.log       corpora prepared and uploaded
  _FAILED.log     something broke; the log says what

The instance terminates itself either way. "terminated" with no marker means it
died before it could report -- check the console output for that case.
DONE
