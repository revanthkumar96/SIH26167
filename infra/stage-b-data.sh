#!/usr/bin/env bash
# Build everything Stage B needs, on one throwaway CPU box.
#
#   ./infra/stage-b-data.sh            # dry run
#   ./infra/stage-b-data.sh --apply    # launches a billable instance
#
# Two jobs share this box because they share a download. Re-preparing the
# BigEarthNet rehearsal slice needs the 155 GB store on local disk, and that
# store is already in S3 from the loader run -- pulling it in-region is free and
# fast, while pulling it to a laptop is neither. Staging the benchmark train
# imagery is unrelated work that happens to need the same thing this box already
# is: a fat pipe next to the bucket.
#
# What it produces:
#
#   datasets/prepared/stage-b-rehearsal/train.jsonl   BigEarthNet, preamble-heavy
#   datasets/VRSBench_train/                          annotations + 8.4 GB imagery
#   datasets/RSVQA/LR/LR_split_train_*.json           train annotations
#   datasets/prepared/stage-b/train.jsonl             the built, guarded mixture
#   datasets/prepared/stage-b/mixture-report.txt      what went in, and warnings
#
# The GPU box then downloads a corpus that has already been checked, rather than
# spending GPU-rate minutes assembling one.
#
# Why re-prepare at all: Stage A ran at --preamble-rate 0.45, giving 36.5%
# coverage. Stage B's corpus is mostly benchmark records, and no RGB benchmark
# image can carry a preamble at all, so that 36.5% dilutes to about 8% of the
# mixture -- well under the 20% train_lora.py warns at, and the controller
# prepends a preamble on every query at inference. The fix is a preamble-heavy
# slice, not a bigger one: enlarging it would just rerun Stage A.
#
# The rehearsal slice is written to its own prefix rather than over
# datasets/prepared/train/. That corpus is what Stage A trained on and what
# results/2026-09-10-stage-a/README.md cites; overwriting it would quietly
# invalidate the provenance of a number already reported.
#
# Same headless pattern as load-bigearthnet.sh: no key pair, no inbound rule,
# nothing to connect to. It runs from user-data and terminates itself, and
# reports by writing to S3 because there is no way to ask it anything.
set -euo pipefail

REGION=${AWS_DEFAULT_REGION:-ap-south-1}
BUCKET=${SATQUERY_BUCKET:?set SATQUERY_BUCKET}
PROFILE=${SATQUERY_INSTANCE_PROFILE:-satquery-data-loader}
INSTANCE_TYPE=${SATQUERY_LOADER_TYPE:-c7i.2xlarge}
DISK=${SATQUERY_DISK:-320}
SEED=${SATQUERY_SEED:-1234}
PER_TYPE=${SATQUERY_PER_TYPE:-25000}

# 0.85 lands the mixture near 21% evidence. The ceiling is about 0.81 of the
# rate -- choose_evidence() refuses any preamble it cannot show agrees with the
# record's own gold answer, and that refusal is the safety property, so the
# remaining fraction is not something to tune away.
PREAMBLE_RATE=${SATQUERY_PREAMBLE_RATE:-0.85}

# CDVQA train is 52 GB and its tile overlap with the test split is unresolved,
# so it is off by default. Set a shard count to include it; the guard will
# settle the overlap question on whatever arrives, and if the splits share
# tiles the build fails loudly rather than training on them.
CDVQA_SHARDS=${SATQUERY_CDVQA_SHARDS:-0}

MODE=${1:-}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cat <<PLAN
plan
  region     ${REGION}
  instance   ${INSTANCE_TYPE}, headless, self-terminating
  disk       ${DISK} GB gp3 (deleted with the instance)

  1. pull the BigEarthNet store from s3://${BUCKET}/datasets/bigearthnet/
  2. re-prepare the train split at --preamble-rate ${PREAMBLE_RATE}
     (same --per-type ${PER_TYPE} and --seed ${SEED}, so the same patches)
  3. pull VRSBench train annotations + imagery (~8.4 GB) and RSVQA train
  4. pull the test splits the guard checks against
  5. build the Stage B mixture, contamination-checked, and report the warnings
  6. push everything to S3
  7. terminate

  CDVQA train shards: ${CDVQA_SHARDS}$([ "$CDVQA_SHARDS" = "0" ] && echo "  (skipped; 52 GB, overlap unresolved)")

  Roughly 2-3 hours at about \$0.09/hr, plus the EBS while it runs.
PLAN

if [ "$MODE" != "--apply" ]; then
    echo
    echo "dry run. Re-run with --apply to launch."
    exit 0
fi

# The instance has no git credentials and the repo may be private, so the source
# travels through the bucket it already has a role for. Scratch lives beside the
# repo: Git Bash hands a POSIX /tmp path to a Windows aws.exe that cannot open
# it, while a relative path is understood by both.
echo "== staging source =="
SCRATCH=.satquery-tmp
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
BUNDLE="$SCRATCH/satquery-src.tar.gz"
tar -czf "$BUNDLE" -C "$REPO_ROOT" src scripts configs pyproject.toml README.md
aws s3 cp "$BUNDLE" "s3://${BUCKET}/bootstrap/satquery-src.tar.gz" --only-show-errors
echo "   $(du -h "$BUNDLE" | cut -f1) to s3://${BUCKET}/bootstrap/"

AMI=$(aws ec2 describe-images --region "$REGION" --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" \
            "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)

USER_DATA=$(cat <<CLOUDINIT
#!/bin/bash
exec > >(tee /var/log/satquery-stage-b.log) 2>&1
set -x

# The AWS CLI first and outside the trap: every status and failure report below
# goes through it, and Ubuntu's cloud image does not ship it.
apt-get update
apt-get install -y python3-pip python3-venv unzip curl
curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install
aws --version || { shutdown -h now; }

STATUS="s3://${BUCKET}/datasets/prepared/stage-b"
note() { echo "\$1" | aws s3 cp - "\${STATUS}/_STATUS.txt" || true; }
fail() {
  aws s3 cp /var/log/satquery-stage-b.log "\${STATUS}/_FAILED.log" || true
  shutdown -h now
}
trap fail ERR

aws configure set default.s3.multipart_chunksize 256MB
aws configure set default.s3.max_concurrent_requests 20

note "installing"
python3 -m venv /opt/venv
# The layout mirrors the GPU box exactly: repo at WORK, data at WORK/data. The
# train configs say root: data/VRSBench_train, which is relative to the working
# directory, so getting this wrong does not fail -- it silently resolves
# annotations somewhere else and the build reports zero records.
mkdir -p /data/work/data && cd /data/work
aws s3 cp "s3://${BUCKET}/bootstrap/satquery-src.tar.gz" /data/src.tar.gz
tar -xzf /data/src.tar.gz -C /data/work
/opt/venv/bin/pip install --quiet -e "/data/work[data]"
PY=/opt/venv/bin/python
WORK=/data/work

# --- 1. the store, from S3 rather than HuggingFace ----------------------
note "pulling the BigEarthNet store from S3"
mkdir -p /data/store
aws s3 sync "s3://${BUCKET}/datasets/bigearthnet/" /data/store --only-show-errors
du -sh /data/store

LMDB=\$(find /data/store -maxdepth 2 -name "*.lmdb" -o -maxdepth 2 -name "BENv2*" -type d | head -1)
META=\$(find /data/store -maxdepth 2 -name "metadata*.parquet" | head -1)
echo "lmdb=\${LMDB} metadata=\${META}"
if [ -z "\${LMDB}" ] || [ -z "\${META}" ]; then
  note "FAILED: could not locate the store or its metadata"
  ls -R /data/store | head -50
  false
fi

# --- 2. the preamble-heavy rehearsal slice ------------------------------
# Same --per-type and --seed as the Stage A corpus, so the same patches are
# selected and the renderings are byte-identical. Only the preamble assignment
# differs, which is why the images are not re-rendered into S3 below -- the
# corpus points at the tree the loader run already uploaded.
note "re-preparing the rehearsal slice at rate ${PREAMBLE_RATE}"
\$PY \$WORK/scripts/prepare_bigearthnet.py \
    --lmdb "\${LMDB}" --metadata "\${META}" \
    --out /data/rehearsal --split train \
    --per-type ${PER_TYPE} --seed ${SEED} --cache /data/cache \
    --preamble-rate ${PREAMBLE_RATE}

# The mixture rewrites a slice's image paths relative to the data root, using
# where the slice file itself sits. Placing it at data/prepared/train/ makes its
# rows read prepared/train/images/<patch>_optical.png -- which is exactly where
# the loader run put those renderings in S3, so the GPU box finds them without
# this box re-uploading ten gigabytes of identical PNGs.
mkdir -p \$WORK/data/prepared/train
cp /data/rehearsal/train.jsonl \$WORK/data/prepared/train/train.jsonl

# --- 3. the benchmark train splits --------------------------------------
note "pulling benchmark train splits"
\$PY -m satquery.cli data pull vrsbench_train --with-images --data-root \$WORK/data
\$PY -m satquery.cli data pull rsvqa_lr_train --data-root \$WORK/data
if [ "${CDVQA_SHARDS}" != "0" ]; then
  \$PY -m satquery.cli data pull cdvqa_train --shards ${CDVQA_SHARDS} --data-root \$WORK/data
fi

# --- 4. the test splits, for the guard ----------------------------------
# Annotations only. Fingerprinting reads image ids and question text and never
# opens a file, so the evaluation imagery stays in S3 where it already is.
note "pulling test splits for the guard"
aws s3 sync "s3://${BUCKET}/datasets/CDVQA/" \$WORK/data/CDVQA --only-show-errors \
    --exclude "im1/*" --exclude "im2/*"
aws s3 sync "s3://${BUCKET}/datasets/RSVQA/" \$WORK/data/RSVQA --only-show-errors \
    --exclude "*Images_LR/*"
\$PY -m satquery.cli data pull vrsbench --data-root \$WORK/data
aws s3 sync "s3://${BUCKET}/datasets/prepared/bench/" \$WORK/data/prepared/bench \
    --only-show-errors --exclude "images/*"

# --- 5. the mixture -----------------------------------------------------
# No --root or --test-root override: each config carries its own root already
# and a single override would flatten all of them onto one directory. The
# working directory is what makes "data/VRSBench_train" resolve.
#
# --no-image-check because the rehearsal slice's renderings are deliberately not
# on this box; every other source has its imagery local.
note "building the Stage B mixture"
mkdir -p /data/out
cd \$WORK
\$PY -m satquery.cli data instruct \
    --config "configs/train/*.yaml" \
    --test-config "configs/bench/*.yaml" \
    --data-root \$WORK/data \
    --include "bigearthnet=\$WORK/data/prepared/train/train.jsonl" \
    --out /data/out/train.jsonl \
    --no-image-check 2>&1 | tee /data/out/mixture-report.txt

# Again on the written file, so the marker in S3 means the corpus in S3 was
# checked -- not that some earlier in-memory copy of it was.
note "verifying the written corpus"
\$PY -m satquery.cli data check-contamination /data/out/train.jsonl \
    --test-config "configs/bench/*.yaml" 2>&1 | tee -a /data/out/mixture-report.txt

# --- 6. upload ----------------------------------------------------------
note "uploading"
aws s3 cp /data/rehearsal/train.jsonl \
    "s3://${BUCKET}/datasets/prepared/stage-b-rehearsal/train.jsonl"
aws s3 cp /data/rehearsal/manifest.json \
    "s3://${BUCKET}/datasets/prepared/stage-b-rehearsal/manifest.json" || true
aws s3 sync /data/out "\${STATUS}/" --only-show-errors

# The benchmark imagery, which is the part the GPU box cannot reconstruct.
aws s3 sync \$WORK/data/VRSBench_train "s3://${BUCKET}/datasets/VRSBench_train/" \
    --only-show-errors
aws s3 sync \$WORK/data/RSVQA "s3://${BUCKET}/datasets/RSVQA/" --only-show-errors
if [ "${CDVQA_SHARDS}" != "0" ]; then
  aws s3 sync \$WORK/data/CDVQA_train "s3://${BUCKET}/datasets/CDVQA_train/" \
      --only-show-errors
fi


aws s3 cp /var/log/satquery-stage-b.log "\${STATUS}/_DONE.log"
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
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=satquery-stage-b},{Key=Project,Value=SIH26167}]' \
    --query 'Instances[0].InstanceId' --output text)

cat <<DONE

launched ${INSTANCE_ID}

No SSH, no inbound rule, no key. Follow it from S3:

  aws s3 cp s3://${BUCKET}/datasets/prepared/stage-b/_STATUS.txt -
  aws s3 cp s3://${BUCKET}/datasets/prepared/stage-b/mixture-report.txt -
  aws ec2 describe-instances --region ${REGION} --instance-ids ${INSTANCE_ID} \\
      --query 'Reservations[].Instances[].State.Name' --output text

  _DONE.log             mixture built and everything uploaded
  _FAILED.log           something broke; the log says what
  mixture-report.txt    per-source counts, the two composition warnings, and
                        the contamination verdict on the written file

Read mixture-report.txt before spending GPU hours. A build that succeeded while
warning about optical-SAR records or the evidence share is a corpus that will
train, finish, and give back a smaller delta than the one before it.

The instance terminates itself either way. "terminated" with no marker means it
died before it could report -- check the console output for that case.
DONE
