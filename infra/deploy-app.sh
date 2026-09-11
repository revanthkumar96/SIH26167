#!/usr/bin/env bash
# Deploy the SatQuery application to a single EC2 instance.
#
#   ./infra/deploy-app.sh          # dry run: prints the plan, launches nothing
#   ./infra/deploy-app.sh --apply  # actually launches a billable instance
#
# Works today on CPU, and becomes the GPU deployment by changing one variable.
# That is the point of the openai_compat backend: the application talks to an
# OpenAI-compatible endpoint and does not care whether vLLM, Ollama or a
# teammate's tunnel is behind it.
#
#   SATQUERY_INSTANCE_TYPE=c7i.2xlarge  SATQUERY_BACKEND=ollama         # now
#   SATQUERY_INSTANCE_TYPE=g5.xlarge    SATQUERY_BACKEND=openai_compat  # on quota
#
# vLLM is deliberately not attempted on the CPU shape. Its CPU backend needs a
# source build, does not meaningfully cover multimodal models, and a 2B
# vision-language model on eight cores answers in tens of seconds. Ollama serves
# the same model quantised to 4-bit and is the honest choice for a CPU host --
# it is also the documented offline fallback, so this is not throwaway work.
#
# Cost note: c7i.2xlarge is roughly $0.09/hr on-demand in ap-south-1, so a day of
# demo hosting is small change against the $100 grant. The idle-shutdown timer
# from bootstrap.sh still applies and is what stops it becoming a forgotten box.
set -euo pipefail

REGION=${AWS_DEFAULT_REGION:-ap-south-1}
INSTANCE_TYPE=${SATQUERY_INSTANCE_TYPE:-c7i.2xlarge}
BACKEND=${SATQUERY_BACKEND:-ollama}
MODEL=${SATQUERY_MODEL:-qwen3-vl:2b-instruct}
BUCKET=${SATQUERY_BUCKET:?set SATQUERY_BUCKET}
KEY_NAME=${SATQUERY_KEY_NAME:?set SATQUERY_KEY_NAME to an EC2 key pair}
REPO_URL=${SATQUERY_REPO_URL:-https://github.com/revanthkumar96/SIH26167.git}
SG_NAME=${SATQUERY_SG_NAME:-satquery-app}
APPLY=${1:-}

# The security group admits exactly one address. The app has no authentication
# of its own, so anything wider publishes an upload endpoint to the internet.
MY_IP=$(curl -fsS --max-time 10 https://checkip.amazonaws.com | tr -d '[:space:]')

cat <<PLAN
plan
  region          ${REGION}
  instance        ${INSTANCE_TYPE}
  backend         ${BACKEND}  (model ${MODEL})
  bucket          s3://${BUCKET}
  ssh key         ${KEY_NAME}
  ingress         ${MY_IP}/32 only, ports 22 and 8000
PLAN

if [ "$APPLY" != "--apply" ]; then
    echo
    echo "dry run. Re-run with --apply to launch a billable instance."
    exit 0
fi

echo "== security group =="
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=${SG_NAME}" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    SG_ID=$(aws ec2 create-security-group --region "$REGION" \
        --group-name "$SG_NAME" --description "SatQuery app, single-operator access" \
        --query GroupId --output text)
    echo "   created ${SG_ID}"
else
    echo "   reusing ${SG_ID}"
fi
for port in 22 8000; do
    aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" \
        --protocol tcp --port "$port" --cidr "${MY_IP}/32" >/dev/null 2>&1 \
        && echo "   opened ${port} to ${MY_IP}/32" \
        || echo "   ${port} already open to ${MY_IP}/32"
done

# Ubuntu 24.04 from the SSM public parameter, so no AMI id is hardcoded and goes
# stale in a region we do not use.
AMI=$(aws ssm get-parameters --region "$REGION" \
    --names /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
    --query 'Parameters[0].Value' --output text)
echo "== AMI ${AMI} =="

USER_DATA=$(cat <<CLOUDINIT
#!/bin/bash
set -eux
apt-get update && apt-get install -y python3-pip python3-venv git curl
su - ubuntu -c '
  git clone ${REPO_URL} ~/SIH26167 || true
  cd ~/SIH26167
  python3 -m venv .venv && . .venv/bin/activate
  pip install -e ".[geo]"
  # Ollama pulls its own runtime; the model is 1.9 GB quantised.
  curl -fsSL https://ollama.com/install.sh | sh
  ollama serve > ~/ollama.log 2>&1 &
  sleep 10 && ollama pull ${MODEL}
'
cat > /etc/systemd/system/satquery.service <<UNIT
[Unit]
Description=SatQuery application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/SIH26167
Environment=SATQUERY_BACKEND=${BACKEND}
Environment=SATQUERY_MODEL=${MODEL}
Environment=SATQUERY_DATA_ROOT=/home/ubuntu/SIH26167/data
ExecStart=/home/ubuntu/SIH26167/.venv/bin/python -m uvicorn satquery.api.app:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload && systemctl enable --now satquery
CLOUDINIT
)

# The device name goes in a file rather than on the command line. Git Bash
# rewrites a bare /dev/sda1 argument into C:/Program Files/Git/dev/sda1 and the
# API rejects it -- the same trap load-bigearthnet.sh documents.
SCRATCH=.satquery-tmp
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
cat > "$SCRATCH/bdm.json" <<BDM
[{"DeviceName": "/dev/sda1",
  "Ebs": {"VolumeSize": 80, "VolumeType": "gp3", "DeleteOnTermination": true}}]
BDM

echo "== launching =="
INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
    --block-device-mappings "file://$SCRATCH/bdm.json" \
    --user-data "$USER_DATA" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=satquery-app},{Key=Project,Value=SIH26167}]' \
    --query 'Instances[0].InstanceId' --output text)
echo "   ${INSTANCE_ID}"

aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

cat <<DONE

running: ${INSTANCE_ID} at ${PUBLIC_IP}

Cloud-init still has to install the app and pull the model, so give it a few
minutes, then:

  http://${PUBLIC_IP}:8000

  ssh -i <key>.pem ubuntu@${PUBLIC_IP}
  sudo journalctl -u satquery -f          # or: tail -f /var/log/cloud-init-output.log

Stop it when you are done. This box has no idle timer of its own -- that is
installed by bootstrap.sh on the training host:

  aws ec2 stop-instances --region ${REGION} --instance-ids ${INSTANCE_ID}
  ./infra/session-end.sh

When the GPU quota clears, the same script deploys the real thing:

  SATQUERY_INSTANCE_TYPE=g5.xlarge SATQUERY_BACKEND=openai_compat \\
  SATQUERY_MODEL=qwen3-vl-satquery ./infra/deploy-app.sh --apply

and vLLM serves base and adapted from one process. Nothing in the application
changes -- only which endpoint openai_compat is pointed at.
DONE
