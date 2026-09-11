#!/usr/bin/env bash
# Self-host the adapted model on AWS: vLLM behind the SatQuery API, on a G
# instance in the region where the GPU quota actually exists.
#
#   ./infra/serve-model.sh            # dry run
#   ./infra/serve-model.sh --apply    # launches a billable instance
#
# This is the deployment half of the programme. Training stays on rented
# marketplace GPUs because it is bulk compute where price dominates and a
# machine vanishing mid-run costs one restart. Serving is the opposite: it has
# to be up when someone points at it, which is worth paying AWS rates for.
#
# Two things about this that are easy to get wrong, so they are handled here
# rather than left to whoever runs it at 2am before a demo:
#
# REGION. The G/VT quota was granted in us-east-1. The bucket, the corpus, the
# adapters and the 155 GB store are all in ap-south-1. The merged model is
# copied across once (~4.5 GB, about $0.09 in transfer) into a us-east-1 bucket
# so the serving box never pays that again on restart.
#
# IDLE. infra/idle-shutdown.sh halts a box whose GPU is quiet, which is correct
# for a training host and actively wrong here: a serving box sits at 0% GPU
# between requests, so that timer would kill the endpoint while it waits for a
# query. This installs a request-based timer instead -- idle means no HTTP
# request for IDLE_MINUTES, read from the API's own access log.
set -euo pipefail

SRC_REGION=${SATQUERY_SRC_REGION:-ap-south-1}      # where the artefacts live
REGION=${AWS_DEFAULT_REGION:-us-east-1}            # where the GPU quota lives
INSTANCE_TYPE=${SATQUERY_INSTANCE_TYPE:-g6.xlarge}
SRC_BUCKET=${SATQUERY_BUCKET:?set SATQUERY_BUCKET (the ap-south-1 bucket)}
DST_BUCKET=${SATQUERY_SERVE_BUCKET:-${SRC_BUCKET}-use1}
MODEL_PREFIX=${SATQUERY_MODEL_PREFIX:-models/satquery-stage-b}
KEY_NAME=${SATQUERY_KEY_NAME:?set SATQUERY_KEY_NAME to a us-east-1 EC2 key pair}
REPO_URL=${SATQUERY_REPO_URL:-https://github.com/revanthkumar96/SIH26167.git}
SG_NAME=${SATQUERY_SG_NAME:-satquery-serve}
IDLE_MINUTES=${SATQUERY_IDLE_MINUTES:-60}
SERVED_NAME=${SATQUERY_SERVED_NAME:-satquery}
APPLY=${1:-}

# The app has no authentication of its own, so the security group is the only
# thing between an upload endpoint and the open internet. One address.
MY_IP=$(curl -fsS --max-time 10 https://checkip.amazonaws.com | tr -d '[:space:]')

cat <<PLAN
plan
  serve region    ${REGION}        (where the G/VT quota was granted)
  artefact region ${SRC_REGION}    (where the merged model currently is)
  instance        ${INSTANCE_TYPE}
  model           s3://${DST_BUCKET}/${MODEL_PREFIX}
  ingress         ${MY_IP}/32 only, ports 22 and 8000
  idle shutdown   ${IDLE_MINUTES} min with no HTTP request

  g6.xlarge is ~\$0.805/hr on-demand. Left running it is ~\$19/day, so the
  idle timer below is the part that matters most in this script.
PLAN

if [ "$APPLY" != "--apply" ]; then
    echo
    echo "dry run. Re-run with --apply to launch a billable instance."
    exit 0
fi

# --- 1. get the model into the serving region ---------------------------
echo "== staging the merged model into ${REGION} =="
if ! aws s3api head-bucket --bucket "$DST_BUCKET" --region "$REGION" 2>/dev/null; then
    aws s3api create-bucket --bucket "$DST_BUCKET" --region "$REGION" \
        $([ "$REGION" = "us-east-1" ] || echo "--create-bucket-configuration LocationConstraint=$REGION")
    echo "   created s3://${DST_BUCKET}"
fi
if aws s3 ls "s3://${DST_BUCKET}/${MODEL_PREFIX}/config.json" --region "$REGION" >/dev/null 2>&1; then
    echo "   model already in ${REGION}, not paying for the transfer again"
else
    echo "   copying from ${SRC_REGION} (one time, ~4.5 GB)"
    aws s3 cp "s3://${SRC_BUCKET}/results/merged-b/" "s3://${DST_BUCKET}/${MODEL_PREFIX}/" \
        --recursive --source-region "$SRC_REGION" --region "$REGION" --only-show-errors
fi

# --- 2. network ----------------------------------------------------------
echo "== security group =="
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=${SG_NAME}" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    SG_ID=$(aws ec2 create-security-group --region "$REGION" \
        --group-name "$SG_NAME" --description "SatQuery serving, single-operator access" \
        --query GroupId --output text)
    echo "   created ${SG_ID}"
else
    echo "   reusing ${SG_ID}"
fi
for port in 22 8000; do
    aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" \
        --protocol tcp --port "$port" --cidr "${MY_IP}/32" >/dev/null 2>&1 \
        && echo "   opened ${port} to ${MY_IP}/32" || echo "   ${port} already open"
done

# The Deep Learning Base AMI ships the NVIDIA driver and CUDA. Installing those
# onto a stock Ubuntu image by hand is twenty minutes of apt and a reboot, and
# it is the step most likely to fail silently and leave a GPU box running on CPU.
AMI=$(aws ec2 describe-images --region "$REGION" --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
            "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
echo "== AMI ${AMI} =="

USER_DATA=$(cat <<CLOUDINIT
#!/bin/bash
exec > >(tee /var/log/satquery-serve.log) 2>&1
set -eux

apt-get update && apt-get install -y python3-venv git awscli

install -d -o ubuntu -g ubuntu /opt/satquery
su - ubuntu -c '
  set -eux
  git clone ${REPO_URL} ~/SIH26167 || true
  cd ~/SIH26167
  python3 -m venv .venv && . .venv/bin/activate
  pip install -q --upgrade pip
  pip install -q -e ".[geo]"
  # vLLM gets its own environment. The application only speaks HTTP to it, so
  # there is no reason for a serving engine and an app to share dependencies --
  # and Stage A lost an afternoon to exactly that collision.
  python3 -m venv ~/vllm-env
  ~/vllm-env/bin/pip install -q --upgrade pip
  ~/vllm-env/bin/pip install -q vllm
  aws s3 sync s3://${DST_BUCKET}/${MODEL_PREFIX}/ /opt/satquery/model/ --only-show-errors
'

cat > /etc/systemd/system/vllm.service <<UNIT
[Unit]
Description=vLLM serving the adapted SatQuery model
After=network.target

[Service]
User=ubuntu
ExecStart=/home/ubuntu/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \\
    --model /opt/satquery/model --served-model-name ${SERVED_NAME} \\
    --max-model-len 8192 --port 8001 --host 127.0.0.1
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/systemd/system/satquery.service <<UNIT
[Unit]
Description=SatQuery application
After=vllm.service
Requires=vllm.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/SIH26167
Environment=SATQUERY_BACKEND=openai_compat
Environment=SATQUERY_MODEL=${SERVED_NAME}
Environment=SATQUERY_VLM_BASE_URL=http://127.0.0.1:8001/v1
Environment=SATQUERY_VLM_API_KEY=EMPTY
Environment=SATQUERY_DATA_ROOT=/home/ubuntu/SIH26167/data
ExecStart=/home/ubuntu/SIH26167/.venv/bin/python -m uvicorn satquery.api.app:app \\
    --host 0.0.0.0 --port 8000 --access-log
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT

# Request-based idle shutdown. A serving box is idle when nobody is asking it
# anything -- not when the GPU is quiet, which is its normal resting state and
# would have this halt the endpoint mid-demo while it waits for a question.
cat > /usr/local/bin/serve-idle-check <<'IDLE'
#!/bin/bash
set -euo pipefail
IDLE_MINUTES=\${IDLE_MINUTES:-60}
last=\$(journalctl -u satquery --since "\${IDLE_MINUTES} min ago" --no-pager 2>/dev/null \\
        | grep -c "HTTP/1.1" || true)
if [ "\${last:-0}" -gt 0 ]; then exit 0; fi
if [ -n "\$(who 2>/dev/null)" ]; then exit 0; fi
logger -t satquery-idle "no HTTP request in \${IDLE_MINUTES} min and no session; halting"
shutdown -h now
IDLE
chmod +x /usr/local/bin/serve-idle-check

cat > /etc/systemd/system/serve-idle.service <<UNIT
[Unit]
Description=Halt this serving box when nothing is asking it anything
[Service]
Type=oneshot
Environment=IDLE_MINUTES=${IDLE_MINUTES}
ExecStart=/usr/local/bin/serve-idle-check
UNIT
cat > /etc/systemd/system/serve-idle.timer <<UNIT
[Unit]
Description=Idle check every 10 minutes
[Timer]
OnBootSec=30min
OnUnitActiveSec=10min
[Install]
WantedBy=timers.target
UNIT

systemctl daemon-reload
systemctl enable --now vllm satquery serve-idle.timer
CLOUDINIT
)

echo "== launching =="
INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3}' \
    --user-data "$USER_DATA" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=satquery-serve},{Key=Project,Value=SIH26167}]' \
    --query 'Instances[0].InstanceId' --output text)
echo "   ${INSTANCE_ID}"

aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

cat <<DONE

running: ${INSTANCE_ID} at ${PUBLIC_IP}

vLLM has to load ~4.5 GB of weights before the first answer, and cloud-init
still has to install it, so give this five to ten minutes:

  http://${PUBLIC_IP}:8000
  curl http://${PUBLIC_IP}:8000/api/health

  ssh -i <key>.pem ubuntu@${PUBLIC_IP}
  sudo journalctl -u vllm -f
  sudo journalctl -u satquery -f
  sudo tail -f /var/log/satquery-serve.log

It halts itself after ${IDLE_MINUTES} minutes with no HTTP request and nobody
logged in. To stop it now, or to start it again for a demo without rebuilding:

  aws ec2 stop-instances  --region ${REGION} --instance-ids ${INSTANCE_ID}
  aws ec2 start-instances --region ${REGION} --instance-ids ${INSTANCE_ID}

Stopping keeps the EBS volume (~\$0.08/GB-month for 120 GB, about \$10/month) and
the model stays on it, so a restart is a minute rather than a rebuild. The
public IP changes on restart unless you attach an Elastic IP.
DONE
