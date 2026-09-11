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
# WHERE THE MODEL COMES FROM. The Hugging Face repo, not S3. The merged weights
# were published there and S3 never kept a copy -- the results sync deliberately
# excludes merged*/ because a 4.3 GB artefact that can be rebuilt from an adapter
# is not worth the storage. Pulling from the Hub also avoids a cross-region copy,
# since the quota is us-east-1 and the bucket is ap-south-1.
#
# The repo is private, so the box needs a token. It is pushed over stdin after
# boot rather than embedded in user-data: user-data is readable by anything that
# can reach the instance metadata endpoint, which is a poor place for a
# credential that can write to your model repos.
#
# IDLE. infra/idle-shutdown.sh halts a box whose GPU is quiet, which is correct
# for a training host and actively wrong here: a serving box sits at 0% GPU
# between requests, so that timer would kill the endpoint while it waits for a
# query. This installs a request-based timer instead -- idle means no HTTP
# request for IDLE_MINUTES, read from the API's own access log.
set -euo pipefail

REGION=${AWS_DEFAULT_REGION:-us-east-1}            # where the GPU quota lives
INSTANCE_TYPE=${SATQUERY_INSTANCE_TYPE:-g6.xlarge}
MODEL_REPO=${SATQUERY_MODEL_REPO:-Siddu2004-2006/satquery-qwen3vl-2b}
# NOTE if creating a new key pair on Windows: aws.exe writes CRLF, and OpenSSH
# rejects a CRLF private key with "error in libcrypto", which names neither the
# file nor the cause. Convert to LF and set the ACL to owner-read (chmod is a
# no-op on NTFS) before using it.
#
# Created for this box specifically, in the serving region. A key pair is
# regional: an ap-south-1 key cannot open a us-east-1 instance, and AWS will
# not re-issue a private key, so losing this one means rebuilding the box.
KEY_NAME=${SATQUERY_KEY_NAME:-satquery-serve-use1}
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
  instance        ${INSTANCE_TYPE}
  model           ${MODEL_REPO}  (private HF repo, 4.26 GB)
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

# --- 1. network ----------------------------------------------------------
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

apt-get update && apt-get install -y python3-venv git awscli software-properties-common

# The Deep Learning AMI is Ubuntu 22.04, which ships Python 3.10, and pyproject
# requires >=3.11 -- schema.py uses StrEnum and the code uses datetime.UTC. This
# exact fault stopped the training box first; not carrying the fix here cost the
# same debugging twice.
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-dev

install -d -o ubuntu -g ubuntu /opt/satquery
su - ubuntu -c '
  set -eux
  git clone ${REPO_URL} ~/SIH26167 || true
  cd ~/SIH26167
  python3.11 -m venv .venv && . .venv/bin/activate
  pip install -q --upgrade pip
  pip install -q -e ".[geo]"
  # vLLM gets its own environment. The application only speaks HTTP to it, so
  # there is no reason for a serving engine and an app to share dependencies --
  # and Stage A lost an afternoon to exactly that collision.
  python3 -m venv ~/vllm-env
  ~/vllm-env/bin/pip install -q --upgrade pip
  ~/vllm-env/bin/pip install -q vllm
  ~/vllm-env/bin/pip install -q "huggingface_hub[hf_transfer]"
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
# Enabled, not started: vLLM would exit immediately against an empty model
# directory and systemd would then back off its restarts. The post-boot step
# starts both once the weights are actually on disk.
systemctl enable vllm satquery serve-idle.timer
systemctl start serve-idle.timer
touch /workspace-ready 2>/dev/null || touch /opt/satquery/PROVISIONED
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
  "Ebs": {"VolumeSize": 120, "VolumeType": "gp3", "DeleteOnTermination": true}}]
BDM
# USER_DATA is built far above; it has to reach disk before run-instances can
# reference it. Rewriting the launch block once dropped this line, and the
# resulting ParamValidation error was reported by the retry loop as "no
# capacity" until the real stderr was surfaced.
printf %s "$USER_DATA" > "$SCRATCH/user-data.sh"

echo "== launching =="
# Two things learned the hard way here.
#
# Do not pin a subnet. Naming one fails with InsufficientInstanceCapacity in
# every AZ -- each error helpfully suggests the others, which also fail -- while
# the identical request with no subnet succeeds immediately. EC2 places across
# AZs itself and is better at finding G capacity than a loop is.
#
# Fall through instance types. G capacity is genuinely tight; every shape the
# quota covers holds a 4.3 GB model with room for a KV cache, so a fallback
# costs throughput rather than capability.
TYPES=${SATQUERY_INSTANCE_TYPES:-"${INSTANCE_TYPE} g5.xlarge g6.2xlarge g5.2xlarge g4dn.xlarge"}

INSTANCE_ID=""
for itype in $TYPES; do
    printf '   trying %-14s ... ' "$itype"
    INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
        --image-id "$AMI" --instance-type "$itype" \
        --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
        --block-device-mappings "file://$SCRATCH/bdm.json" \
        --user-data "file://$SCRATCH/user-data.sh" \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=satquery-serve},{Key=Project,Value=SIH26167}]' \
        --query 'Instances[0].InstanceId' --output text 2>"$SCRATCH/launch.err") || INSTANCE_ID=""
    if [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ]; then
        echo "got $INSTANCE_ID"
        INSTANCE_TYPE="$itype"
        break
    fi
    # The reason is printed, not swallowed: a capacity shortage and a bad
    # request look identical when stderr goes to /dev/null, and that cost a
    # round of debugging the first time.
    echo "failed -- $(tail -1 "$SCRATCH/launch.err" | sed 's/.*): //' | cut -c1-90)"
done

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "None" ]; then
    echo
    echo "No G capacity in ${REGION} for any of: ${TYPES}"
    echo "Last error:"; tail -2 "$SCRATCH/launch.err"
    exit 1
fi
echo "   launched ${INSTANCE_ID} (${INSTANCE_TYPE})"

aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

cat <<WAIT

launched ${INSTANCE_ID} at ${PUBLIC_IP}

Cloud-init installs vLLM and the app, which takes several minutes. The model is
NOT fetched by cloud-init: the repo is private, and a token in user-data is
readable by anything that can reach the instance metadata endpoint.
WAIT

echo "== waiting for ssh =="
for _ in $(seq 1 60); do
    ssh -i "$HOME/.ssh/${KEY_NAME}.pem" -o ConnectTimeout=10 -o BatchMode=yes         -o StrictHostKeyChecking=accept-new "ubuntu@${PUBLIC_IP}" 'echo ok' >/dev/null 2>&1 && break
    sleep 20
done

echo "== waiting for provisioning =="
for _ in $(seq 1 90); do
    ssh -i "$HOME/.ssh/${KEY_NAME}.pem" -o BatchMode=yes "ubuntu@${PUBLIC_IP}"         'test -f /opt/satquery/PROVISIONED' 2>/dev/null && break
    sleep 20
done

# The token goes over stdin and into a 0600 file owned by the service user. It
# never appears in a command line, in shell history, or in instance metadata.
echo "== pushing the Hub token and fetching the model =="
HF=$(python -c "
for ln in open('${SATQUERY_ENV_FILE:-$HOME/OneDrive/Desktop/SIH26167/.env}', encoding='utf-8-sig'):
    if ln.strip().startswith('HF_TOKEN'):
        print(ln.split('=', 1)[1].strip().strip('\"').strip(\"'\"))" 2>/dev/null)
if [ -z "$HF" ]; then
    echo "no HF_TOKEN found; set SATQUERY_ENV_FILE to the .env holding it"
    exit 1
fi
printf '%s' "$HF" | ssh -i "$HOME/.ssh/${KEY_NAME}.pem" -o BatchMode=yes "ubuntu@${PUBLIC_IP}"     'umask 077 && cat > ~/.hf_token && echo "  token written ($(stat -c %a ~/.hf_token))"'

ssh -i "$HOME/.ssh/${KEY_NAME}.pem" -o BatchMode=yes "ubuntu@${PUBLIC_IP}" "
  set -eu
  export HF_TOKEN=\$(cat ~/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=1
  ~/vllm-env/bin/python -c \"
from huggingface_hub import snapshot_download
import os
p = snapshot_download('${MODEL_REPO}', token=os.environ['HF_TOKEN'],
                      local_dir='/opt/satquery/model')
print('  model at', p)
\"
  sudo systemctl start vllm satquery
  echo '  services started'
"

cat <<DONE

serving: ${INSTANCE_ID} at ${PUBLIC_IP}

vLLM loads 4.3 GB of weights before the first answer, so give it a minute:

  curl http://${PUBLIC_IP}:8000/api/health
  http://${PUBLIC_IP}:8000

  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}
  sudo journalctl -u vllm -f
  sudo journalctl -u satquery -f

It halts itself after ${IDLE_MINUTES} minutes with no HTTP request and nobody
logged in. To stop it now, or start it again for a demo without rebuilding:

  aws ec2 stop-instances  --region ${REGION} --instance-ids ${INSTANCE_ID}
  aws ec2 start-instances --region ${REGION} --instance-ids ${INSTANCE_ID}

Stopping keeps the EBS volume and the model on it, so a restart is a minute
rather than a rebuild. The public IP changes on restart unless you attach an
Elastic IP.
DONE
