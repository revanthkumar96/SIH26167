#!/usr/bin/env bash
# Prepare a freshly launched g5.xlarge for the programme.
#
# Run on the instance after first connecting. Idempotent: safe to re-run after a
# spot reclamation replaces the box.
set -euo pipefail

REPO=${SATQUERY_REPO:-/opt/satquery}
BUCKET=${SATQUERY_BUCKET:?set SATQUERY_BUCKET}

echo "== confirming the GPU is present =="
# The first thing to check, and the first thing to go wrong: a spot request can
# be satisfied by an instance whose driver did not come up.
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

echo
echo "== installing the project =="
cd "$REPO"
pip install -e ".[data,geo,hf,vllm]"
pip install "transformers==4.57.1" "peft==0.17.1" "accelerate==1.7.0" "qwen-vl-utils==0.0.14"

echo
echo "== installing the idle auto-shutdown timer =="
sudo install -m 0755 "$REPO/infra/idle-shutdown.sh" /opt/satquery/infra/idle-shutdown.sh
sudo install -m 0644 "$REPO/infra/satquery-idle.service" /etc/systemd/system/
sudo install -m 0644 "$REPO/infra/satquery-idle.timer" /etc/systemd/system/
sudo sed -i "s|^Environment=SATQUERY_S3_CHECKPOINTS=.*|Environment=SATQUERY_S3_CHECKPOINTS=s3://${BUCKET}/checkpoints/|" \
    /etc/systemd/system/satquery-idle.service
sudo systemctl daemon-reload
sudo systemctl enable --now satquery-idle.timer
systemctl list-timers satquery-idle.timer --no-pager

echo
echo "== checkpoint round trip =="
# Proves the instance role can actually write to the bucket, before eight hours
# of training discovers it cannot.
printf 'roundtrip %s\n' "$(date -Iseconds)" > /tmp/satquery-probe.txt
aws s3 cp /tmp/satquery-probe.txt "s3://${BUCKET}/checkpoints/.probe"
aws s3 cp "s3://${BUCKET}/checkpoints/.probe" /tmp/satquery-probe-back.txt
diff /tmp/satquery-probe.txt /tmp/satquery-probe-back.txt && echo "S3 round trip ok"
aws s3 rm "s3://${BUCKET}/checkpoints/.probe"

cat <<'NOTE'

Phase 0 is done when all three of these held:
  - nvidia-smi showed the A10G                      (above)
  - a checkpoint round-tripped to S3                (above)
  - the idle timer demonstrably halts the box

The third is not proven by installing it. Verify it once, deliberately:
  sudo IDLE_CYCLES=1 /opt/satquery/infra/idle-shutdown.sh
and confirm the box halts. Do that before trusting it with a budget.
NOTE
