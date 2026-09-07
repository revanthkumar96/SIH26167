# Infrastructure

One `g5.xlarge` trains the adapters and serves the model. Budget: **$100 in
credits**, of which the programme spends roughly half.

**None of this has been applied.** The scripts are written and syntax-checked;
no AWS credentials were available in the environment they were written in, so no
bucket, role, instance or budget exists yet. Treat the first run of each as a
first run.

---

## What is here

| file | what it does |
| --- | --- |
| `budgets.sh` | AWS Budgets alarms at $40 and $75 |
| `idle-shutdown.sh` | halts the box after 30 idle minutes |
| `satquery-idle.{service,timer}` | systemd units that run the check every minute |
| `bootstrap.sh` | prepares a freshly launched instance, and proves it works |
| `session-end.sh` | sync, stop, and show what is still running |

---

## Order

Cost control goes in **before** the first training run, not after the first
surprise.

```bash
export AWS_DEFAULT_REGION=ap-south-1
export SATQUERY_ALERT_EMAIL=you@example.com
export SATQUERY_BUCKET=satquery-<suffix>
./infra/budgets.sh
```

Confirm the instance family exists in this region first — spot capacity for
A10G is thinner in Mumbai than in `us-east-1`, which is the price of operating
the box from the same continent as the operator:

```bash
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters Name=instance-type,Values=g5.xlarge \
  --region ap-south-1 \
  --query 'InstanceTypeOfferings[].Location' --output table
```

Then launch the instance and, on it:

```bash
export SATQUERY_BUCKET=satquery-<suffix>
./infra/bootstrap.sh
```

### Prove the idle timer before trusting it

Installing the timer is not evidence that it works, and the whole point of it is
that nobody is watching when it matters:

```bash
sudo IDLE_CYCLES=1 /opt/satquery/infra/idle-shutdown.sh
```

The box should halt. Phase 0 is not done until it has.

---

## Shape

| | |
| --- | --- |
| Instance | `g5.xlarge` — NVIDIA A10G, 24 GB |
| Pricing | spot (~$0.30–0.40/hr), on-demand (~$1.01/hr) fallback |
| Region | `ap-south-1` (Mumbai) — operated from India; see `docs/AWS.md` |
| Storage | S3 for datasets and checkpoints, 200 GB gp3 EBS for working data |
| AMI | Deep Learning AMI, PyTorch 2.x / CUDA 12.x |

**Stop, do not terminate**, between sessions. The EBS volume holds the 155 GB
LMDB, and re-pulling it is hours. HuggingFace-to-EC2 ingress is free, so that
pull costs time rather than money — but only once.

---

## Why the idle timer is the important one

The likeliest way to lose the grant is not an expensive mistake, it is a
forgotten instance: ~$24 a night at on-demand rates, a quarter of the budget,
for nothing.

Budget alarms cannot stop anything — they send email to someone who may be
asleep. `idle-shutdown.sh` is the control, and it is written so that it does not
depend on anyone remembering.

It requires **all** of these to be quiet before it counts a cycle as idle:

- GPU utilisation below 5%
- no login session
- no `train_lora.py`, `train_landcover.py`, `prepare_bigearthnet.py` or
  `aws s3` process running

The third clause matters more than it looks. A 155 GB dataset pull and a
checkpoint sync are both real work with no GPU load at all, and a timer that
killed the box mid-pull would cost more than the one it saved.

---

## Storage layout

```
s3://satquery-<suffix>/
  datasets/
    bigearthnet-txt/BigEarthNet.txt.parquet
    prepared/{train,bench,dev}/{*.jsonl, images/, manifest.json}
  checkpoints/
    stage-a/<run-id>/  stage-b/<run-id>/  cnn/<run-id>/
  adapters/
    qwen3-vl-satquery/<version>/
  results/
    baseline/   # the "before" matrix
    adapted/    # the "after" matrix
```

Keep `results/` in git as well. Predictions are large and regenerable; the
results table is the source of the submission's numbers and must not live only
in a bucket.

---

## Serving

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --enable-lora --lora-modules qwen3-vl-satquery=/opt/adapters/qwen3-vl-satquery \
  --max-model-len 8192 --port 8000
```

Base and adapted are then two model names on one process, which is exactly what
the benchmark matrix consumes. Point the app at it:

```bash
export SATQUERY_BACKEND=openai_compat
export SATQUERY_VLM_BASE_URL=http://<host>:8000/v1
export SATQUERY_VLM_API_KEY=<token>
```

The port must **not** be open to the world — restrict the security group to
known IPs or reach it over an SSH tunnel, and set an API key even then.

---

## Session end

```bash
export SATQUERY_BUCKET=satquery-<suffix>
export SATQUERY_INSTANCE_ID=i-...
./infra/session-end.sh
```

It syncs, stops, and then lists every instance in the region. The listing is the
point: a `stop-instances` that silently failed looks exactly like one that
worked.

---

## Demo day

Rehearse all three before the day, not on it.

1. **Primary** — EC2 running, adapter loaded, app pointed at it.
2. **Degraded** — instance unreachable; `SATQUERY_BACKEND=ollama` and carry on
   with the local 1.9 GB model.
3. **Fully offline** — no network; bundled GeoTIFF samples and the local model.

Start the instance well before the slot. A cold spot request can be rejected,
and finding that out five minutes before presenting is avoidable.
