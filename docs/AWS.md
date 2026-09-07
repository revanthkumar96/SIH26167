# AWS

One GPU instance trains the adapters and serves the model. Budget: **$100 in
credits**, of which the programme in `ROADMAP.md` spends roughly half.

The single likeliest way to lose that budget is a forgotten running instance.
Cost control is therefore a requirement in this document, not an afterthought at
the end of it.

---

## Shape

| | |
| --- | --- |
| Instance | `g5.xlarge` — NVIDIA A10G, 24 GB |
| Pricing | **spot** primary (~$0.30–0.40/hr), on-demand (~$1.01/hr) fallback |
| Region | **`ap-south-1`** (Mumbai) — see below |
| Storage | S3 for datasets and checkpoints, EBS gp3 200 GB for working data |
| AMI | Deep Learning AMI, PyTorch 2.x / CUDA 12.x |
| Serving | vLLM OpenAI-compatible server |

### Why this and not the alternatives

**vLLM on EC2, not SageMaker endpoints.** vLLM serves LoRA adapters directly via
`--enable-lora`, so base and adapted models are two `model` names against one
process — exactly the shape the benchmark matrix wants. SageMaker costs more per
hour and adds moving parts for no benefit at this scale. Bedrock does not host
custom Qwen weights at all.

**One instance type for both jobs.** The same `g5.xlarge` trains and serves, so
there is one environment to build and debug rather than two.

**A10G over the T4 the competitor used.** Ampere means **bf16**, so no fp16
GradScaler and one fewer source of training instability. 24 GB instead of 14.6
means `max_pixels` can rise, which is the lever grounding accuracy needs.

**`ap-south-1`, reversing an earlier call.** This document originally chose
`us-east-1` on the grounds that A10G spot capacity is deeper there and we are
not serving users, so latency is irrelevant. The second half of that was wrong.
We are not serving users, but we are *operating* the box from India — every SSH
round-trip, every interactive debugging session and the demo itself run over
that link, and a transatlantic hop makes all of them worse.

The capacity argument still stands and is the cost of the decision: a spot
request in Mumbai is likelier to be rejected or reclaimed. Two consequences to
plan for rather than be surprised by — budget for the on-demand fallback at
~$1.01/hr, and confirm the instance family is offered here at all before
provisioning anything:

```bash
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters Name=instance-type,Values=g5.xlarge \
  --region ap-south-1 \
  --query 'InstanceTypeOfferings[].Location' --output table
```

Empty output means g5 is not available in Mumbai, and the choice is a different
family or a different region -- not a thing to discover after the bucket and the
200 GB volume are already there.

---

## Cost control

Set these up **before** the first training run, not after the first surprise.

| guardrail | why |
| --- | --- |
| AWS Budgets alarms at **$40** and **$75** | early warning, then review |
| Idle auto-shutdown: systemd timer halting the box after 30 min with no GPU utilisation and no active SSH session | the forgotten-instance failure, solved structurally |
| Checkpoints stream to S3 during training, never only to EBS | spot reclamation gives two minutes' notice |
| `aws ec2 describe-instances` in the daily routine | catch anything the timer missed |
| Stop, do not terminate, between sessions | keeps the EBS volume and its cached datasets |

Budget estimate:

| item | rate | allocation |
| --- | --- | --- |
| training, ~20 GPU h spot | $0.35/hr | ~$7 |
| benchmarking, ~12 GPU h spot | $0.35/hr | ~$4 |
| demo serving, ~8 h on-demand | $1.01/hr | ~$8 |
| CNN training, ~4 GPU h | $0.35/hr | ~$1.50 |
| S3 storage and requests | | ~$3 |
| EBS gp3 200 GB, prorated | ~$16/mo | ~$8 |
| **subtotal** | | **~$32** |

Leaves headroom for a reclaimed spot run, a second training pass after ablation,
and demo-day contingency. Note **HuggingFace-to-EC2 ingress is free**; only
egress out of AWS is billed, so the 155 GB dataset pull costs time, not money.

---

## Storage layout

```
s3://satquery-<suffix>/
  datasets/
    bigearthnet-txt/BigEarthNet.txt.parquet
    prepared/train/{train.jsonl, images/, manifest.json}
    prepared/dev/...
  checkpoints/
    stage-a/<run-id>/
    stage-b/<run-id>/
    cnn/<run-id>/
  adapters/
    qwen3-vl-satquery/<version>/
  results/
    baseline/    # the "before" matrix
    adapted/     # the "after" matrix
```

Keep `results/` versioned in git as well. Predictions are large and regenerable;
the results table is the source of the submission numbers and must not live only
in a bucket.

The 155 GB LMDB is **one monolithic file** — there is no partial fetch. Pull it
to EBS once, keep the volume, and stage development on the 2.5 GB slice
(`DATA.md`) so the big pull happens exactly once.

---

## Serving contract

A new backend, `openai_compat`, joins `BACKENDS` alongside `ollama`, `hf`,
`vllm` and `echo`. Configuration:

```
SATQUERY_BACKEND=openai_compat
SATQUERY_VLM_BASE_URL=http://<host>:8000/v1
SATQUERY_VLM_API_KEY=<token>
SATQUERY_MODEL=qwen3-vl-satquery
```

It is close to free to build: `build_messages()` in `eval/backends/base.py`
already emits OpenAI-format content with base64 `image_url` parts. Because it
speaks plain OpenAI, the same code path works against a local vLLM, the EC2 box,
or a teammate's tunnel.

### Keep Ollama as the offline fallback

Not as the default — as insurance. Demo day with no network and a stopped
instance is a real failure mode, and a 1.9 GB local model answering in ~20 s is a
far better outcome than a stack trace in front of judges. The backend already
exists and is tested, so this costs nothing but the decision to keep it.

---

## Security

- The vLLM port must **not** be open to the world. Security group restricted to
  known IPs, or reached over an SSH tunnel.
- An API key on the endpoint even when access is IP-restricted.
- Credentials in environment variables or the instance role. Never in the repo —
  the pre-commit secret scanner will catch it, but do not rely on that.
- The S3 bucket is private. Nothing here needs public read.

---

## Runbook

**Session start**

```bash
aws ec2 start-instances --instance-ids <id>
# wait for status ok, then ssh in
nvidia-smi                      # confirm the A10G is present
aws s3 sync s3://<bucket>/datasets/prepared/ ~/data/prepared/
```

**Serving**

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --enable-lora --lora-modules satquery=/opt/adapters/qwen3-vl-satquery \
  --max-model-len 8192 --port 8000
```

Base and adapted are then two model names on one process, which is what the
benchmark matrix consumes.

**Session end**

```bash
aws s3 sync ~/checkpoints/ s3://<bucket>/checkpoints/
aws ec2 stop-instances --instance-ids <id>
aws ec2 describe-instances \
  --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output table
```

That last command is the habit worth building. Run it at the end of every
session, and once more before going to bed.

---

## Demo day

Rehearse all three of these before the day, not on it:

1. **Primary** — EC2 running, adapter loaded, app pointed at it.
2. **Degraded** — instance unreachable; switch `SATQUERY_BACKEND=ollama` and
   carry on with the local model.
3. **Fully offline** — no network; bundled GeoTIFF samples and the local model,
   no Planetary Computer feed.

Start the instance well before the slot — a cold spot request can be rejected,
and discovering that five minutes before presenting is avoidable.
