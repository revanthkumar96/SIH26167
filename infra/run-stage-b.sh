#!/bin/bash
# Stage B on the rented GPU box: instruction tuning on the task mixture.
#
#   scp this to the box, then:
#   TRAIN_HOURS=3.0 bash /workspace/run-stage-b.sh
#
# Stage A adapted the model to the domain and to optical-SAR pairs, and the
# measured delta was +0.0760 overall. Two columns it did not fix: captioning
# regressed (CIDEr-D 0.128 -> 0.003, captions 66.8 -> 79.9 words against a 47.4
# word reference) and grounding stayed at 0.000. Both are output-format failures
# rather than perception failures, which is what this stage is for.
#
# It resumes from the Stage A adapter rather than starting from base. That is
# why the corpus carries a BigEarthNet rehearsal slice instead of a full second
# pass: the cross-modal ability is already in the weights and the slice is there
# to stop it being trained away.
#
# The corpus is NOT built here. It was built and contamination-checked on a CPU
# box and is downloaded whole -- 44 MB. Rebuilding it at GPU rates would repeat
# half an hour of pure metadata work.
exec > >(tee /workspace/logs/stage-b.log) 2>&1
# -e and an ERR trap, because the first attempt at this run had three staging
# commands fail with ModuleNotFoundError and carried on regardless -- the log
# was correct and nobody was reading it. A run that continues past a failed
# stage produces a model trained on whatever happened to arrive.
set -eEuo pipefail
set -x

fail() {
    local code=$? line=${BASH_LINENO[0]}
    set +x
    echo "### FAILED at line ${line} (exit ${code})"
    echo "FAILED at line ${line} (exit ${code})"         | aws s3 cp - "s3://${BUCKET:-satquery-869987460914}/results/_STATUS_B.txt" || true
    touch /workspace/logs/STAGE_B_FAILED
    exit "$code"
}
trap fail ERR

cd /workspace/satquery
BUCKET=satquery-869987460914
PY=/opt/train-env/bin/python
LIMIT=${LIMIT:-200}
TRAIN_HOURS=${TRAIN_HOURS:-3.0}
# Set to a Trainer checkpoint to continue a run split across rentals. The
# step count, optimizer state and LR schedule position all come back, so the
# cosine curve completes instead of restarting.
RESUME_CKPT=${RESUME_CKPT:-}
# The baseline is the unadapted base model; it does not change between
# attempts. Re-running it would append a second set of rows for the same
# model to the results CSV, and the scorer would then be averaging one
# model against two different measurements of the same thing.
SKIP_BASELINE=${SKIP_BASELINE:-0}
BASE=Qwen/Qwen3-VL-2B-Instruct
REV=89644892e4d85e24eaac8bacfd4f463576704203
DATA=/workspace/data
FINAL=/workspace/runs/stage-b-results.csv

mark() {
    echo "$1" | aws s3 cp - "s3://${BUCKET}/results/_STATUS_B.txt" || true
    echo "### $1 at $(date -u +%H:%M:%S)"
}

mkdir -p "$DATA" /workspace/runs /workspace/logs
# Every benchmark config carries its own relative root -- data/VRSBench,
# data/RSVQA, data/CDVQA, data/prepared/bench. A single --root override
# replaces all of them with one directory and each config then looks for
# its annotations somewhere they are not. The symlink makes the relative
# roots resolve exactly as written, from the repo directory the CLI runs in.
ln -sfn "$DATA" /workspace/satquery/data
# Ships with the run so the watcher always has something to call.
cp -f infra/box-status.sh /workspace/status.sh 2>/dev/null || true

# --- staging -------------------------------------------------------------
# Each source comes from wherever it is cheapest. VRSBench imagery is 8.4 GB and
# is pulled from HuggingFace rather than from S3 on purpose: S3 egress to a
# rented box leaves the region and is billed per GB, while HuggingFace is free
# and at least as fast. S3 carries only what cannot be re-derived -- the
# rendered BigEarthNet pairs and the corpus itself.
mark "staging corpus"
mkdir -p "$DATA/prepared/stage-b"
aws s3 cp "s3://${BUCKET}/datasets/prepared/stage-b/train.jsonl" \
    "$DATA/prepared/stage-b/train.jsonl"
aws s3 cp "s3://${BUCKET}/datasets/prepared/stage-b/mixture-report.txt" - || true

# One 5 GB object, not 139,122 small ones. The corpus references 36,452 of those
# renderings; fetching them individually is thousands of round trips paid for at
# GPU rates while the card sits idle.
mark "staging BigEarthNet renderings"
if [ ! -d "$DATA/prepared/train/images" ]; then
    mkdir -p "$DATA/prepared"
    aws s3 cp "s3://${BUCKET}/datasets/prepared-train.tar.gz" - \
        | tar -xz -C "$DATA/prepared/"
fi

mark "staging benchmark imagery"
pids=()
$PY -m satquery.cli data pull vrsbench_train --with-images --data-root "$DATA" & pids+=($!)
$PY -m satquery.cli data pull rsvqa_lr_train --data-root "$DATA" & pids+=($!)
# The *test* annotations too: rsvqa_lr_train ships only the train files,
# and the benchmark config reads LR_split_test_*.json. Without this the
# rsvqa_lr split is unreadable and both the guard and the sweep lose it.
# The shared Images_LR archive is skipped once it is already on disk.
$PY -m satquery.cli data pull rsvqa_lr --data-root "$DATA" & pids+=($!)
$PY -m satquery.cli data pull vrsbench --with-images --data-root "$DATA" & pids+=($!)
aws s3 sync "s3://${BUCKET}/datasets/CDVQA/" "$DATA/CDVQA/" --only-show-errors & pids+=($!)
aws s3 sync "s3://${BUCKET}/datasets/prepared/bench/" "$DATA/prepared/bench/" --only-show-errors & pids+=($!)
$PY -c "
from huggingface_hub import snapshot_download
snapshot_download('${BASE}', revision='${REV}')" & pids+=($!)

# Bare `wait` returns 0 even when a background job failed, which is exactly
# how three ModuleNotFoundError pulls went unnoticed on the first attempt.
# Each pid is waited on individually so a failure is a failure.
staging_failed=0
for pid in "${pids[@]}"; do wait "$pid" || staging_failed=1; done
[ "$staging_failed" -eq 0 ] || { echo "### a staging job failed"; false; }

mark "staging the Stage A adapter"
mkdir -p /workspace/runs/adapters
aws s3 sync "s3://${BUCKET}/adapters/stage-a/" /workspace/runs/adapters/stage-a/ \
    --exclude "checkpoint-*/*" --only-show-errors
test -f /workspace/runs/adapters/stage-a/adapter_model.safetensors || {
    echo "REFUSING: no Stage A adapter to resume from"
    exit 1
}

# Staging "succeeded" is not the same as "the corpus can find its images".
# Checked on a sample rather than all 54,626 so it costs seconds, and fatal
# because every missing image is a record that trains on nothing.
$PY - <<PYEOF
import json, os, sys, collections
DATA = "$DATA"
missing = collections.Counter(); seen = collections.Counter()
with open(os.path.join(DATA, "prepared/stage-b/train.jsonl"), encoding="utf-8") as fh:
    for n, line in enumerate(fh):
        if n >= 3000:
            break
        for image in json.loads(line)["images"]:
            root = image.split("/")[0]
            seen[root] += 1
            if not os.path.exists(os.path.join(DATA, image)):
                missing[root] += 1
for root in sorted(seen):
    print(f"  {root:<16} {seen[root]:>6} refs {missing[root]:>6} missing")
if missing:
    sys.exit("image roots are incomplete -- staging did not deliver what the corpus references")
print("  every sampled image resolves")
PYEOF

# --- the gate ------------------------------------------------------------
# Checked again here even though the CPU box checked it. The corpus crossed a
# network and a tarball since then, and the whole point of the guard is that a
# contaminated corpus is indistinguishable from a clean one by every other
# signal in the run.
mark "contamination guard"
$PY -m satquery.cli data check-contamination \
    "$DATA/prepared/stage-b/train.jsonl" \
    --test-config "configs/bench/*.yaml" || {
        echo "REFUSING TO TRAIN: the guard did not clear this corpus."
        echo "(exit 2 means it could not read its splits; exit 1 means overlap)"
        exit 1
    }

# --- baseline ------------------------------------------------------------
# Re-measured rather than cited from the Stage A run. The claim this programme
# has to support is base vs adapted on identical splits through identical code,
# and two sweeps in one session is a cheaper way to guarantee that than an
# argument about whether anything changed in between.
if [ "$SKIP_BASELINE" = "1" ] && [ -s "$FINAL" ]; then
    mark "baseline sweep skipped (rows already in $FINAL)"
    grep -c . "$FINAL"
else
    mark "baseline sweep"
    $PY -m satquery.cli bench run --config "configs/bench/*.yaml" \
        --backend hf --model "$BASE" --limit "$LIMIT" --seed 1234 \
        --out /workspace/runs/b-base --results "$FINAL"
    aws s3 cp "$FINAL" "s3://${BUCKET}/results/stage-b-results.csv" || true
fi

# --- train ---------------------------------------------------------------
# --image-root is not optional: the corpus spans four source trees, so its paths
# are relative to data/ rather than sitting beside the jsonl the way Stage A's
# did. train_lora.py would otherwise default to the corpus's own directory and
# fail to open every image in the run.
mark "training stage b, ${TRAIN_HOURS}h budget"
$PY scripts/train_lora.py --stage b \
    --resume /workspace/runs/adapters/stage-a \
    ${RESUME_CKPT:+--resume-from-checkpoint "$RESUME_CKPT"} \
    --data "$DATA/prepared/stage-b/train.jsonl" \
    --image-root "$DATA" \
    --out /workspace/runs/adapters/stage-b \
    --max-hours "$TRAIN_HOURS" --save-steps 200 --batch-size 4 \
    --dataloader-workers 8
aws s3 sync /workspace/runs/adapters/stage-b/ "s3://${BUCKET}/adapters/stage-b/" \
    --only-show-errors

mark "merging adapter"
$PY scripts/merge_adapter.py \
    --adapter /workspace/runs/adapters/stage-b --out /workspace/runs/merged-b

mark "adapted sweep"
$PY -m satquery.cli bench run --config "configs/bench/*.yaml" \
    --backend hf --model /workspace/runs/merged-b --limit "$LIMIT" --seed 1234 \
    --out /workspace/runs/b-adapted --results "$FINAL"

mark "scoring"
$PY -m satquery.cli bench score --results "$FINAL" \
    --baseline "$BASE" --json /workspace/runs/stage-b-scores.json \
    | tee /workspace/runs/stage-b-delta.txt

# The aggregate hides the two defects Stage A was criticised for, so they are
# pulled out explicitly rather than left for someone to notice later.
$PY - <<'PYEOF' | tee /workspace/runs/stage-b-quality.txt
import json, pathlib
for run in ("b-base", "b-adapted"):
    root = pathlib.Path("/workspace/runs") / run
    print(f"== {run}")
    for cell in sorted(root.rglob("predictions.jsonl")):
        rows = [json.loads(x) for x in cell.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not rows:
            continue
        empty = sum(1 for r in rows if not (r.get("raw_text") or "").strip())
        unparsed = sum(1 for r in rows if r.get("bbox") is None and r.get("reference_bbox"))
        words = [len((r.get("raw_text") or "").split()) for r in rows]
        mean = sum(words) / len(words) if words else 0
        print(f"   {cell.parent.name:<22} n={len(rows):<5} empty={empty:<4} "
              f"unparsed_box={unparsed:<4} mean_words={mean:.1f}")
PYEOF

aws s3 sync /workspace/runs/ "s3://${BUCKET}/results/" \
    --exclude "*predictions.jsonl" --exclude "merged*/*" --only-show-errors

if [ -n "${HF_REPO:-}" ]; then
    mark "publishing ${HF_REPO}"
    $PY scripts/publish_adapter.py \
        --adapter /workspace/runs/adapters/stage-b \
        --repo "$HF_REPO" --results "$FINAL" \
        --baseline-model "$BASE" --adapted-model /workspace/runs/merged-b \
        --merge --merge-dir /workspace/runs/merged-b
fi

mark "stage b finished"
touch /workspace/logs/STAGE_B_DONE
