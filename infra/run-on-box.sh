#!/bin/bash
# Baseline -> train -> merge -> adapted -> delta, on the rented box.
#
# Everything runs through the `hf` backend. vLLM could not start on this host --
# its flashinfer sampler JIT-compiles kernels with ninja and that build failed --
# and the comparison does not require vLLM. It requires that base and adapted be
# measured by the *same* harness through the *same* backend, which one backend
# for both satisfies exactly.
#
# The adapter is merged before the adapted sweep so the hf backend can load it
# as an ordinary model, and so the artefact that gets benchmarked is the same
# artefact that gets published.
exec > >(tee /workspace/logs/run.log) 2>&1
set -x

cd /workspace/satquery
BUCKET=satquery-869987460914
PY=/opt/train-env/bin/python
LIMIT=${LIMIT:-200}
TRAIN_HOURS=${TRAIN_HOURS:-4.0}
BASE=Qwen/Qwen3-VL-2B-Instruct
REV=89644892e4d85e24eaac8bacfd4f463576704203

mark() {
    echo "$1" | aws s3 cp - "s3://${BUCKET}/results/_STATUS.txt" || true
    echo "### $1 at $(date -u +%H:%M:%S)"
}

mark "baseline sweep"
$PY -m satquery.cli bench run --config "configs/bench/*.yaml" \
    --backend hf --model "$BASE" --limit "$LIMIT" --seed 1234 \
    --out /workspace/runs/base --results /workspace/runs/results.csv
aws s3 cp /workspace/runs/results.csv "s3://${BUCKET}/results/results.csv"

# Checked before the GPU hours are committed: a corpus that overlaps the splits
# it is scored on produces a better number that means nothing, and every other
# signal in the run agrees with it.
mark "contamination guard"
$PY -m satquery.cli data check-contamination \
    /workspace/data/prepared/train/train.jsonl \
    --test-config "configs/bench/*.yaml" || {
        echo "REFUSING TO TRAIN: corpus overlaps the benchmark test splits"
        exit 1
    }

mark "training ${TRAIN_HOURS}h"
$PY scripts/train_lora.py --stage a \
    --data /workspace/data/prepared/train/train.jsonl \
    --image-root /workspace/data/prepared/train \
    --out /workspace/runs/adapters/stage-a \
    --max-hours "$TRAIN_HOURS" --save-steps 200 --batch-size 4
aws s3 sync /workspace/runs/adapters/ "s3://${BUCKET}/adapters/" --only-show-errors

mark "merging adapter"
$PY /workspace/merge.py

mark "adapted sweep"
$PY -m satquery.cli bench run --config "configs/bench/*.yaml" \
    --backend hf --model /workspace/runs/merged --limit "$LIMIT" --seed 1234 \
    --out /workspace/runs/adapted --results /workspace/runs/results.csv

mark "scoring"
$PY -m satquery.cli bench score --results /workspace/runs/results.csv \
    --baseline "$BASE" --json /workspace/runs/scores.json \
    | tee /workspace/runs/delta.txt

aws s3 sync /workspace/runs/ "s3://${BUCKET}/results/" \
    --exclude "*predictions.jsonl" --exclude "merged/*" --only-show-errors
mark "done"
touch /workspace/logs/RUN_DONE
