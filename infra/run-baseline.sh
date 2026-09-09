#!/usr/bin/env bash
# Produce the "before" column: stock Qwen3-VL across every benchmark.
#
# This is the most important artefact the project produces. Everything after it
# is measured against it, and without it "we fine-tuned the model" is a claim
# nobody can check. Run it before training anything.
#
#   ./infra/run-baseline.sh
#
# Serves the base weights under the name `qwen3-vl-base`, which is what the
# model catalogue and the openai_compat backend expect. When the adapter exists,
# the same server gains `--lora-modules qwen3-vl-satquery=<path>` and the adapted
# row is measured by the identical harness against the identical server -- which
# is the whole point. A comparison across two backends or two precisions is not
# a comparison.
set -euo pipefail

BUCKET=${SATQUERY_BUCKET:?set SATQUERY_BUCKET}
REPO=${SATQUERY_REPO:-$HOME/SIH26167}
DATA=${SATQUERY_DATA_ROOT:-$REPO/data}
LIMIT=${SATQUERY_BENCH_LIMIT:-200}
SEED=${SATQUERY_BENCH_SEED:-1234}
MODEL=Qwen/Qwen3-VL-2B-Instruct
REVISION=89644892e4d85e24eaac8bacfd4f463576704203
PORT=${SATQUERY_VLM_PORT:-8000}

cd "$REPO"

echo "== staging data =="
mkdir -p "$DATA"
# The small splits come from S3; they were uploaded from the laptop, where they
# already were.
aws s3 sync "s3://${BUCKET}/datasets/CDVQA/" "$DATA/CDVQA/" --only-show-errors
aws s3 sync "s3://${BUCKET}/datasets/RSVQA/" "$DATA/RSVQA/" --only-show-errors

# VRSBench imagery is 3.8 GB and is pulled from source rather than from S3:
# HuggingFace-to-EC2 ingress is free and this link is far fatter than the one it
# was originally downloaded over. Pushed back to S3 afterwards so the next box
# syncs in-region instead of repeating the download.
if [ ! -d "$DATA/VRSBench/Images_val" ] || [ -z "$(ls -A "$DATA/VRSBench/Images_val" 2>/dev/null)" ]; then
    echo "== pulling VRSBench imagery from source =="
    python -m satquery.cli data pull vrsbench --with-images --data-root "$DATA"
    aws s3 sync "$DATA/VRSBench/" "s3://${BUCKET}/datasets/VRSBench/" --only-show-errors
else
    echo "   VRSBench imagery already present"
fi

echo
echo "== what the harness can see =="
# Run before spending a GPU-hour: a config whose images are missing scores zero
# and looks like model failure.
python -m satquery.cli bench validate --config "configs/bench/*.yaml" \
    | grep -E '"name"|"num_samples"|"images_missing"|"error"' || true

echo
echo "== starting vLLM =="
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --revision "$REVISION" \
    --served-model-name qwen3-vl-base \
    --max-model-len 8192 \
    --port "$PORT" \
    > "$REPO/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo -n "   waiting for the server"
for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo " up"
        break
    fi
    # A cold start compiles CUDA graphs and can take minutes; the log is where
    # an OOM or a bad revision actually shows up.
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo " died -- last lines of vllm.log:"
        tail -30 "$REPO/vllm.log"
        exit 1
    fi
    echo -n "."
    sleep 5
done
curl -s "http://127.0.0.1:${PORT}/v1/models" | python -c \
    "import sys,json;print('   serving:', [m['id'] for m in json.load(sys.stdin)['data']])"

export SATQUERY_VLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export SATQUERY_VLM_API_KEY=EMPTY

echo
echo "== benchmarks, limit=${LIMIT} seed=${SEED} =="
# Every config the repo declares, so a criterion is never quietly skipped.
# bigearthnet_bench is included and will report an error until the BigEarthNet
# split is prepared -- visible rather than absent, which is the point.
python -m satquery.cli bench run \
    --config "configs/bench/*.yaml" \
    --backend openai_compat \
    --model qwen3-vl-base \
    --limit "$LIMIT" \
    --seed "$SEED" \
    --out "$REPO/runs/baseline" \
    --results "$REPO/runs/results.csv"

echo
echo "== normalised aggregate =="
python -m satquery.cli bench score \
    --results "$REPO/runs/results.csv" \
    --baseline qwen3-vl-base \
    --json "$REPO/runs/baseline/scores.json"

echo
echo "== publishing =="
aws s3 sync "$REPO/runs/" "s3://${BUCKET}/results/baseline/" --only-show-errors
echo "   s3://${BUCKET}/results/baseline/"

cat <<NOTE

Commit runs/results.csv and runs/baseline/scores.json to git as well. Predictions
are large and regenerable; the results table is the source of the submission's
numbers and must not live only in a bucket.

Report with every number: n, the seed (${SEED}), the sample limit (${LIMIT}), the
prompt version, and the model revision (${REVISION:0:8}). A score without them
cannot be reproduced or defended, and the final evaluation is against a hidden
reference there is no arguing with.
NOTE
