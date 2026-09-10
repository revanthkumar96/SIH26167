#!/usr/bin/env bash
# The whole adaptation programme inside a fixed rented-GPU budget.
#
#   SATQUERY_HOURS=8 ./infra/run-programme.sh
#
# Written for vast.ai or Colab: one box, one session, a hard deadline, and no
# second attempt if the budget runs out with nothing to show.
#
# The order is chosen so the *comparison* survives a squeeze. Adaptation with no
# before-and-after on identical splits is an unverifiable claim, so the two
# benchmark sweeps are reserved first and training is given what remains --
# never the other way round. A shorter fine-tune with a measured delta is worth
# more than a longer one nobody can check.
#
#   1. stage data and weights          ~25 min
#   2. baseline sweep      (reserved)  ~40 min
#   3. train               (remainder) ~5 h at 8h budget
#   4. adapted sweep       (reserved)  ~40 min
#   5. merge and publish   (reserved)  ~30 min
#
# Every stage writes to S3 as it finishes, so an interruption loses one stage
# rather than the session.
set -euo pipefail

HOURS=${SATQUERY_HOURS:-8}
BUCKET=${SATQUERY_BUCKET:?set SATQUERY_BUCKET}
HF_REPO=${SATQUERY_HF_REPO:-}
LIMIT=${SATQUERY_BENCH_LIMIT:-200}
SEED=${SATQUERY_BENCH_SEED:-1234}
WORK=${SATQUERY_WORK:-$HOME/satquery}
PORT=${SATQUERY_VLM_PORT:-8000}
BASE=Qwen/Qwen3-VL-2B-Instruct
REV=89644892e4d85e24eaac8bacfd4f463576704203

# Fixed reservations, deducted before training is offered anything.
SETUP_H=0.45; BENCH_H=0.7; PUBLISH_H=0.5
TRAIN_H=$(awk -v h="$HOURS" -v s="$SETUP_H" -v b="$BENCH_H" -v p="$PUBLISH_H" 'BEGIN{t=h-s-2*b-p; printf "%.2f", (t<0.5?0.5:t)}')

START=$(date +%s)
elapsed() { awk -v s="$START" 'BEGIN{printf "%.2f", (systime()-s)/3600}'; }
step() { echo; echo "=== [$(elapsed)h/${HOURS}h] $* ==="; }

step "plan"
cat <<PLAN
  budget      ${HOURS} h
  training    ${TRAIN_H} h  (whatever is left after the reservations below)
  benchmarks  ${BENCH_H} h each, before and after
  publish     ${PUBLISH_H} h
  bench limit ${LIMIT} samples, seed ${SEED}
PLAN

mkdir -p "$WORK" && cd "$WORK"

# --- 1. stage -----------------------------------------------------------
step "staging corpus and weights"
pip install -q -e ".[geo]" 2>/dev/null || true
pip install -q "transformers==4.57.1" "peft==0.17.1" "accelerate==1.7.0" \
               "qwen-vl-utils==0.0.14" vllm

# One tarball, not 141,090 objects: individually GETting small files is the
# slowest part of the whole run.
if [ ! -d "$WORK/data/prepared/train" ]; then
    mkdir -p "$WORK/data/prepared"
    if aws s3 ls "s3://${BUCKET}/datasets/prepared-train.tar.gz" >/dev/null 2>&1; then
        aws s3 cp "s3://${BUCKET}/datasets/prepared-train.tar.gz" -             | tar -xz -C "$WORK/data/prepared/" &
    else
        # 141,090 individual GETs. Works, but it is the slowest thing here --
        # build the tarball ahead of time when there is a chance to.
        echo "   no tarball; syncing objects individually (slow)"
        aws s3 sync "s3://${BUCKET}/datasets/prepared/train/"             "$WORK/data/prepared/train/" --only-show-errors &
    fi
fi
aws s3 sync "s3://${BUCKET}/datasets/CDVQA/" "$WORK/data/CDVQA/" --only-show-errors &
aws s3 sync "s3://${BUCKET}/datasets/RSVQA/" "$WORK/data/RSVQA/" --only-show-errors &
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${BASE}', revision='${REV}')" &
wait
python3 -m satquery.cli data pull vrsbench --with-images --data-root "$WORK/data" || true

serve() {  # $1 = extra vLLM args
    python3 -m vllm.entrypoints.openai.api_server --model "$BASE" --revision "$REV" \
        --served-model-name qwen3-vl-base --max-model-len 8192 --port "$PORT" $1 \
        > "$WORK/vllm.log" 2>&1 &
    VLLM_PID=$!
    for _ in $(seq 1 150); do
        curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && return 0
        kill -0 $VLLM_PID 2>/dev/null || { tail -30 "$WORK/vllm.log"; return 1; }
        sleep 4
    done
    return 1
}
export SATQUERY_VLM_BASE_URL="http://127.0.0.1:${PORT}/v1" SATQUERY_VLM_API_KEY=EMPTY

sweep() {  # $1 = model name
    python3 -m satquery.cli bench run --config "configs/bench/*.yaml" \
        --backend openai_compat --model "$1" --limit "$LIMIT" --seed "$SEED" \
        --out "$WORK/runs/$1" --results "$WORK/runs/results.csv"
    aws s3 sync "$WORK/runs/" "s3://${BUCKET}/results/" --only-show-errors
}

# --- 2. baseline, before anything is trained ----------------------------
step "baseline sweep"
serve "" || { echo "vLLM failed to start"; exit 1; }
sweep qwen3-vl-base
kill $VLLM_PID 2>/dev/null || true; sleep 10

# --- 3. train on the remaining budget -----------------------------------
step "training, ${TRAIN_H}h budget"
python3 scripts/train_lora.py --stage a \
    --data "$WORK/data/prepared/train/train.jsonl" \
    --out "$WORK/runs/adapters/stage-a" \
    --max-hours "$TRAIN_H" --save-steps 100 \
    --s3-checkpoints "s3://${BUCKET}/checkpoints/stage-a/"
aws s3 sync "$WORK/runs/adapters/" "s3://${BUCKET}/adapters/" --only-show-errors

# --- 4. adapted sweep, identical harness --------------------------------
step "adapted sweep"
serve "--enable-lora --lora-modules qwen3-vl-satquery=$WORK/runs/adapters/stage-a" \
    || { echo "vLLM failed with the adapter"; exit 1; }
sweep qwen3-vl-satquery
kill $VLLM_PID 2>/dev/null || true

step "normalised delta"
python3 -m satquery.cli bench score --results "$WORK/runs/results.csv" \
    --baseline qwen3-vl-base --json "$WORK/runs/scores.json"
aws s3 sync "$WORK/runs/" "s3://${BUCKET}/results/" --only-show-errors

# --- 5. publish ---------------------------------------------------------
if [ -n "$HF_REPO" ]; then
    step "publishing to ${HF_REPO}"
    python3 scripts/publish_adapter.py --adapter "$WORK/runs/adapters/stage-a" \
        --repo "$HF_REPO" --results "$WORK/runs/results.csv" --merge
fi

step "done in $(elapsed)h"
echo "results  s3://${BUCKET}/results/"
echo "adapter  s3://${BUCKET}/adapters/"
cat <<'NOTE'

Report the delta with n, the seed, the sample limit, the prompt version and the
model revision. The adapter metadata records whether training stopped on the
time budget rather than finishing its epochs -- say so if it did. A short run
with an honest delta is defensible; a long one described as complete is not.
NOTE
