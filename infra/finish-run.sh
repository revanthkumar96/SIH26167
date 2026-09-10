#!/bin/bash
# Re-score both rows against one parser, then publish.
#
# The baseline was measured before the bounding-box parser was fixed, and the
# adapted sweep after. Comparing across that change would not be a comparison,
# so both sweeps are simply redone into a fresh results file -- each costs about
# two minutes, which is far cheaper than an argument about whether the delta is
# real.
#
# Run after run.sh finishes: it needs the merged model to exist.
exec > >(tee /workspace/logs/finish.log) 2>&1
set -x

cd /workspace/satquery
BUCKET=satquery-869987460914
PY=/opt/train-env/bin/python
LIMIT=${LIMIT:-200}
BASE=Qwen/Qwen3-VL-2B-Instruct
FINAL=/workspace/runs/final.csv

mark() {
    echo "$1" | aws s3 cp - "s3://${BUCKET}/results/_STATUS.txt" || true
    echo "### $1 at $(date -u +%H:%M:%S)"
}

rm -f "$FINAL"

mark "final baseline sweep (fixed parser)"
$PY -m satquery.cli bench run --config "configs/bench/*.yaml" \
    --backend hf --model "$BASE" --limit "$LIMIT" --seed 1234 \
    --out /workspace/runs/final-base --results "$FINAL"

mark "final adapted sweep (fixed parser)"
$PY -m satquery.cli bench run --config "configs/bench/*.yaml" \
    --backend hf --model /workspace/runs/merged --limit "$LIMIT" --seed 1234 \
    --out /workspace/runs/final-adapted --results "$FINAL"

mark "scoring"
$PY -m satquery.cli bench score --results "$FINAL" \
    --baseline "$BASE" --json /workspace/runs/scores.json \
    | tee /workspace/runs/delta.txt

# Parse and truncation rates are defects the aggregate hides, so they are pulled
# out explicitly rather than left for someone to notice.
$PY - <<'PYEOF' | tee /workspace/runs/quality.txt
import json, pathlib
for run in ("final-base", "final-adapted"):
    root = pathlib.Path("/workspace/runs") / run
    print(f"== {run}")
    for cell in sorted(root.rglob("predictions.jsonl")):
        rows = [json.loads(x) for x in cell.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not rows:
            continue
        empty = sum(1 for r in rows if not (r.get("raw_text") or "").strip())
        parse_fail = sum(1 for r in rows if r.get("bbox") is None and r.get("reference_bbox"))
        print(f"   {cell.parent.name:<22} n={len(rows):<5} empty={empty:<4} unparsed_box={parse_fail}")
PYEOF

aws s3 sync /workspace/runs/ "s3://${BUCKET}/results/" \
    --exclude "*predictions.jsonl" --exclude "merged/*" --only-show-errors

if [ -n "${HF_REPO:-}" ]; then
    mark "publishing ${HF_REPO}"
    # The merged weights are what was benchmarked, so they are what is published.
    $PY scripts/publish_adapter.py \
        --adapter /workspace/runs/adapters/stage-a \
        --repo "$HF_REPO" --results "$FINAL" \
        --baseline-model "$BASE" --adapted-model /workspace/runs/merged \
        --merge --merge-dir /workspace/runs/merged
fi

mark "finished"
touch /workspace/logs/FINISH_DONE
