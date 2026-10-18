#!/usr/bin/env bash
# Run the Docker image on all three spot-check datasets, then score the runs.
#
# SCRIPT=predict_v2.py selects version 2.
# Needs an OpenAI-compatible endpoint reachable from Docker, e.g. Ollama:
#   OPENAI_BASE_URL=http://host.docker.internal:11434/v1 OPENAI_MODEL=qwen2.5:7b ./run_local.sh
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-ucuagent-task2}"
SCRIPT="${SCRIPT:-predict.py}"   # or predict_v2.py
OUT="${OUT:-runs/$(date +%Y%m%d-%H%M%S)}"
: "${OPENAI_BASE_URL:?set OPENAI_BASE_URL}"

runs=()
for dataset in business-trip-spot-check task-2-procurement-spot-check task-2-expense-reimbursement-spot-check; do
  mkdir -p "$OUT/$dataset"
  echo "=== $dataset -> $OUT/$dataset"
  docker run --rm \
    --env OPENAI_BASE_URL --env OPENAI_API_KEY --env OPENAI_MODEL --env OPENAI_REASONING_EFFORT --env PROMPT_VARIANT --env CASE_BUDGET_SECONDS --env RUN_BUDGET_SECONDS \
    --volume "$PWD/datasets/$dataset-inputs:/input:ro" \
    --volume "$PWD/$OUT/$dataset:/output" \
    --entrypoint "/app/$SCRIPT" "$IMAGE" --input /input --output /output
  runs+=("$OUT/$dataset")
done

python3 evaluate_local.py "${runs[@]}" | tee "$OUT/evaluation.txt"
