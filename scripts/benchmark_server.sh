#!/usr/bin/env bash
set -euo pipefail

# Run vLLM's online benchmark inside the already-running server container.
# The model server must be healthy before this script is called.
ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
COMPOSE_FILE=${COMPOSE_FILE:-"$ROOT_DIR/qwen3.6_35b-int8/docker-compose.yml"}
SERVICE=${SERVICE:-vllm-35b-int8}
CONTAINER=${CONTAINER:-vllm-35b-int8}
RESULT_DIR=${RESULT_DIR:-"$ROOT_DIR/benchmarks"}
RESULT_FILENAME=${RESULT_FILENAME:-single.json}
NUM_PROMPTS=${NUM_PROMPTS:-1}
NUM_WARMUPS=${NUM_WARMUPS:-0}
INPUT_LEN=${INPUT_LEN:-256}
OUTPUT_LEN=${OUTPUT_LEN:-128}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-1}

mkdir -p "$RESULT_DIR"

if ! curl --fail --silent --show-error --max-time 10 \
  http://127.0.0.1:8000/health >/dev/null; then
  echo "vLLM health check failed; wait for the 10–15 minute startup and retry." >&2
  exit 1
fi

docker compose -f "$COMPOSE_FILE" exec -T "$SERVICE" \
  vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model qwen3.6-35b-a3b-int8 \
  --tokenizer /models/Qwen3.6-35B-A3B-tokenizer \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --random-range-ratio 0 \
  --seed 0 \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --request-rate inf \
  --num-warmups "$NUM_WARMUPS" \
  --ready-check-timeout-sec 0 \
  --temperature 0 \
  --disable-tqdm \
  --save-result \
  --save-detailed \
  --result-dir /tmp/qwen-bench \
  --result-filename "$RESULT_FILENAME"

docker cp "$CONTAINER:/tmp/qwen-bench/$RESULT_FILENAME" \
  "$RESULT_DIR/$RESULT_FILENAME"
printf 'Saved %s\n' "$RESULT_DIR/$RESULT_FILENAME"
