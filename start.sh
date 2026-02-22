#!/bin/bash
# Start vLLM server + FastAPI app.
# Runs setup automatically if needed.
#
# Usage: bash start.sh

set -e

# Run setup (idempotent — skips already-done steps)
bash "$(dirname "$0")/setup-runpod.sh"

VLM_PORT="${VLM_SERVER_PORT:-8080}"

echo ""
echo "=== Starting vLLM server for PaddleOCR-VL-1.5 on port $VLM_PORT ==="

vllm serve PaddlePaddle/PaddleOCR-VL-1.5 \
    --port "$VLM_PORT" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.3 \
    --trust-remote-code &

VLM_PID=$!
echo "vLLM server PID: $VLM_PID"

trap "kill $VLM_PID 2>/dev/null; wait $VLM_PID 2>/dev/null" EXIT

echo "Waiting for vLLM server to start (this may take a minute)..."
for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:$VLM_PORT/v1/models" > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died. Check logs above."
        echo ""
        echo "Fallback: add VLM_BACKEND=local to .env, then run: uv run poe serve"
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://127.0.0.1:$VLM_PORT/v1/models" > /dev/null 2>&1; then
    echo "WARNING: vLLM server not ready after 120s, starting app anyway..."
fi

echo ""
echo "=== Starting FastAPI app ==="
uv run poe serve
