#!/bin/bash
# Start vLLM server for PaddleOCR-VL, then launch the FastAPI app.
#
# vLLM runs with system Python/PyTorch (NOT in the uv venv).
# The FastAPI app runs in the uv venv and connects via HTTP.
#
# First-time setup on RunPod:
#   pip install vllm
#
# Usage: bash start.sh

set -e

VLM_PORT="${VLM_SERVER_PORT:-8080}"

echo "=== Starting vLLM server for PaddleOCR-VL-1.5 on port $VLM_PORT ==="

# Run vLLM with system python — completely outside the uv venv
vllm serve PaddlePaddle/PaddleOCR-VL-1.5 \
    --port "$VLM_PORT" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.3 &

VLM_PID=$!
echo "vLLM server PID: $VLM_PID"

# Cleanup on exit
trap "kill $VLM_PID 2>/dev/null; wait $VLM_PID 2>/dev/null" EXIT

# Wait for vLLM server to be ready (model loading takes 30-90s)
echo "Waiting for vLLM server to start (this may take a minute)..."
for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:$VLM_PORT/v1/models" > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died. Check logs above."
        echo ""
        echo "If vLLM fails, you can still run without it:"
        echo "  Add VLM_BACKEND=local to .env"
        echo "  uv run poe serve"
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://127.0.0.1:$VLM_PORT/v1/models" > /dev/null 2>&1; then
    echo "WARNING: vLLM server not ready after 120s, starting app anyway..."
fi

echo "=== Starting FastAPI app ==="
uv run poe serve
