#!/bin/bash
# Start vLLM server for PaddleOCR-VL, then launch the FastAPI app.
#
# vLLM runs system-wide (uses system PyTorch from RunPod image).
# The FastAPI app runs in the uv venv and connects via HTTP.
#
# First-time setup on RunPod:
#   pip install vllm
#
# Usage: bash start.sh

set -e

VLM_PORT="${VLM_SERVER_PORT:-8080}"

# Check if vLLM is available system-wide
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "=== vLLM not found. Installing system-wide... ==="
    pip install vllm
fi

echo "=== Starting vLLM server for PaddleOCR-VL-1.5 on port $VLM_PORT ==="

# Use system python (not uv run) for the vLLM server
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
paddleocr genai_server \
    --model_name PaddleOCR-VL-1.5-0.9B \
    --backend vllm \
    --port "$VLM_PORT" &

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
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://127.0.0.1:$VLM_PORT/v1/models" > /dev/null 2>&1; then
    echo "WARNING: vLLM server not ready after 120s, starting app anyway..."
fi

echo "=== Starting FastAPI app ==="
uv run poe serve
