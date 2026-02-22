#!/bin/bash
# Start vLLM server for PaddleOCR-VL, then launch the FastAPI app.
# Usage: bash start.sh
#
# First-time setup on RunPod:
#   paddleocr install_genai_server_deps vllm
#
# This installs vLLM 0.10.2 with compatible CUDA libs.
# Do NOT add vLLM to pyproject.toml — it conflicts with PaddlePaddle GPU.

set -e

VLM_PORT="${VLM_SERVER_PORT:-8080}"

# Check if vLLM deps are installed
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "=== vLLM not found. Installing genai_server deps... ==="
    paddleocr install_genai_server_deps vllm
fi

echo "=== Starting vLLM server for PaddleOCR-VL-1.5 on port $VLM_PORT ==="
paddleocr genai_server \
    --model_name PaddleOCR-VL-1.5-0.9B \
    --backend vllm \
    --port "$VLM_PORT" &

VLM_PID=$!
echo "vLLM server PID: $VLM_PID"

# Cleanup on exit
trap "kill $VLM_PID 2>/dev/null; wait $VLM_PID 2>/dev/null" EXIT

# Wait for vLLM server to be ready (may take 30-90s to load model)
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
