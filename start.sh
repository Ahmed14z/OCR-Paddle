#!/bin/bash
# Start vLLM server for PaddleOCR-VL, then launch the FastAPI app.
# Usage: bash start.sh

set -e

VLM_PORT="${VLM_SERVER_PORT:-8080}"

echo "=== Starting vLLM server for PaddleOCR-VL-1.5 on port $VLM_PORT ==="
paddleocr genai_server \
    --model_name PaddleOCR-VL-1.5-0.9B \
    --backend vllm \
    --port "$VLM_PORT" &

VLM_PID=$!
echo "vLLM server PID: $VLM_PID"

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to start..."
for i in $(seq 1 60); do
    if curl -s "http://127.0.0.1:$VLM_PORT/v1/models" > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died"
        exit 1
    fi
    sleep 1
done

# Verify it's actually up
if ! curl -s "http://127.0.0.1:$VLM_PORT/v1/models" > /dev/null 2>&1; then
    echo "WARNING: vLLM server may not be ready yet, starting app anyway..."
fi

echo "=== Starting FastAPI app ==="
uv run poe serve

# Cleanup on exit
trap "kill $VLM_PID 2>/dev/null" EXIT
