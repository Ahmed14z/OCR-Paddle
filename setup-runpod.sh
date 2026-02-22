#!/bin/bash
# One-time setup for RunPod environment.
# Idempotent — safe to run multiple times, skips already-done steps.
#
# Usage: bash setup-runpod.sh

set -e

echo "=== RunPod OCR Setup ==="

# Fix hf_transfer issue (RunPod enables it but doesn't install it)
if ! python3 -c "import hf_transfer" 2>/dev/null; then
    echo "[1/5] Installing hf_transfer..."
    pip install hf_transfer
else
    echo "[1/5] hf_transfer already installed"
fi

# Install vLLM system-wide (uses RunPod's PyTorch)
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "[2/5] Installing vLLM..."
    pip install vllm
else
    echo "[2/5] vLLM already installed"
fi

# Install pip in the uv venv (needed by paddleocr internals)
if ! .venv/bin/python -m pip --version 2>/dev/null; then
    echo "[3/5] Installing pip in uv venv..."
    uv pip install pip
else
    echo "[3/5] pip already in uv venv"
fi

# Pre-download the PaddleOCR-VL-1.5 model for vLLM
if python3 -c "from huggingface_hub import try_to_load_from_cache; assert try_to_load_from_cache('PaddlePaddle/PaddleOCR-VL-1.5', 'config.json') is not None" 2>/dev/null; then
    echo "[4/5] PaddleOCR-VL-1.5 model already downloaded"
else
    echo "[4/5] Downloading PaddleOCR-VL-1.5 model (~2GB)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('PaddlePaddle/PaddleOCR-VL-1.5')
print('Model downloaded successfully')
"
fi

# Ensure .env doesn't override hardcoded DPI/VLM settings
echo "[5/5] Cleaning .env overrides..."
sed -i '/^PDF_DPI=/d; /^VLM_MAX_PIXELS=/d' .env 2>/dev/null || true

echo "=== Setup complete ==="
