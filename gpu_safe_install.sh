#!/usr/bin/env bash
set -euo pipefail

echo "=== GPU SAFE INSTALL ==="

if ! command -v nvidia-smi &>/dev/null; then
    echo "❌ No NVIDIA GPU detected"
    exit 1
fi

CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "Detected CUDA: ${CUDA_VERSION}"

if [[ "${CUDA_VERSION}" == 12* ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "${CUDA_VERSION}" == 11* ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "⚠️ Unsupported CUDA version, using CPU fallback"
    pip install torch torchvision torchaudio
fi
