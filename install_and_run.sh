#!/usr/bin/env bash
set -euo pipefail

MAX_RETRIES=3

retry_cmd() {
    local cmd="$1"
    local description="$2"
    local attempt=1

    while (( attempt <= MAX_RETRIES )); do
        echo "[$attempt/$MAX_RETRIES] ${description}"
        if eval "$cmd"; then
            return 0
        fi

        echo "⚠️ ${description} failed"
        (( attempt++ ))
        sleep 2
    done

    echo "❌ ${description} failed after ${MAX_RETRIES} attempts"
    return 1
}

echo "=== FULL DEFENSIVE PIPELINE START ==="

retry_cmd "python preflight_check.py" "Preflight checks" || exit 1

python -m venv venv
source venv/bin/activate

retry_cmd "bash gpu_safe_install.sh" "GPU-safe torch install" || exit 1

retry_cmd "git clone https://github.com/comfyanonymous/ComfyUI" "Clone ComfyUI" || exit 1
cd ComfyUI

retry_cmd "pip install -r requirements.txt" "Install ComfyUI requirements" || exit 1

cd custom_nodes
retry_cmd "git clone https://github.com/Kijai/ComfyUI-WanVideoWrapper" "Clone WanVideo wrapper" || exit 1
cd ..

python ../network_check.py

echo "🚀 Launching ComfyUI on 0.0.0.0:8188"
exec python main.py --listen 0.0.0.0 --port 8188
