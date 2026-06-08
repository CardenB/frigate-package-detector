#!/usr/bin/env bash
# One-time environment setup. Creates a venv, installs a Blackwell-compatible
# torch (cu128) for the RTX 5090, then the rest of the deps.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
echo ">> Creating venv (.venv) with $PY"
$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel

echo ">> Installing torch (CUDA 12.8 wheels for RTX 5090 / Blackwell sm_120)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo ">> Installing project requirements"
pip install -r requirements.txt

echo ">> Verifying GPU is visible to torch"
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo
echo "Done. Next:"
echo "  source .venv/bin/activate"
echo "  cp .env.example .env   # add ROBOFLOW_API_KEY"
echo "  make data && make analyze && make train && make export"
