#!/usr/bin/env bash
# Isolated env for RF-DETR (deps conflict with the Ultralytics main venv).
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
echo ">> Creating projects/rfdetr/.venv"
$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# Blackwell/5090: cu128 torch first so rfdetr doesn't pull a non-Blackwell wheel.
echo ">> Installing torch (cu128 for RTX 5090)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo ">> Installing rfdetr + project deps"
pip install -r requirements.txt

python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())
try:
    import rfdetr; print("rfdetr", getattr(rfdetr, "__version__", "?"))
except Exception as e:
    print("rfdetr import FAILED:", e)
PY
echo "Done. Next: convert data, then train. See PROJECT.md."
