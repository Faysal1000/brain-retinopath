#!/usr/bin/env bash
# Create the virtual environment in .venv and install everything. Safe to run again.
#   bash scripts/setup.sh                     macOS (Apple silicon) or Linux
#   PYTHON=python3.12 bash scripts/setup.sh   use a specific interpreter
# Linux installs the CUDA 13 build of PyTorch (aarch64 and x86_64 wheels exist); macOS installs the default build (MPS).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
command -v "$PY" >/dev/null || { echo "Python not found. Linux: sudo apt install -y python3 python3-venv python3-pip   macOS: brew install python@3.12"; exit 1; }
echo "== $($PY --version) on $(uname -s) $(uname -m)"
$PY -c 'import sys; assert (3, 12) <= sys.version_info[:2] <= (3, 13), "need Python 3.12 or 3.13"'

[ -d .venv ] || $PY -m venv .venv || { rm -rf .venv; echo "venv failed. Linux: sudo apt install -y python3-venv"; exit 1; }
.venv/bin/pip install --quiet --upgrade pip

if [ "$(uname -s)" = "Linux" ]; then
  .venv/bin/pip install torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
  .venv/bin/pip install -r requirements-dgx.txt
else
  .venv/bin/pip install -r requirements.txt
fi
.venv/bin/pip install -e .
.venv/bin/python -m ipykernel install --user --name retinopath --display-name retinopath

.venv/bin/python - <<'PY'
import torch, dipy
from retinopath.core import get_device
dev = get_device()
print("torch", torch.__version__, "| dipy", dipy.__version__, "| device:", dev)
if dev.type == "cuda":
    for i in range(torch.cuda.device_count()):
        print("  GPU", i, torch.cuda.get_device_name(i), round(torch.cuda.mem_get_info(i)[1] / 1e9), "GB, CUDA", torch.version.cuda)
x = torch.randn(1, 8, 16, 16, 16, device=dev)
torch.nn.Conv3d(8, 8, 3).to(dev)(x).sum().backward()
print("  3D convolution forward/backward OK on", dev)
PY
[ -f .env ] || echo "Next: cp .env.example .env  and add your HCP keys (docs/data.md)"
