#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data_root}"

echo "[1/3] Python environment"
python --version
python - <<'PY'
import importlib.util

required = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "tqdm": "tqdm",
}
missing = [pkg for module, pkg in required.items() if importlib.util.find_spec(module) is None]
if missing:
    raise SystemExit("Missing packages: " + ", ".join(missing))

import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("gpu_count:", torch.cuda.device_count())
PY

echo "[2/3] Dataset layout under ${DATA_ROOT}"
python tools/check_data.py --data_root "${DATA_ROOT}" --dataset pacs
python tools/check_data.py --data_root "${DATA_ROOT}" --dataset vlcs
python tools/check_data.py --data_root "${DATA_ROOT}" --dataset office_home

echo "[3/3] Ready"
