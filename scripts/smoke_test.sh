#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data_root}"
BACKBONE="${BACKBONE:-resnet18}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_DIR="${SAVE_DIR:-outputs/smoke}"

mkdir -p "${SAVE_DIR}"

echo "[1/2] Smoke test: ERM, PACS/photo, 1 epoch"
python tools/train.py \
  --dataset pacs \
  --test_domain photo \
  --data_root "${DATA_ROOT}" \
  --method erm \
  --backbone "${BACKBONE}" \
  --epochs 1 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --save_dir "${SAVE_DIR}"

echo "[2/2] Smoke test: M2-CL, PACS/photo, 1 epoch"
python tools/train.py \
  --dataset pacs \
  --test_domain photo \
  --data_root "${DATA_ROOT}" \
  --method m2cl \
  --backbone "${BACKBONE}" \
  --epochs 1 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --save_dir "${SAVE_DIR}"

python tools/summarize_results.py \
  --metrics_dir "${SAVE_DIR}" \
  --output_csv "${SAVE_DIR}/smoke_results.csv"

echo "Smoke test finished. Outputs: ${SAVE_DIR}"
