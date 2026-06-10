#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data_root}"
BACKBONE="${BACKBONE:-resnet18}"
SEEDS="${SEEDS:-0 1 2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_DIR="${SAVE_DIR:-outputs/ablation_tables}"
DATASETS="${DATASETS:-pacs vlcs}"
TABLES="${TABLES:-all}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-0.001}"

read -r -a SEED_ARGS <<< "${SEEDS}"
read -r -a DATASET_ARGS <<< "${DATASETS}"
read -r -a TABLE_ARGS <<< "${TABLES}"

python tools/run_ablations.py \
  --tables "${TABLE_ARGS[@]}" \
  --datasets "${DATASET_ARGS[@]}" \
  --data_root "${DATA_ROOT}" \
  --backbone "${BACKBONE}" \
  --seeds "${SEED_ARGS[@]}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --hparams_profile paper \
  --holdout_fraction 0.2 \
  --scheduler none \
  --num_workers "${NUM_WORKERS}" \
  --save_dir "${SAVE_DIR}" \
  --skip_existing

python tools/summarize_results.py \
  --metrics_dir "${SAVE_DIR}" \
  --output_csv "${SAVE_DIR}/results.csv"

echo "Ablations finished."
echo "Checkpoints: ${SAVE_DIR}"
echo "Summary CSV: ${SAVE_DIR}/results.csv"
