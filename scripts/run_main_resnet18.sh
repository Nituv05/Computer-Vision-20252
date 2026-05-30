#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data_root}"
BACKBONE="${BACKBONE:-resnet18}"
SEEDS="${SEEDS:-0 1 2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_DIR="${SAVE_DIR:-outputs/checkpoints}"
METHODS="${METHODS:-erm m2 m2cl}"
DATASETS="${DATASETS:-pacs vlcs office_home}"
EPOCHS="${EPOCHS:-}"
LR="${LR:-}"

read -r -a SEED_ARGS <<< "${SEEDS}"
read -r -a METHOD_ARGS <<< "${METHODS}"
read -r -a DATASET_ARGS <<< "${DATASETS}"

COMMON_ARGS=(
  --data_root "${DATA_ROOT}"
  --methods "${METHOD_ARGS[@]}"
  --backbones "${BACKBONE}"
  --seeds "${SEED_ARGS[@]}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --save_dir "${SAVE_DIR}"
  --skip_existing
)

if [[ -n "${EPOCHS}" ]]; then
  COMMON_ARGS+=(--epochs "${EPOCHS}")
fi

if [[ -n "${LR}" ]]; then
  COMMON_ARGS+=(--lr "${LR}")
fi

for dataset in "${DATASET_ARGS[@]}"; do
  echo "Running ${dataset}: methods=${METHODS}, seeds=${SEEDS}, backbone=${BACKBONE}"
  python run_experiments.py --dataset "${dataset}" "${COMMON_ARGS[@]}"
done

python summarize_results.py \
  --metrics_dir "${SAVE_DIR}" \
  --output_csv "outputs/main_results.csv"

echo "Main experiments finished."
echo "Checkpoints: ${SAVE_DIR}"
echo "Summary CSV: outputs/main_results.csv"
