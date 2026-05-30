#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data_root}"
BACKBONE="${BACKBONE:-resnet18}"
SEEDS="${SEEDS:-0 1 2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_DIR="${SAVE_DIR:-outputs/architecture}"
DATASETS="${DATASETS:-pacs}"
EPOCHS="${EPOCHS:-}"

read -r -a SEED_ARGS <<< "${SEEDS}"
read -r -a DATASET_ARGS <<< "${DATASETS}"

COMMON_ARGS=(
  --data_root "${DATA_ROOT}"
  --backbone "${BACKBONE}"
  --seeds "${SEED_ARGS[@]}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --save_dir "${SAVE_DIR}"
  --skip_existing
)

if [[ -n "${EPOCHS}" ]]; then
  COMMON_ARGS+=(--epochs "${EPOCHS}")
fi

for dataset in "${DATASET_ARGS[@]}"; do
  echo "Running architecture ablations for ${dataset}: seeds=${SEEDS}, backbone=${BACKBONE}"
  python architecture_baselines.py --dataset "${dataset}" "${COMMON_ARGS[@]}"
done

python summarize_results.py \
  --metrics_dir "${SAVE_DIR}" \
  --output_csv "outputs/architecture_results.csv"

echo "Architecture ablations finished."
echo "Checkpoints: ${SAVE_DIR}"
echo "Summary CSV: outputs/architecture_results.csv"
